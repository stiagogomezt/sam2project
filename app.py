"""
=============================================================================
SAM 2 GeoExtractor: Extracción Inteligente de Objetos Geoespaciales con IA
=============================================================================
Flujo de trabajo (5 etapas / 10 pasos):
  1. Zona de estudio -> 2. Imagen -> 3. Segmentar y refinar -> 4. Validar -> 5. Analizar y exportar

La lógica vive en módulos independientes de la interfaz (`src/` y `geosam/`); este archivo solo
orquesta el estado de sesión y los widgets.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_folium import st_folium

from config import (
    DEFAULT_BBOX,
    DEFAULT_ZOOM,
    MIN_ZOOM,
    MAX_ZOOM,
    SAM2_MODELS,
    DEFAULT_SAM2_MODEL_KEY,
    SAM2_INPUT_SIZE,
    CRS_WEB_MERCATOR,
)
from src.tile_manager import get_tiles_info, TileManagerError
from src.imagery_loader import download_tiles_with_report
from src.mosaic_builder import build_mosaic
from src.sam_engine import (
    get_sam2_engine,
    ensure_checkpoint,
    is_checkpoint_downloaded,
    can_load_model,
    max_models_in_ram,
)
from src.segmentation import coordinates_to_pixel, pixel_to_coordinates, extract_cutout, create_mask_overlay
from src.geospatial import mask_to_geodataframe, derived_geometries, spatial_relations
from src.interactive import InteractiveSegmenter
from src.postprocess import refine_mask, touches_border
from src.semantic import CATEGORIES, REMOTECLIP_INFO, SemanticClassifier, classifier_available, object_crop
from src.objects import ObjectCatalog, STATUSES
from src.visualization import create_base_map, build_dynamic_layers
from src.exporter import (
    export_geojson,
    export_geotiff,
    export_png,
    export_shapefile_zip,
    export_csv,
    export_geopackage,
    export_label_geotiff,
    export_coco_dataset,
)
from src.ui import (
    EXPORT_DESCRIPTIONS,
    inject_global_css,
    render_topbar,
    render_stepper,
    render_tiles_kpis,
    render_model_card,
    render_map_legend,
    render_empty_inspector,
)

st.set_page_config(
    page_title="SAM 2 GeoExtractor",
    page_icon=":material/satellite_alt:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

MODE_LABELS = {"pos": "Incluir", "neg": "Excluir", "box": "Caja (2 clics)"}
STATUS_LABELS = {"aceptado": "aceptado", "propuesto": "propuesto", "rechazado": "rechazado"}

# =============================================================================
# ESTADO DE SESIÓN
# =============================================================================
_DEFAULTS = {
    "mosaic_data": None,        # img_array, transforms, bounds (sin PIL duplicado)
    "emb": None,                # ImageEmbedding de ESTA sesión (no se comparte entre usuarios)
    "tile_failures": [],
    "seg": None,                # InteractiveSegmenter del objeto en edición
    "catalog": None,            # ObjectCatalog
    "editing_object_id": None,  # objeto del catálogo que se está refinando
    "last_seen_click": None,    # último clic recibido del mapa (evita re-disparos)
    "map_center": [
        (DEFAULT_BBOX["min_lat"] + DEFAULT_BBOX["max_lat"]) / 2,
        (DEFAULT_BBOX["min_lon"] + DEFAULT_BBOX["max_lon"]) / 2,
    ],
    "map_zoom": DEFAULT_ZOOM,
    "focus_object_id": None,
    "label_default": "Sin categoría",
    "suggestions": None,        # (clave de candidata, [(categoría, prob), ...]) de RemoteCLIP
    "suggested_pick": None,     # categoría sugerida que el usuario adoptó
    "cand_cache": None,
    "active_model_key": DEFAULT_SAM2_MODEL_KEY,
    "last_model_timings": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if st.session_state.seg is None:
    st.session_state.seg = InteractiveSegmenter()
if st.session_state.catalog is None:
    st.session_state.catalog = ObjectCatalog()

seg: InteractiveSegmenter = st.session_state.seg
catalog: ObjectCatalog = st.session_state.catalog


def _reset_editing() -> None:
    st.session_state.seg = InteractiveSegmenter(refine_with_logits=st.session_state.seg.refine_with_logits)
    st.session_state.editing_object_id = None
    st.session_state.cand_cache = None
    st.session_state.suggestions = None
    st.session_state.suggested_pick = None


@st.cache_resource(show_spinner=False, max_entries=1)
def get_semantic_classifier() -> SemanticClassifier:
    ensure_checkpoint(REMOTECLIP_INFO)
    return SemanticClassifier()


def shape_options() -> dict:
    """Opciones de forma de la huella (widgets del panel; se leen antes de dibujar el mapa)."""
    return {
        "fill_holes": bool(st.session_state.get("shape_fill", False)),
        "close_px": float(st.session_state.get("shape_close", 0)),
        "keep_largest": bool(st.session_state.get("shape_largest", False)),
    }


def _ensure_active_engine(model_key: str):
    """Carga el motor activo sin bloquear toda la pantalla si falta descargar pesos."""
    allowed, reason = can_load_model(model_key)
    if not allowed:
        raise RuntimeError(reason)
    info = SAM2_MODELS[model_key]
    if not is_checkpoint_downloaded(info):
        # La primera carga (solo si falta el .pt) muestra descarga paso a paso en st.status
        with st.status(f"Preparando {info['name']}...", expanded=True) as status:
            st.write(f"Descargando checkpoint ({info['size_mb']} MB)...")
            pbar = st.progress(0, text=f"Descargando {info['name']}: 0.0 / {info['size_mb']} MB")
            ensure_checkpoint(
                info,
                progress_callback=lambda d, t: pbar.progress(
                    min(100, int(d * 100 / t)) if t else 0,
                    text=f"Descargando {info['name']}: {d / 1e6:.1f} / {(t / 1e6) if t else info['size_mb']:.1f} MB",
                ),
            )
            st.write("Inicializando arquitectura y pesos en memoria...")
            eng = get_sam2_engine(model_key=model_key)
            status.update(label=f"{info['name']} listo ({eng.load_time:.2f} s)", state="complete", expanded=False)
            return eng
    return get_sam2_engine(model_key=model_key)


def _uniform_candidate_thumb(img: np.ndarray, mask: np.ndarray, crop_box: tuple[int, int, int, int], size: int = 180) -> np.ndarray:
    """Genera una miniatura cuadrada uniforme para comparar candidatas de forma consistente."""
    x0, y0, x1, y1 = crop_box
    overlay = create_mask_overlay(img[y0:y1, x0:x1], mask[y0:y1, x0:x1])
    h, w = overlay.shape[:2]
    side = max(h, w, 1)
    canvas = np.full((side, side, 3), 24, dtype=np.uint8)
    oy = (side - h) // 2
    ox = (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = overlay
    pil_thumb = Image.fromarray(canvas).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil_thumb)


# =============================================================================
# MOTOR ACTIVO Y BARRA LATERAL
# =============================================================================
engine = None
engine_error = None
try:
    engine = _ensure_active_engine(st.session_state.active_model_key)
except Exception as exc:
    engine_error = str(exc)

with st.sidebar:
    st.markdown("#### :material/memory: 1. Motor de IA")
    model_keys = list(SAM2_MODELS.keys())
    selected_model_key = st.selectbox(
        "Modelo SAM 2:",
        model_keys,
        index=model_keys.index(st.session_state.active_model_key),
        format_func=lambda k: SAM2_MODELS[k]["name"],
        help="Seleccionar una variante muestra su ficha técnica sin recargar el motor hasta pulsar 'Cargar modelo'.",
    )
    sel_info = SAM2_MODELS[selected_model_key]
    sel_downloaded = is_checkpoint_downloaded(sel_info)
    sel_is_active = selected_model_key == st.session_state.active_model_key and engine is not None
    render_model_card(sel_info, downloaded=sel_downloaded, is_active=sel_is_active)

    allowed_sel, deny_reason = can_load_model(selected_model_key)
    if not allowed_sel and deny_reason:
        st.caption(f":material/warning: {deny_reason}")

    btn_load_model = st.button(
        "Cargar modelo",
        icon=":material/model_training:",
        width="stretch",
        disabled=(not allowed_sel) or sel_is_active,
        help="Descarga los pesos (si faltan), inicializa el modelo y recalcula el embedding si ya hay ortofoto.",
    )
    if btn_load_model:
        try:
            with st.status(f"Cargando {sel_info['name']}...", expanded=True) as status:
                t_dl0 = time.perf_counter()
                if not is_checkpoint_downloaded(sel_info):
                    lbl_dl = st.empty()
                    pbar_dl = st.progress(0, text=f"Descargando {sel_info['name']}: 0.0 / {sel_info['size_mb']} MB")

                    def _dl_cb(done_b: int, total_b: int) -> None:
                        done_mb = done_b / 1e6
                        tot_mb = (total_b / 1e6) if total_b else float(sel_info["size_mb"])
                        pct = min(100, int(done_b * 100 / total_b)) if total_b else 0
                        msg = f"Descargando {sel_info['name']}: {done_mb:.1f} / {tot_mb:.1f} MB"
                        lbl_dl.caption(msg)
                        pbar_dl.progress(pct, text=msg)

                    ensure_checkpoint(sel_info, progress_callback=_dl_cb)
                    dl_s = time.perf_counter() - t_dl0
                    st.write(f":material/check_circle: Descarga completada en {dl_s:.1f} s")
                else:
                    dl_s = 0.0
                    st.write(":material/check_circle: Checkpoint verificado en disco")

                t_ld0 = time.perf_counter()
                new_engine = get_sam2_engine(model_key=selected_model_key)
                ld_s = time.perf_counter() - t_ld0
                st.write(f":material/check_circle: Pesos cargados en {new_engine.device_name} ({ld_s:.2f} s)")

                emb_s = 0.0
                if st.session_state.mosaic_data is not None and not new_engine.is_compatible(st.session_state.emb):
                    st.write("Calculando embedding para la ortofoto actual...")
                    new_emb, emb_s = new_engine.embed(st.session_state.mosaic_data["img_array"])
                    st.session_state.emb = new_emb
                    _reset_editing()
                    st.write(f":material/check_circle: Embedding actualizado en {emb_s:.2f} s")

                st.session_state.active_model_key = selected_model_key
                st.session_state.last_model_timings = {
                    "model_name": sel_info["name"],
                    "download_s": dl_s,
                    "load_s": ld_s,
                    "embed_s": emb_s,
                }
                status.update(
                    label=f"{sel_info['name']} activo ({ld_s:.2f} s carga)",
                    state="complete",
                    expanded=False,
                )
            st.toast(f"Modelo activo: {sel_info['name']}", icon=":material/check_circle:")
            st.rerun()
        except Exception as exc:
            st.error(f"No se pudo cargar {sel_info['name']}: {exc}")

    if st.session_state.last_model_timings:
        tm = st.session_state.last_model_timings
        parts = [f"carga {tm['load_s']:.2f} s"]
        if tm["download_s"] > 0:
            parts.insert(0, f"descarga {tm['download_s']:.1f} s")
        if tm["embed_s"] > 0:
            parts.append(f"embedding {tm['embed_s']:.2f} s")
        st.caption(f"Último cambio ({tm['model_name']}): " + " · ".join(parts))

    if engine_error:
        st.error(f"Error en motor activo: {engine_error}")
    st.caption(f"Caché de modelos en RAM: máx. {max_models_in_ram()} (`GEOSAM_MAX_MODELS`).")

    st.markdown("---")

    # 2. Zona de estudio -------------------------------------------------------
    st.markdown("#### :material/map: 2. Zona de estudio (WGS84)")
    c1, c2 = st.columns(2)
    with c1:
        min_lon = st.number_input("Oeste (min lon)", value=DEFAULT_BBOX["min_lon"], format="%.5f", step=0.001)
        min_lat = st.number_input("Sur (min lat)", value=DEFAULT_BBOX["min_lat"], format="%.5f", step=0.001)
    with c2:
        max_lon = st.number_input("Este (max lon)", value=DEFAULT_BBOX["max_lon"], format="%.5f", step=0.001)
        max_lat = st.number_input("Norte (max lat)", value=DEFAULT_BBOX["max_lat"], format="%.5f", step=0.001)
    zoom_level = st.slider("Zoom de teselas:", min_value=MIN_ZOOM, max_value=MAX_ZOOM, value=DEFAULT_ZOOM)

    tiles_estimate = None
    try:
        tiles_estimate = get_tiles_info(min_lon, min_lat, max_lon, max_lat, zoom=zoom_level)
        h_px, w_px = tiles_estimate["pixel_shape"]
        eff_res = tiles_estimate["res_m_per_px"] * max(1.0, max(h_px, w_px) / SAM2_INPUT_SIZE)
        render_tiles_kpis(
            n_tiles=tiles_estimate["num_tiles"],
            width_px=w_px,
            height_px=h_px,
            sam_px=SAM2_INPUT_SIZE,
            gs_m=eff_res,
        )
        if max(h_px, w_px) > SAM2_INPUT_SIZE * 1.5 or tiles_estimate["is_large"]:
            st.warning(
                "Área amplia: SAM 2 escala a 1024 px y el cálculo tardará más. "
                "Reduzca extensión o zoom para objetos pequeños."
            )
    except TileManagerError as exc:
        st.error(f"Zona inválida: {exc}")

    confirm_reset = True
    if len(catalog) > 0:
        confirm_reset = st.checkbox(
            f"Reiniciar el catálogo actual ({len(catalog)} objetos) al cargar otra ortofoto"
        )
    download_disabled = tiles_estimate is None or engine is None or not confirm_reset
    btn_download = st.button(
        "Descargar y cargar ortofoto",
        icon=":material/satellite_alt:",
        type="primary",
        width="stretch",
        disabled=download_disabled,
    )
    if download_disabled:
        if tiles_estimate is None:
            st.caption("Corrija la zona de estudio para habilitar la descarga.")
        elif engine is None:
            st.caption("Cargue primero un modelo SAM 2 válido.")
        elif not confirm_reset:
            st.caption("Marque la casilla superior para reemplazar el catálogo actual.")

    if btn_download and tiles_estimate and engine:
        with st.status("Descargando ortofoto e indexando con SAM 2...", expanded=True) as dl_status:
            progress = st.progress(0, text="Descargando teselas...")

            def _progress(done, total):
                progress.progress(int(done / total * 100), text=f"Teselas: {done}/{total}")

            try:
                t_dl0 = time.perf_counter()
                tile_images, failed = download_tiles_with_report(tiles_estimate["tiles"], progress_callback=_progress)
                if len(failed) == len(tiles_estimate["tiles"]):
                    raise RuntimeError("No se pudo descargar ninguna tesela (revise la conexión o el proveedor).")
                mosaic = build_mosaic(tiles_estimate, tile_images)
                mosaic.pop("pil_image", None)
                mosaic["zoom"] = zoom_level
                dl_tiles_s = time.perf_counter() - t_dl0
                st.write(f":material/check_circle: Mosaico armado en {dl_tiles_s:.1f} s")

                progress.progress(100, text="Calculando embedding de SAM 2...")
                emb, embed_time = engine.embed(mosaic["img_array"])
                st.write(f":material/check_circle: Embedding calculado en {embed_time:.2f} s")

                st.session_state.mosaic_data = mosaic
                st.session_state.emb = emb
                st.session_state.tile_failures = [(t.z, t.x, t.y, e) for t, e in failed]
                st.session_state.map_center = list(tiles_estimate["centroid"])
                st.session_state.map_zoom = zoom_level
                catalog.clear()
                _reset_editing()
                dl_status.update(label=f"Ortofoto lista (embedding {embed_time:.1f} s)", state="complete", expanded=False)
                st.toast(f"Ortofoto lista · embedding en {embed_time:.1f} s", icon=":material/check_circle:")
                st.rerun()
            except Exception as exc:
                dl_status.update(label="Error al cargar la ortofoto", state="error")
                st.error(f"Error durante la carga: {exc}")

    st.markdown("---")

    # 3. Vectorización ---------------------------------------------------------
    st.markdown("#### :material/polyline: 3. Vectorización")
    min_pixel_area = st.number_input(
        "Área mínima de cada parte (px)",
        min_value=0,
        max_value=10000,
        value=16,
        step=4,
        help="Descarta fragmentos de la máscara más pequeños que este valor.",
    )
    simplify_px = st.slider(
        "Suavizado del contorno (px)",
        0.0,
        3.0,
        1.0,
        0.25,
        help="Douglas-Peucker. 0 = borde ráster exacto; 1 px corrige el efecto escalera con error de área < 1 %.",
    )

# =============================================================================
# COHERENCIA MODELO <-> EMBEDDING
# =============================================================================
mosaic = st.session_state.mosaic_data
mosaic_loaded = mosaic is not None
if mosaic_loaded and engine is not None and not engine.is_compatible(st.session_state.emb):
    with st.status("Recalculando embedding de la ortofoto para el motor activo...", expanded=False) as emb_st:
        st.session_state.emb, _t = engine.embed(mosaic["img_array"])
        emb_st.update(label=f"Embedding recalculado ({_t:.2f} s)", state="complete")
    _reset_editing()
    seg = st.session_state.seg
emb = st.session_state.emb

# =============================================================================
# CABECERA Y STEPPER
# =============================================================================
active_info = SAM2_MODELS[st.session_state.active_model_key]
render_topbar(
    model_name=active_info["name"],
    device_name=engine.device_name if engine is not None else "No cargado",
    is_mock=bool(engine is not None and engine.is_mock),
    model_loaded=engine is not None,
)
render_stepper(
    has_mosaic=mosaic_loaded,
    has_embedding=emb is not None,
    has_prompts=not seg.is_empty,
    has_candidate=seg.result is not None,
    catalog_count=len(catalog),
)

if engine is not None and engine.is_mock:
    st.error(
        "**MODO SIMULADO.** PyTorch o SAM 2 no están instalados en este servidor; las máscaras son "
        f"figuras sintéticas y NO provienen de SAM 2. Motivo: `{engine.mock_reason}`",
        icon=":material/warning:",
    )
if st.session_state.tile_failures:
    st.warning(
        f"{len(st.session_state.tile_failures)} teselas no se pudieron descargar y aparecen en gris; "
        "no segmente sobre esas zonas.",
        icon=":material/broken_image:",
    )


# =============================================================================
# FUNCIONES AUXILIARES DE LA INTERFAZ
# =============================================================================
def _cand_key(idx: int | None = None):
    idx = seg.selected_idx if idx is None else idx
    so = shape_options()
    return (id(seg.result), idx, min_pixel_area, simplify_px, so["fill_holes"], so["close_px"], so["keep_largest"])


def candidate_mask_and_gdf(idx: int | None = None):
    """(máscara post-procesada, GeoDataFrame) de la candidata, cacheados por prompts y parámetros."""
    if seg.result is None:
        return None, None
    idx = seg.selected_idx if idx is None else idx
    key = _cand_key(idx)
    cache = st.session_state.cand_cache or {}
    if key not in cache:
        cand = seg.candidates[idx]
        mask = refine_mask(cand.mask, min_part_px=min_pixel_area, **shape_options())
        gdf = mask_to_geodataframe(
            mask,
            mosaic["transform_3857"],
            confidence=cand.score,
            min_pixel_area=min_pixel_area,
            simplify_px=simplify_px,
        )
        cache[key] = (mask, gdf)
        st.session_state.cand_cache = cache
    return cache[key]


def candidate_gdf(idx: int | None = None):
    return candidate_mask_and_gdf(idx)[1]


def _pick_suggestion(cat: str, key: str) -> None:
    st.session_state[key] = cat
    st.session_state.suggested_pick = cat


def _crop_box(masks, pad=12):
    h, w = masks[0].shape
    union = np.zeros((h, w), dtype=bool)
    for m in masks:
        union |= m
    ys, xs = np.where(union)
    if len(xs) == 0:
        return 0, 0, w, h
    return (max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1))


def _prompt_overlays():
    T = mosaic["transform_3857"]
    pts = []
    for e in seg.events:
        if e.kind == "point":
            lon, lat = pixel_to_coordinates(e.x, e.y, T)
            pts.append((lat, lon, e.label))
    box_ll = None
    boxes = [e.box for e in seg.events if e.kind == "box"]
    if boxes:
        x0, y0, x1, y1 = boxes[-1]
        lon0, lat0 = pixel_to_coordinates(x0, y1, T)
        lon1, lat1 = pixel_to_coordinates(x1, y0, T)
        box_ll = (lat0, lon0, lat1, lon1)
    pending = None
    if seg.pending_corner is not None:
        lon, lat = pixel_to_coordinates(*seg.pending_corner, T)
        pending = (lat, lon)
    return pts, box_ll, pending


def start_refine(object_id: str) -> None:
    obj = catalog.get(object_id)
    new = InteractiveSegmenter(refine_with_logits=seg.refine_with_logits)
    if obj.prompts and obj.prompts.get("events"):
        new = InteractiveSegmenter.from_dict(obj.prompts, engine, emb)
    else:  # objeto automático: se parte de su caja envolvente
        x0, y0, x1, y1 = obj.bbox_px
        new.click_box_corner(engine, emb, x0, y0)
        new.click_box_corner(engine, emb, x1, y1)
    st.session_state.seg = new
    st.session_state.editing_object_id = object_id
    st.session_state.cand_cache = None


# =============================================================================
# 3–6. BARRA DE HERRAMIENTAS, MAPA E INSPECTOR DERECHO (65 / 35)
# =============================================================================
mode = st.session_state.get("prompt_mode", "pos") or "pos"
if mosaic_loaded:
    tb1, tb2, tb3, tb4 = st.columns([3.3, 1.9, 0.9, 0.9], vertical_alignment="bottom")
    with tb1:
        mode = (
            st.segmented_control(
                "Herramienta de prompt",
                options=list(MODE_LABELS),
                key="prompt_mode",
                format_func=lambda k: MODE_LABELS[k],
                default="pos",
                help="Incluir = punto positivo · Excluir = punto negativo · Caja = dos clics en esquinas opuestas.",
            )
            or "pos"
        )
    with tb2:
        seg.refine_with_logits = st.toggle(
            "Refinar con máscara previa",
            value=seg.refine_with_logits,
            help="Entrega a SAM 2 los logits de la máscara elegida (mask_input) en cada nuevo prompt.",
        )
    with tb3:
        if st.button("Deshacer", icon=":material/undo:", width="stretch", disabled=seg.is_empty):
            seg.undo(engine, emb)
            st.session_state.cand_cache = None
            st.rerun()
    with tb4:
        if st.button(
            "Limpiar",
            icon=":material/ink_eraser:",
            width="stretch",
            disabled=seg.is_empty and not st.session_state.editing_object_id,
        ):
            _reset_editing()
            st.rerun()

col_map, col_panel = st.columns([65, 35], gap="medium")

with col_map:
    base_center = st.session_state.map_center
    base_map = create_base_map(
        center=base_center,
        zoom_start=st.session_state.map_zoom,
        bounds_4326=mosaic["bounds_4326"] if mosaic_loaded else None,
        native_zoom=mosaic.get("zoom") if mosaic_loaded else None,
    )
    layers = []
    if mosaic_loaded:
        visible = [
            {"gdf": o.gdf, "color": o.color, "object_id": o.object_id, "label": o.label, "status": o.status}
            for o in catalog.filter(("aceptado", "propuesto"), visible_only=True)
            if o.object_id != st.session_state.editing_object_id
        ]
        pts, box_ll, pending = _prompt_overlays()
        layers = build_dynamic_layers(
            objects=visible,
            candidate_gdf=candidate_gdf(),
            points_latlon=pts,
            box_latlon=box_ll,
            pending_corner_latlon=pending,
            highlight_id=st.session_state.focus_object_id,
        )
    map_out = st_folium(
        base_map,
        key="visor",
        height=575,
        use_container_width=True,
        returned_objects=["last_clicked"],
        feature_group_to_add=layers or None,
        center=st.session_state.map_center,
        zoom=st.session_state.map_zoom,
    )
    render_map_legend(seg.summary() if mosaic_loaded else {"positivos": 0, "negativos": 0, "caja": 0})

# ---- Procesamiento del clic (solo clics NUEVOS) -----------------------------
if mosaic_loaded and engine is not None and map_out and map_out.get("last_clicked"):
    lc = map_out["last_clicked"]
    click = (round(lc["lat"], 7), round(lc["lng"], 7))
    if click != st.session_state.last_seen_click:
        st.session_state.last_seen_click = click
        w_deg, s_deg, e_deg, n_deg = mosaic["bounds_4326"]
        if not (s_deg <= click[0] <= n_deg and w_deg <= click[1] <= e_deg):
            st.toast("El clic debe estar dentro del rectángulo de la ortofoto.", icon=":material/warning:")
        else:
            try:
                col, row = coordinates_to_pixel(
                    click[1],
                    click[0],
                    mosaic["transform_3857"],
                    mosaic["width"],
                    mosaic["height"],
                )
                with st.spinner("SAM 2 segmentando..."):
                    if mode == "box":
                        seg.click_box_corner(engine, emb, col, row)
                    elif mode == "neg" and not seg.events:
                        st.toast("Empiece con un punto positivo o una caja; después excluya zonas.", icon=":material/info:")
                    else:
                        seg.add_point(engine, emb, col, row, positive=(mode != "neg"))
                st.session_state.cand_cache = None
                st.rerun()
            except ValueError as exc:
                st.toast(str(exc), icon=":material/warning:")

# ---- Panel derecho (Inspector SIG) ------------------------------------------
with col_panel:
    editing = st.session_state.editing_object_id
    with st.container(border=True):
        if editing:
            st.markdown(f"#### :material/edit: Refinando {editing} · v{catalog.get(editing).version}")
        else:
            st.markdown("#### :material/target: Inspector de segmentación")

        if seg.pending_corner is not None:
            st.info("Primera esquina marcada: haga clic en la esquina opuesta de la caja.", icon=":material/crop_free:")

        if not mosaic_loaded or seg.result is None:
            render_empty_inspector(has_mosaic=mosaic_loaded, mode=mode, refining_id=editing)
        else:
            cands = seg.candidates
            if len(cands) > 1:
                st.markdown("**Candidatas de SAM 2**")
                crop_box = _crop_box([c.mask for c in cands])
                ccols = st.columns(len(cands), gap="small")
                for i, (cc, cand) in enumerate(zip(ccols, cands)):
                    with cc:
                        is_sel = i == seg.selected_idx
                        with st.container(border=True):
                            thumb = _uniform_candidate_thumb(mosaic["img_array"], cand.mask, crop_box)
                            st.image(thumb, width="stretch")
                            prefix = "✓ " if is_sel else ""
                            st.caption(f"{prefix}{i + 1} · IoU pred. {cand.score:.2f}")
                choice = st.radio(
                    "Candidata elegida",
                    options=list(range(len(cands))),
                    horizontal=True,
                    index=seg.selected_idx,
                    format_func=lambda i: f"#{i + 1}",
                    key=f"cand_radio_{len(seg.events)}_{id(seg.result)}",
                )
                if choice != seg.selected_idx:
                    seg.select(choice)
                    st.rerun()
                st.caption(
                    "¿Solo tomó una parte del edificio? En edificaciones compuestas (iglesias, conventos, colegios) "
                    "use **▭ Caja** alrededor de todo el conjunto y añada **➕** sobre las partes que falten."
                )

    if mosaic_loaded and seg.result is not None:
        mask_c, gdf_c = candidate_mask_and_gdf()
        cand = seg.selected_candidate
        with st.container(border=True):
            if gdf_c is None or gdf_c.empty:
                st.warning("La máscara elegida no produce polígonos con el área mínima configurada.")
            else:
                r = gdf_c.iloc[0]
                if touches_border(mask_c):
                    st.warning(
                        "El objeto toca el borde de la ortofoto: probablemente está **recortado** y sus medidas "
                        "quedarán incompletas. Amplíe la zona de estudio para incluirlo entero.",
                        icon=":material/crop:",
                    )
                m1, m2 = st.columns(2)
                m1.metric("Área", f"{r['area_m2']:,.1f} m²", help=f"Área planar en {r['metric_crs']}")
                m2.metric(
                    "Perímetro exterior",
                    f"{r['perimeter_ext_m']:,.1f} m",
                    help="Contorno de la huella, sin bordes de patios ni huecos interiores.",
                )
                m3, m4 = st.columns(2)
                m3.metric(
                    "IoU predicho",
                    f"{cand.score:.3f}",
                    help="Estimación interna de SAM 2 de la calidad de la máscara (IoU esperado).",
                )
                m4.metric(
                    "Inferencia",
                    f"{seg.result['inference_time'] * 1000:.0f} ms",
                    help="Latencia del decodificador de máscaras sobre el embedding en memoria.",
                )
                if r["n_holes"] or r["n_parts"] > 1:
                    st.caption(
                        f"{int(r['n_parts'])} parte(s) · {int(r['n_holes'])} hueco(s) · perímetro total con "
                        f"huecos: {r['perimeter_m']:,.1f} m"
                    )

                with st.expander("Forma de la huella", expanded=bool(r["n_holes"] or r["n_parts"] > 1)):
                    st.checkbox(
                        "Rellenar patios y huecos interiores",
                        key="shape_fill",
                        help="Huella exterior de la edificación: incluye claustros y patios.",
                    )
                    st.slider(
                        "Unir partes separadas (px)",
                        0,
                        15,
                        key="shape_close",
                        help="Cierre morfológico: une partes separadas por menos del doble de este valor.",
                    )
                    st.checkbox("Conservar solo la parte principal", key="shape_largest")

                # Categoría (asignada por el usuario; sugerencia opcional con RemoteCLIP)
                cat_key = f"cat_{editing}"
                custom = sorted({o.label for o in catalog} - set(CATEGORIES) - {"Sin categoría"})
                options = ["Sin categoría"] + CATEGORIES + custom
                if cat_key not in st.session_state:
                    st.session_state[cat_key] = (
                        catalog.get(editing).label if editing else st.session_state.label_default
                    )
                if st.session_state[cat_key] not in options:
                    options.append(st.session_state[cat_key])
                label = st.selectbox(
                    "Categoría (la asigna usted; SAM 2 no clasifica)",
                    options,
                    key=cat_key,
                    accept_new_options=True,
                    help="Elija de la lista o escriba una nueva.",
                )
                label = (label or "Sin categoría").strip() or "Sin categoría"

                ok_sem, why_sem = classifier_available()
                ckey = _cand_key()
                sugg = st.session_state.suggestions
                if sugg is None or sugg[0] != ckey:
                    if st.button(
                        "Sugerir categoría (RemoteCLIP)",
                        icon=":material/auto_fix_high:",
                        width="stretch",
                        disabled=not ok_sem,
                        help=(
                            "Clasificador zero-shot de teledetección. Es una sugerencia: usted decide."
                            if ok_sem
                            else why_sem
                        ),
                    ):
                        x0, y0, x1, y1 = _crop_box([mask_c], pad=0)
                        crop = object_crop(mosaic["img_array"], (x0, y0, x1, y1))
                        try:
                            with st.status(
                                f"Clasificando recorte con RemoteCLIP (≈{REMOTECLIP_INFO['size_mb']} MB primera vez)...",
                                expanded=False,
                            ) as sem_st:
                                st.session_state.suggestions = (ckey, get_semantic_classifier().classify(crop))
                                sem_st.update(label="Sugerencias listas", state="complete")
                        except Exception as exc:
                            st.error(f"No se pudo usar RemoteCLIP: {exc}")
                        st.rerun()
                else:
                    st.caption(
                        "Sugerencias de RemoteCLIP — es una ayuda, no una clasificación verificada. "
                        "Pulse una para adoptarla:"
                    )
                    for cat, prob in sugg[1]:
                        st.button(
                            f"{cat} · {prob:.0%}",
                            key=f"sug_{cat}",
                            width="stretch",
                            on_click=_pick_suggestion,
                            args=(cat, cat_key),
                        )

                a1, a2 = st.columns(2)
                with a1:
                    if st.button("Aceptar objeto", icon=":material/check_circle:", type="primary", width="stretch"):
                        backend_name = type(engine.backend).__name__
                        src_label = (
                            "RemoteCLIP (confirmada por el usuario)"
                            if st.session_state.suggested_pick == label
                            else "usuario"
                        )
                        if editing:
                            catalog.replace_geometry(
                                editing,
                                mask_c,
                                gdf_c,
                                cand.score,
                                seg.to_dict(),
                                engine.model_key,
                                backend_name,
                            )
                            catalog.update(editing, label=label, label_source=src_label)
                            st.toast(f"{editing} actualizado.", icon=":material/edit:")
                        else:
                            obj = catalog.add_from_mask(
                                mask_c,
                                gdf_c,
                                cand.score,
                                engine.model_key,
                                backend_name,
                                label=label,
                                prompts=seg.to_dict(),
                                label_source=src_label,
                            )
                            st.session_state.focus_object_id = obj.object_id
                            st.toast(f"{obj.object_id} añadido al catálogo.", icon=":material/add_circle:")
                        st.session_state.label_default = label
                        st.session_state.pop(cat_key, None)
                        _reset_editing()
                        st.rerun()
                with a2:
                    if st.button("Descartar", icon=":material/delete_outline:", width="stretch"):
                        st.session_state.pop(cat_key, None)
                        _reset_editing()
                        st.toast("Segmentación descartada.", icon=":material/info:")
                        st.rerun()


# =============================================================================
# 7–10. CATÁLOGO, MÉTRICAS, ESTADÍSTICAS, RELACIONES, EXPORTACIÓN
# =============================================================================
if mosaic_loaded:
    st.markdown("---")
    tab_obj, tab_det, tab_stats, tab_rel, tab_exp, tab_auto = st.tabs([
        f":material/layers: Objetos ({len(catalog)})",
        ":material/straighten: Detalle y métricas",
        ":material/bar_chart: Estadísticas",
        ":material/hub: Relaciones espaciales",
        ":material/download: Exportar",
        ":material/auto_awesome: Segmentación automática",
    ])

    # ---- 9. Administración ----------------------------------------------------
    with tab_obj:
        st.markdown("#### Catálogo de objetos")
        if len(catalog) == 0:
            st.caption("Aún no hay objetos. Acepte una segmentación en el inspector para añadirla al catálogo.")
        else:
            table = catalog.table().drop(columns=["color"])
            editor_key = f"catalog_editor_{catalog.revision}"
            row_ids = table["object_id"].tolist()

            def _apply_edits(key=editor_key, ids=row_ids):
                edits = st.session_state[key].get("edited_rows", {})
                for row_idx, changes in edits.items():
                    fields = {k: v for k, v in changes.items() if k in ("label", "status", "ver")}
                    if "ver" in fields:
                        fields["visible"] = bool(fields.pop("ver"))
                    if "label" in fields:
                        fields["label"] = str(fields["label"]).strip() or "Sin categoría"
                        fields["label_source"] = "usuario"
                    if fields:
                        catalog.update(ids[int(row_idx)], **fields)

            st.data_editor(
                table,
                key=editor_key,
                on_change=_apply_edits,
                hide_index=True,
                width="stretch",
                disabled=["object_id", "area_m2", "perimeter_m", "pred_iou", "compactness", "origin", "version"],
                column_config={
                    "ver": st.column_config.CheckboxColumn("Ver", width="small"),
                    "object_id": st.column_config.TextColumn("ID"),
                    "label": st.column_config.TextColumn("Etiqueta"),
                    "status": st.column_config.SelectboxColumn("Estado", options=list(STATUSES), required=True),
                    "area_m2": st.column_config.NumberColumn("Área (m²)", format="%.2f"),
                    "perimeter_m": st.column_config.NumberColumn("Perímetro (m)", format="%.2f"),
                    "pred_iou": st.column_config.NumberColumn("IoU pred.", format="%.3f"),
                    "compactness": st.column_config.NumberColumn("Compacidad", format="%.3f"),
                    "origin": st.column_config.TextColumn("Origen"),
                    "version": st.column_config.NumberColumn("Versión", format="%d"),
                },
            )
            ids = catalog.ids()
            focus = st.session_state.focus_object_id if st.session_state.focus_object_id in ids else ids[0]
            sel = st.selectbox(
                "Objeto seleccionado",
                ids,
                index=ids.index(focus),
                format_func=lambda i: f"{i} · {catalog.get(i).label} · {STATUS_LABELS[catalog.get(i).status]}",
            )
            st.session_state.focus_object_id = sel
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if st.button("Centrar en el mapa", icon=":material/center_focus_strong:", width="stretch"):
                    r = catalog.get(sel).gdf.iloc[0]
                    st.session_state.map_center = [float(r["centroid_lat"]), float(r["centroid_lon"])]
                    st.session_state.map_zoom = max(int(mosaic.get("zoom", 18)), 18)
                    st.rerun()
            with b2:
                if st.button("Refinar", icon=":material/edit:", width="stretch", disabled=engine is None):
                    with st.spinner("Reconstruyendo prompts..."):
                        start_refine(sel)
                    st.rerun()
            with b3:
                if st.button("Eliminar", icon=":material/delete:", width="stretch"):
                    if st.session_state.editing_object_id == sel:
                        _reset_editing()
                    catalog.remove(sel)
                    st.rerun()
            with b4:
                n_rej = len(catalog.filter(["rechazado"]))
                if st.button(f"Eliminar rechazados ({n_rej})", icon=":material/delete_sweep:", width="stretch", disabled=n_rej == 0):
                    for o in catalog.filter(["rechazado"]):
                        catalog.remove(o.object_id)
                    st.rerun()

            st.markdown("---")
            st.markdown("**Fusionar objetos** — p. ej. nave, claustro y alas de una misma edificación")
            f1, f2 = st.columns([3, 1])
            to_merge = f1.multiselect(
                "Objetos a fusionar",
                ids,
                label_visibility="collapsed",
                format_func=lambda i: f"{i} · {catalog.get(i).label}",
                placeholder="Elija dos o más objetos",
            )
            if f2.button("Fusionar", icon=":material/link:", width="stretch", disabled=len(to_merge) < 2):
                so = shape_options()
                conf = min(catalog.get(i).pred_iou for i in to_merge)
                try:
                    new = catalog.merge(
                        to_merge,
                        label=catalog.get(to_merge[0]).label,
                        postprocess=lambda m: refine_mask(m, min_part_px=min_pixel_area, **so),
                        gdf_builder=lambda m: mask_to_geodataframe(
                            m,
                            mosaic["transform_3857"],
                            confidence=conf,
                            min_pixel_area=min_pixel_area,
                            simplify_px=simplify_px,
                        ),
                    )
                    st.session_state.focus_object_id = new.object_id
                    st.toast(f"{len(to_merge)} objetos fusionados en {new.object_id}.", icon=":material/link:")
                except ValueError as exc:
                    st.error(str(exc))
                st.rerun()
            st.caption("La fusión usa las opciones de *Forma de la huella* del panel (p. ej. rellenar patios).")

            unl = [o for o in catalog if o.label in ("sin etiquetar", "Sin categoría")]
            ok_sem, why_sem = classifier_available()
            if unl and st.button(
                f"Sugerir categoría a {len(unl)} objeto(s) sin categoría (RemoteCLIP)",
                icon=":material/auto_fix_high:",
                disabled=not ok_sem,
                help=None if ok_sem else why_sem,
            ):
                with st.status("Clasificando recortes con RemoteCLIP...", expanded=False) as clf_st:
                    clf = get_semantic_classifier()
                    for o in unl:
                        top = clf.classify(object_crop(mosaic["img_array"], o.bbox_px), top_k=1)[0]
                        catalog.update(
                            o.object_id,
                            label=top[0],
                            label_source=f"RemoteCLIP {top[1]:.0%} (sin confirmar)",
                        )
                    clf_st.update(label="Categorías sugeridas", state="complete")
                st.toast("Categorías sugeridas: revíselas antes de aceptar los objetos.", icon=":material/auto_fix_high:")
                st.rerun()

    # ---- 8. Detalle y métricas -------------------------------------------------
    with tab_det:
        fid = st.session_state.focus_object_id
        if not fid or fid not in catalog:
            st.markdown("#### Detalle y métricas")
            st.caption("Seleccione un objeto en la pestaña Objetos.")
        else:
            obj = catalog.get(fid)
            r = obj.gdf.iloc[0]
            st.markdown(f"#### {obj.object_id} · {obj.label} · {STATUS_LABELS[obj.status]} · v{obj.version}")
            st.caption(f"Categoría «{obj.label}» · fuente: {obj.label_source}")
            if obj.touches_edge:
                st.warning(
                    "Este objeto toca el borde de la ortofoto: puede estar recortado.",
                    icon=":material/crop:",
                )
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Área planar", f"{r['area_m2']:,.2f} m²", help=f"En {r['metric_crs']}")
            d2.metric("Área geodésica", f"{r['area_geod_m2']:,.2f} m²", help="Sobre el elipsoide WGS84")
            d3.metric("Perímetro", f"{r['perimeter_m']:,.2f} m")
            d4.metric("IoU predicho", f"{obj.pred_iou:.3f}")
            metrics_df = pd.DataFrame(
                [
                    ("Perímetro exterior (sin patios)", f"{float(r.get('perimeter_ext_m', r['perimeter_m'])):,.2f}", "m"),
                    ("Perímetro geodésico", f"{float(r['perimeter_geod_m']):,.2f}", "m"),
                    ("Área exacta de la máscara", f"{float(r['pixel_area_m2']):,.2f}", "m²"),
                    ("Píxeles de la máscara", f"{int(r['pixel_count']):,}", "px"),
                    ("GSD real del terreno", f"{float(r['gsd_m']):.4f}", "m/px"),
                    ("Área del convex hull", f"{float(r['hull_area_m2']):,.2f}", "m²"),
                    ("Solidez (A / A hull)", f"{float(r['solidity']):.3f}", "—"),
                    ("Compacidad Polsby-Popper (4πA/P²)", f"{float(r['compactness']):.3f}", "—"),
                    ("Rectangularidad (A / A rect. mín.)", f"{float(r['rectangularity']):.3f}", "—"),
                    ("Elongación (1 − ancho/largo)", f"{float(r['elongation']):.3f}", "—"),
                    ("Relación de aspecto (largo/ancho)", f"{float(r['aspect_ratio']):.2f}", "—"),
                    ("Orientación del eje largo (acimut)", f"{float(r['orientation_deg']):.1f}", "°"),
                    ("Largo × ancho del rectángulo mínimo", f"{float(r['mrr_length_m']):.2f} × {float(r['mrr_width_m']):.2f}", "m"),
                    ("Partes / huecos", f"{int(r['n_parts'])} / {int(r['n_holes'])}", "—"),
                    ("Centroide (lat, lon)", f"{float(r['centroid_lat']):.7f}, {float(r['centroid_lon']):.7f}", "°"),
                    ("Bounding box (O, S, E, N)", f"{float(r['min_lon']):.6f}, {float(r['min_lat']):.6f}, {float(r['max_lon']):.6f}, {float(r['max_lat']):.6f}", "°"),
                    ("CRS métrico", str(r["metric_crs"]), "—"),
                    ("Modelo / backend", f"{obj.model_id} / {obj.backend}", "—"),
                    ("Origen", str(obj.origin), "—"),
                    ("Fuente de la categoría", str(obj.label_source), "—"),
                ],
                columns=["Métrica", "Valor", "Unidad"],
            )
            st.dataframe(metrics_df, hide_index=True, width="stretch")
            mask = obj.mask()
            cut, _bb = extract_cutout(mosaic["img_array"], mask, crop_to_bbox=True)
            i1, i2 = st.columns(2)
            i1.image(cut, caption="Recorte con canal alfa", width="stretch")
            x0, y0, x1, y1 = _crop_box([mask], pad=40)
            i2.image(
                create_mask_overlay(mosaic["img_array"][y0:y1, x0:x1], mask[y0:y1, x0:x1]),
                caption="Máscara sobre la ortofoto",
                width="stretch",
            )

    # ---- Estadísticas ---------------------------------------------------------
    with tab_stats:
        st.markdown("#### Estadísticas del catálogo")
        stats = catalog.statistics(("aceptado",))
        counts = {s_: len(catalog.filter([s_])) for s_ in STATUSES}
        st.caption(" · ".join(f"{STATUS_LABELS[k]}: {v}" for k, v in counts.items()))
        if stats["n"] == 0:
            st.caption("Las estadísticas se calculan sobre los objetos aceptados.")
        else:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Objetos aceptados", stats["n"])
            s2.metric("Área total", f"{stats['area_total_m2']:,.1f} m²")
            s3.metric("Área media", f"{stats['area_media_m2']:,.1f} m²")
            s4.metric("Compacidad media", f"{stats['compacidad_media']:.3f}")
            st.markdown("**Resumen por etiqueta**")
            st.dataframe(
                stats["por_etiqueta"].rename(
                    columns={"label": "Etiqueta", "n": "Objetos", "area_m2": "Área total (m²)"}
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Objetos": st.column_config.NumberColumn("Objetos", format="%d"),
                    "Área total (m²)": st.column_config.NumberColumn("Área total (m²)", format="%.2f"),
                },
            )
            gacc = catalog.to_geodataframe(("aceptado",))
            st.markdown("**Área por objeto (m²)**")
            st.bar_chart(gacc.set_index("object_id")["area_m2"], height=220)

    # ---- Relaciones espaciales ------------------------------------------------
    with tab_rel:
        st.markdown("#### Relaciones espaciales")
        gacc = catalog.to_geodataframe(("aceptado",))
        if len(gacc) < 2:
            st.caption("Se necesitan al menos dos objetos aceptados.")
        else:
            dmax = st.number_input("Listar pares a menos de (m)", min_value=0.0, value=50.0, step=10.0)
            summary, pairs = spatial_relations(gacc, max_pairs_distance_m=dmax)
            st.markdown("**Vecino más cercano y contactos**")
            st.dataframe(
                summary[["object_id", "label", "nn_id", "nn_dist_m", "n_contacts"]].rename(
                    columns={
                        "object_id": "ID",
                        "label": "Etiqueta",
                        "nn_id": "Vecino más cercano",
                        "nn_dist_m": "Distancia (m)",
                        "n_contacts": "Contactos",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Distancia (m)": st.column_config.NumberColumn("Distancia (m)", format="%.2f"),
                    "Contactos": st.column_config.NumberColumn("Contactos", format="%d"),
                },
            )
            st.markdown(f"**Pares a menos de {dmax:g} m**")
            st.dataframe(pairs, hide_index=True, width="stretch")
            st.caption(
                "Distancias euclidianas en el CRS métrico (EPSG:9377 o UTM); "
                "relación según el modelo topológico DE-9IM (shapely)."
            )

    # ---- 10. Exportación ------------------------------------------------------
    with tab_exp:
        st.markdown("#### Exportación de resultados")
        scope = st.multiselect(
            "Estados a exportar",
            list(STATUSES),
            default=["aceptado"],
            format_func=lambda s_: STATUS_LABELS[s_],
        )
        gexp = catalog.to_geodataframe(scope)
        objs_exp = catalog.filter(scope)
        T = mosaic["transform_3857"]
        img = mosaic["img_array"]
        if gexp.empty:
            st.caption("No hay objetos con los estados seleccionados para los formatos vectoriales y etiquetas.")

        st.markdown("##### Vectorial")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.download_button(
                on_click="ignore",
                label="GeoJSON",
                icon=":material/public:",
                data=lambda g=gexp: export_geojson(g),
                file_name="objetos_sam2.geojson",
                mime="application/geo+json",
                width="stretch",
                disabled=gexp.empty,
            )
            st.markdown(f'<p class="gis-export-desc">{EXPORT_DESCRIPTIONS["geojson"]}</p>', unsafe_allow_html=True)
        with v2:
            st.download_button(
                on_click="ignore",
                label="GeoPackage",
                icon=":material/layers:",
                width="stretch",
                disabled=gexp.empty,
                file_name="sam2_geoextractor.gpkg",
                mime="application/geopackage+sqlite3",
                data=lambda g=gexp: export_geopackage({"objetos": g, **derived_geometries(g)}),
            )
            st.markdown(f'<p class="gis-export-desc">{EXPORT_DESCRIPTIONS["gpkg"]}</p>', unsafe_allow_html=True)
        with v3:
            st.download_button(
                on_click="ignore",
                label="Shapefile (.zip)",
                icon=":material/folder_zip:",
                data=lambda g=gexp: export_shapefile_zip(g),
                file_name="objetos_sam2_shp.zip",
                mime="application/zip",
                width="stretch",
                disabled=gexp.empty,
            )
            st.markdown(f'<p class="gis-export-desc">{EXPORT_DESCRIPTIONS["shp"]}</p>', unsafe_allow_html=True)
        with v4:
            st.download_button(
                on_click="ignore",
                label="CSV de atributos",
                icon=":material/table_chart:",
                data=lambda g=gexp: export_csv(g),
                file_name="objetos_sam2.csv",
                mime="text/csv",
                width="stretch",
                disabled=gexp.empty,
            )
            st.markdown(f'<p class="gis-export-desc">{EXPORT_DESCRIPTIONS["csv"]}</p>', unsafe_allow_html=True)

        st.markdown("##### Ráster")
        r1, r2, r3, _r4 = st.columns(4)
        with r1:
            st.download_button(
                on_click="ignore",
                label="Ortofoto GeoTIFF",
                icon=":material/map:",
                data=lambda: export_geotiff(img, T, CRS_WEB_MERCATOR),
                file_name="ortofoto_3857.tif",
                mime="image/tiff",
                width="stretch",
            )
            st.markdown(f'<p class="gis-export-desc">{EXPORT_DESCRIPTIONS["ortho"]}</p>', unsafe_allow_html=True)
        with r2:
            st.download_button(
                on_click="ignore",
                label="Máscaras GeoTIFF",
                icon=":material/grid_on:",
                width="stretch",
                disabled=not objs_exp,
                file_name="mascaras_etiquetas.tif",
                mime="image/tiff",
                data=lambda o=objs_exp: export_label_geotiff([x.mask() for x in o], [x.raster_value for x in o], T),
            )
            st.markdown(f'<p class="gis-export-desc">{EXPORT_DESCRIPTIONS["masks"]}</p>', unsafe_allow_html=True)
        with r3:
            fid = st.session_state.focus_object_id
            fobj = catalog.get(fid) if fid in catalog else None
            st.download_button(
                on_click="ignore",
                label="PNG del objeto",
                icon=":material/image:",
                width="stretch",
                disabled=fobj is None,
                mime="image/png",
                file_name=f"{fid}.png" if fobj else "objeto.png",
                data=lambda o=fobj: export_png(extract_cutout(img, o.mask(), crop_to_bbox=True)[0]),
            )
            st.markdown(
                '<p class="gis-export-desc">Recorte RGBA transparente del objeto enfocado en Objetos.</p>',
                unsafe_allow_html=True,
            )

        st.markdown("##### Dataset")
        ds1, _ds2 = st.columns([1, 3])
        with ds1:
            st.download_button(
                on_click="ignore",
                label="Dataset COCO (.zip)",
                icon=":material/dataset:",
                width="stretch",
                disabled=not objs_exp,
                file_name="dataset_coco_sam2.zip",
                mime="application/zip",
                data=lambda o=objs_exp: export_coco_dataset(
                    img,
                    [
                        {
                            "object_id": x.object_id,
                            "label": x.label,
                            "mask_rle": x.mask_rle,
                            "bbox_px": x.bbox_px,
                            "pixel_count": x.pixel_count,
                            "pred_iou": x.pred_iou,
                            "area_m2": float(x.gdf["area_m2"].iloc[0]),
                        }
                        for x in o
                    ],
                    T,
                ),
            )
            st.markdown(f'<p class="gis-export-desc">{EXPORT_DESCRIPTIONS["coco"]}</p>', unsafe_allow_html=True)

    # ---- Segmentación automática ----------------------------------------------
    with tab_auto:
        st.markdown("#### Segmentación automática del mosaico")
        st.markdown(
            "Genera máscaras para **todo** el mosaico con `SAM2AutomaticMaskGenerator`. "
            "Los resultados entran al catálogo como **propuestos** para revisión humana. "
            "En CPU (2 vCPU), 16×16 puntos tardan ~35–40 s."
        )
        g1, g2, g3, g4 = st.columns(4)
        pps = g1.selectbox("Puntos por lado", [8, 16, 24, 32], index=1)
        iou_t = g2.slider("Umbral IoU pred.", 0.5, 0.99, 0.8, 0.01)
        stab_t = g3.slider("Umbral de estabilidad", 0.5, 0.99, 0.92, 0.01)
        min_m2 = g4.number_input("Área mínima (m²)", min_value=0.0, value=5.0, step=1.0)
        if st.button("Ejecutar segmentación automática", icon=":material/auto_awesome:", disabled=engine is None):
            with st.status(f"Ejecutando segmentación automática ({pps}×{pps} puntos)...", expanded=False) as auto_st:
                masks, secs = engine.segment_everything(
                    img,
                    points_per_side=int(pps),
                    points_per_batch=64,
                    pred_iou_thresh=float(iou_t),
                    stability_score_thresh=float(stab_t),
                )
                auto_st.update(label=f"Completado en {secs:.1f} s", state="complete")
            added = 0
            for cand in sorted(masks, key=lambda c: -c.score)[:300]:
                g = mask_to_geodataframe(
                    cand.mask,
                    T,
                    confidence=cand.score,
                    min_pixel_area=min_pixel_area,
                    simplify_px=simplify_px,
                )
                if g.empty or float(g["area_m2"].iloc[0]) < min_m2:
                    continue
                catalog.add_from_mask(
                    cand.mask,
                    g,
                    cand.score,
                    engine.model_key,
                    type(engine.backend).__name__,
                    label="sin etiquetar",
                    origin="automático",
                    status="propuesto",
                )
                added += 1
            st.toast(f"{added} objetos propuestos en {secs:.0f} s.", icon=":material/auto_awesome:")
            st.rerun()
