"""
=============================================================================
SAM 2 GeoExtractor: Extracción Inteligente de Objetos Geoespaciales con IA
=============================================================================
Flujo de trabajo:
  1. Zona de estudio  2. Ortofoto  3. Selección  4. Segmentación SAM 2  5. Refinamiento
  6. Validación  7. Objeto geoespacial  8. Métricas  9. Administración  10. Exportación

La lógica vive en módulos independientes de la interfaz (src/ y geosam/); este archivo solo
orquesta el estado de sesión y los widgets.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from config import (
    DEFAULT_BBOX, DEFAULT_ZOOM, MIN_ZOOM, MAX_ZOOM, SAM2_MODELS, DEFAULT_SAM2_MODEL_KEY, SAM2_INPUT_SIZE,
    CRS_WEB_MERCATOR,
)
from src.tile_manager import get_tiles_info, TileManagerError
from src.imagery_loader import download_tiles_with_report
from src.mosaic_builder import build_mosaic
from src.sam_engine import get_sam2_engine
from src.segmentation import coordinates_to_pixel, pixel_to_coordinates, extract_cutout, create_mask_overlay
from src.geospatial import mask_to_geodataframe, derived_geometries, spatial_relations
from src.interactive import InteractiveSegmenter
from src.postprocess import refine_mask, touches_border
from src.semantic import CATEGORIES, REMOTECLIP_INFO, SemanticClassifier, classifier_available, object_crop
from src.sam_engine import ensure_checkpoint
import geopandas as gpd
from src.products import (
    SupervisedClassifier, color_stats, handcrafted_features, training_eligible, coverage_raster,
    coverage_stats, coverage_rgb, category_color, mosaic_pixel_area_m2, vertex_table, azimuth_to_dms,
    zone_indicators, units_indicators, is_built, technical_report_html, UNLABELED, SHAPE_KEYS, COLOR_KEYS,
)
SHAPE_AND_COLOR = SHAPE_KEYS + COLOR_KEYS
from src.objects import ObjectCatalog, STATUSES
from src.visualization import create_base_map, build_dynamic_layers
from src.exporter import (
    export_geojson, export_geotiff, export_png, export_shapefile_zip, export_csv,
    export_geopackage, export_label_geotiff, export_coco_dataset, export_dxf, export_kml,
    export_coverage_geotiff,
)

st.set_page_config(
    page_title="SAM 2 GeoExtractor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ESTILOS
# =============================================================================
st.markdown(
    """
    <style>
    .status-badge { display:inline-block; padding:4px 10px; border-radius:9999px;
                    font-size:0.82rem; font-weight:600; }
    .badge-gpu  { background-color:#dcfce7; color:#15803d; }
    .badge-cpu  { background-color:#fef9c3; color:#854d0e; }
    .badge-mock { background-color:#fee2e2; color:#b91c1c; }
    .prompt-chip { display:inline-block; padding:2px 9px; margin-right:6px; border-radius:6px;
                   font-size:0.8rem; background:#f1f5f9; border:1px solid #e2e8f0; }
    iframe { cursor: crosshair !important; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODE_LABELS = {"pos": "➕ Incluir", "neg": "➖ Excluir", "box": "▭ Caja (2 clics)"}
STATUS_LABELS = {"aceptado": "✅ aceptado", "propuesto": "🟡 propuesto", "rechazado": "⛔ rechazado"}

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
    "map_center": [(DEFAULT_BBOX["min_lat"] + DEFAULT_BBOX["max_lat"]) / 2,
                   (DEFAULT_BBOX["min_lon"] + DEFAULT_BBOX["max_lon"]) / 2],
    "map_zoom": DEFAULT_ZOOM,
    "focus_object_id": None,
    "label_default": "Sin categoría",
    "suggestions": None,        # (clave de candidata, [(categoría, prob), ...]) de RemoteCLIP
    "suggested_pick": None,     # categoría sugerida que el usuario adoptó
    "feat_cache": {},           # descriptores por (object_id, versión, con_clip)
    "sup_model": None,          # (SupervisedClassifier, con_clip)
    "sup_cv": None,             # resultado de validación cruzada
    "cand_cache": None,
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


@st.cache_resource(show_spinner=f"Cargando RemoteCLIP (≈{REMOTECLIP_INFO['size_mb']} MB la primera vez)...", max_entries=1)
def get_semantic_classifier() -> SemanticClassifier:
    ensure_checkpoint(REMOTECLIP_INFO)
    return SemanticClassifier()


def shape_options() -> dict:
    """Opciones de forma de la huella (widgets del panel; se leen antes de dibujar el mapa)."""
    return {"fill_holes": bool(st.session_state.get("shape_fill", False)),
            "close_px": float(st.session_state.get("shape_close", 0)),
            "keep_largest": bool(st.session_state.get("shape_largest", False))}


# =============================================================================
# BARRA LATERAL
# =============================================================================
with st.sidebar:
    st.title("🛰️ SAM 2 GeoExtractor")
    st.caption("Segmentación interactiva de objetos geoespaciales con SAM 2")
    st.markdown("---")

    # 1. Motor de IA -----------------------------------------------------------
    st.subheader("1. Motor de IA")
    model_keys = list(SAM2_MODELS.keys())
    selected_model_key = st.selectbox(
        "Modelo:", model_keys, index=model_keys.index(DEFAULT_SAM2_MODEL_KEY),
        format_func=lambda k: SAM2_MODELS[k]["name"],
        help="Los pesos se descargan la primera vez. Solo se mantiene un modelo en memoria.",
    )
    engine = None
    try:
        engine = get_sam2_engine(model_key=selected_model_key)
        badge = "badge-mock" if engine.is_mock else ("badge-gpu" if engine.device == "cuda" else "badge-cpu")
        st.markdown(f"**Dispositivo:** <span class='status-badge {badge}'>{engine.device_name}</span>",
                    unsafe_allow_html=True)
    except Exception as exc:  # descarga o carga de pesos fallida
        st.error(f"No se pudo inicializar SAM 2: {exc}")

    st.markdown("---")

    # 2. Zona de estudio -------------------------------------------------------
    st.subheader("2. Zona de estudio (WGS84)")
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
        # SAM 2 redimensiona toda imagen a 1024×1024: por encima, pierde detalle
        eff_res = tiles_estimate["res_m_per_px"] * max(1.0, max(h_px, w_px) / SAM2_INPUT_SIZE)
        st.info(
            f"📦 {tiles_estimate['num_tiles']} teselas · {w_px} × {h_px} px · ~{tiles_estimate['estimated_mb']} MB\n\n"
            f"📏 Resolución de la imagen: ~{tiles_estimate['res_m_per_px']} m/px\n\n"
            f"🧠 Resolución efectiva para SAM 2: ~{eff_res:.2f} m/px"
        )
        if max(h_px, w_px) > SAM2_INPUT_SIZE * 1.5:
            st.warning("SAM 2 reduce la imagen a 1024 px: con esta extensión pierde detalle. "
                       "Para objetos pequeños reduzca el área o el zoom.")
        if tiles_estimate["is_large"]:
            st.warning("Área extensa: la descarga y el embedding tardarán más.")
    except TileManagerError as exc:
        st.error(f"❌ {exc}")

    confirm_reset = True
    if len(catalog) > 0:
        confirm_reset = st.checkbox(f"Reiniciar el catálogo actual ({len(catalog)} objetos) al cargar otra ortofoto")
    btn_download = st.button("📥 Descargar y cargar ortofoto", type="primary", width="stretch",
                             disabled=(tiles_estimate is None or engine is None or not confirm_reset))

    if btn_download and tiles_estimate and engine:
        progress = st.progress(0, text="Descargando teselas...")

        def _progress(done, total):
            progress.progress(int(done / total * 100), text=f"Teselas: {done}/{total}")

        try:
            tile_images, failed = download_tiles_with_report(tiles_estimate["tiles"], progress_callback=_progress)
            if len(failed) == len(tiles_estimate["tiles"]):
                raise RuntimeError("No se pudo descargar ninguna tesela (revise la conexión o el proveedor).")
            mosaic = build_mosaic(tiles_estimate, tile_images)
            mosaic.pop("pil_image", None)  # evita duplicar la imagen en memoria
            mosaic["zoom"] = zoom_level
            progress.progress(100, text="Calculando embeddings de SAM 2...")
            emb, embed_time = engine.embed(mosaic["img_array"])
            st.session_state.mosaic_data = mosaic
            st.session_state.emb = emb
            st.session_state.tile_failures = [(t.z, t.x, t.y, e) for t, e in failed]
            st.session_state.map_center = list(tiles_estimate["centroid"])
            st.session_state.map_zoom = zoom_level
            catalog.clear()
            _reset_editing()
            progress.empty()
            st.toast(f"Ortofoto lista · embedding en {embed_time:.1f} s", icon="✅")
            st.rerun()
        except Exception as exc:
            progress.empty()
            st.error(f"Error durante la carga: {exc}")

    st.markdown("---")

    # 3. Vectorización ---------------------------------------------------------
    st.subheader("3. Vectorización")
    min_pixel_area = st.number_input("Área mínima de cada parte (px)", min_value=0, max_value=10000, value=16,
                                     step=4, help="Descarta fragmentos de la máscara más pequeños que este valor.")
    simplify_px = st.slider("Suavizado del contorno (px)", 0.0, 3.0, 1.0, 0.25,
                            help="Douglas-Peucker. 0 = borde ráster exacto (perímetro sobrestimado ~28 %); "
                                 "1 px corrige el efecto escalera con error de área < 1 %.")

# =============================================================================
# COHERENCIA MODELO ↔ EMBEDDING
# =============================================================================
mosaic = st.session_state.mosaic_data
mosaic_loaded = mosaic is not None
if mosaic_loaded and engine is not None and not engine.is_compatible(st.session_state.emb):
    with st.spinner("Modelo cambiado: recalculando embeddings de la ortofoto..."):
        st.session_state.emb, _t = engine.embed(mosaic["img_array"])
    _reset_editing()
    seg = st.session_state.seg
    st.toast("Embeddings recalculados para el nuevo modelo.", icon="🔁")
emb = st.session_state.emb

# =============================================================================
# CABECERA
# =============================================================================
st.title("Extracción inteligente de objetos geoespaciales")
if engine is not None and engine.is_mock:
    st.error(
        "**MODO SIMULADO.** PyTorch o SAM 2 no están instalados en este servidor; las máscaras son "
        f"figuras sintéticas y NO provienen de SAM 2. Motivo: `{engine.mock_reason}`",
        icon="🚨",
    )
if st.session_state.tile_failures:
    st.warning(f"{len(st.session_state.tile_failures)} teselas no se pudieron descargar y aparecen en gris; "
               "no segmente sobre esas zonas.")

if not mosaic_loaded:
    st.info(
        "👋 **Para comenzar:** defina la zona de estudio en la barra lateral y pulse "
        "**Descargar y cargar ortofoto**. Después haga clic sobre un objeto para segmentarlo con SAM 2, "
        "refínelo con puntos o una caja, valídelo y expórtelo."
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
        gdf = mask_to_geodataframe(mask, mosaic["transform_3857"], confidence=cand.score,
                                   min_pixel_area=min_pixel_area, simplify_px=simplify_px)
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
# 3–6. VISOR, PROMPTS Y VALIDACIÓN
# =============================================================================
if mosaic_loaded:
    tb1, tb2, tb3, tb4 = st.columns([3.2, 1.6, 1, 1])
    with tb1:
        mode = st.segmented_control("Herramienta de prompt", options=list(MODE_LABELS), key="prompt_mode",
                                    format_func=lambda k: MODE_LABELS[k], default="pos",
                                    help="Incluir = punto positivo · Excluir = punto negativo · "
                                         "Caja = dos clics en esquinas opuestas.") or "pos"
    with tb2:
        seg.refine_with_logits = st.toggle("Refinar con máscara previa", value=seg.refine_with_logits,
                                           help="Entrega a SAM 2 los logits de la máscara elegida (mask_input) "
                                                "en cada nuevo prompt.")
    with tb3:
        if st.button("↩️ Deshacer", width="stretch", disabled=seg.is_empty):
            seg.undo(engine, emb)
            st.session_state.cand_cache = None
            st.rerun()
    with tb4:
        if st.button("🧹 Limpiar", width="stretch", disabled=seg.is_empty and not st.session_state.editing_object_id):
            _reset_editing()
            st.rerun()

col_map, col_panel = st.columns([3, 2], gap="medium")

with col_map:
    base_center = st.session_state.map_center
    base_map = create_base_map(
        center=base_center, zoom_start=st.session_state.map_zoom,
        bounds_4326=mosaic["bounds_4326"] if mosaic_loaded else None,
        native_zoom=mosaic.get("zoom") if mosaic_loaded else None,
    )
    layers = []
    if mosaic_loaded:
        visible = [{"gdf": o.gdf, "color": o.color, "object_id": o.object_id, "label": o.label, "status": o.status}
                   for o in catalog.filter(("aceptado", "propuesto"), visible_only=True)
                   if o.object_id != st.session_state.editing_object_id]
        pts, box_ll, pending = _prompt_overlays()
        layers = build_dynamic_layers(objects=visible, candidate_gdf=candidate_gdf(), points_latlon=pts,
                                      box_latlon=box_ll, pending_corner_latlon=pending,
                                      highlight_id=st.session_state.focus_object_id)
    map_out = st_folium(
        base_map, key="visor", height=600, use_container_width=True,
        returned_objects=["last_clicked"], feature_group_to_add=layers or None,
        center=st.session_state.map_center, zoom=st.session_state.map_zoom,
    )

# ---- Procesamiento del clic (solo clics NUEVOS) -----------------------------
if mosaic_loaded and engine is not None and map_out and map_out.get("last_clicked"):
    lc = map_out["last_clicked"]
    click = (round(lc["lat"], 7), round(lc["lng"], 7))
    if click != st.session_state.last_seen_click:
        st.session_state.last_seen_click = click
        w_deg, s_deg, e_deg, n_deg = mosaic["bounds_4326"]
        if not (s_deg <= click[0] <= n_deg and w_deg <= click[1] <= e_deg):
            st.toast("El clic debe estar dentro del rectángulo rojo de la ortofoto.", icon="⚠️")
        else:
            try:
                col, row = coordinates_to_pixel(click[1], click[0], mosaic["transform_3857"],
                                                mosaic["width"], mosaic["height"])
                with st.spinner("SAM 2 segmentando..."):
                    if mode == "box":
                        seg.click_box_corner(engine, emb, col, row)
                    elif mode == "neg" and not seg.events:
                        st.toast("Empiece con un punto positivo o una caja; después excluya zonas.", icon="ℹ️")
                    else:
                        seg.add_point(engine, emb, col, row, positive=(mode != "neg"))
                st.session_state.cand_cache = None
                st.rerun()
            except ValueError as exc:
                st.toast(str(exc), icon="⚠️")

# ---- Panel de edición / validación ------------------------------------------
with col_panel:
    if not mosaic_loaded:
        st.markdown("#### Flujo de trabajo")
        st.markdown("Zona de estudio → Ortofoto → Clic sobre el objeto → SAM 2 → Refinar → Validar → "
                    "Objeto GIS → Métricas → Administrar → Exportar")
    else:
        editing = st.session_state.editing_object_id
        if editing:
            st.markdown(f"#### ✏️ Refinando `{editing}` (v{catalog.get(editing).version})")
        else:
            st.markdown("#### 🎯 Objeto en edición")
        s = seg.summary()
        st.markdown(
            f"<span class='prompt-chip'>➕ {s['positivos']}</span>"
            f"<span class='prompt-chip'>➖ {s['negativos']}</span>"
            f"<span class='prompt-chip'>▭ {s['caja']}</span>",
            unsafe_allow_html=True,
        )
        if seg.pending_corner is not None:
            st.info("Primera esquina marcada: haga clic en la esquina opuesta de la caja.")

        if seg.result is None:
            st.caption({"pos": "Haga clic sobre un objeto del mapa para segmentarlo.",
                        "neg": "Seleccione primero un objeto con ➕ o ▭.",
                        "box": "Haga dos clics en esquinas opuestas del objeto."}[mode])
        else:
            cands = seg.candidates
            if len(cands) > 1:
                st.markdown("**Candidatas de SAM 2** (el modelo resuelve la ambigüedad de un solo clic):")
                x0, y0, x1, y1 = _crop_box([c.mask for c in cands])
                crop = mosaic["img_array"][y0:y1, x0:x1]
                ccols = st.columns(len(cands))
                for i, (cc, cand) in enumerate(zip(ccols, cands)):
                    with cc:
                        st.image(create_mask_overlay(crop, cand.mask[y0:y1, x0:x1]), width="stretch",
                                 caption=f"{i + 1} · IoU pred. {cand.score:.2f}")
                choice = st.radio("Candidata elegida", options=list(range(len(cands))), horizontal=True,
                                  index=seg.selected_idx, format_func=lambda i: f"#{i + 1}",
                                  key=f"cand_radio_{len(seg.events)}_{id(seg.result)}")
                if choice != seg.selected_idx:
                    seg.select(choice)
                    st.rerun()
                st.caption("¿Solo tomó una parte del edificio? En edificaciones compuestas (iglesias, conventos, "
                           "colegios) use **▭ Caja** alrededor de todo el conjunto y añada **➕** sobre las partes "
                           "que falten.")

            mask_c, gdf_c = candidate_mask_and_gdf()
            cand = seg.selected_candidate
            if gdf_c is None or gdf_c.empty:
                st.warning("La máscara elegida no produce polígonos con el área mínima configurada.")
            else:
                r = gdf_c.iloc[0]
                if touches_border(mask_c):
                    st.warning("El objeto toca el borde de la ortofoto: probablemente está **recortado** y sus "
                               "medidas quedarán incompletas. Amplíe la zona de estudio para incluirlo entero.")
                m1, m2 = st.columns(2)
                m1.metric("Área", f"{r['area_m2']:,.1f} m²")
                m2.metric("Perímetro exterior", f"{r['perimeter_ext_m']:,.1f} m",
                          help="Contorno de la huella, sin bordes de patios ni huecos interiores.")
                m3, m4 = st.columns(2)
                m3.metric("IoU predicho", f"{cand.score:.3f}",
                          help="Estimación de SAM 2 de la calidad de la máscara (IoU esperado con el objeto "
                               "real). No es una probabilidad de clase.")
                m4.metric("Inferencia", f"{seg.result['inference_time'] * 1000:.0f} ms")
                if r["n_holes"] or r["n_parts"] > 1:
                    st.caption(f"{int(r['n_parts'])} parte(s) · {int(r['n_holes'])} hueco(s) · perímetro total con "
                               f"huecos: {r['perimeter_m']:,.1f} m")

                with st.expander("Forma de la huella", expanded=bool(r["n_holes"] or r["n_parts"] > 1)):
                    st.checkbox("Rellenar patios y huecos interiores", key="shape_fill",
                                help="Huella exterior de la edificación: incluye claustros y patios.")
                    st.slider("Unir partes separadas (px)", 0, 15, key="shape_close",
                              help="Cierre morfológico: une partes separadas por menos del doble de este valor.")
                    st.checkbox("Conservar solo la parte principal", key="shape_largest")

                # ---- Categoría (asignada por el usuario; sugerencia opcional con RemoteCLIP) ----
                cat_key = f"cat_{editing}"
                custom = sorted({o.label for o in catalog} - set(CATEGORIES) - {"Sin categoría"})
                options = ["Sin categoría"] + CATEGORIES + custom
                if cat_key not in st.session_state:
                    st.session_state[cat_key] = (catalog.get(editing).label if editing
                                                 else st.session_state.label_default)
                if st.session_state[cat_key] not in options:
                    options.append(st.session_state[cat_key])
                label = st.selectbox("Categoría (la asigna usted; SAM 2 no clasifica)", options, key=cat_key,
                                     accept_new_options=True, help="Elija de la lista o escriba una nueva.")
                label = (label or "Sin categoría").strip() or "Sin categoría"

                ok_sem, why_sem = classifier_available()
                ckey = _cand_key()
                sugg = st.session_state.suggestions
                if sugg is None or sugg[0] != ckey:
                    if st.button("✨ Sugerir categoría (RemoteCLIP)", width="stretch", disabled=not ok_sem,
                                 help="Clasificador zero-shot de teledetección. Es una sugerencia: usted decide."
                                 if ok_sem else why_sem):
                        x0, y0, x1, y1 = _crop_box([mask_c], pad=0)
                        crop = object_crop(mosaic["img_array"], (x0, y0, x1, y1))
                        try:
                            with st.spinner("Clasificando el recorte..."):
                                st.session_state.suggestions = (ckey, get_semantic_classifier().classify(crop))
                        except Exception as exc:
                            st.error(f"No se pudo usar RemoteCLIP: {exc}")
                        st.rerun()
                else:
                    st.caption("Sugerencias de RemoteCLIP — es una ayuda, no una clasificación verificada. "
                               "Pulse una para adoptarla:")
                    for cat, prob in sugg[1]:
                        st.button(f"{cat} · {prob:.0%}", key=f"sug_{cat}", width="stretch",
                                  on_click=_pick_suggestion, args=(cat, cat_key))

                a1, a2 = st.columns(2)
                with a1:
                    if st.button("✅ Aceptar objeto", type="primary", width="stretch"):
                        backend_name = type(engine.backend).__name__
                        src_label = ("RemoteCLIP (confirmada por el usuario)"
                                     if st.session_state.suggested_pick == label else "usuario")
                        if editing:
                            catalog.replace_geometry(editing, mask_c, gdf_c, cand.score, seg.to_dict(),
                                                     engine.model_key, backend_name)
                            catalog.update(editing, label=label, label_source=src_label)
                            st.toast(f"{editing} actualizado.", icon="✏️")
                        else:
                            obj = catalog.add_from_mask(mask_c, gdf_c, cand.score, engine.model_key, backend_name,
                                                        label=label, prompts=seg.to_dict(), label_source=src_label)
                            st.session_state.focus_object_id = obj.object_id
                            st.toast(f"{obj.object_id} añadido al catálogo.", icon="🎉")
                        st.session_state.label_default = label
                        st.session_state.pop(cat_key, None)
                        _reset_editing()
                        st.rerun()
                with a2:
                    if st.button("🗑️ Descartar", width="stretch"):
                        st.session_state.pop(cat_key, None)
                        _reset_editing()
                        st.toast("Segmentación descartada.", icon="ℹ️")
                        st.rerun()

# =============================================================================
# 7–10. CATÁLOGO, MÉTRICAS, ESTADÍSTICAS, RELACIONES, EXPORTACIÓN
# =============================================================================
if mosaic_loaded:
    st.markdown("---")
    tab_obj, tab_det, tab_stats, tab_rel, tab_prod, tab_exp, tab_auto = st.tabs([
        f"🗂️ Objetos ({len(catalog)})", "📐 Detalle y métricas", "📊 Estadísticas",
        "🔗 Relaciones espaciales", "🧩 Productos", "💾 Exportar", "🤖 Segmentación automática",
    ])

    # ---- 9. Administración ----------------------------------------------------
    with tab_obj:
        if len(catalog) == 0:
            st.caption("Aún no hay objetos. Acepte una segmentación para añadirla al catálogo.")
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
                    oid = ids[int(row_idx)]
                    if "label" in fields:
                        fields["label"] = str(fields["label"]).strip() or "Sin categoría"
                        fields["label_source"] = "usuario"  # editada a mano: muestra confirmada
                    elif fields.get("status") == "aceptado" and "sin confirmar" in catalog.get(oid).label_source:
                        # aceptar un objeto con categoría propuesta equivale a confirmarla
                        src = catalog.get(oid).label_source.split(" ")[0]
                        fields["label_source"] = f"{src} (confirmada por el usuario)"
                    if fields:
                        catalog.update(oid, **fields)

            st.data_editor(
                table, key=editor_key, on_change=_apply_edits, hide_index=True, width="stretch",
                disabled=["object_id", "area_m2", "perimeter_m", "pred_iou", "compactness", "origin", "version"],
                column_config={
                    "ver": st.column_config.CheckboxColumn("Ver", width="small"),
                    "object_id": st.column_config.TextColumn("ID"),
                    "label": st.column_config.TextColumn("Etiqueta"),
                    "status": st.column_config.SelectboxColumn("Estado", options=list(STATUSES), required=True),
                    "area_m2": st.column_config.NumberColumn("Área (m²)", format="%.1f"),
                    "perimeter_m": st.column_config.NumberColumn("Perímetro (m)", format="%.1f"),
                    "pred_iou": st.column_config.NumberColumn("IoU pred.", format="%.3f"),
                    "compactness": st.column_config.NumberColumn("Compacidad", format="%.3f"),
                    "origin": st.column_config.TextColumn("Origen"),
                    "version": st.column_config.NumberColumn("Versión"),
                },
            )
            ids = catalog.ids()
            focus = st.session_state.focus_object_id if st.session_state.focus_object_id in ids else ids[0]
            sel = st.selectbox("Objeto seleccionado", ids, index=ids.index(focus),
                               format_func=lambda i: f"{i} · {catalog.get(i).label} · {STATUS_LABELS[catalog.get(i).status]}")
            st.session_state.focus_object_id = sel
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if st.button("🔎 Centrar en el mapa", width="stretch"):
                    r = catalog.get(sel).gdf.iloc[0]
                    st.session_state.map_center = [float(r["centroid_lat"]), float(r["centroid_lon"])]
                    st.session_state.map_zoom = max(int(mosaic.get("zoom", 18)), 18)
                    st.rerun()
            with b2:
                if st.button("✏️ Refinar", width="stretch", disabled=engine is None):
                    with st.spinner("Reconstruyendo prompts..."):
                        start_refine(sel)
                    st.rerun()
            with b3:
                if st.button("🗑️ Eliminar", width="stretch"):
                    if st.session_state.editing_object_id == sel:
                        _reset_editing()
                    catalog.remove(sel)
                    st.rerun()
            with b4:
                n_rej = len(catalog.filter(["rechazado"]))
                if st.button(f"Eliminar rechazados ({n_rej})", width="stretch", disabled=n_rej == 0):
                    for o in catalog.filter(["rechazado"]):
                        catalog.remove(o.object_id)
                    st.rerun()

            st.markdown("**Fusionar objetos** — p. ej. nave, claustro y alas de una misma edificación")
            f1, f2 = st.columns([3, 1])
            to_merge = f1.multiselect("Objetos a fusionar", ids, label_visibility="collapsed",
                                      format_func=lambda i: f"{i} · {catalog.get(i).label}",
                                      placeholder="Elija dos o más objetos")
            if f2.button("🔗 Fusionar", width="stretch", disabled=len(to_merge) < 2):
                so = shape_options()
                conf = min(catalog.get(i).pred_iou for i in to_merge)
                try:
                    new = catalog.merge(
                        to_merge, label=catalog.get(to_merge[0]).label,
                        postprocess=lambda m: refine_mask(m, min_part_px=min_pixel_area, **so),
                        gdf_builder=lambda m: mask_to_geodataframe(m, mosaic["transform_3857"], confidence=conf,
                                                                   min_pixel_area=min_pixel_area,
                                                                   simplify_px=simplify_px))
                    st.session_state.focus_object_id = new.object_id
                    st.toast(f"{len(to_merge)} objetos fusionados en {new.object_id}.", icon="🔗")
                except ValueError as exc:
                    st.error(str(exc))
                st.rerun()
            st.caption("La fusión usa las opciones de *Forma de la huella* del panel (p. ej. rellenar patios).")

            unl = [o for o in catalog if o.label in ("sin etiquetar", "Sin categoría")]
            ok_sem, why_sem = classifier_available()
            if unl and st.button(f"✨ Sugerir categoría a {len(unl)} objeto(s) sin categoría (RemoteCLIP)",
                                 disabled=not ok_sem, help=None if ok_sem else why_sem):
                with st.spinner("Clasificando recortes..."):
                    clf = get_semantic_classifier()
                    for o in unl:
                        top = clf.classify(object_crop(mosaic["img_array"], o.bbox_px), top_k=1)[0]
                        catalog.update(o.object_id, label=top[0], label_source=f"RemoteCLIP {top[1]:.0%} (sin confirmar)")
                st.toast("Categorías sugeridas: revíselas antes de aceptar los objetos.", icon="✨")
                st.rerun()

    # ---- 8. Detalle y métricas -------------------------------------------------
    with tab_det:
        fid = st.session_state.focus_object_id
        if not fid or fid not in catalog:
            st.caption("Seleccione un objeto en la pestaña Objetos.")
        else:
            obj = catalog.get(fid)
            r = obj.gdf.iloc[0]
            st.markdown(f"#### {obj.object_id} · {obj.label} · {STATUS_LABELS[obj.status]} · v{obj.version}")
            st.caption(f"Categoría «{obj.label}» · fuente: {obj.label_source}")
            if obj.touches_edge:
                st.warning("Este objeto toca el borde de la ortofoto: puede estar recortado.")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Área planar", f"{r['area_m2']:,.2f} m²", help=f"En {r['metric_crs']}")
            d2.metric("Área geodésica", f"{r['area_geod_m2']:,.2f} m²", help="Sobre el elipsoide WGS84")
            d3.metric("Perímetro", f"{r['perimeter_m']:,.2f} m")
            d4.metric("IoU predicho", f"{obj.pred_iou:.3f}")
            metrics_df = pd.DataFrame([
                ("Perímetro exterior (sin patios)", r.get("perimeter_ext_m", r["perimeter_m"]), "m"),
                ("Perímetro geodésico", r["perimeter_geod_m"], "m"),
                ("Área exacta de la máscara", r["pixel_area_m2"], "m²"),
                ("Píxeles de la máscara", r["pixel_count"], "px"),
                ("GSD real del terreno", r["gsd_m"], "m/px"),
                ("Área del convex hull", r["hull_area_m2"], "m²"),
                ("Solidez (A / A hull)", r["solidity"], "—"),
                ("Compacidad Polsby-Popper (4πA/P²)", r["compactness"], "—"),
                ("Rectangularidad (A / A rect. mín.)", r["rectangularity"], "—"),
                ("Elongación (1 − ancho/largo)", r["elongation"], "—"),
                ("Relación de aspecto (largo/ancho)", r["aspect_ratio"], "—"),
                ("Orientación del eje largo (acimut de cuadrícula)", r["orientation_deg"], "°"),
                ("Largo × ancho del rectángulo mínimo", f"{r['mrr_length_m']:.2f} × {r['mrr_width_m']:.2f}", "m"),
                ("Partes / huecos", f"{r['n_parts']} / {r['n_holes']}", "—"),
                ("Centroide (lat, lon)", f"{r['centroid_lat']:.7f}, {r['centroid_lon']:.7f}", "°"),
                ("Bounding box (O, S, E, N)", f"{r['min_lon']:.6f}, {r['min_lat']:.6f}, {r['max_lon']:.6f}, {r['max_lat']:.6f}", "°"),
                ("Color dominante (cubierta/superficie)",
                 color_stats(mosaic["img_array"], obj.mask())["color_dominante"], ""),
                ("CRS métrico", r["metric_crs"], ""),
                ("Modelo / backend", f"{obj.model_id} / {obj.backend}", ""),
                ("Origen", obj.origin, ""),
                ("Fuente de la categoría", obj.label_source, ""),
            ], columns=["Métrica", "Valor", "Unidad"])
            metrics_df["Valor"] = metrics_df["Valor"].astype(str)
            st.dataframe(metrics_df, hide_index=True, width="stretch")
            mask = obj.mask()
            cut, _bb = extract_cutout(mosaic["img_array"], mask, crop_to_bbox=True)
            i1, i2 = st.columns(2)
            i1.image(cut, caption="Recorte con canal alfa", width="stretch")
            x0, y0, x1, y1 = _crop_box([mask], pad=40)
            i2.image(create_mask_overlay(mosaic["img_array"][y0:y1, x0:x1], mask[y0:y1, x0:x1]),
                     caption="Máscara sobre la ortofoto", width="stretch")

    # ---- Estadísticas -------------------------------------------------------------
    with tab_stats:
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
            st.markdown("**Por etiqueta**")
            st.dataframe(stats["por_etiqueta"].rename(columns={"label": "Etiqueta", "n": "Objetos",
                                                                "area_m2": "Área total (m²)"}),
                         hide_index=True, width="stretch")
            gacc = catalog.to_geodataframe(("aceptado",))
            st.markdown("**Área por objeto (m²)**")
            st.bar_chart(gacc.set_index("object_id")["area_m2"], height=220)

    # ---- Relaciones espaciales --------------------------------------------------
    with tab_rel:
        gacc = catalog.to_geodataframe(("aceptado",))
        if len(gacc) < 2:
            st.caption("Se necesitan al menos dos objetos aceptados.")
        else:
            dmax = st.number_input("Listar pares a menos de (m)", min_value=0.0, value=50.0, step=10.0)
            summary, pairs = spatial_relations(gacc, max_pairs_distance_m=dmax)
            st.markdown("**Vecino más cercano y contactos**")
            st.dataframe(summary[["object_id", "label", "nn_id", "nn_dist_m", "n_contacts"]].rename(columns={
                "object_id": "ID", "label": "Etiqueta", "nn_id": "Vecino más cercano",
                "nn_dist_m": "Distancia (m)", "n_contacts": "Contactos"}), hide_index=True, width="stretch")
            st.markdown(f"**Pares a menos de {dmax:g} m**")
            st.dataframe(pairs, hide_index=True, width="stretch")
            st.caption("Distancias euclidianas en el CRS métrico (EPSG:9377 o UTM); "
                       "relación según el modelo topológico DE-9IM (shapely).")


    # ---- Productos ---------------------------------------------------------------
    with tab_prod:
        prod = st.segmented_control(
            "Producto", ["sup", "cov", "coord", "ind"], default="sup", key="product_sel",
            format_func={"sup": "🎓 Clasificación supervisada", "cov": "🗺️ Mapa de coberturas",
                         "coord": "📍 Coordenadas y linderos", "ind": "🏙️ Indicadores e informe"}.get) or "sup"
        T = mosaic["transform_3857"]
        img = mosaic["img_array"]

        def _features(o, use_clip):
            key = (o.object_id, o.version, use_clip)
            fc = st.session_state.feat_cache
            if key not in fc:
                mk = o.mask()
                v = handcrafted_features(o.attributes(), color_stats(img, mk))
                if use_clip:
                    v = np.concatenate([v, get_semantic_classifier().embed(object_crop(img, o.bbox_px))])
                fc[key] = v
            return fc[key]

        if prod == "sup":
            st.markdown(
                "Clasificación **supervisada orientada a objetos (GEOBIA)**: SAM 2 produce los segmentos, sus "
                "categorías confirmadas son las **muestras de entrenamiento** y una regresión logística multinomial "
                "aprende de descriptores de **forma** y **color** (y opcionalmente del embedding de RemoteCLIP) para "
                "clasificar el resto (p. ej. los objetos propuestos por la segmentación automática).")
            train = [o for o in catalog if training_eligible(o.label, o.status, o.label_source)]
            counts = pd.Series([o.label for o in train], dtype=object).value_counts()
            targets = [o for o in catalog if not training_eligible(o.label, o.status, o.label_source)
                       and o.status != "rechazado"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Muestras de entrenamiento", len(train))
            c2.metric("Categorías", len(counts))
            c3.metric("Objetos por clasificar", len(targets))
            if len(counts):
                st.dataframe(counts.rename_axis("Categoría").reset_index(name="Muestras"), hide_index=True,
                             width="stretch")
            ok_sem, _why = classifier_available()
            use_clip = st.checkbox("Añadir descriptores de RemoteCLIP", value=False, disabled=not ok_sem,
                                   help="En la prueba con 71 objetos reales no mejoró la exactitud frente a forma + color "
                                        "y descarga 605 MB; útil con más muestras.")
            ready = len(counts) >= 2 and counts.min() >= 3
            if not ready:
                st.info("Para entrenar necesita al menos **2 categorías con 3 o más objetos aceptados** cada una "
                        "(recomendado: 10 o más por categoría). Edite la categoría y el estado en la pestaña Objetos; "
                        "las categorías *sin confirmar* no cuentan como muestras.")
            b1, b2 = st.columns(2)
            if b1.button("🎓 Entrenar y validar", type="primary", width="stretch", disabled=not ready):
                with st.spinner("Calculando descriptores y validando..."):
                    X = np.stack([_features(o, use_clip) for o in train])
                    y = np.array([o.label for o in train])
                    st.session_state.sup_cv = SupervisedClassifier.cross_validate(X, y)
                    st.session_state.sup_model = (SupervisedClassifier().fit(X, y), use_clip)
                st.rerun()
            cv = st.session_state.sup_cv
            if cv is not None:
                v1, v2, v3 = st.columns(3)
                v1.metric("Exactitud global", f"{cv['exactitud']:.1%}")
                v2.metric("Índice kappa", f"{cv['kappa']:.3f}", help="Acuerdo corregido por azar: >0,6 sustancial, >0,8 casi perfecto.")
                v3.metric("Muestras validadas", cv["n"])
                st.caption(f"Método: {cv['metodo']}. Medido con objetos no vistos por el modelo en cada pliegue.")
                st.dataframe(cv["por_clase"].rename(columns={"categoria": "Categoría", "muestras": "Muestras",
                             "precision": "Precisión", "exhaustividad": "Exhaustividad", "f1": "F1"}),
                             hide_index=True, width="stretch")
                with st.expander("Matriz de confusión (filas = real, columnas = predicha)"):
                    st.dataframe(cv["matriz_confusion"], width="stretch")
            model = st.session_state.sup_model
            min_p = st.slider("Probabilidad mínima para asignar categoría", 0.3, 0.95, 0.6, 0.05)
            if b2.button(f"🏷️ Clasificar {len(targets)} objeto(s)", width="stretch",
                         disabled=model is None or not targets):
                clf, with_clip = model
                with st.spinner("Clasificando..."):
                    X = np.stack([_features(o, with_clip) for o in targets])
                    n_ok = 0
                    for o, (cat, p) in zip(targets, clf.predict(X)):
                        if p >= min_p:
                            catalog.update(o.object_id, label=cat, label_source=f"Supervisada {p:.0%} (sin confirmar)")
                            n_ok += 1
                        else:
                            catalog.update(o.object_id, label="Sin categoría",
                                           label_source=f"Supervisada: confianza baja ({p:.0%})")
                st.toast(f"{n_ok} objetos clasificados; revíselos y acéptelos en la pestaña Objetos.", icon="🏷️")
                st.rerun()
            if train:
                feats = pd.DataFrame([{"object_id": o.object_id, "categoria": o.label,
                                       **dict(zip(SHAPE_AND_COLOR, _features(o, False)))} for o in train])
                st.download_button(on_click="ignore", label="⬇️ Muestras de entrenamiento (CSV)",
                                   data=feats.to_csv(index=False).encode("utf-8"), file_name="muestras_entrenamiento.csv",
                                   mime="text/csv")

        elif prod == "cov":
            st.markdown("Mapa temático de **coberturas** a partir de las categorías de los objetos "
                        "(lo no segmentado queda *Sin clasificar*).")
            incl = st.multiselect("Estados incluidos", list(STATUSES), default=["aceptado", "propuesto"],
                                  format_func=lambda s_: STATUS_LABELS[s_], key="cov_status")
            objs_c = [o for o in catalog.filter(incl) if o.label not in UNLABELED]
            objs_c.sort(key=lambda o: (o.status == "aceptado", -o.pixel_count))  # grandes primero, aceptados encima
            if not objs_c:
                st.caption("No hay objetos con categoría en los estados elegidos.")
            else:
                raster, legend = coverage_raster([(o.mask(), o.label) for o in objs_c], img.shape[:2])
                px_area, zone_area, crs_m = mosaic_pixel_area_m2(T, mosaic["width"], mosaic["height"])
                stats_c = coverage_stats(raster, legend, px_area)
                cc1, cc2 = st.columns([3, 2])
                rgb = coverage_rgb(raster, legend)
                cc1.image((0.55 * rgb + 0.45 * img).astype(np.uint8), caption="Coberturas sobre la ortofoto",
                          width="stretch")
                leg = "".join(f"<div><span style='display:inline-block;width:12px;height:12px;margin-right:6px;"
                              f"background:rgb{category_color(n)}'></span>{n}</div>" for n in legend.values())
                cc2.markdown(leg, unsafe_allow_html=True)
                cc2.dataframe(stats_c.rename(columns={"categoria": "Categoría", "pixeles": "Píxeles",
                              "area_m2": "Área (m²)", "porcentaje": "%"}), hide_index=True, width="stretch")
                cc2.caption(f"Área de la zona: {zone_area:,.0f} m² · píxel medio {px_area:.3f} m² ({crs_m}).")
                colors = {k: category_color(v) for k, v in legend.items()}
                d1, d2 = st.columns(2)
                d1.download_button(on_click="ignore", label="🗺️ Coberturas GeoTIFF (con paleta)", width="stretch",
                                   data=lambda r_=raster, l_=legend, c_=colors: export_coverage_geotiff(r_, l_, c_, T),
                                   file_name="coberturas.tif", mime="image/tiff")
                d2.download_button(on_click="ignore", label="📋 Estadísticas de coberturas (CSV)", width="stretch",
                                   data=stats_c.to_csv(index=False).encode("utf-8"), file_name="coberturas.csv",
                                   mime="text/csv")

        elif prod == "coord":
            st.markdown("**Cuadro de coordenadas y linderos** del contorno de un objeto en coordenadas planas "
                        "(EPSG:9377 en Colombia), numerado en sentido horario desde el vértice más al norte.")
            st.caption("Los vértices son bordes vistos en la ortofoto (p. ej. aleros de cubierta), no linderos legales: "
                       "insumo preliminar que no sustituye un levantamiento topográfico.")
            ids_c = catalog.ids()
            if not ids_c:
                st.caption("No hay objetos.")
            else:
                fid = st.session_state.focus_object_id if st.session_state.focus_object_id in ids_c else ids_c[0]
                oid = st.selectbox("Objeto", ids_c, index=ids_c.index(fid), key="coord_obj",
                                   format_func=lambda i: f"{i} · {catalog.get(i).label}")
                tol = st.slider("Generalización del contorno (m)", 0.1, 3.0, 0.5, 0.1,
                                help="Tolerancia Douglas-Peucker: menos vértices con valores mayores.")
                o = catalog.get(oid)
                vt = vertex_table(o.gdf.geometry.iloc[0], tolerance_m=tol)
                vt_show = vt.assign(acimut_dms=vt["acimut_deg"].map(azimuth_to_dms))
                q1, q2, q3 = st.columns(3)
                q1.metric("Vértices", len(vt))
                q2.metric("Área (polígono generalizado)", f"{vt.attrs['area_m2']:,.2f} m²")
                q3.metric("Perímetro", f"{vt.attrs['perimetro_m']:,.2f} m")
                st.dataframe(vt_show.rename(columns={"vertice": "Vértice", "este_x": "Este X (m)", "norte_y": "Norte Y (m)",
                             "lindero": "Lindero", "distancia_m": "Distancia (m)", "acimut_deg": "Acimut (°)",
                             "acimut_dms": "Acimut (° ' \")"}), hide_index=True, width="stretch")
                st.caption(f"CRS: {vt.attrs['crs']} · tolerancia {tol} m.")
                e1, e2 = st.columns(2)
                e1.download_button(on_click="ignore", label="📋 Cuadro de coordenadas (CSV)", width="stretch",
                                   data=vt_show.to_csv(index=False).encode("utf-8"),
                                   file_name=f"coordenadas_{oid}.csv", mime="text/csv")
                e2.download_button(on_click="ignore", label="📐 Objeto en DXF", width="stretch",
                                   data=lambda g=o.gdf.assign(label=o.label): export_dxf(g),
                                   file_name=f"{oid}_9377.dxf", mime="application/dxf")

        else:  # indicadores
            st.markdown("**Indicadores urbanos** de la zona y, si carga una capa de predios o manzanas, por unidad.")
            incl = st.multiselect("Estados incluidos", list(STATUSES), default=["aceptado"],
                                  format_func=lambda s_: STATUS_LABELS[s_], key="ind_status")
            gi = catalog.to_geodataframe(incl)
            _px, zone_area, crs_m = mosaic_pixel_area_m2(T, mosaic["width"], mosaic["height"])
            ind = zone_indicators(gi, zone_area)
            if ind.get("n_objetos", 0) == 0:
                st.caption("No hay objetos en los estados elegidos.")
            else:
                i1, i2, i3, i4 = st.columns(4)
                i1.metric("Índice de ocupación", f"{ind['indice_ocupacion']:.3f}",
                          help="Área construida en planta / área de la zona (sin alturas: no es índice de construcción).")
                i2.metric("Edificaciones", ind["n_edificaciones"], help=f"{ind['densidad_edificaciones_ha']:.2f} por hectárea")
                i3.metric("Área construida", f"{ind['area_construida_m2']:,.0f} m²")
                i4.metric("Cobertura verde", f"{ind['cobertura_verde_pct']:.1f} %")
                st.caption(f"Zona analizada: {zone_area:,.0f} m² ({crs_m}). Edificación = categorías «Edificación…», "
                           "«Edificio en altura» y «Cubierta*»; verde = «Zona verde» y «Arbolado».")
                built_objs = [o for o in catalog.filter(incl) if is_built(o.label)]
                if built_objs:
                    tip = pd.Series([color_stats(img, o.mask())["color_dominante"] for o in built_objs]).value_counts()
                    st.markdown("**Tipología de cubiertas por color dominante** (insumo para caracterización constructiva)")
                    st.dataframe(tip.rename_axis("Color").reset_index(name="Edificaciones"), hide_index=True,
                                 width="stretch")

            st.markdown("**Análisis por predio o manzana**")
            up = st.file_uploader("Capa de unidades (GeoJSON, GeoPackage o Shapefile en .zip)",
                                  type=["geojson", "json", "gpkg", "zip"], key="units_upload")
            if up is not None and not gi.empty:
                import tempfile as _tf, os as _os
                try:
                    with _tf.TemporaryDirectory() as td:
                        path = _os.path.join(td, up.name)
                        open(path, "wb").write(up.getbuffer())
                        units = gpd.read_file(f"zip://{path}" if up.name.endswith(".zip") else path)
                    fields = [c for c in units.columns if c != "geometry"]
                    if not fields:
                        units["unidad"] = [f"U{i + 1}" for i in range(len(units))]
                        fields = ["unidad"]
                    idf = st.selectbox("Campo identificador", fields)
                    res = units_indicators(gi, units, idf)
                    st.dataframe(res.drop(columns="geometry"), hide_index=True, width="stretch")
                    st.download_button(on_click="ignore", label="📦 Indicadores por unidad (GeoPackage)",
                                       data=lambda r_=res: export_geopackage({"indicadores_unidad": r_}),
                                       file_name="indicadores_por_unidad.gpkg", mime="application/geopackage+sqlite3")
                except Exception as exc:
                    st.error(f"No se pudo procesar la capa: {exc}")

            if ind.get("n_objetos", 0):
                cov_df = None
                objs_c = [o for o in catalog.filter(incl) if o.label not in UNLABELED]
                if objs_c:
                    rr, lg = coverage_raster([(o.mask(), o.label) for o in objs_c], img.shape[:2])
                    cov_df = coverage_stats(rr, lg, _px)
                meta = {
                    "Fecha (UTC)": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
                    "Fuente de imagen": f"Esri World Imagery, zoom {mosaic.get('zoom')}",
                    "Extensión (O, S, E, N)": ", ".join(f"{v:.6f}" for v in mosaic["bounds_4326"]),
                    "Tamaño del mosaico": f"{mosaic['width']} × {mosaic['height']} px",
                    "CRS métrico": crs_m, "Modelo de segmentación": engine.model_key if engine else "—",
                    "Estados incluidos": ", ".join(incl),
                }
                tbl = gi.drop(columns="geometry")[[c for c in ["object_id", "label", "status", "area_m2",
                                                               "perimeter_ext_m", "pred_iou", "label_source"]
                                                   if c in gi.columns]]
                st.download_button(
                    on_click="ignore", label="📄 Informe técnico (HTML, imprimible a PDF)", type="primary",
                    data=lambda m_=meta, i_=ind, c_=cov_df, t_=tbl, v_=st.session_state.sup_cv:
                        technical_report_html(m_, i_, c_, t_, v_).encode("utf-8"),
                    file_name="informe_tecnico_zona.html", mime="text/html")

    # ---- 10. Exportación --------------------------------------------------------
    with tab_exp:
        scope = st.multiselect("Estados a exportar", list(STATUSES), default=["aceptado"],
                               format_func=lambda s_: STATUS_LABELS[s_])
        gexp = catalog.to_geodataframe(scope)
        objs_exp = catalog.filter(scope)
        T = mosaic["transform_3857"]
        img = mosaic["img_array"]
        if gexp.empty:
            st.caption("No hay objetos con los estados seleccionados.")
        st.markdown("**Vectores**")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.download_button(on_click="ignore", label="🗺️ GeoJSON", data=lambda g=gexp: export_geojson(g), file_name="objetos_sam2.geojson",
                               mime="application/geo+json", width="stretch", disabled=gexp.empty)
        with v2:
            st.download_button(on_click="ignore", label=
                "📦 GeoPackage", width="stretch", disabled=gexp.empty, file_name="sam2_geoextractor.gpkg",
                mime="application/geopackage+sqlite3",
                data=lambda g=gexp: export_geopackage({"objetos": g, **derived_geometries(g)}),
                help="Capas: objetos, centroides, bbox, convex_hull, rect_min_rotado.")
        with v3:
            st.download_button(on_click="ignore", label="📁 Shapefile (.zip)", data=lambda g=gexp: export_shapefile_zip(g),
                               file_name="objetos_sam2_shp.zip", mime="application/zip", width="stretch",
                               disabled=gexp.empty, help="Incluye campos.csv con los nombres completos.")
        with v4:
            st.download_button(on_click="ignore", label="📋 CSV de atributos", data=lambda g=gexp: export_csv(g), file_name="objetos_sam2.csv",
                               mime="text/csv", width="stretch", disabled=gexp.empty)
        k1, k2, _k3, _k4 = st.columns(4)
        with k1:
            st.download_button(on_click="ignore", label="📐 DXF (CAD, EPSG:9377)", data=lambda g=gexp: export_dxf(g),
                               file_name="objetos_sam2_9377.dxf", mime="application/dxf", width="stretch",
                               disabled=gexp.empty, help="Una capa CAD por categoría, coordenadas Origen Nacional.")
        with k2:
            st.download_button(on_click="ignore", label="🌍 KML (Google Earth)", data=lambda g=gexp: export_kml(g),
                               file_name="objetos_sam2.kml", mime="application/vnd.google-earth.kml+xml",
                               width="stretch", disabled=gexp.empty)
        st.markdown("**Ráster y datasets**")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.download_button(on_click="ignore", label="🌐 Ortofoto GeoTIFF", data=lambda: export_geotiff(img, T, CRS_WEB_MERCATOR),
                               file_name="ortofoto_3857.tif", mime="image/tiff", width="stretch")
        with r2:
            st.download_button(on_click="ignore", label=
                "🏷️ Máscaras GeoTIFF", width="stretch", disabled=not objs_exp, file_name="mascaras_etiquetas.tif",
                mime="image/tiff", help="uint16: 0 = fondo; cada objeto con su raster_value.",
                data=lambda o=objs_exp: export_label_geotiff([x.mask() for x in o], [x.raster_value for x in o], T))
        with r3:
            st.download_button(on_click="ignore", label=
                "🧠 Dataset COCO (.zip)", width="stretch", disabled=not objs_exp, file_name="dataset_coco_sam2.zip",
                mime="application/zip", help="Imagen PNG + anotaciones RLE + georreferencia.",
                data=lambda o=objs_exp: export_coco_dataset(img, [{
                    "object_id": x.object_id, "label": x.label, "mask_rle": x.mask_rle, "bbox_px": x.bbox_px,
                    "pixel_count": x.pixel_count, "pred_iou": x.pred_iou,
                    "area_m2": float(x.gdf["area_m2"].iloc[0])} for x in o], T))
        with r4:
            fid = st.session_state.focus_object_id
            fobj = catalog.get(fid) if fid in catalog else None
            st.download_button(on_click="ignore", label=
                "🖼️ PNG del objeto", width="stretch", disabled=fobj is None, mime="image/png",
                file_name=f"{fid}.png" if fobj else "objeto.png",
                help="Recorte RGBA del objeto seleccionado en la pestaña Objetos.",
                data=lambda o=fobj: export_png(extract_cutout(img, o.mask(), crop_to_bbox=True)[0]))

    # ---- Segmentación automática --------------------------------------------------
    with tab_auto:
        st.markdown(
            "Genera máscaras para **todo** el mosaico con `SAM2AutomaticMaskGenerator` (función oficial de SAM 2). "
            "Los resultados entran al catálogo como **propuestos** para que usted los acepte o rechace. "
            "Es costoso en CPU: en 2 vCPU, 16×16 puntos tardan del orden de 35-40 s."
        )
        g1, g2, g3, g4 = st.columns(4)
        pps = g1.selectbox("Puntos por lado", [8, 16, 24, 32], index=1)
        iou_t = g2.slider("Umbral IoU pred.", 0.5, 0.99, 0.8, 0.01)
        stab_t = g3.slider("Umbral de estabilidad", 0.5, 0.99, 0.92, 0.01)
        min_m2 = g4.number_input("Área mínima (m²)", min_value=0.0, value=5.0, step=1.0)
        if st.button("🤖 Ejecutar segmentación automática", disabled=engine is None):
            with st.spinner(f"Generando máscaras ({pps}×{pps} puntos)..."):
                masks, secs = engine.segment_everything(
                    img, points_per_side=int(pps), points_per_batch=64,
                    pred_iou_thresh=float(iou_t), stability_score_thresh=float(stab_t))
            added = 0
            for cand in sorted(masks, key=lambda c: -c.score)[:300]:
                g = mask_to_geodataframe(cand.mask, T, confidence=cand.score,
                                         min_pixel_area=min_pixel_area, simplify_px=simplify_px)
                if g.empty or float(g["area_m2"].iloc[0]) < min_m2:
                    continue
                catalog.add_from_mask(cand.mask, g, cand.score, engine.model_key, type(engine.backend).__name__,
                                      label="sin etiquetar", origin="automático", status="propuesto")
                added += 1
            st.toast(f"{added} objetos propuestos en {secs:.0f} s.", icon="🤖")
            st.rerun()
