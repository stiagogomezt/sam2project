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
from src.objects import ObjectCatalog, STATUSES
from src.visualization import create_base_map, build_dynamic_layers
from src.exporter import (
    export_geojson, export_geotiff, export_png, export_shapefile_zip, export_csv,
    export_geopackage, export_label_geotiff, export_coco_dataset,
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
    "label_default": "objeto",
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
def candidate_gdf(idx: int | None = None):
    """GeoDataFrame de la candidata (cacheado por estado de prompts y parámetros)."""
    if seg.result is None:
        return None
    idx = seg.selected_idx if idx is None else idx
    key = (id(seg.result), idx, min_pixel_area, simplify_px)
    cache = st.session_state.cand_cache or {}
    if key not in cache:
        cand = seg.candidates[idx]
        cache[key] = mask_to_geodataframe(cand.mask, mosaic["transform_3857"], confidence=cand.score,
                                          min_pixel_area=min_pixel_area, simplify_px=simplify_px)
        st.session_state.cand_cache = cache
    return cache[key]


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
        st.markdown("1. Zona de estudio · 2. Ortofoto · 3. Clic sobre el objeto · 4. SAM 2 · "
                    "5. Refinar · 6. Validar · 7. Objeto GIS · 8. Métricas · 9. Administrar · 10. Exportar")
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

            gdf_c = candidate_gdf()
            cand = seg.selected_candidate
            if gdf_c is None or gdf_c.empty:
                st.warning("La máscara elegida no produce polígonos con el área mínima configurada.")
            else:
                r = gdf_c.iloc[0]
                m1, m2 = st.columns(2)
                m1.metric("Área", f"{r['area_m2']:,.1f} m²")
                m2.metric("Perímetro", f"{r['perimeter_m']:,.1f} m")
                m3, m4 = st.columns(2)
                m3.metric("IoU predicho", f"{cand.score:.3f}",
                          help="Estimación de SAM 2 de la calidad de la máscara (IoU esperado con el objeto "
                               "real). No es una probabilidad de clase.")
                m4.metric("Inferencia", f"{seg.result['inference_time'] * 1000:.0f} ms")

                label_value = catalog.get(editing).label if editing else st.session_state.label_default
                label = st.text_input("Etiqueta (asignada por usted; SAM 2 no clasifica)", value=label_value,
                                      key=f"label_input_{editing}")
                a1, a2 = st.columns(2)
                with a1:
                    if st.button("✅ Aceptar objeto", type="primary", width="stretch"):
                        backend_name = type(engine.backend).__name__
                        if editing:
                            catalog.replace_geometry(editing, cand.mask, gdf_c, cand.score, seg.to_dict(),
                                                     engine.model_key, backend_name)
                            catalog.update(editing, label=label.strip() or "objeto")
                            st.toast(f"{editing} actualizado.", icon="✏️")
                        else:
                            obj = catalog.add_from_mask(cand.mask, gdf_c, cand.score, engine.model_key, backend_name,
                                                        label=label.strip() or "objeto", prompts=seg.to_dict())
                            st.session_state.focus_object_id = obj.object_id
                            st.toast(f"{obj.object_id} añadido al catálogo.", icon="🎉")
                        st.session_state.label_default = label.strip() or "objeto"
                        _reset_editing()
                        st.rerun()
                with a2:
                    if st.button("🗑️ Descartar", width="stretch"):
                        _reset_editing()
                        st.toast("Segmentación descartada.", icon="ℹ️")
                        st.rerun()

# =============================================================================
# 7–10. CATÁLOGO, MÉTRICAS, ESTADÍSTICAS, RELACIONES, EXPORTACIÓN
# =============================================================================
if mosaic_loaded:
    st.markdown("---")
    tab_obj, tab_det, tab_stats, tab_rel, tab_exp, tab_auto = st.tabs([
        f"🗂️ Objetos ({len(catalog)})", "📐 Detalle y métricas", "📊 Estadísticas",
        "🔗 Relaciones espaciales", "💾 Exportar", "🤖 Segmentación automática",
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
                    if "label" in fields:
                        fields["label"] = str(fields["label"]).strip() or "objeto"
                    if fields:
                        catalog.update(ids[int(row_idx)], **fields)

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

    # ---- 8. Detalle y métricas -------------------------------------------------
    with tab_det:
        fid = st.session_state.focus_object_id
        if not fid or fid not in catalog:
            st.caption("Seleccione un objeto en la pestaña Objetos.")
        else:
            obj = catalog.get(fid)
            r = obj.gdf.iloc[0]
            st.markdown(f"#### {obj.object_id} · {obj.label} · {STATUS_LABELS[obj.status]} · v{obj.version}")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Área planar", f"{r['area_m2']:,.2f} m²", help=f"En {r['metric_crs']}")
            d2.metric("Área geodésica", f"{r['area_geod_m2']:,.2f} m²", help="Sobre el elipsoide WGS84")
            d3.metric("Perímetro", f"{r['perimeter_m']:,.2f} m")
            d4.metric("IoU predicho", f"{obj.pred_iou:.3f}")
            metrics_df = pd.DataFrame([
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
                ("CRS métrico", r["metric_crs"], ""),
                ("Modelo / backend", f"{obj.model_id} / {obj.backend}", ""),
                ("Origen", obj.origin, ""),
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
