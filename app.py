"""
=============================================================================
SAM2_STREAMLIT_APP: Extracción Inteligente de Objetos Geoespaciales con IA
=============================================================================
Aplicación SIG avanzada para la delimitación, segmentación y extracción
de objetos vectoriales y ráster georreferenciados a partir de ortofotos
de alta resolución mediante Segment Anything Model 2 (SAM 2).
"""
import io
import time
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_folium import st_folium

from config import (
    DEFAULT_BBOX,
    DEFAULT_ZOOM,
    MIN_ZOOM,
    MAX_ZOOM,
    SAM2_MODELS,
    DEFAULT_SAM2_MODEL_KEY
)
from src.tile_manager import get_tiles_info, TileManagerError
from src.imagery_loader import download_tiles_concurrently
from src.mosaic_builder import build_mosaic
from src.sam_engine import get_sam2_engine
from src.segmentation import coordinates_to_pixel, extract_cutout, create_mask_overlay
from src.geospatial import mask_to_geodataframe
from src.visualization import create_geospatial_map
from src.exporter import (
    export_geojson,
    export_geotiff,
    export_png,
    export_shapefile_zip,
    export_csv
)

# Configuración de página
st.set_page_config(
    page_title="SAM 2 - Extracción Inteligente Geoespacial",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# INYECCIÓN DE ESTILOS CSS PROFESIONALES
# =============================================================================
st.markdown(
    """
    <style>
    /* Estilos generales y tipografía */
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-gpu { background-color: #dcfce7; color: #15803d; }
    .badge-cpu { background-color: #fef9c3; color: #854d0e; }
    iframe { cursor: crosshair !important; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
# =============================================================================
if "mosaic_data" not in st.session_state:
    st.session_state.mosaic_data = None # Almacena img_array, transforms, bounds
if "embeddings_computed" not in st.session_state:
    st.session_state.embeddings_computed = False
if "candidate_result" not in st.session_state:
    st.session_state.candidate_result = None # Objeto en evaluación (máscara, score, gdf)
if "accepted_result" not in st.session_state:
    st.session_state.accepted_result = None # Objeto aceptado para exportación
if "last_processed_click" not in st.session_state:
    st.session_state.last_processed_click = None
if "map_center" not in st.session_state:
    st.session_state.map_center = [
        (DEFAULT_BBOX["min_lat"] + DEFAULT_BBOX["max_lat"]) / 2.0,
        (DEFAULT_BBOX["min_lon"] + DEFAULT_BBOX["max_lon"]) / 2.0
    ]

# =============================================================================
# BARRA LATERAL (PANEL DE CONTROL GEOESPACIAL)
# =============================================================================
with st.sidebar:
    st.title("🛰️ SAM 2 GeoExtractor")
    st.caption("Extracción Inteligente de Objetos Geoespaciales con IA")
    st.markdown("---")
    
    # 1. Configuración del Modelo SAM 2
    st.subheader("1. Motor de IA (SAM 2)")
    model_options = list(SAM2_MODELS.keys())
    model_labels = [SAM2_MODELS[k]["name"] for k in model_options]
    
    selected_model_idx = st.selectbox(
        "Arquitectura del Modelo:",
        range(len(model_options)),
        format_func=lambda i: model_labels[i],
        index=0
    )
    selected_model_key = model_options[selected_model_idx]
    
    # Carga en caché del motor de IA
    try:
        engine = get_sam2_engine(model_key=selected_model_key)
        device_badge = "badge-gpu" if engine.device == "cuda" else "badge-cpu"
        st.markdown(
            f"**Dispositivo Activo:** <span class='status-badge {device_badge}'>{engine.device_name}</span>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error al inicializar SAM 2: {e}")
        engine = None
        
    st.markdown("---")
    
    # 2. Definición de la Zona de Estudio (Bounding Box)
    st.subheader("2. Zona de Estudio (BBOX)")
    st.caption("Ingrese las coordenadas geográficas en grados decimales (WGS84).")
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        min_lon = st.number_input("Oeste (Min Lon)", value=DEFAULT_BBOX["min_lon"], format="%.5f", step=0.001)
        min_lat = st.number_input("Sur (Min Lat)", value=DEFAULT_BBOX["min_lat"], format="%.5f", step=0.001)
    with c_col2:
        max_lon = st.number_input("Este (Max Lon)", value=DEFAULT_BBOX["max_lon"], format="%.5f", step=0.001)
        max_lat = st.number_input("Norte (Max Lat)", value=DEFAULT_BBOX["max_lat"], format="%.5f", step=0.001)
        
    zoom_level = st.slider("Nivel de Zoom de Teselas:", min_value=MIN_ZOOM, max_value=MAX_ZOOM, value=DEFAULT_ZOOM)
    
    # Validación y estimación previa de recursos
    tiles_estimate = None
    try:
        tiles_estimate = get_tiles_info(min_lon, min_lat, max_lon, max_lat, zoom=zoom_level)
        st.info(
            f"📦 **Estimación:** {tiles_estimate['num_tiles']} teselas "
            f"({tiles_estimate['pixel_shape'][1]} × {tiles_estimate['pixel_shape'][0]} px)\n\n"
            f"💾 Tamaño aprox: ~{tiles_estimate['estimated_mb']} MB | "
            f"Resolución: ~{tiles_estimate['res_m_per_px']} m/px"
        )
        if tiles_estimate["is_large"]:
            st.warning("⚠️ Área extensa seleccionada. La descarga y el procesamiento pueden tomar unos momentos.")
    except TileManagerError as e:
        st.error(f"❌ {e}")
        
    btn_download = st.button(
        "📥 Descargar y Cargar Ortofoto",
        use_container_width=True,
        type="primary",
        disabled=(tiles_estimate is None)
    )
    
    if btn_download and tiles_estimate and engine:
        progress_bar = st.progress(0, text="Iniciando descarga de teselas...")
        
        def update_progress(done, total):
            pct = int((done / total) * 100)
            progress_bar.progress(pct, text=f"Descargando teselas: {done}/{total} ({pct}%)")
            
        try:
            # Descarga y ensamblado del mosaico
            tile_images = download_tiles_concurrently(
                tiles_estimate["tiles"],
                progress_callback=update_progress
            )
            progress_bar.progress(100, text="Ensamblando mosaico y calculando georreferenciación...")
            
            mosaic = build_mosaic(tiles_estimate, tile_images)
            st.session_state.mosaic_data = mosaic
            st.session_state.map_center = [tiles_estimate["centroid"][0], tiles_estimate["centroid"][1]]
            st.session_state.candidate_result = None
            st.session_state.accepted_result = None
            st.session_state.last_processed_click = None
            
            # Pre-cálculo de Image Embeddings en SAM 2 (Crítico para inferencia rápida)
            progress_bar.progress(100, text="Precalculando embeddings de SAM 2...")
            embed_time = engine.set_image(mosaic["img_array"])
            st.session_state.embeddings_computed = True
            
            progress_bar.empty()
            st.success(f"✅ Ortofoto cargada y embeddings listos en {embed_time:.2f} s.")
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            st.error(f"Error durante el procesamiento: {e}")

# =============================================================================
# PANEL PRINCIPAL
# =============================================================================
st.title("Extracción Inteligente de Objetos Geoespaciales")

mosaic_loaded = st.session_state.mosaic_data is not None

if not mosaic_loaded:
    st.info(
        "👋 **Bienvenido**: Configure el Bounding Box en el panel lateral y haga clic en "
        "**'Descargar y Cargar Ortofoto'** para comenzar la extracción con IA."
    )
else:
    st.markdown(
        "> 🎯 **Instrucción**: Haga clic sobre cualquier objeto visible en la imagen "
        "(edificios, estructuras, árboles, vías, etc.) para que SAM 2 lo segmente automáticamente."
    )

# -----------------------------------------------------------------------------
# VISOR CARTOGRÁFICO INTERACTIVO (FOLIUM)
# -----------------------------------------------------------------------------
mosaic = st.session_state.mosaic_data
bounds_4326 = mosaic["bounds_4326"] if mosaic_loaded else None

# Determinar qué polígono mostrar en el mapa
active_gdf = None
if st.session_state.candidate_result is not None:
    active_gdf = st.session_state.candidate_result["gdf"]
elif st.session_state.accepted_result is not None:
    active_gdf = st.session_state.accepted_result["gdf"]

clicked_coords = st.session_state.last_processed_click

map_obj = create_geospatial_map(
    center=st.session_state.map_center,
    zoom_start=18,
    bounds_4326=bounds_4326,
    current_polygon=active_gdf,
    clicked_latlon=clicked_coords
)

map_interaction = st_folium(
    map_obj,
    width="100%",
    height=550,
    returned_objects=["last_clicked"]
)

# -----------------------------------------------------------------------------
# CAPTURA Y PROCESAMIENTO DEL CLIC DE USUARIO
# -----------------------------------------------------------------------------
if mosaic_loaded and map_interaction and map_interaction.get("last_clicked") and engine:
    click_data = map_interaction["last_clicked"]
    click_lat = click_data["lat"]
    click_lon = click_data["lng"]
    current_click = (round(click_lat, 6), round(click_lon, 6))
    
    # Procesar únicamente si es un clic nuevo
    if current_click != st.session_state.last_processed_click:
        w_deg, s_deg, e_deg, n_deg = mosaic["bounds_4326"]
        
        # Validar si el clic cae dentro del área descargada
        if s_deg <= click_lat <= n_deg and w_deg <= click_lon <= e_deg:
            try:
                # 1. Transformación rigurosa a coordenadas píxel
                col, row = coordinates_to_pixel(
                    click_lon,
                    click_lat,
                    mosaic["transform_3857"],
                    mosaic["width"],
                    mosaic["height"]
                )
                
                # 2. Inferencia con SAM 2
                with st.spinner("Ejecutando inferencia SAM 2 sobre el objeto..."):
                    pred_res = engine.predict_point(col, row, multimask_output=True)
                    mask = pred_res["best_mask"]
                    score = pred_res["best_score"]
                    inf_time = pred_res["inference_time"]
                    
                    # 3. Vectorización y cálculo geoespacial métrico
                    gdf = mask_to_geodataframe(
                        mask=mask,
                        transform_3857=mosaic["transform_3857"],
                        confidence=score
                    )
                    
                    # 4. Extracción de recorte y overlay visual
                    cutout_img, bbox_px = extract_cutout(mosaic["img_array"], mask, crop_to_bbox=True)
                    overlay_arr = create_mask_overlay(mosaic["img_array"], mask)
                    
                    st.session_state.candidate_result = {
                        "mask": mask,
                        "gdf": gdf,
                        "score": score,
                        "inference_time": inf_time,
                        "device": pred_res["device"],
                        "click_latlon": (click_lat, click_lon),
                        "pixel_colrow": (col, row),
                        "cutout_image": cutout_img,
                        "overlay_array": overlay_arr,
                        "bbox_px": bbox_px
                    }
                    st.session_state.last_processed_click = current_click
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error durante la inferencia: {e}")
        else:
            st.warning("⚠️ El clic debe realizarse dentro del rectángulo rojo de la ortofoto descargada.")

# -----------------------------------------------------------------------------
# PANEL DE VALIDACIÓN Y CONTROL DE CALIDAD DEL OBJETO SEGMENTADO
# -----------------------------------------------------------------------------
candidate = st.session_state.candidate_result

if candidate is not None:
    st.markdown("---")
    st.subheader("🔍 Validación de la Segmentación")
    st.info(
        "Examine la máscara obtenida en el mapa o en el visor inferior. "
        "Puede **Aceptar el Objeto** para consolidarlo y habilitar su exportación, "
        "o hacer un **Nuevo Clic** en el mapa para segmentar otro elemento."
    )
    
    val_col1, val_col2, val_col3 = st.columns([2, 2, 4])
    with val_col1:
        if st.button("✅ Aceptar Objeto", type="primary", use_container_width=True):
            st.session_state.accepted_result = candidate
            st.session_state.candidate_result = None
            st.toast("¡Objeto aceptado correctamente!", icon="🎉")
            st.rerun()
            
    with val_col2:
        if st.button("🔄 Descartar / Reintentar", use_container_width=True):
            st.session_state.candidate_result = None
            st.session_state.last_processed_click = None
            st.toast("Selección descartada. Puede hacer un nuevo clic.", icon="ℹ️")
            st.rerun()

# -----------------------------------------------------------------------------
# PANEL DE RESULTADOS Y EXTRACCIÓN (OBJETO CONFIRMADO O EN EVALUACIÓN)
# -----------------------------------------------------------------------------
active_res = st.session_state.accepted_result if st.session_state.accepted_result else candidate

if active_res is not None and not active_res["gdf"].empty:
    st.markdown("---")
    res_title = "📊 Objeto Extraído Confirmado" if st.session_state.accepted_result else "🔬 Evaluación de Objeto Segmentado"
    st.subheader(res_title)
    
    gdf_item = active_res["gdf"]
    row_data = gdf_item.iloc[0]
    
    # Métricas clave en tarjetas
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("📐 Superficie / Área", f"{row_data['area_m2']:,.2f} m²")
    with m_col2:
        st.metric("📏 Perímetro", f"{row_data['perimeter_m']:,.2f} m")
    with m_col3:
        st.metric("🎯 Confianza SAM 2", f"{row_data['confidence'] * 100:.1f} %")
    with m_col4:
        st.metric("⚡ Tiempo Inferencia", f"{active_res['inference_time']:.3f} s")
        
    tab_inspect, tab_cutout, tab_export = st.tabs([
        "📋 Atributos y Coordenadas",
        "✂️ Objeto Extraído (Canal Alfa)",
        "💾 Centro de Exportación"
    ])
    
    with tab_inspect:
        i_col1, i_col2 = st.columns(2)
        with i_col1:
            st.markdown(f"**Identificador:** `{row_data['object_id']}`")
            st.markdown(f"**Etiqueta:** {row_data['label']}")
            st.markdown(f"**Centroide:** Lat `{row_data['centroid_lat']:.6f}`, Lon `{row_data['centroid_lon']:.6f}`")
            st.markdown(f"**Sistema Métrico Utilizado:** `{row_data['metric_crs']}`")
        with i_col2:
            st.markdown(f"**Bounding Box Oeste / Este:** `[{row_data['min_lon']:.5f}, {row_data['max_lon']:.5f}]`")
            st.markdown(f"**Bounding Box Sur / Norte:** `[{row_data['min_lat']:.5f}, {row_data['max_lat']:.5f}]`")
            st.markdown(f"**Píxel Prompt de Inferencia:** `col={active_res['pixel_colrow'][0]}, row={active_res['pixel_colrow'][1]}`")
            st.markdown(f"**Dispositivo de Cómputo:** `{active_res['device']}`")
            
        st.dataframe(gdf_item.drop(columns=["geometry"]), use_container_width=True)
        
    with tab_cutout:
        c_col_a, c_col_b = st.columns(2)
        with c_col_a:
            st.write("**Objeto Extraído (Transparencia RGBA):**")
            st.image(active_res["cutout_image"], use_container_width=True)
        with c_col_b:
            st.write("**Ortofoto con Máscara Superpuesta:**")
            st.image(active_res["overlay_array"], use_container_width=True)
            
    with tab_export:
        st.write("Seleccione el formato para descargar el resultado geoespacial:")
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        
        # 1. GeoJSON
        with e_col1:
            geojson_str = export_geojson(gdf_item)
            st.download_button(
                "🗺️ GeoJSON Vectorial",
                data=geojson_str,
                file_name=f"{row_data['object_id']}.geojson",
                mime="application/geo+json",
                use_container_width=True
            )
            
        # 2. GeoTIFF georreferenciado
        with e_col2:
            geotiff_bytes = export_geotiff(
                img_array=mosaic["img_array"],
                transform=mosaic["transform_3857"],
                crs_str="EPSG:3857",
                mask=active_res["mask"]
            )
            st.download_button(
                "🌐 GeoTIFF Ráster",
                data=geotiff_bytes,
                file_name=f"{row_data['object_id']}.tif",
                mime="image/tiff",
                use_container_width=True
            )
            
        # 3. PNG con canal alfa
        with e_col3:
            png_bytes = export_png(active_res["cutout_image"])
            st.download_button(
                "🖼️ PNG Transparente",
                data=png_bytes,
                file_name=f"{row_data['object_id']}.png",
                mime="image/png",
                use_container_width=True
            )
            
        # 4. Shapefile comprimido en ZIP
        with e_col4:
            try:
                shp_zip = export_shapefile_zip(gdf_item)
                st.download_button(
                    "📁 Shapefile (.zip)",
                    data=shp_zip,
                    file_name=f"{row_data['object_id']}_shp.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generando Shapefile: {e}")
