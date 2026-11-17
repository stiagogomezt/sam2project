"""
Módulo de visualización cartográfica e interactiva con Folium.

v2: el mapa se divide en
- un MAPA BASE estable (`create_base_map`): solo cambia al cargar otra ortofoto, de modo que
  streamlit-folium no lo vuelve a renderizar y el usuario conserva zoom y encuadre;
- CAPAS DINÁMICAS (`build_dynamic_layers`): objetos, candidata y prompts, que se envían con
  `st_folium(..., feature_group_to_add=...)` sin reconstruir el mapa.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import folium
import folium.plugins
import geopandas as gpd

from config import ESRI_IMAGERY_URL

MAP_MAX_ZOOM = 21

def create_geospatial_map(
    center: Tuple[float, float],
    zoom_start: int = 18,
    bounds_4326: Optional[Tuple[float, float, float, float]] = None,
    current_polygon: Optional[gpd.GeoDataFrame] = None,
    clicked_latlon: Optional[Tuple[float, float]] = None,
    show_bounds: bool = True,
    show_mask: bool = True
) -> folium.Map:
    """
    Construye un mapa Folium profesional con herramientas geoespaciales avanzadas:
    - Capa satelital de alta resolución (Esri World Imagery)
    - Posición del cursor en tiempo real (MousePosition)
    - Barra de escala métrica
    - Límites de la ortofoto descargada
    - Renderizado de la máscara vectorial SAM 2
    - Marcador del punto de interacción (Prompt)
    """
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        max_zoom=20,
        control_scale=True
    )
    
    # Capa base Esri World Imagery
    esri_layer = folium.TileLayer(
        tiles=ESRI_IMAGERY_URL,
        attr="Esri World Imagery",
        name="Esri Satelital (Alta Resolución)",
        max_zoom=20,
    )
    esri_layer.add_to(m)
    
    # Plugin: Posición del cursor (Lat / Lon) en pantalla
    folium.plugins.MousePosition(
        position="bottomright",
        separator=" | Longitud: ",
        empty_string="Mueva el cursor sobre el mapa",
        lng_first=False,
        num_digits=6,
        prefix="Latitud: "
    ).add_to(m)
    
    # Rectángulo delimitador de la ortofoto descargada
    if show_bounds and bounds_4326 is not None:
        west, south, east, north = bounds_4326
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#FF3333",
            weight=2,
            dash_array="5, 5",
            fill=False,
            tooltip="Área de Ortofoto Descargada",
            name="Límites de Ortofoto"
        ).add_to(m)
        
    # Renderizado de la máscara vectorial
    if show_mask and current_polygon is not None and not current_polygon.empty:
        folium.GeoJson(
            current_polygon,
            name="Máscara Extraída (SAM 2)",
            style_function=lambda x: {
                "fillColor": "#00FFFF",
                "color": "#00FFFF",
                "weight": 3,
                "fillOpacity": 0.55
            },
            tooltip="Objeto Segmentado por SAM 2",
            popup=folium.GeoJsonPopup(
                fields=["object_id", "area_m2", "perimeter_m", "confidence"],
                aliases=["ID:", "Área (m²):", "Perímetro (m):", "Confianza:"]
            )
        ).add_to(m)
        
    # Marcador visual del punto donde se hizo clic
    if clicked_latlon is not None:
        c_lat, c_lon = clicked_latlon
        folium.CircleMarker(
            location=[c_lat, c_lon],
            radius=6,
            color="#FFD700", # Dorado
            fill=True,
            fill_color="#FF0000", # Rojo
            fill_opacity=1.0,
            tooltip=f"Prompt SAM 2: [{c_lat:.5f}, {c_lon:.5f}]"
        ).add_to(m)
        
    folium.LayerControl(position="topright").add_to(m)
    return m


def create_base_map(
    center: Tuple[float, float],
    zoom_start: int,
    bounds_4326: Optional[Tuple[float, float, float, float]] = None,
    native_zoom: Optional[int] = None,
    overlay: Optional[Tuple[Any, Tuple[float, float, float, float], str]] = None,
) -> folium.Map:
    """Mapa base estable.

    `native_zoom` = zoom de las teselas descargadas. Con `max_native_zoom`, Leaflet no pide teselas
    más detalladas que las procesadas por SAM 2: al acercarse amplía ESAS MISMAS teselas, así que lo
    que el usuario ve coincide con la imagen segmentada.
    """
    m = folium.Map(location=center, zoom_start=zoom_start, max_zoom=MAP_MAX_ZOOM,
                   control_scale=True, tiles=None)
    tile_kwargs: Dict[str, Any] = {"max_zoom": MAP_MAX_ZOOM}
    if native_zoom is not None:
        tile_kwargs["max_native_zoom"] = int(native_zoom)
    folium.TileLayer(tiles=ESRI_IMAGERY_URL, attr="Esri World Imagery",
                     name="Esri World Imagery", **tile_kwargs).add_to(m)
    folium.plugins.MousePosition(position="bottomright", separator=" | Lon: ",
                                 empty_string="", lng_first=False, num_digits=6, prefix="Lat: ").add_to(m)
    if overlay is not None:
        # Imagen procesada (p. ej. Sentinel-2 en grilla EPSG:3857): se dibuja encima del mapa base para
        # que lo que el usuario ve y donde hace clic coincida con lo que segmenta SAM 2.
        arr, (west, south, east, north), attribution = overlay
        folium.raster_layers.ImageOverlay(image=arr, bounds=[[south, west], [north, east]], opacity=1.0,
                                          interactive=False, cross_origin=False, zindex=1,
                                          name="Imagen procesada", attr=attribution).add_to(m)
    if bounds_4326 is not None:
        west, south, east, north = bounds_4326
        folium.Rectangle(bounds=[[south, west], [north, east]], color="#FF3333", weight=2,
                         dash_array="5, 5", fill=False, interactive=False).add_to(m)
    return m


def build_dynamic_layers(
    objects: Sequence[Dict[str, Any]] = (),
    candidate_gdf: Optional[gpd.GeoDataFrame] = None,
    points_latlon: Sequence[Tuple[float, float, int]] = (),
    box_latlon: Optional[Tuple[float, float, float, float]] = None,
    pending_corner_latlon: Optional[Tuple[float, float]] = None,
    highlight_id: Optional[str] = None,
) -> List[folium.FeatureGroup]:
    """Capas que cambian con la interacción.

    `objects`: dicts con `gdf`, `color`, `object_id`, `label`, `status`.
    Todas las geometrías se declaran `interactive=False` para que los clics atraviesen los polígonos
    y lleguen al mapa (imprescindible para añadir puntos negativos DENTRO de una máscara).
    """
    fg_objects = folium.FeatureGroup(name="Objetos")
    for o in objects:
        gdf = o["gdf"]
        if gdf is None or gdf.empty:
            continue
        color = o["color"]
        dashed = o.get("status") == "propuesto"
        weight = 4 if o["object_id"] == highlight_id else 2
        folium.GeoJson(
            gdf[["object_id", "label", "geometry"]].to_json(),
            style_function=lambda _x, c=color, d=dashed, w=weight: {
                "fillColor": c, "color": c, "weight": w, "fillOpacity": 0.25,
                "dashArray": "6, 4" if d else None},
            interactive=False,
        ).add_to(fg_objects)

    fg_edit = folium.FeatureGroup(name="Edición")
    if candidate_gdf is not None and not candidate_gdf.empty:
        folium.GeoJson(
            candidate_gdf[["geometry"]].to_json(),
            style_function=lambda _x: {"fillColor": "#00FFFF", "color": "#FFFFFF", "weight": 2,
                                       "fillOpacity": 0.45},
            interactive=False,
        ).add_to(fg_edit)
    if box_latlon is not None:
        south, west, north, east = box_latlon
        folium.Rectangle(bounds=[[south, west], [north, east]], color="#FFD600", weight=2,
                         dash_array="4, 4", fill=False, interactive=False).add_to(fg_edit)
    for lat, lon, lbl in points_latlon:
        folium.CircleMarker(location=[lat, lon], radius=6, weight=2, color="#FFFFFF",
                            fill=True, fill_opacity=1.0,
                            fill_color="#00C853" if lbl == 1 else "#D50000",
                            interactive=False).add_to(fg_edit)
    if pending_corner_latlon is not None:
        folium.CircleMarker(location=list(pending_corner_latlon), radius=5, color="#FFD600",
                            fill=True, fill_color="#FFD600", fill_opacity=1.0,
                            interactive=False).add_to(fg_edit)
    return [fg_objects, fg_edit]
