"""
Módulo de visualización cartográfica e interactiva con Folium.
"""
from typing import List, Tuple, Optional
import folium
import folium.plugins
import geopandas as gpd

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
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satelital (Alta Resolución)",
        max_zoom=20,
        subdomains=["server", "services"]
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
