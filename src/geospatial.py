"""
Módulo geoespacial: vectorización de máscaras, reproyección rigurosa y cálculo de métricas en metros.
"""
from datetime import datetime
import uuid
from typing import Dict, Any, Optional
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
import rasterio
import rasterio.features
from config import CRS_GEOGRAPHIC, CRS_WEB_MERCATOR, CRS_COLOMBIA_METRIC

def mask_to_geodataframe(
    mask: np.ndarray,
    transform_3857: rasterio.Affine,
    confidence: float = 1.0,
    min_pixel_area: int = 4
) -> gpd.GeoDataFrame:
    """
    Convierte una máscara binaria en un GeoDataFrame georreferenciado con métricas espaciales exactas.
    
    Flujo:
        Máscara píxel (H, W) 
        -> shapes en EPSG:3857 
        -> GeoDataFrame en EPSG:4326 
        -> Reproyección métrica (EPSG:3116 o UTM) para cálculo de Área (m²) y Perímetro (m).
    """
    mask_uint8 = mask.astype(np.uint8)
    
    # 1. Vectorizar en el espacio nativo de las teselas (EPSG:3857)
    generator = rasterio.features.shapes(mask_uint8, transform=transform_3857)
    polygons = []
    
    for geom_dict, val in generator:
        if val == 1:
            geom = shape(geom_dict)
            if geom.is_valid and not geom.is_empty and geom.area > 0:
                polygons.append(geom)
                
    if not polygons:
        # GeoDataFrame vacío si no se detectaron polígonos
        empty_gdf = gpd.GeoDataFrame(
            columns=["object_id", "label", "confidence", "area_m2", "perimeter_m", 
                     "centroid_lat", "centroid_lon", "metric_crs", "timestamp", "geometry"],
            geometry="geometry",
            crs=CRS_GEOGRAPHIC
        )
        return empty_gdf
        
    # Unir componentes conexos si corresponden al mismo objeto
    merged_geom = unary_union(polygons)
    
    # Crear GeoDataFrame nativo en EPSG:3857
    gdf_3857 = gpd.GeoDataFrame(geometry=[merged_geom], crs=CRS_WEB_MERCATOR)
    
    # Reproyectar a EPSG:4326 (WGS84) para compatibilidad global y Leaflet
    gdf_4326 = gdf_3857.to_crs(CRS_GEOGRAPHIC)
    primary_geom_4326 = gdf_4326.geometry.iloc[0]
    
    # Centroide geográfico
    centroid_pt = primary_geom_4326.centroid
    centroid_lon = float(centroid_pt.x)
    centroid_lat = float(centroid_pt.y)
    
    # 2. Selección de Sistema de Coordenadas Métrico Proyectado (NO calcular en grados)
    metric_crs = None
    # Si está dentro de Colombia (Lat: -4.5 a 13.5, Lon: -79.5 a -66.5), usar EPSG:3116 (MAGNA-SIRGAS Origen Nacional)
    if -79.5 <= centroid_lon <= -66.5 and -4.5 <= centroid_lat <= 13.5:
        metric_crs = CRS_COLOMBIA_METRIC
    else:
        try:
            metric_crs = gdf_4326.estimate_utm_crs().to_string()
        except Exception:
            metric_crs = CRS_WEB_MERCATOR
            
    # Reproyección para cálculo métrico real
    try:
        gdf_metric = gdf_4326.to_crs(metric_crs)
        area_m2 = float(gdf_metric.area.iloc[0])
        perimeter_m = float(gdf_metric.length.iloc[0])
    except Exception:
        # Fallback a UTM estimado
        gdf_metric = gdf_4326.to_crs(gdf_4326.estimate_utm_crs())
        area_m2 = float(gdf_metric.area.iloc[0])
        perimeter_m = float(gdf_metric.length.iloc[0])
        metric_crs = gdf_metric.crs.to_string()
        
    bounds = primary_geom_4326.bounds # (minx, miny, maxx, maxy)
    obj_id = f"OBJ-{uuid.uuid4().hex[:8].upper()}"
    timestamp_str = datetime.utcnow().isoformat() + "Z"
    
    # Crear GeoDataFrame final con atributos enriquecidos
    gdf_final = gpd.GeoDataFrame({
        "object_id": [obj_id],
        "label": ["Objeto segmentado mediante SAM 2"],
        "confidence": [round(confidence, 4)],
        "area_m2": [round(area_m2, 2)],
        "perimeter_m": [round(perimeter_m, 2)],
        "centroid_lat": [round(centroid_lat, 6)],
        "centroid_lon": [round(centroid_lon, 6)],
        "min_lon": [round(bounds[0], 6)],
        "min_lat": [round(bounds[1], 6)],
        "max_lon": [round(bounds[2], 6)],
        "max_lat": [round(bounds[3], 6)],
        "metric_crs": [metric_crs],
        "timestamp": [timestamp_str],
        "geometry": [primary_geom_4326]
    }, geometry="geometry", crs=CRS_GEOGRAPHIC)
    
    return gdf_final
