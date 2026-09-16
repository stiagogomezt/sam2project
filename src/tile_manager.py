"""
Módulo para validación de Bounding Box, cálculo de teselas y estimaciones de recursos.
"""
import math
import mercantile
from typing import Dict, Any, List, Tuple
from config import MAX_TILES_LIMIT, WARNING_TILES_LIMIT, AVG_TILE_SIZE_KB

class TileManagerError(Exception):
    """Excepción para errores de validación de teselas y límites espaciales."""
    pass

def validate_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> None:
    """
    Valida que las coordenadas del Bounding Box sean consistentes y estén dentro de rangos válidos.
    """
    # Rango de coordenadas globales
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise TileManagerError("Las longitudes deben estar dentro del rango [-180.0, 180.0].")
    
    # Rango para Web Mercator (aproximadamente +/- 85.0511)
    if not (-85.0511 <= min_lat <= 85.0511 and -85.0511 <= max_lat <= 85.0511):
        raise TileManagerError("Las latitudes deben estar dentro del rango [-85.05, 85.05] para proyección Web Mercator.")
    
    # Bounding Box invertido
    if min_lon >= max_lon:
        raise TileManagerError(f"La longitud mínima ({min_lon:.5f}) debe ser menor que la máxima ({max_lon:.5f}).")
    
    if min_lat >= max_lat:
        raise TileManagerError(f"La latitud mínima ({min_lat:.5f}) debe ser menor que la máxima ({max_lat:.5f}).")
    
    # Área degenerada o prácticamente cero
    if (max_lon - min_lon) < 1e-5 or (max_lat - min_lat) < 1e-5:
        raise TileManagerError("El área del Bounding Box es demasiado pequeña para el procesamiento.")

def get_tiles_info(min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int) -> Dict[str, Any]:
    """
    Calcula la lista de teselas mercantile y devuelve métricas detalladas para la interfaz:
    - Lista de teselas
    - Cantidad total
    - Dimensiones en píxeles
    - Estimación de tamaño en MB
    - Resolución espacial estimada (metros/píxel)
    - Centroide
    """
    validate_bbox(min_lon, min_lat, max_lon, max_lat)
    
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zoom))
    
    if not tiles:
        raise TileManagerError("No se generaron teselas para las coordenadas y zoom indicados.")
        
    num_tiles = len(tiles)
    if num_tiles > MAX_TILES_LIMIT:
        raise TileManagerError(
            f"El área seleccionada requiere {num_tiles} teselas (el límite máximo seguro es {MAX_TILES_LIMIT}). "
            "Por favor reduce el Bounding Box o disminuye el nivel de zoom."
        )
        
    x_coords = [t.x for t in tiles]
    y_coords = [t.y for t in tiles]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    tiles_x = max_x - min_x + 1
    tiles_y = max_y - min_y + 1
    
    width_px = tiles_x * 256
    height_px = tiles_y * 256
    
    # Estimación de descarga
    estimated_mb = (num_tiles * AVG_TILE_SIZE_KB) / 1024.0
    
    # Centroide
    centroid_lat = (min_lat + max_lat) / 2.0
    centroid_lon = (min_lon + max_lon) / 2.0
    
    # Resolución en el ecuador: ~156543.03 m/px a zoom 0
    # En latitud phi: 156543.03 * cos(phi) / (2^zoom)
    res_m_per_px = (156543.03392 * math.cos(math.radians(centroid_lat))) / (2 ** zoom)
    
    # Bounds exactos cubiertos por las teselas (EPSG:3857)
    tl_tile = mercantile.Tile(min_x, min_y, zoom)
    br_tile = mercantile.Tile(max_x, max_y, zoom)
    
    tl_xy = mercantile.xy_bounds(tl_tile)
    br_xy = mercantile.xy_bounds(br_tile)
    
    # Bounds 3857: (west_m, south_m, east_m, north_m)
    bounds_3857 = (tl_xy.left, br_xy.bottom, br_xy.right, tl_xy.top)
    
    # Bounds 4326: (west, south, east, north)
    tl_geo = mercantile.bounds(tl_tile)
    br_geo = mercantile.bounds(br_tile)
    bounds_4326 = (tl_geo.west, br_geo.south, br_geo.east, tl_geo.north)
    
    return {
        "tiles": tiles,
        "num_tiles": num_tiles,
        "grid_shape": (tiles_y, tiles_x),
        "pixel_shape": (height_px, width_px),
        "estimated_mb": round(estimated_mb, 2),
        "res_m_per_px": round(res_m_per_px, 3),
        "centroid": (centroid_lat, centroid_lon),
        "bounds_3857": bounds_3857,
        "bounds_4326": bounds_4326,
        "tile_range": {
            "min_x": min_x, "max_x": max_x,
            "min_y": min_y, "max_y": max_y,
            "zoom": zoom
        },
        "is_large": num_tiles > WARNING_TILES_LIMIT
    }
