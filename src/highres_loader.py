"""
Módulo retrocompatible para la carga de ortofotos de alta resolución.
Delega en los módulos especializados: tile_manager, imagery_loader y mosaic_builder.
"""
from typing import Tuple, Any, Optional, Callable
import numpy as np
import rasterio
from src.tile_manager import get_tiles_info
from src.imagery_loader import download_tiles_concurrently
from src.mosaic_builder import build_mosaic

def get_highres_image(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int = 18,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[np.ndarray, rasterio.Affine, Tuple[float, float, float, float]]:
    """
    Descarga y construye el mosaico de alta resolución para un Bounding Box dado.
    
    Returns:
        img_array: Matriz uint8 RGB de la imagen completa.
        transform_3857: Transformación afín en EPSG:3857 (espacio nativo plano).
        bounds_4326: Límites geográficos (oeste, sur, este, norte) en WGS84.
    """
    tiles_info = get_tiles_info(min_lon, min_lat, max_lon, max_lat, zoom=zoom)
    tile_images = download_tiles_concurrently(tiles_info["tiles"], progress_callback=progress_callback)
    mosaic = build_mosaic(tiles_info, tile_images)
    
    # Retornamos img_array, transform en EPSG:3857 (crítico para precisión de clics) y bounds_4326
    return mosaic["img_array"], mosaic["transform_3857"], mosaic["bounds_4326"]

def get_highres_mosaic_bundle(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int = 18,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> dict:
    """
    Retorna el bundle completo con metadatos espaciales, ambos transforms (3857 y 4326) y métricas.
    """
    tiles_info = get_tiles_info(min_lon, min_lat, max_lon, max_lat, zoom=zoom)
    tile_images = download_tiles_concurrently(tiles_info["tiles"], progress_callback=progress_callback)
    return build_mosaic(tiles_info, tile_images)
