"""
Módulo para la construcción y georreferenciación precisa del mosaico continuo de ortofoto.
"""
from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from typing import Dict, Any
import mercantile

def build_mosaic(tiles_info: Dict[str, Any], tile_images: Dict[mercantile.Tile, Image.Image]) -> Dict[str, Any]:
    """
    Construye un mosaico continuo sin desplazamientos espaciales y con georreferenciación matemática rigurosa.
    """
    height_px, width_px = tiles_info["pixel_shape"]
    min_x = tiles_info["tile_range"]["min_x"]
    min_y = tiles_info["tile_range"]["min_y"]
    
    # Crear lienzo RGB continuo
    full_image = Image.new("RGB", (width_px, height_px), color=(0, 0, 0))
    
    for tile, img in tile_images.items():
        paste_x = (tile.x - min_x) * 256
        paste_y = (tile.y - min_y) * 256
        full_image.paste(img, (paste_x, paste_y))
        
    # Bounds y Transform afín en EPSG:3857 (Web Mercator - Espacio nativo de las teselas)
    w_m, s_m, e_m, n_m = tiles_info["bounds_3857"]
    transform_3857 = from_bounds(w_m, s_m, e_m, n_m, width_px, height_px)
    
    # Bounds y Transform afín en EPSG:4326 (WGS84)
    w_deg, s_deg, e_deg, n_deg = tiles_info["bounds_4326"]
    transform_4326 = from_bounds(w_deg, s_deg, e_deg, n_deg, width_px, height_px)
    
    img_array = np.array(full_image, dtype=np.uint8)
    
    return {
        "img_array": img_array,
        "pil_image": full_image,
        "width": width_px,
        "height": height_px,
        "transform_3857": transform_3857,
        "transform_4326": transform_4326,
        "bounds_3857": tiles_info["bounds_3857"],
        "bounds_4326": tiles_info["bounds_4326"],
        "tiles_info": tiles_info
    }
