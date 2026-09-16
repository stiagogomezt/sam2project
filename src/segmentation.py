"""
Módulo de coordinación de segmentación, transformación exacta de clics y extracción de objetos recortados.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import mercantile
import rasterio
from rasterio.transform import rowcol
from PIL import Image, ImageFilter

def coordinates_to_pixel(
    lon: float,
    lat: float,
    transform_3857: rasterio.Affine,
    width: int,
    height: int
) -> Tuple[int, int]:
    """
    Transformación rigurosa de coordenadas geográficas (WGS84) a índice de píxel (col, row).
    Utiliza el espacio proyectado nativo EPSG:3857 para eliminar la distorsión de latitud de Mercator.
    """
    xm, ym = mercantile.xy(lon, lat)
    row_f, col_f = rowcol(transform_3857, xm, ym)
    row, col = int(np.round(row_f)), int(np.round(col_f))
    
    # Asegurar que esté dentro de los límites
    if not (0 <= col < width and 0 <= row < height):
        raise ValueError(
            f"El punto seleccionado ({lon:.5f}, {lat:.5f}) cae fuera de la imagen descargada "
            f"(Píxel calculado: col={col}, row={row}, tamaño={width}x{height})."
        )
        
    return col, row

def extract_cutout(
    img_array: np.ndarray,
    mask: np.ndarray,
    crop_to_bbox: bool = True,
    padding: int = 10
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Extrae el objeto segmentado como una imagen RGBA con fondo transparente.
    Si crop_to_bbox=True, recorta la imagen ajustándose al Bounding Box del objeto con un margen.
    
    Returns:
        rgba_image: Objeto PIL.Image en formato RGBA.
        bbox_px: (min_col, min_row, max_col, max_row) en coordenadas de la imagen original.
    """
    h, w = img_array.shape[:2]
    mask_bool = mask.astype(bool)
    
    # Crear canal alfa (255 para el objeto, 0 para el fondo)
    alpha = (mask_bool * 255).astype(np.uint8)
    rgba = np.dstack((img_array, alpha))
    
    # Coordenadas de los píxeles positivos
    rows, cols = np.where(mask_bool)
    if len(rows) == 0 or len(cols) == 0:
        return Image.fromarray(rgba, "RGBA"), (0, 0, w, h)
        
    min_row = max(0, int(np.min(rows)) - padding)
    max_row = min(h, int(np.max(rows)) + padding + 1)
    min_col = max(0, int(np.min(cols)) - padding)
    max_col = min(w, int(np.max(cols)) + padding + 1)
    
    bbox_px = (min_col, min_row, max_col, max_row)
    
    if crop_to_bbox:
        cropped_rgba = rgba[min_row:max_row, min_col:max_col]
        return Image.fromarray(cropped_rgba, "RGBA"), bbox_px
    else:
        return Image.fromarray(rgba, "RGBA"), bbox_px

def create_mask_overlay(
    img_array: np.ndarray,
    mask: np.ndarray,
    color_rgb: Tuple[int, int, int] = (0, 255, 255), # Cian
    alpha: float = 0.45,
    draw_contour: bool = True
) -> np.ndarray:
    """
    Superpone la máscara sobre la imagen con color semitransparente y contorno nítido.
    """
    overlay = img_array.copy()
    mask_bool = mask.astype(bool)
    
    # Pintar área de la máscara con transparencia
    color_arr = np.array(color_rgb, dtype=np.uint8)
    overlay[mask_bool] = (
        overlay[mask_bool].astype(float) * (1.0 - alpha) + 
        color_arr.astype(float) * alpha
    ).astype(np.uint8)
    
    if draw_contour and np.any(mask_bool):
        # Detección de bordes con Pillow
        mask_img = Image.fromarray((mask_bool * 255).astype(np.uint8))
        edges = np.array(mask_img.filter(ImageFilter.FIND_EDGES)) > 0
        
        # Engrosar borde ligeramente aplicando max pooling 3x3
        edge_thick = edges.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(edges, dy, axis=0), dx, axis=1)
                edge_thick |= shifted
                
        # Pintar contorno exterior con amarillo brillante
        overlay[edge_thick] = [255, 230, 0]
        
    return overlay
