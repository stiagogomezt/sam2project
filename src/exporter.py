"""
Módulo de exportación multiformato para objetos geoespaciales y datos derivados.
Formatos soportados: GeoJSON, GeoTIFF georreferenciado, PNG (transparente), Shapefile (.zip) y CSV.
"""
import io
import os
import zipfile
import tempfile
from typing import Optional
import numpy as np
from PIL import Image
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from config import CRS_GEOGRAPHIC, CRS_WEB_MERCATOR

def export_geojson(gdf: gpd.GeoDataFrame) -> str:
    """Exporta el GeoDataFrame como cadena formateada en formato GeoJSON."""
    return gdf.to_json(indent=2)

def export_csv(gdf: gpd.GeoDataFrame) -> str:
    """Exporta los atributos del GeoDataFrame (sin la columna geometría) en formato CSV."""
    df_no_geom = gdf.drop(columns=["geometry"]) if "geometry" in gdf.columns else gdf
    return df_no_geom.to_csv(index=False)

def export_png(pil_image: Image.Image) -> bytes:
    """Exporta una imagen PIL a bytes en formato PNG."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()

def export_geotiff(
    img_array: np.ndarray,
    transform: rasterio.Affine,
    crs_str: str = CRS_WEB_MERCATOR,
    mask: Optional[np.ndarray] = None
) -> bytes:
    """
    Genera un GeoTIFF georreferenciado en memoria.
    Si se proporciona una máscara, se incluye como cuarto canal (Alfa) para preservar la extracción.
    """
    h, w = img_array.shape[:2]
    
    if mask is not None:
        # 4 canales: RGB + Alpha
        alpha = (mask.astype(bool) * 255).astype(np.uint8)
        rgba = np.dstack((img_array, alpha))
        count = 4
        data = np.moveaxis(rgba, -1, 0) # (4, H, W)
    else:
        # 3 canales: RGB
        count = 3
        data = np.moveaxis(img_array, -1, 0) # (3, H, W)
        
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=h,
        width=w,
        count=count,
        dtype=np.uint8,
        crs=crs_str,
        transform=transform,
        compress="deflate"
    ) as dst:
        dst.write(data)
        
    return memfile.read()

def export_shapefile_zip(gdf: gpd.GeoDataFrame) -> bytes:
    """
    Empaqueta el GeoDataFrame como un conjunto de archivos ESRI Shapefile comprimidos en un archivo .zip.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base_name = "objeto_extraido_sam2"
        shp_path = os.path.join(tmpdir, f"{base_name}.shp")
        gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(tmpdir):
                full_path = os.path.join(tmpdir, fname)
                zf.write(full_path, arcname=fname)
                
        buf.seek(0)
        return buf.getvalue()
