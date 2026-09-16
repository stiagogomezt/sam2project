import numpy as np
from rasterio.features import rasterize
import pandas as pd

def classify_polygons(gdf, ndvi_array, transform):
    """
    Clasifica los polígonos generados por SAM 2 utilizando el índice NDVI.
    Calcula el NDVI promedio dentro de cada polígono rasterizando la geometría.
    
    Args:
        gdf: GeoDataFrame con las geometrías extraídas.
        ndvi_array: Matriz numpy con los valores de NDVI (2D).
        transform: Affine transform del raster recortado.
        
    Returns:
        GeoDataFrame actualizado con columnas 'clase', 'color' y 'ndvi_mean'.
    """
    if gdf.empty:
        return gdf
        
    clases = []
    colores = []
    ndvi_means = []
    
    for geom in gdf.geometry:
        # Si la geometría es inválida, usar valores por defecto
        if not geom.is_valid or geom.is_empty:
            clases.append("Desconocido")
            colores.append("#FFFFFF")
            ndvi_means.append(0.0)
            continue
            
        # Crear una máscara de 1s para el área del polígono
        mask = rasterize(
            [(geom, 1)],
            out_shape=ndvi_array.shape,
            transform=transform,
            fill=0,
            dtype='uint8'
        )
        
        # Extraer los píxeles de NDVI que caen dentro del polígono
        ndvi_vals = ndvi_array[mask == 1]
        
        if len(ndvi_vals) > 0:
            mean_ndvi = float(np.nanmean(ndvi_vals))
        else:
            mean_ndvi = 0.0
            
        ndvi_means.append(mean_ndvi)
        
        # Árbol de decisión simple basado en NDVI
        if mean_ndvi > 0.4:
            clases.append("Bosque / Vegetación Densa")
            colores.append("#006400") # DarkGreen
        elif mean_ndvi > 0.1:
            clases.append("Pastos / Vegetación Ligera")
            colores.append("#9ACD32") # YellowGreen
        elif mean_ndvi < -0.05:
            clases.append("Cuerpos de Agua")
            colores.append("#4169E1") # RoyalBlue
        else:
            clases.append("Suelo Desnudo / Urbano")
            colores.append("#A9A9A9") # DarkGray
            
    gdf['ndvi_mean'] = ndvi_means
    gdf['clase'] = clases
    gdf['color'] = colores
    
    return gdf


def spectral_profiler(geom, img_array, transform):
    """
    Calcula firmas espectrales (NDVI, NDWI, etc.) para un solo polígono interactivo.
    Asume que img_array es un array HxWxC con bandas: 0=Blue, 1=Green, 2=Red, 3=NIR.
    """
    if not geom or not geom.is_valid or geom.is_empty:
        return {"Error": "Geometría inválida"}
        
    mask = rasterize(
        [(geom, 1)],
        out_shape=img_array.shape[:2],
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    
    # Índices de bandas:
    # B02 (Blue) = img_array[:,:,0]
    # B03 (Green) = img_array[:,:,1]
    # B04 (Red) = img_array[:,:,2]
    # B08 (NIR) = img_array[:,:,3]
    
    b2_vals = img_array[:,:,0][mask == 1]
    b3_vals = img_array[:,:,1][mask == 1]
    b4_vals = img_array[:,:,2][mask == 1]
    b8_vals = img_array[:,:,3][mask == 1]
    
    if len(b8_vals) == 0:
        return {"Error": "Polígono fuera de los límites de la imagen"}
        
    # Promedios
    b2_mean = float(np.nanmean(b2_vals))
    b3_mean = float(np.nanmean(b3_vals))
    b4_mean = float(np.nanmean(b4_vals))
    b8_mean = float(np.nanmean(b8_vals))
    
    # Prevenir divisiones por cero
    denom_ndvi = (b8_mean + b4_mean) if (b8_mean + b4_mean) != 0 else 1e-10
    denom_ndwi = (b3_mean + b8_mean) if (b3_mean + b8_mean) != 0 else 1e-10
    
    ndvi = (b8_mean - b4_mean) / denom_ndvi
    ndwi = (b3_mean - b8_mean) / denom_ndwi # NDWI using Green and NIR
    
    clase_estimada = "Desconocido"
    if ndvi > 0.4: clase_estimada = "Bosque"
    elif ndwi > 0.1: clase_estimada = "Cuerpo de Agua"
    elif ndvi > 0.1: clase_estimada = "Pastos"
    else: clase_estimada = "Urbano / Suelo Desnudo"
    
    return {
        "Clase Estimada": clase_estimada,
        "NDVI (Vegetación)": round(ndvi, 3),
        "NDWI (Agua)": round(ndwi, 3),
        "NIR Reflectancia": round(b8_mean, 1),
        "Red Reflectancia": round(b4_mean, 1)
    }
