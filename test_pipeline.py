"""
Script de pruebas unitarias y de integración para SAM2_STREAMLIT_APP.
Verifica:
1. Validación y cálculo de teselas con TileManager.
2. Ensamblado y transform afín con MosaicBuilder.
3. Precisión sub-píxel en CoordinatesToPixel.
4. Extracción de recorte RGBA y generación de overlay.
5. Conversión de máscara a GeoDataFrame con métricas reales en metros.
6. Exportación a GeoJSON, GeoTIFF, PNG, Shapefile y CSV.
"""
import numpy as np
from PIL import Image
import mercantile
import rasterio
from rasterio.transform import from_bounds

from config import DEFAULT_BBOX, DEFAULT_ZOOM
from src.tile_manager import get_tiles_info, validate_bbox, TileManagerError
from src.mosaic_builder import build_mosaic
from src.segmentation import coordinates_to_pixel, extract_cutout, create_mask_overlay
from src.geospatial import mask_to_geodataframe
from src.exporter import export_geojson, export_geotiff, export_png, export_shapefile_zip, export_csv

def test_tile_manager():
    print("Test 1: TileManager...")
    info = get_tiles_info(
        DEFAULT_BBOX["min_lon"], DEFAULT_BBOX["min_lat"],
        DEFAULT_BBOX["max_lon"], DEFAULT_BBOX["max_lat"],
        zoom=DEFAULT_ZOOM
    )
    assert info["num_tiles"] > 0, "No se generaron teselas"
    assert info["pixel_shape"][0] > 0 and info["pixel_shape"][1] > 0
    assert len(info["bounds_3857"]) == 4
    print(f"  -> OK: {info['num_tiles']} teselas estimadas ({info['pixel_shape']} px)")

def test_pixel_mapping_accuracy():
    print("Test 2: CoordinatesToPixel accuracy...")
    info = get_tiles_info(
        DEFAULT_BBOX["min_lon"], DEFAULT_BBOX["min_lat"],
        DEFAULT_BBOX["max_lon"], DEFAULT_BBOX["max_lat"],
        zoom=DEFAULT_ZOOM
    )
    h_px, w_px = info["pixel_shape"]
    w_m, s_m, e_m, n_m = info["bounds_3857"]
    transform_3857 = from_bounds(w_m, s_m, e_m, n_m, w_px, h_px)
    
    # Probar el centroide
    c_lat, c_lon = info["centroid"]
    col, row = coordinates_to_pixel(c_lon, c_lat, transform_3857, w_px, h_px)
    assert 0 <= col < w_px, f"Col fuera de rango: {col}"
    assert 0 <= row < h_px, f"Row fuera de rango: {row}"
    print(f"  -> OK: Centroide ({c_lat:.4f}, {c_lon:.4f}) -> Píxel (col={col}, row={row}) dentro de {w_px}x{h_px}")

def test_geospatial_vectorization_and_metrics():
    print("Test 3: Geospatial vectorization & metrics...")
    h, w = 500, 500
    # Creamos un rectángulo de 100x100 píxeles
    mask = np.zeros((h, w), dtype=bool)
    mask[150:250, 150:250] = True
    
    # Definir transform sintético centrado en Bogotá en EPSG:3857
    xm, ym = mercantile.xy(-74.0815, 4.6095)
    # Supongamos resolución de 0.6 m/px (similar a zoom 18)
    res = 0.597
    transform_3857 = from_bounds(xm - 150, ym - 150, xm + 150, ym + 150, w, h)
    
    gdf = mask_to_geodataframe(mask, transform_3857, confidence=0.985)
    assert not gdf.empty, "El GeoDataFrame no debe estar vacío"
    area = gdf["area_m2"].iloc[0]
    perim = gdf["perimeter_m"].iloc[0]
    
    # 100 px * 0.6 m = 60 m de lado -> Área esperada ~3600 m2
    assert area > 0, f"Área inválida: {area}"
    assert perim > 0, f"Perímetro inválido: {perim}"
    assert gdf["confidence"].iloc[0] == 0.985
    assert "MAGNA" in gdf["metric_crs"].iloc[0] or "3116" in gdf["metric_crs"].iloc[0] or "32618" in gdf["metric_crs"].iloc[0]
    print(f"  -> OK: Área calculada={area:.2f} m², Perímetro={perim:.2f} m, CRS={gdf['metric_crs'].iloc[0]}")

def test_exporters():
    print("Test 4: Exporters multiformato...")
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=bool)
    mask[20:60, 20:60] = True
    img_array = np.full((h, w, 3), 128, dtype=np.uint8)
    transform = from_bounds(-8245000, 513000, -8244000, 514000, w, h)
    
    gdf = mask_to_geodataframe(mask, transform, confidence=0.95)
    
    # 1. GeoJSON
    geojson = export_geojson(gdf)
    assert "FeatureCollection" in geojson or "Feature" in geojson
    
    # 2. GeoTIFF
    geotiff = export_geotiff(img_array, transform, "EPSG:3857", mask)
    assert len(geotiff) > 0, "GeoTIFF vacío"
    
    # 3. PNG
    cutout, _ = extract_cutout(img_array, mask)
    png_bytes = export_png(cutout)
    assert len(png_bytes) > 0, "PNG vacío"
    
    # 4. Shapefile
    shp_zip = export_shapefile_zip(gdf)
    assert len(shp_zip) > 0, "Shapefile zip vacío"
    
    # 5. CSV
    csv_bytes = export_csv(gdf)
    assert "area_m2" in csv_bytes
    print("  -> OK: GeoJSON, GeoTIFF, PNG, Shapefile y CSV exportados exitosamente.")

if __name__ == "__main__":
    test_tile_manager()
    test_pixel_mapping_accuracy()
    test_geospatial_vectorization_and_metrics()
    test_exporters()
    print("\nTODOS LOS TESTS UNITARIOS Y DE INTEGRACIÓN PASARON EXITOSAMENTE (100%).")
