"""Métricas geométricas sobre geometrías sintéticas de dimensiones conocidas (en metros)."""
import math

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.affinity import rotate
from shapely.geometry import Point, box

from src.geospatial import (
    compute_shape_metrics, mask_to_geodataframe, select_metric_crs, spatial_relations, derived_geometries,
)

# Transformación EPSG:3857 de 0.5 m/píxel con origen en Bogotá
BOGOTA_X, BOGOTA_Y = -8246700.0, 513800.0
TR = from_origin(BOGOTA_X, BOGOTA_Y, 0.5, 0.5)


def test_rectangle_metrics():
    m = compute_shape_metrics(box(0, 0, 40, 10))
    assert m["area_m2"] == pytest.approx(400)
    assert m["perimeter_m"] == pytest.approx(100)
    assert m["rectangularity"] == pytest.approx(1.0)
    assert m["solidity"] == pytest.approx(1.0)
    assert m["elongation"] == pytest.approx(0.75)
    assert m["aspect_ratio"] == pytest.approx(4.0)
    assert m["compactness"] == pytest.approx(4 * math.pi * 400 / 100 ** 2)
    assert m["orientation_deg"] == pytest.approx(90.0)  # eje largo este-oeste


def test_orientation_rotated_rectangle():
    g = rotate(box(0, 0, 40, 10), 30, origin="centroid")  # antihorario 30° → acimut 60°
    m = compute_shape_metrics(g)
    assert m["orientation_deg"] == pytest.approx(60.0, abs=0.5)
    assert m["mrr_length_m"] == pytest.approx(40, rel=1e-3)
    assert m["mrr_width_m"] == pytest.approx(10, rel=1e-3)


def test_circle_metrics():
    m = compute_shape_metrics(Point(0, 0).buffer(10, quad_segs=256))
    assert m["compactness"] == pytest.approx(1.0, abs=1e-3)
    assert m["elongation"] == pytest.approx(0.0, abs=1e-2)
    assert m["rectangularity"] == pytest.approx(math.pi / 4, abs=1e-2)


def test_concave_solidity():
    l_shape = box(0, 0, 20, 20).difference(box(10, 10, 20, 20))
    m = compute_shape_metrics(l_shape)
    assert m["solidity"] < 0.9
    assert m["area_m2"] == pytest.approx(300)


def test_metric_crs_selection():
    assert select_metric_crs(-74.08, 4.61) == "EPSG:9377"        # Bogotá
    assert select_metric_crs(-81.7, 12.5) == "EPSG:9377"         # San Andrés (dentro del área de uso)
    assert select_metric_crs(-3.70, 40.42) == "EPSG:32630"       # Madrid → UTM 30N
    assert select_metric_crs(-58.4, -34.6) == "EPSG:32721"       # Buenos Aires → UTM 21S


def _disk_mask(r_px: int, size: int = 400) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= r_px ** 2


def test_mask_vectorization_square_known_area():
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:150, 40:120] = True  # 80 x 100 px = 40 m x 50 m nominales en 3857
    gdf = mask_to_geodataframe(mask, TR, confidence=0.9)
    row = gdf.iloc[0]
    # Referencia: área geodésica del mismo rectángulo sobre WGS84 (calculada con pyproj.Geod de
    # forma independiente: 1973.95 m²). 9377 es planar con k0=0.9992 → ~0.12 % menor en Bogotá.
    assert row["area_geod_m2"] == pytest.approx(1973.95, abs=0.05)
    assert row["area_m2"] == pytest.approx(row["area_geod_m2"], rel=3e-3)
    assert row["area_m2"] < row["area_geod_m2"]
    assert row["metric_crs"] == "EPSG:9377"
    assert row["pred_iou"] == 0.9
    assert row["rectangularity"] == pytest.approx(1.0, abs=1e-3)
    assert row["pixel_count"] == 8000


def test_perimeter_staircase_correction():
    """El contorno ráster sobrestima el perímetro ~28 %; con simplify_px=1 el error baja a ~1-3 %."""
    r_px = 60
    mask = _disk_mask(r_px)
    raw = mask_to_geodataframe(mask, TR, simplify_px=0).iloc[0]
    smooth = mask_to_geodataframe(mask, TR, simplify_px=1.0).iloc[0]
    gsd = smooth["gsd_m"]
    true_p = 2 * math.pi * r_px * gsd
    assert raw["perimeter_m"] / true_p - 1 > 0.2
    assert abs(smooth["perimeter_m"] / true_p - 1) < 0.03
    assert abs(smooth["area_m2"] / raw["pixel_area_m2"] - 1) < 0.01
    assert smooth["compactness"] > 0.95


def test_min_pixel_area_filters_noise():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:40, 10:40] = True
    mask[80, 80] = True  # ruido de 1 píxel
    keep = mask_to_geodataframe(mask, TR, min_pixel_area=1).iloc[0]
    drop = mask_to_geodataframe(mask, TR, min_pixel_area=4).iloc[0]
    assert keep["n_parts"] == 2
    assert drop["n_parts"] == 1


def test_empty_mask_returns_empty_gdf():
    assert mask_to_geodataframe(np.zeros((50, 50), dtype=bool), TR).empty


def test_spatial_relations_and_derived_layers():
    m1 = np.zeros((200, 200), dtype=bool); m1[10:60, 10:60] = True
    m2 = np.zeros((200, 200), dtype=bool); m2[10:60, 60:110] = True     # toca a m1
    m3 = np.zeros((200, 200), dtype=bool); m3[150:190, 150:190] = True  # lejos
    import pandas as pd
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(pd.concat(
        [mask_to_geodataframe(m, TR, object_id=f"O{i}") for i, m in enumerate([m1, m2, m3])]),
        crs="EPSG:4326")
    summary, pairs = spatial_relations(gdf, max_pairs_distance_m=1000)
    rel = {(r.id_a, r.id_b): r.relacion for r in pairs.itertuples()}
    assert rel[("O0", "O1")] == "toca"
    assert rel[("O0", "O2")] == "disjunto"
    assert summary.set_index("object_id").loc["O2", "nn_dist_m"] > 30
    layers = derived_geometries(gdf)
    assert set(layers) == {"centroides", "bbox", "convex_hull", "rect_min_rotado"}
    assert len(layers["bbox"]) == 3
