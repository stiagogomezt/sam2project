"""Productos: clasificación supervisada, coberturas, cuadro de coordenadas, indicadores, DXF/KML."""
import io
import math
import os
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from src.exporter import export_coverage_geotiff, export_dxf, export_kml
from src.geospatial import mask_to_geodataframe
from src.products import (
    SupervisedClassifier, color_stats, coverage_raster, coverage_stats, handcrafted_features,
    is_built, mosaic_pixel_area_m2, training_eligible, units_indicators, vertex_table, zone_indicators,
    azimuth_to_dms,
)

TR = from_origin(-8246700.0, 513800.0, 0.5, 0.5)


def test_color_names():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    m = np.ones((10, 10), dtype=bool)
    img[:] = (190, 70, 45)
    assert color_stats(img, m)["color_dominante"] == "rojo / teja de barro"
    img[:] = (40, 90, 200)
    assert color_stats(img, m)["color_dominante"] == "azul"
    img[:] = (128, 128, 130)
    assert color_stats(img, m)["color_dominante"] == "gris / concreto"
    img[:] = (240, 240, 240)
    assert color_stats(img, m)["color_dominante"] == "blanco / claro"


def _synthetic_samples(n_per=12, seed=0):
    """Tres clases separables por forma y color (como techo rojo, césped y vía gris)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    specs = {"Edificación residencial": ((190, 70, 45), 300, 0.7),
             "Zona verde": ((90, 150, 60), 2000, 0.5),
             "Vía": ((120, 120, 125), 900, 0.1)}
    for cat, (rgb, area, comp) in specs.items():
        for _ in range(n_per):
            img = np.clip(np.array(rgb) + rng.normal(0, 12, size=(20, 20, 3)), 0, 255).astype(np.uint8)
            col = color_stats(img, np.ones((20, 20), bool))
            attrs = {"area_m2": area * rng.uniform(0.6, 1.4), "perimeter_m": 4 * math.sqrt(area) / max(comp, .05) ** .5,
                     "compactness": comp + rng.normal(0, .05), "solidity": .9, "rectangularity": .8,
                     "elongation": .2 if cat != "Vía" else .9}
            X.append(handcrafted_features(attrs, col)); y.append(cat)
    return np.stack(X), y


def test_supervised_classifier_fit_predict_and_cv():
    X, y = _synthetic_samples()
    clf = SupervisedClassifier().fit(X, y)
    pred = [c for c, _ in clf.predict(X)]
    assert np.mean(np.array(pred) == np.array(y)) == 1.0
    p = clf.predict_proba(X)
    assert np.allclose(p.sum(1), 1)
    cv = SupervisedClassifier.cross_validate(X, np.array(y), k=5)
    assert cv["exactitud"] > 0.9 and cv["kappa"] > 0.85
    assert cv["matriz_confusion"].values.sum() == len(y)
    assert set(cv["por_clase"]["categoria"]) == set(y)
    assert "estratificada" in cv["metodo"]
    # pocas muestras → dejando uno fuera
    Xs, ys = _synthetic_samples(n_per=3)
    assert "dejando uno fuera" in SupervisedClassifier.cross_validate(Xs, np.array(ys))["metodo"]
    with pytest.raises(ValueError):
        SupervisedClassifier().fit(X[:3], ["a", "a", "a"])


def test_training_eligibility():
    assert training_eligible("Vía", "aceptado", "usuario")
    assert training_eligible("Vía", "aceptado", "RemoteCLIP (confirmada por el usuario)")
    assert not training_eligible("Vía", "propuesto", "usuario")
    assert not training_eligible("Sin categoría", "aceptado", "usuario")
    assert not training_eligible("Vía", "aceptado", "Supervisada 80% (sin confirmar)")


def test_coverage_raster_and_stats():
    a = np.zeros((10, 10), bool); a[:5] = True
    b = np.zeros((10, 10), bool); b[5:, :5] = True
    r, legend = coverage_raster([(a, "Zona verde"), (b, "Vía"), (b, "Sin categoría")], (10, 10))
    stats = coverage_stats(r, legend, pixel_area_m2=0.25).set_index("categoria")
    assert stats.loc["Zona verde", "porcentaje"] == 50 and stats.loc["Vía", "porcentaje"] == 25
    assert stats.loc["Sin clasificar", "area_m2"] == 25 * 0.25
    tif = export_coverage_geotiff(r, legend, {1: (0, 255, 0), 2: (100, 100, 100)}, TR)
    import rasterio
    with rasterio.MemoryFile(tif) as mf, mf.open() as ds:
        assert ds.colormap(1)[1][:3] == (0, 255, 0)
        assert ds.tags()["clase_1"] == "Zona verde"


def test_pixel_area_matches_geodesy():
    px, total, crs = mosaic_pixel_area_m2(TR, 200, 200)
    assert crs == "EPSG:9377"
    # 0.5 m en 3857 a 4.6° de latitud → ≈0.498 m en el terreno → ≈0.248 m² por píxel
    assert px == pytest.approx(0.248, rel=0.01)


def test_vertex_table_square():
    # Cuadrado de 20 m × 20 m en 9377 construido en lat/lon
    sq = gpd.GeoSeries([box(4884000, 2075000, 4884020, 2075020)], crs="EPSG:9377").to_crs(4326).iloc[0]
    vt = vertex_table(sq, tolerance_m=0.01)
    assert len(vt) == 4
    assert vt["distancia_m"].tolist() == pytest.approx([20, 20, 20, 20], abs=0.01)
    assert vt.attrs["area_m2"] == pytest.approx(400, abs=0.1)
    # horario desde el vértice más al norte (empate → el más al oeste): NO → NE → SE → SO
    assert vt.loc[0, "este_x"] == pytest.approx(4884000, abs=0.01) and vt.loc[0, "norte_y"] == pytest.approx(2075020, abs=0.01)
    assert vt["acimut_deg"].tolist() == pytest.approx([90, 180, 270, 0], abs=0.01)
    assert azimuth_to_dms(90.5) == '90°30\'00.00"'


def test_zone_and_units_indicators():
    m1 = np.zeros((200, 200), bool); m1[10:50, 10:50] = True       # edificación
    m2 = np.zeros((200, 200), bool); m2[100:180, 100:180] = True   # zona verde
    rows = []
    for m, lab in ((m1, "Edificación residencial"), (m2, "Zona verde")):
        g = mask_to_geodataframe(m, TR, object_id=lab[:3]); g["label"] = lab; rows.append(g)
    objs = gpd.GeoDataFrame(pd.concat(rows), crs="EPSG:4326")
    ind = zone_indicators(objs, zone_area_m2=10_000)
    assert ind["n_edificaciones"] == 1 and 0 < ind["indice_ocupacion"] < 1
    assert ind["cobertura_verde_pct"] > ind["indice_ocupacion"] * 100
    # unidades: una que contiene la edificación y otra la zona verde
    xmin, ymin, xmax, ymax = objs.total_bounds
    midx = (xmin + xmax) / 2
    units = gpd.GeoDataFrame({"predio": ["A", "B"]},
                             geometry=[box(xmin - 1e-4, ymin - 1e-4, midx, ymax + 1e-4),
                                       box(midx, ymin - 1e-4, xmax + 1e-4, ymax + 1e-4)], crs=4326)
    res = units_indicators(objs, units, "predio").set_index("predio")
    assert res.loc["A", "n_edificaciones"] == 1 and res.loc["B", "n_edificaciones"] == 0
    assert res.loc["A", "area_construida_m2"] == pytest.approx(objs.iloc[0]["area_m2"], rel=0.02)
    assert res.loc["B", "pct_verde"] > 0
    assert is_built("Edificio en altura") and not is_built("Vía")


def test_dxf_and_kml():
    m = np.zeros((100, 100), bool); m[10:40, 10:40] = True
    g = mask_to_geodataframe(m, TR, label="Edificación religiosa")
    g["status"] = "aceptado"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.dxf"); open(p, "wb").write(export_dxf(g))
        back = gpd.read_file(p)
        assert len(back) >= 1 and back["Layer"].iloc[0] == "Edificacion religiosa"
        k = os.path.join(d, "a.kml"); open(k, "wb").write(export_kml(g))
        kb = gpd.read_file(k)
        assert len(kb) == 1 and "Edificación religiosa" in kb["Name"].iloc[0]
