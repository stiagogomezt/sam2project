import numpy as np
import pytest

from src import landcover as lc
from src.examples import EXAMPLES
from src.sentinel_loader import grid_3857, stretch_rgb


def test_reglas_espectrales_clases_basicas():
    assert lc.spectral_rule(ndvi=-0.1, mndwi=0.4, ndmi=-0.2, red=0.03)[0] == "Agua"
    assert lc.spectral_rule(ndvi=0.85, mndwi=-0.5, ndmi=0.35, red=0.02)[0] == "Bosque y arbolado"
    assert lc.spectral_rule(ndvi=0.7, mndwi=-0.4, ndmi=0.15, red=0.05)[0] == "Pastos y herbazales"
    assert lc.spectral_rule(ndvi=0.1, mndwi=-0.2, ndmi=-0.1, red=0.12)[0] == "Suelo desnudo o construido"
    assert lc.spectral_rule(ndvi=float("nan"), mndwi=0, ndmi=0, red=0)[0] == "Sin datos"


def test_clasificacion_espectral_por_objeto():
    h = w = 20
    bands = {k: np.full((h, w), 0.05, "float32") for k in ("blue", "green", "red", "nir", "swir16", "swir22")}
    bands["nir"][:, :10] = 0.35          # mitad izquierda: vegetación densa
    bands["red"][:, :10] = 0.02
    bands["swir16"][:, :10] = 0.15
    bands["green"][:, 10:] = 0.08        # mitad derecha: agua
    bands["nir"][:, 10:] = 0.02
    bands["swir16"][:, 10:] = 0.01
    left = np.zeros((h, w), bool); left[:, :10] = True
    out = lc.classify_spectral([left, ~left], bands)
    assert [o[0] for o in out] == ["Bosque y arbolado", "Agua"]


def test_verdor_corrige_a_remoteclip():
    probs = {"Edificaciones": 0.6, "Pastos y herbazales": 0.3, "Bosque y arbolado": 0.1}
    assert lc.combine_rgb(probs, exg=0.15)[0] == "Pastos y herbazales"
    assert lc.combine_rgb(probs, exg=-0.05)[0] == "Edificaciones"


def test_pintado_respeta_objetos_pequenos_y_tabla():
    big = np.ones((10, 10), bool)
    small = np.zeros((10, 10), bool); small[2:4, 2:4] = True
    raster, legend = lc.paint([small, big], ["Edificaciones", "Pastos y herbazales"], (10, 10))
    assert legend[raster[3, 3]] == "Edificaciones" and legend[raster[8, 8]] == "Pastos y herbazales"
    tab = lc.area_table(raster, legend, pixel_area_m2=100.0)
    assert abs(tab["hectareas"].sum() - 1.0) < 1e-6


def test_grilla_sentinel_y_limite():
    T, w, h, b = grid_3857((-74.11, 2.38, -74.01, 2.48))
    assert 1000 < w < 1200 and 1000 < h < 1200
    with pytest.raises(ValueError):
        grid_3857((-75.0, 2.0, -74.0, 3.0))


def test_realce_rgb_uint8():
    r = np.linspace(0, 0.3, 100, dtype="float32").reshape(10, 10)
    out = stretch_rgb(r, r, r)
    assert out.dtype == np.uint8 and out.max() == 255


def test_ejemplos_validos():
    for name, ex in EXAMPLES.items():
        w, s, e, n = ex["bbox"]
        assert w < e and s < n and ex["source"] in ("Ortoimagen", "Sentinel-2")
        if ex["source"] == "Sentinel-2":
            grid_3857(ex["bbox"])


def test_smooth_dark_object_is_not_forest():
    probs = {"Bosque y arbolado": 0.6, "Pastos y herbazales": 0.3, "Cultivos e invernaderos": 0.1}
    name, _ = lc.combine_rgb(probs, 0.2, {"exg": 0.2, "tex": 2.0})
    assert name != "Bosque y arbolado"
    name, _ = lc.combine_rgb(probs, 0.2, {"exg": 0.2, "tex": 6.0})
    assert name == "Bosque y arbolado"


def test_spectral_map_background_goes_pixelwise():
    h = w = 20
    nan = np.full((h, w), 0.05, "float32")
    bands = {"blue": nan.copy(), "green": nan.copy(), "red": np.full((h, w), 0.02, "float32"),
             "nir": np.full((h, w), 0.4, "float32"), "swir16": np.full((h, w), 0.1, "float32")}
    bands["green"][:, :5] = 0.1; bands["nir"][:, :5] = 0.02; bands["swir16"][:, :5] = 0.01; bands["red"][:, :5] = 0.05
    fondo = np.ones((h, w), bool)
    obj = np.zeros((h, w), bool); obj[15:18, 15:18] = True
    r, leg, labels, info = lc.spectral_map([fondo, obj], bands)
    assert labels[0][0] == "Fondo"
    names = {leg[c] for c in np.unique(r) if c}
    assert "Agua" in names and "Bosque y arbolado" in names


def test_grid_bounds_are_west_south_east_north():
    from src.sentinel_loader import grid_3857
    from pyproj import Transformer
    bbox = (-73.86, 7.05, -73.84, 7.07)
    T, w, h, (x0, y0, x1, y1) = grid_3857(bbox)
    assert x1 > x0 and y1 > y0
    tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon0, lat0 = tr.transform(x0, y0)
    lon1, lat1 = tr.transform(x1, y1)
    assert abs(lon0 - bbox[0]) < 1e-3 and abs(lat0 - bbox[1]) < 2e-3
    assert abs(lon1 - bbox[2]) < 2e-3 and abs(lat1 - bbox[3]) < 1e-3
    # la esquina noroeste de la grilla coincide con el origen del transform
    assert abs(T.c - x0) < 1e-6 and abs(T.f - y1) < 1e-6


def test_placeholder_tile_detection():
    from PIL import Image
    from src.imagery_loader import is_placeholder_tile
    gray = Image.new("RGB", (256, 256), (205, 205, 205))
    assert is_placeholder_tile(gray)
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    assert not is_placeholder_tile(photo)
    green = Image.new("RGB", (256, 256), (60, 90, 50))
    assert not is_placeholder_tile(green)
