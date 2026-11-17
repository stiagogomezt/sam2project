"""
Sentinel-2 L2A desde el catálogo público Earth Search (AWS, sin cuenta).

Busca la escena con menos nubes sobre la zona y el periodo pedidos, lee solo la ventana necesaria de
los COG (sin descargar la escena completa) y la reproyecta a una grilla EPSG:3857 de 10 m, la misma
que usa el resto de la aplicación para el mapa y los clics. Devuelve:

* `img_array`: RGB uint8 con realce para SAM 2 (que trabaja con imágenes RGB);
* `bands`: reflectancias de superficie (0–1) de azul, verde, rojo, NIR, SWIR1 y SWIR2, y la máscara
  SCL, para los índices espectrales y la clasificación de coberturas;
* los mismos campos de georreferencia que `mosaic_builder.build_mosaic`.

Copernicus Sentinel-2: datos abiertos (atribución «Contiene datos modificados de Copernicus Sentinel»).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
BANDS = {"blue": "blue", "green": "green", "red": "red", "nir": "nir", "swir16": "swir16", "swir22": "swir22", "scl": "scl"}
CLOUD_SCL = (3, 8, 9, 10)          # sombra de nube, nube media, nube alta, cirro
MAX_SIDE_M = 20_000                # 20 km: tope de la ventana (≈2000 × 2000 px)
ATTRIBUTION = "Contiene datos modificados de Copernicus Sentinel"


def search_scenes(bbox_4326: Tuple[float, float, float, float], start: str, end: str,
                  max_cloud: float = 30.0, limit: int = 50, session=None) -> List[dict]:
    """Escenas L2A que cubren la caja, ordenadas de menos a más nubes (nubosidad de la escena)."""
    import requests

    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox_4326),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": limit,
    }
    r = (session or requests).post(STAC_URL, json=body, timeout=60)
    r.raise_for_status()
    feats = [f for f in r.json().get("features", []) if str(f["assets"]["red"]["href"]).startswith("http")]
    return sorted(feats, key=lambda f: f["properties"].get("eo:cloud_cover", 100))


def grid_3857(bbox_4326: Tuple[float, float, float, float], res: float = 10.0):
    """Grilla EPSG:3857 que cubre la caja: (transform, ancho, alto, bounds_3857).

    El tamaño de píxel en metros de Mercator se corrige por la latitud para que equivalga a ~`res` m
    sobre el terreno."""
    from pyproj import Transformer
    from rasterio.transform import from_origin

    w, s, e, n = bbox_4326
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = tr.transform(w, s)
    x1, y1 = tr.transform(e, n)
    lat = np.radians((s + n) / 2)
    px = res / np.cos(lat)
    width = int(np.ceil((x1 - x0) / px))
    height = int(np.ceil((y1 - y0) / px))
    side_m = max(width, height) * res
    if side_m > MAX_SIDE_M:
        raise ValueError(f"La zona mide ~{side_m / 1000:.1f} km de lado; el máximo para Sentinel-2 es {MAX_SIDE_M / 1000:.0f} km.")
    return from_origin(x0, y1, px, px), width, height, (x0, y0, x0 + width * px, y1 - height * px)


def stretch_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """RGB uint8 con realce por percentiles 1–97 (común a las tres bandas, conserva el color)."""
    stack = np.dstack([red, green, blue]).astype("float32")
    valid = stack.sum(axis=2) > 0
    if not valid.any():
        return np.zeros(stack.shape, dtype=np.uint8)
    # Realce común a las tres bandas (conserva el color), con techo en el percentil 97 para que la arena,
    # las nubes sueltas o los techos claros no oscurezcan el resto, y gamma < 1 para levantar la vegetación.
    lo, hi = np.percentile(stack[valid], [1, 97])
    out = np.clip((stack - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.75
    return (out * 255).astype(np.uint8)


def load_scene(feature: dict, bbox_4326: Tuple[float, float, float, float], res: float = 10.0) -> Dict:
    """Lee la escena sobre la grilla 3857 y arma el «mosaico» que consume la aplicación."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    T, width, height, b3857 = grid_3857(bbox_4326, res)
    bands: Dict[str, np.ndarray] = {}
    for name, key in BANDS.items():
        href = feature["assets"][key]["href"]
        resampling = Resampling.nearest if name == "scl" else Resampling.bilinear
        with rasterio.open(href) as src:
            # El VRT se define directamente sobre la grilla de salida: si la escena no cubre toda la zona,
            # lo que falta queda en 0 (sin dato) en vez de fallar.
            with WarpedVRT(src, crs="EPSG:3857", transform=T, width=width, height=height,
                           resampling=resampling, nodata=0) as vrt:
                arr = vrt.read(1)
        if name == "scl":
            bands[name] = arr.astype(np.uint8)
        else:
            # Línea base ≥ 04.00: reflectancia = (DN − 1000) / 10000 (Earth Search ya trae el desplazamiento
            # en raster:bands; se aplica aquí de forma explícita).
            off = feature["assets"][key].get("raster:bands", [{}])[0].get("offset", 0.0) or 0.0
            scale = feature["assets"][key].get("raster:bands", [{}])[0].get("scale", 1e-4) or 1e-4
            valid = arr[arr > 0]
            if off < 0 and valid.size and np.percentile(valid, 1) < -off / scale:
                # Los COG de Earth Search ya vienen sin el +1000 aunque los metadatos digan lo contrario
                # (un bosque tiene ND≈200 en el rojo): aplicar el desplazamiento daría reflectancias negativas.
                off = 0.0
            refl = arr.astype("float32") * scale + off
            refl[arr == 0] = np.nan
            bands[name] = refl
    rgb = stretch_rgb(np.nan_to_num(bands["red"]), np.nan_to_num(bands["green"]), np.nan_to_num(bands["blue"]))
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    w4, s4 = tr.transform(b3857[0], b3857[1])
    e4, n4 = tr.transform(b3857[2], b3857[3])
    scl = bands["scl"]
    cloud = float(np.isin(scl, CLOUD_SCL).sum() / max((scl > 0).sum(), 1))
    return {
        "img_array": rgb,
        "bands": bands,
        "width": width,
        "height": height,
        "transform_3857": T,
        "transform_4326": from_bounds(w4, s4, e4, n4, width, height),
        "bounds_3857": b3857,
        "bounds_4326": (w4, s4, e4, n4),
        "source": "sentinel-2",
        "scene_id": feature["id"],
        "scene_date": feature["properties"]["datetime"][:10],
        "scene_cloud": float(feature["properties"].get("eo:cloud_cover", np.nan)),
        "window_cloud": cloud,
        "res_m": res,
        "attribution": ATTRIBUTION,
    }


def best_scene(bbox_4326, start: str, end: str, max_cloud: float = 30.0, max_tries: int = 6,
               window_cloud_max: float = 0.05) -> Dict:
    """Primera escena (de menos nubes) cuya ventana tenga menos de `window_cloud_max` de nubes según SCL."""
    feats = search_scenes(bbox_4326, start, end, max_cloud)
    if not feats:
        raise ValueError("No hay escenas Sentinel-2 en ese periodo con esa nubosidad.")
    best: Optional[Dict] = None
    for f in feats[:max_tries]:
        scene = load_scene(f, bbox_4326)
        valid = float((scene["bands"]["scl"] > 0).mean())
        if valid < 0.95:
            continue
        if best is None or scene["window_cloud"] < best["window_cloud"]:
            best = scene
        if scene["window_cloud"] <= window_cloud_max:
            break
    if best is None:
        raise ValueError("Las escenas encontradas no cubren completa la zona; amplíe el periodo.")
    return best
