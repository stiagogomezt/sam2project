"""
Módulo para descarga concurrente, resiliente y cacheada de teselas de imágenes aéreas/satelitales.
"""
import io
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from typing import List, Dict, Callable, Optional, Tuple
import mercantile
from config import ESRI_IMAGERY_URL

def _create_session() -> requests.Session:
    """Crea una sesión de requests con política de reintentos y timeouts robustos."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "SAM2_Streamlit_GIS_App/1.0 (Research/Geospatial)"
    })
    return session

NEUTRAL_TILE_COLOR = (40, 40, 40)
NO_DATA_ERROR = "sin imagen a este zoom"


def is_placeholder_tile(img: Image.Image) -> bool:
    """Esri devuelve un gris liso («Map data not yet available») donde no hay imagen a ese zoom."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    mean = arr.reshape(-1, 3).mean(axis=0)
    return bool(arr.std() < 10 and abs(mean - 205).max() < 12 and np.ptp(mean) < 3)


def probe_max_zoom(lon: float, lat: float, z_max: int = 19, z_min: int = 15,
                   url_template: str = ESRI_IMAGERY_URL) -> Optional[int]:
    """Zoom más alto con imagen real en el punto (lon, lat); None si no responde el proveedor."""
    session = _create_session()
    answered = False
    for z in range(z_max, z_min - 1, -1):
        _t, img, err = _fetch_tile(mercantile.tile(lon, lat, z), session, url_template)
        if img is not None:
            return z
        if err == NO_DATA_ERROR:
            answered = True
    return z_min if answered else None


def _fetch_tile(tile: mercantile.Tile, session: requests.Session, url_template: str) -> Tuple[mercantile.Tile, Optional[Image.Image], Optional[str]]:
    """Descarga una tesela. Devuelve (tile, imagen o None, mensaje de error o None)."""
    url = url_template.format(z=tile.z, y=tile.y, x=tile.x)
    try:
        response = session.get(url, timeout=12)
        if response.status_code != 200:
            return tile, None, f"HTTP {response.status_code}"
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256))
        if is_placeholder_tile(img):
            return tile, None, NO_DATA_ERROR
        return tile, img, None
    except Exception as exc:  # red, decodificación, timeout
        return tile, None, type(exc).__name__


def download_single_tile(tile: mercantile.Tile, session: requests.Session, url_template: str = ESRI_IMAGERY_URL) -> Tuple[mercantile.Tile, Optional[Image.Image]]:
    """Descarga una sola tesela; ante error devuelve una tesela neutra (compatibilidad v1)."""
    tile, img, _err = _fetch_tile(tile, session, url_template)
    if img is None:
        img = Image.new("RGB", (256, 256), color=NEUTRAL_TILE_COLOR)
    return tile, img


def download_tiles_with_report(
    tiles: List[mercantile.Tile],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_workers: int = 8,
    url_template: str = ESRI_IMAGERY_URL,
) -> Tuple[Dict[mercantile.Tile, Image.Image], List[Tuple[mercantile.Tile, str]]]:
    """Descarga concurrente que además informa qué teselas fallaron.

    Las teselas fallidas se rellenan con un gris neutro para no romper la georreferenciación del
    mosaico, pero se devuelven en la lista `failed` para que la interfaz avise al usuario.
    """
    session = _create_session()
    results: Dict[mercantile.Tile, Image.Image] = {}
    failed: List[Tuple[mercantile.Tile, str]] = []
    total = len(tiles)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_tile, t, session, url_template) for t in tiles]
        for future in as_completed(futures):
            tile, img, err = future.result()
            if img is None:
                failed.append((tile, err or "desconocido"))
                img = Image.new("RGB", (256, 256), color=NEUTRAL_TILE_COLOR)
            results[tile] = img
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
    return results, failed


def download_tiles_concurrently(
    tiles: List[mercantile.Tile],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_workers: int = 8,
    url_template: str = ESRI_IMAGERY_URL
) -> Dict[mercantile.Tile, Image.Image]:
    """
    Descarga una lista de teselas de forma concurrente informando el progreso (API v1).
    Use `download_tiles_with_report` para conocer las teselas que fallaron.
    """
    results, _failed = download_tiles_with_report(tiles, progress_callback, max_workers, url_template)
    return results
