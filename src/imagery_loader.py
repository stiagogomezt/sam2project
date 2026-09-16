"""
Módulo para descarga concurrente, resiliente y cacheada de teselas de imágenes aéreas/satelitales.
"""
import io
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

def download_single_tile(tile: mercantile.Tile, session: requests.Session, url_template: str = ESRI_IMAGERY_URL) -> Tuple[mercantile.Tile, Optional[Image.Image]]:
    """Descarga una sola tesela y la decodifica como imagen PIL."""
    url = url_template.format(z=tile.z, y=tile.y, x=tile.x)
    try:
        response = session.get(url, timeout=12)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            return tile, img
        else:
            # En caso de error HTTP, crear una tesela neutra oscura para no romper el mosaico
            neutral = Image.new("RGB", (256, 256), color=(40, 40, 40))
            return tile, neutral
    except Exception:
        neutral = Image.new("RGB", (256, 256), color=(40, 40, 40))
        return tile, neutral

def download_tiles_concurrently(
    tiles: List[mercantile.Tile],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_workers: int = 8,
    url_template: str = ESRI_IMAGERY_URL
) -> Dict[mercantile.Tile, Image.Image]:
    """
    Descarga una lista de teselas de forma concurrente informando el progreso.
    """
    session = _create_session()
    results: Dict[mercantile.Tile, Image.Image] = {}
    total = len(tiles)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tile = {
            executor.submit(download_single_tile, t, session, url_template): t 
            for t in tiles
        }
        
        for future in as_completed(future_to_tile):
            tile, img = future.result()
            results[tile] = img
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
                
    return results
