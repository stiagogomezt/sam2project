"""Zona de estudio inicial desde la URL (`?bbox=oeste,sur,este,norte&z=19`).

Permite abrir la aplicación ya centrada en un caso (por ejemplo, desde un enlace o un código QR en
una presentación). Los valores se validan y, si no son válidos, se usan los predeterminados.
"""
from typing import Dict, Mapping, Optional, Tuple

from config import DEFAULT_BBOX, DEFAULT_ZOOM, MAX_ZOOM, MIN_ZOOM

# Lado máximo de la caja (grados): evita que un enlace dispare descargas enormes.
MAX_SIDE_DEG = 0.02


def _first(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value)


def parse_view_params(params: Mapping) -> Tuple[Dict[str, float], int, bool]:
    """Devuelve (bbox, zoom, viene_de_url) a partir de los parámetros de consulta.

    `bbox` = "oeste,sur,este,norte" en WGS84; `z` = zoom de teselas entre MIN_ZOOM y MAX_ZOOM.
    """
    bbox = dict(DEFAULT_BBOX)
    zoom = DEFAULT_ZOOM
    from_url = False
    raw = _first(params.get("bbox")) if params else None
    if raw:
        try:
            w, s, e, n = (float(v) for v in raw.split(","))
            valid = (-180 <= w < e <= 180 and -90 <= s < n <= 90
                     and (e - w) <= MAX_SIDE_DEG and (n - s) <= MAX_SIDE_DEG)
            if valid:
                bbox = {"min_lon": w, "min_lat": s, "max_lon": e, "max_lat": n}
                from_url = True
        except ValueError:
            pass
    raw_z = _first(params.get("z")) if params else None
    if raw_z:
        try:
            z = int(raw_z)
            if MIN_ZOOM <= z <= MAX_ZOOM:
                zoom = z
                from_url = True
        except ValueError:
            pass
    return bbox, zoom, from_url
