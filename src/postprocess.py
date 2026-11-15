"""
Post-procesamiento de máscaras para objetos compuestos (edificaciones con patios, claustros, alas).

SAM 2 segmenta regiones visualmente homogéneas: en un convento separa nave, alas y patios. Estas
operaciones convierten la máscara en la HUELLA que espera un analista catastral, y la devuelven
como máscara ráster para que máscara, vector, exportación y dataset sigan siendo coherentes:

- close_px: cierre morfológico (buffer +d / −d) que une partes separadas por menos de 2·d píxeles.
- min_part_px: elimina fragmentos sueltos más pequeños que este número de píxeles.
- keep_largest: conserva solo la parte principal.
- fill_holes: rellena huecos interiores (patios, claustros) → contorno exterior.

Las operaciones se hacen sobre geometría en el espacio de la imagen (shapely) y se rasterizan de
vuelta con la misma transformación; no requieren SciPy ni OpenCV.
"""
from __future__ import annotations

import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union


def touches_border(mask: np.ndarray, margin: int = 1) -> bool:
    """True si la máscara toca el borde de la imagen (objeto posiblemente recortado)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return False
    k = max(1, int(margin))
    return bool(m[:k].any() or m[-k:].any() or m[:, :k].any() or m[:, -k:].any())


def refine_mask(
    mask: np.ndarray,
    fill_holes: bool = False,
    close_px: float = 0.0,
    min_part_px: int = 0,
    keep_largest: bool = False,
) -> np.ndarray:
    """Aplica las operaciones de forma y devuelve una máscara booleana del mismo tamaño."""
    m = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    if not m.any() or not (fill_holes or close_px > 0 or min_part_px > 0 or keep_largest):
        return m.astype(bool)

    ident = rasterio.Affine.identity()  # coordenadas = píxeles
    polys = [shape(g) for g, v in rasterio.features.shapes(m, mask=m.astype(bool), transform=ident) if v == 1]
    geom = unary_union(polys)

    if close_px and close_px > 0:
        geom = geom.buffer(close_px, join_style="mitre").buffer(-close_px, join_style="mitre")

    parts = [p for p in (geom.geoms if isinstance(geom, MultiPolygon) else [geom])
             if isinstance(p, Polygon) and not p.is_empty]
    if min_part_px and min_part_px > 0:
        parts = [p for p in parts if p.area >= min_part_px]
    if keep_largest and parts:
        parts = [max(parts, key=lambda p: p.area)]
    if fill_holes:
        parts = [Polygon(p.exterior) for p in parts]
    if not parts:
        return np.zeros(m.shape, dtype=bool)

    out = rasterio.features.rasterize(
        [(p, 1) for p in parts], out_shape=m.shape, transform=ident, fill=0, dtype=np.uint8
    )
    return out.astype(bool)


def merge_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Unión de varias máscaras del mismo tamaño (fusión de objetos)."""
    if not masks:
        raise ValueError("No hay máscaras para fusionar.")
    out = np.zeros_like(np.asarray(masks[0], dtype=bool))
    for mk in masks:
        out |= np.asarray(mk, dtype=bool)
    return out
