"""
Módulo geoespacial: vectorización de máscaras, reproyección rigurosa y métricas en metros.

Reglas (docs/DECISIONES.md, DEC-0004):
- Ninguna métrica se calcula en grados. Toda geometría se proyecta a un CRS métrico:
  EPSG:9377 (MAGNA-SIRGAS 2018 / Origen-Nacional) dentro de su área de uso, o UTM fuera de ella.
- La geometría vectorial se obtiene en EPSG:3857 (espacio nativo de las teselas) y se almacena en
  EPSG:4326.

Área planar vs. geodésica: `area_m2`/`perimeter_m` se miden en el plano del CRS métrico (en Colombia,
EPSG:9377, el sistema oficial para catastro); `area_geod_m2`/`perimeter_geod_m` se miden sobre el
elipsoide WGS84 y no dependen de la proyección. En Bogotá difieren ≈0,1 % por el factor de escala
k0=0,9992 de 9377.

Precisión del contorno: el polígono ráster tiene borde en "escalera", que sobrestima el perímetro
≈28 % en objetos curvos. Por defecto se simplifica con Douglas-Peucker a 1 píxel
(`simplify_px=1.0`), lo que deja el error de perímetro en ≈1 % y el de área por debajo de 1 %
(validado en tests/unit/test_geometry_metrics.py). `pixel_area_m2` conserva el área exacta de la
máscara antes de simplificar.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
from pyproj import CRS, Geod
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from config import CRS_GEOGRAPHIC, CRS_WEB_MERCATOR, CRS_COLOMBIA_METRIC

# Orden de columnas de atributos producidas por mask_to_geodataframe
ATTRIBUTE_COLUMNS = [
    "object_id", "label", "pred_iou",
    "area_m2", "perimeter_m", "perimeter_ext_m", "area_geod_m2", "perimeter_geod_m", "pixel_area_m2", "pixel_count",
    "hull_area_m2", "solidity", "compactness", "rectangularity",
    "elongation", "aspect_ratio", "orientation_deg", "mrr_length_m", "mrr_width_m",
    "n_parts", "n_holes",
    "centroid_lat", "centroid_lon", "min_lon", "min_lat", "max_lon", "max_lat",
    "metric_crs", "gsd_m", "timestamp",
]


_GEOD = Geod(ellps="WGS84")


def geodesic_area_perimeter(geom_4326: BaseGeometry) -> tuple[float, float]:
    """Área (m²) y perímetro (m) sobre el elipsoide WGS84 (referencia independiente de proyección)."""
    area, perim = _GEOD.geometry_area_perimeter(geom_4326)
    return abs(float(area)), float(perim)


@lru_cache(maxsize=8)
def _area_of_use(crs_code: str):
    aou = CRS(crs_code).area_of_use
    return (aou.west, aou.south, aou.east, aou.north) if aou else None


def select_metric_crs(lon: float, lat: float) -> str:
    """CRS proyectado para métricas: EPSG:9377 dentro de Colombia (área de uso oficial) o UTM."""
    bounds = _area_of_use(CRS_COLOMBIA_METRIC)
    if bounds and bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]:
        return CRS_COLOMBIA_METRIC
    zone = int((lon + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def compute_shape_metrics(geom: BaseGeometry) -> Dict[str, Any]:
    """Descriptores de forma de una geometría expresada en un CRS MÉTRICO.

    - perimeter_m incluye los bordes de los huecos (patios); perimeter_ext_m solo el contorno exterior.

    - compactness (Polsby-Popper) = 4πA / P²  → 1 círculo, ≈0.785 cuadrado, →0 formas irregulares.
    - solidity = A / A(convex hull)           → 1 convexo; baja con concavidades.
    - rectangularity = A / A(rectángulo mínimo rotado) → 1 rectángulo.
    - elongation = 1 - ancho/largo del rectángulo mínimo rotado → 0 isótropo, →1 alargado.
    - aspect_ratio = largo/ancho.
    - orientation_deg = acimut de cuadrícula del eje largo, horario desde el norte, en [0, 180).
      (Es acimut de la cuadrícula del CRS; difiere del norte verdadero por la convergencia de
      meridianos, despreciable a la escala de objetos individuales.)
    """
    if geom is None or geom.is_empty:
        raise ValueError("Geometría vacía")
    area = float(geom.area)
    perim = float(geom.length)
    hull = geom.convex_hull
    hull_area = float(hull.area)
    mrr = geom.minimum_rotated_rectangle

    length = width = 0.0
    orientation = 0.0
    if isinstance(mrr, Polygon) and len(mrr.exterior.coords) >= 4:
        c = list(mrr.exterior.coords)
        e1 = (c[1][0] - c[0][0], c[1][1] - c[0][1])
        e2 = (c[2][0] - c[1][0], c[2][1] - c[1][1])
        l1, l2 = math.hypot(*e1), math.hypot(*e2)
        long_e = e1 if l1 >= l2 else e2
        length, width = max(l1, l2), min(l1, l2)
        orientation = math.degrees(math.atan2(long_e[0], long_e[1])) % 180.0  # dx, dy → acimut

    parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    n_holes = sum(len(p.interiors) for p in parts if isinstance(p, Polygon))
    # Perímetro exterior: solo anillos exteriores (sin patios/huecos); es el contorno de la huella.
    perim_ext = float(sum(p.exterior.length for p in parts if isinstance(p, Polygon)))
    return {
        "area_m2": area,
        "perimeter_m": perim,
        "perimeter_ext_m": perim_ext,
        "hull_area_m2": hull_area,
        "solidity": area / hull_area if hull_area > 0 else 0.0,
        "compactness": 4.0 * math.pi * area / (perim ** 2) if perim > 0 else 0.0,
        "rectangularity": area / float(mrr.area) if mrr.area > 0 else 0.0,
        "elongation": 1.0 - (width / length) if length > 0 else 0.0,
        "aspect_ratio": (length / width) if width > 0 else float("inf"),
        "orientation_deg": orientation,
        "mrr_length_m": length,
        "mrr_width_m": width,
        "n_parts": len(parts),
        "n_holes": n_holes,
    }


def vectorize_mask(
    mask: np.ndarray,
    transform_3857: rasterio.Affine,
    min_pixel_area: int = 4,
    simplify_px: float = 1.0,
) -> Optional[BaseGeometry]:
    """Máscara binaria → (Multi)Polygon en EPSG:3857, descartando componentes < min_pixel_area."""
    mask_uint8 = np.ascontiguousarray(mask.astype(np.uint8))
    px_area = abs(transform_3857.a * transform_3857.e)
    polygons = []
    for geom_dict, val in rasterio.features.shapes(mask_uint8, mask=mask_uint8.astype(bool), transform=transform_3857):
        if val != 1:
            continue
        geom = shape(geom_dict)
        if geom.is_empty or not geom.is_valid or geom.area <= 0:
            continue
        if geom.area / px_area < min_pixel_area:
            continue
        polygons.append(geom)
    if not polygons:
        return None
    merged = unary_union(polygons)
    if simplify_px and simplify_px > 0:
        simplified = merged.simplify(simplify_px * abs(transform_3857.a), preserve_topology=True)
        if not simplified.is_empty and simplified.is_valid:
            merged = simplified
    return merged


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=ATTRIBUTE_COLUMNS + ["geometry"], geometry="geometry", crs=CRS_GEOGRAPHIC)


def mask_to_geodataframe(
    mask: np.ndarray,
    transform_3857: rasterio.Affine,
    confidence: float = 1.0,
    min_pixel_area: int = 4,
    simplify_px: float = 1.0,
    label: str = "objeto",
    object_id: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Convierte una máscara binaria en un GeoDataFrame (EPSG:4326) de 1 fila con métricas métricas.

    `confidence` es el IoU predicho por SAM 2 para la máscara (se guarda en `pred_iou`); NO es una
    probabilidad de clase. Devuelve un GeoDataFrame vacío si la máscara no produce polígonos.
    """
    mask_bool = np.asarray(mask).astype(bool)
    merged_3857 = vectorize_mask(mask_bool, transform_3857, min_pixel_area=min_pixel_area, simplify_px=simplify_px)
    if merged_3857 is None:
        return _empty_gdf()

    gdf_3857 = gpd.GeoDataFrame(geometry=[merged_3857], crs=CRS_WEB_MERCATOR)
    gdf_4326 = gdf_3857.to_crs(CRS_GEOGRAPHIC)
    geom_4326 = gdf_4326.geometry.iloc[0]

    # CRS métrico elegido a partir de un punto representativo (siempre dentro del objeto)
    rep = geom_4326.representative_point()
    metric_crs = select_metric_crs(rep.x, rep.y)
    gdf_metric = gdf_4326.to_crs(metric_crs)
    geom_metric = gdf_metric.geometry.iloc[0]
    metrics = compute_shape_metrics(geom_metric)
    area_geod, perim_geod = geodesic_area_perimeter(geom_4326)

    # Centroide calculado en el CRS métrico (correcto) y devuelto a lat/lon
    centroid_4326 = gpd.GeoSeries([geom_metric.centroid], crs=metric_crs).to_crs(CRS_GEOGRAPHIC).iloc[0]

    # Área exacta de la máscara (antes de simplificar) y GSD real del terreno
    pixel_count = int(mask_bool.sum())
    raw = vectorize_mask(mask_bool, transform_3857, min_pixel_area=0, simplify_px=0)
    pixel_area_m2 = float(gpd.GeoSeries([raw], crs=CRS_WEB_MERCATOR).to_crs(metric_crs).area.iloc[0]) if raw else 0.0
    gsd_m = math.sqrt(pixel_area_m2 / pixel_count) if pixel_count else float("nan")

    minx, miny, maxx, maxy = geom_4326.bounds
    row = {
        "object_id": object_id or f"OBJ-{uuid.uuid4().hex[:8].upper()}",
        "label": label,
        "pred_iou": round(float(confidence), 4),
        "area_m2": round(metrics["area_m2"], 2),
        "perimeter_m": round(metrics["perimeter_m"], 2),
        "perimeter_ext_m": round(metrics["perimeter_ext_m"], 2),
        "area_geod_m2": round(area_geod, 2),
        "perimeter_geod_m": round(perim_geod, 2),
        "pixel_area_m2": round(pixel_area_m2, 2),
        "pixel_count": pixel_count,
        "hull_area_m2": round(metrics["hull_area_m2"], 2),
        "solidity": round(metrics["solidity"], 4),
        "compactness": round(metrics["compactness"], 4),
        "rectangularity": round(metrics["rectangularity"], 4),
        "elongation": round(metrics["elongation"], 4),
        "aspect_ratio": round(metrics["aspect_ratio"], 3) if math.isfinite(metrics["aspect_ratio"]) else None,
        "orientation_deg": round(metrics["orientation_deg"], 1),
        "mrr_length_m": round(metrics["mrr_length_m"], 2),
        "mrr_width_m": round(metrics["mrr_width_m"], 2),
        "n_parts": metrics["n_parts"],
        "n_holes": metrics["n_holes"],
        "centroid_lat": round(float(centroid_4326.y), 7),
        "centroid_lon": round(float(centroid_4326.x), 7),
        "min_lon": round(minx, 7),
        "min_lat": round(miny, 7),
        "max_lon": round(maxx, 7),
        "max_lat": round(maxy, 7),
        "metric_crs": metric_crs,
        "gsd_m": round(gsd_m, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return gpd.GeoDataFrame({**{k: [v] for k, v in row.items()}, "geometry": [geom_4326]},
                            geometry="geometry", crs=CRS_GEOGRAPHIC)


def derived_geometries(gdf: gpd.GeoDataFrame) -> Dict[str, gpd.GeoDataFrame]:
    """Capas derivadas (EPSG:4326) por objeto: centroide, bounding box, convex hull y rectángulo
    mínimo rotado. Se construyen en el CRS métrico de cada objeto para que sean exactas."""
    out: Dict[str, list] = {"centroides": [], "bbox": [], "convex_hull": [], "rect_min_rotado": []}
    for _, r in gdf.iterrows():
        crs_m = r.get("metric_crs") or select_metric_crs(r.geometry.centroid.x, r.geometry.centroid.y)
        g = gpd.GeoSeries([r.geometry], crs=gdf.crs).to_crs(crs_m).iloc[0]
        minx, miny, maxx, maxy = g.bounds
        bbox = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
        for name, geom in (("centroides", g.centroid), ("bbox", bbox),
                           ("convex_hull", g.convex_hull), ("rect_min_rotado", g.minimum_rotated_rectangle)):
            g4326 = gpd.GeoSeries([geom], crs=crs_m).to_crs(gdf.crs).iloc[0]
            out[name].append({"object_id": r["object_id"], "label": r.get("label"), "geometry": g4326})
    return {k: gpd.GeoDataFrame(v, geometry="geometry", crs=gdf.crs) if v else
            gpd.GeoDataFrame(columns=["object_id", "label", "geometry"], geometry="geometry", crs=gdf.crs)
            for k, v in out.items()}


RELATION_LABELS = {
    "equals": "iguales", "contains": "contiene", "within": "dentro de",
    "overlaps": "se superpone", "touches": "toca", "disjoint": "disjunto",
}


def spatial_relations(gdf: gpd.GeoDataFrame, max_pairs_distance_m: float = 100.0) -> tuple[gpd.GeoDataFrame, "Any"]:
    """Relaciones espaciales entre objetos, en un CRS métrico común.

    Devuelve (resumen_por_objeto, tabla_de_pares). El resumen añade `nn_id` (vecino más cercano),
    `nn_dist_m`, `n_contacts` (toca o se superpone). La tabla lista los pares a menos de
    `max_pairs_distance_m` con su distancia (m) y relación topológica (DE-9IM simplificado).
    """
    import pandas as pd

    if gdf.empty:
        return gdf.assign(nn_id=None, nn_dist_m=None, n_contacts=0), pd.DataFrame(
            columns=["id_a", "id_b", "distancia_m", "relacion"])
    rep = gdf.union_all().representative_point()
    crs_m = select_metric_crs(rep.x, rep.y)
    gm = gdf.to_crs(crs_m).reset_index(drop=True)
    ids = gm["object_id"].tolist()
    geoms = gm.geometry.tolist()
    n = len(geoms)
    nn_id: list = [None] * n
    nn_d: list = [None] * n
    contacts = [0] * n
    pairs = []
    for i in range(n):
        best = (None, float("inf"))
        for j in range(n):
            if i == j:
                continue
            d = geoms[i].distance(geoms[j])
            if d < best[1]:
                best = (ids[j], d)
            if j > i and d <= max_pairs_distance_m:
                a, b = geoms[i], geoms[j]
                if a.equals(b):
                    rel = "equals"
                elif a.contains(b):
                    rel = "contains"
                elif a.within(b):
                    rel = "within"
                elif a.overlaps(b) or (a.intersects(b) and a.intersection(b).area > 0):
                    rel = "overlaps"
                elif a.touches(b) or d == 0:
                    rel = "touches"
                else:
                    rel = "disjoint"
                if rel != "disjoint":
                    contacts[i] += 1
                    contacts[j] += 1
                pairs.append({"id_a": ids[i], "id_b": ids[j], "distancia_m": round(d, 2),
                              "relacion": RELATION_LABELS[rel]})
        if best[0] is not None:
            nn_id[i], nn_d[i] = best[0], round(best[1], 2)
    summary = gdf.reset_index(drop=True).copy()
    summary["nn_id"] = nn_id
    summary["nn_dist_m"] = nn_d
    summary["n_contacts"] = contacts
    return summary, pd.DataFrame(pairs, columns=["id_a", "id_b", "distancia_m", "relacion"])
