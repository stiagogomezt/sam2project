"""
Productos derivados para Catastro, Geodesia y Ordenamiento (independientes de Streamlit).

1. Clasificación SUPERVISADA orientada a objetos (GEOBIA): SAM 2 aporta los segmentos; las
   categorías confirmadas por el usuario son las muestras de entrenamiento; un clasificador
   (regresión logística multinomial con regularización L2, en PyTorch) aprende de descriptores
   de forma, de color y, si está disponible, del embedding visual de RemoteCLIP. Se valida con
   validación cruzada estratificada (o dejando uno fuera si hay pocas muestras).
2. Mapa de coberturas: ráster clasificado y estadísticas por categoría (m², %).
3. Cuadro de coordenadas y linderos en un CRS métrico (EPSG:9377 en Colombia): vértices E/N,
   distancias y acimutes, numerados en sentido horario desde el vértice más al norte.
4. Indicadores urbanos de la zona y por unidad (predio/manzana cargado por el usuario): índice de
   ocupación, densidad de edificaciones, cobertura verde, tipología de cubiertas por color.

Limitaciones (imagen RGB de Esri, sin infrarrojo ni modelo de elevación): no hay alturas, número de
pisos, índice de construcción ni NDVI. Los vértices son bordes de cubierta vistos en la ortofoto,
no linderos legales: sirven como insumo preliminar y no sustituyen un levantamiento.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

from src.geospatial import select_metric_crs

# --------------------------------------------------------------------------- categorías
BUILT_PREFIXES = ("Edificación", "Edificio en altura", "Cubierta")
GREEN_CATEGORIES = ("Zona verde", "Arbolado")
UNLABELED = {"Sin categoría", "sin etiquetar", "Otro", ""}

CATEGORY_COLORS = {
    "Edificación religiosa": (156, 39, 176), "Edificación institucional": (63, 81, 181),
    "Edificación residencial": (244, 67, 54), "Edificación comercial / industrial": (121, 85, 72),
    "Edificio en altura": (33, 33, 33), "Cubierta": (255, 152, 0), "Patio / claustro": (255, 235, 59),
    "Cancha deportiva": (0, 188, 212), "Zona verde": (139, 195, 74), "Arbolado": (27, 94, 32),
    "Vía": (158, 158, 158), "Vía férrea": (96, 125, 139), "Andén / zona dura": (189, 189, 189),
    "Parqueadero": (120, 144, 156), "Cuerpo de agua": (3, 169, 244),
}


def is_built(category: str) -> bool:
    return str(category).startswith(BUILT_PREFIXES)


_KEYWORD_COLORS = [
    (("teja", "barro"), (191, 87, 47)), (("metál", "metal", "clara", "gris", "concreto"), (120, 144, 175)),
    (("árbol", "arbol", "dosel"), (27, 94, 32)), (("verde", "pasto", "césped", "cesped", "jardín"), (139, 195, 74)),
    (("agua", "río", "rio", "lago", "piscina"), (3, 169, 244)), (("cancha", "deport"), (0, 188, 212)),
    (("vía", "via", "calle", "carretera", "anden", "andén", "dura"), (158, 158, 158)),
    (("edific", "cubierta", "techo", "casa", "vivienda"), (244, 67, 54)),
]


def category_color(category: str) -> Tuple[int, int, int]:
    """Color de la categoría: tabla fija, palabras clave (p. ej. «teja» → terracota) o color estable."""
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    low = str(category).lower()
    for keys, rgb in _KEYWORD_COLORS:
        if any(k in low for k in keys):
            return rgb
    import zlib
    h = zlib.crc32(str(category).encode("utf-8")) % 360  # estable entre ejecuciones (hash() no lo es)  # color estable para categorías escritas por el usuario
    rgb = Image.new("HSV", (1, 1), (int(h / 360 * 255), 160, 210)).convert("RGB").getpixel((0, 0))
    return tuple(rgb)


# --------------------------------------------------------------------------- color de cubierta
def color_stats(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Estadísticos de color dentro de la máscara y nombre del color dominante (voto por píxel)."""
    m = np.asarray(mask, dtype=bool)
    px = img[m]
    if len(px) == 0:
        return {"mean_r": 0, "mean_g": 0, "mean_b": 0, "std_r": 0, "std_g": 0, "std_b": 0,
                "h_sin": 0, "h_cos": 0, "sat": 0, "val": 0, "color_dominante": "sin datos"}
    rgb = px.astype(np.float32) / 255.0
    hsv = np.asarray(Image.fromarray(px.reshape(-1, 1, 3)).convert("HSV")).reshape(-1, 3).astype(np.float32)
    h = hsv[:, 0] / 255.0 * 360.0
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0
    names = np.full(len(px), "gris / concreto", dtype=object)
    names[(s < 0.18) & (v > 0.75)] = "blanco / claro"
    names[v < 0.22] = "oscuro"
    chrom = (s >= 0.18) & (v >= 0.22)
    names[chrom & ((h < 30) | (h >= 330))] = "rojo / teja de barro"
    names[chrom & (h >= 30) & (h < 60)] = "ocre / amarillo"
    names[chrom & (h >= 60) & (h < 170)] = "verde"
    names[chrom & (h >= 170) & (h < 260)] = "azul"
    names[chrom & (h >= 260) & (h < 330)] = "violeta / rosa"
    vals, counts = np.unique(names, return_counts=True)
    rad = np.deg2rad(h)
    w = s  # el tono solo es fiable donde hay saturación
    return {
        "mean_r": float(rgb[:, 0].mean()), "mean_g": float(rgb[:, 1].mean()), "mean_b": float(rgb[:, 2].mean()),
        "std_r": float(rgb[:, 0].std()), "std_g": float(rgb[:, 1].std()), "std_b": float(rgb[:, 2].std()),
        "h_sin": float((np.sin(rad) * w).mean()), "h_cos": float((np.cos(rad) * w).mean()),
        "sat": float(s.mean()), "val": float(v.mean()),
        "color_dominante": str(vals[np.argmax(counts)]),
    }


# --------------------------------------------------------------------------- descriptores
SHAPE_KEYS = ["log_area", "compactness", "solidity", "rectangularity", "elongation", "log_perim_area"]
COLOR_KEYS = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "h_sin", "h_cos", "sat", "val"]


def handcrafted_features(attrs: Dict, color: Dict) -> np.ndarray:
    area = max(float(attrs.get("area_m2") or 0.0), 1e-6)
    perim = max(float(attrs.get("perimeter_m") or 0.0), 1e-6)
    shape = [math.log(area), float(attrs.get("compactness") or 0), float(attrs.get("solidity") or 0),
             float(attrs.get("rectangularity") or 0), float(attrs.get("elongation") or 0),
             math.log(perim / math.sqrt(area))]
    return np.array(shape + [float(color[k]) for k in COLOR_KEYS], dtype=np.float32)


def feature_names(with_embedding: bool, emb_dim: int = 512) -> List[str]:
    names = SHAPE_KEYS + COLOR_KEYS
    return names + ([f"clip_{i}" for i in range(emb_dim)] if with_embedding else [])


# --------------------------------------------------------------------------- clasificador
@dataclass
class SupervisedClassifier:
    """Regresión logística multinomial con L2, entrenada con L-BFGS (PyTorch, CPU)."""

    l2: float = 1e-2
    max_iter: int = 200
    classes_: List[str] = field(default_factory=list)
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    W_: Optional[np.ndarray] = None
    b_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "SupervisedClassifier":
        import torch

        X = np.asarray(X, dtype=np.float32)
        self.classes_ = sorted(set(y))
        if len(self.classes_) < 2:
            raise ValueError("Se necesitan al menos 2 categorías para entrenar.")
        yi = np.array([self.classes_.index(c) for c in y])
        self.mean_ = X.mean(0)
        self.std_ = X.std(0) + 1e-6
        Xs = torch.from_numpy((X - self.mean_) / self.std_)
        Y = torch.from_numpy(yi).long()
        counts = np.bincount(yi, minlength=len(self.classes_)).astype(np.float32)
        weights = torch.from_numpy(len(yi) / (len(self.classes_) * counts))  # clases balanceadas
        torch.manual_seed(0)
        W = torch.zeros(X.shape[1], len(self.classes_), requires_grad=True)
        b = torch.zeros(len(self.classes_), requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=self.max_iter, line_search_fn="strong_wolfe")
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        def closure():
            opt.zero_grad()
            loss = loss_fn(Xs @ W + b, Y) + self.l2 * (W ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.W_ = W.detach().numpy()
        self.b_ = b.detach().numpy()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float32) - self.mean_) / self.std_
        z = Xs @ self.W_ + self.b_
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, X: np.ndarray) -> List[Tuple[str, float]]:
        p = self.predict_proba(X)
        return [(self.classes_[i], float(p[k, i])) for k, i in enumerate(p.argmax(1))]

    @staticmethod
    def cross_validate(X: np.ndarray, y: Sequence[str], l2: float = 1e-2, k: int = 5) -> Dict:
        """Validación cruzada estratificada; dejando uno fuera si alguna clase tiene < k muestras."""
        y = np.asarray(y)
        classes = sorted(set(y))
        min_count = min((y == c).sum() for c in classes)
        n = len(y)
        rng = np.random.default_rng(0)
        if min_count >= k:
            folds = np.zeros(n, dtype=int)
            for c in classes:
                idx = np.where(y == c)[0]
                rng.shuffle(idx)
                folds[idx] = np.arange(len(idx)) % k
            method = f"validación cruzada estratificada ({k} pliegues)"
        else:
            folds = np.arange(n)
            method = "dejando uno fuera (pocas muestras por clase)"
        pred = np.empty(n, dtype=object)
        for f in np.unique(folds):
            te = folds == f
            tr = ~te
            if len(set(y[tr])) < 2:
                pred[te] = y[tr][0]
                continue
            clf = SupervisedClassifier(l2=l2).fit(X[tr], y[tr])
            pred[te] = [c for c, _ in clf.predict(X[te])]
        cm = pd.crosstab(pd.Series(y, name="real"), pd.Series(pred, name="predicha"), dropna=False)
        cm = cm.reindex(index=classes, columns=classes, fill_value=0)
        per_class = []
        for c in classes:
            tp = cm.loc[c, c]
            rec = tp / cm.loc[c].sum() if cm.loc[c].sum() else 0.0
            prec = tp / cm[c].sum() if cm[c].sum() else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class.append({"categoria": c, "muestras": int(cm.loc[c].sum()), "precision": round(prec, 3),
                              "exhaustividad": round(rec, 3), "f1": round(f1, 3)})
        acc = float((pred == y).mean())
        # Kappa de Cohen (acuerdo corregido por azar), habitual en evaluación de clasificaciones
        po = acc
        pe = sum(cm.loc[c].sum() * cm[c].sum() for c in classes) / (n * n)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        return {"metodo": method, "n": n, "exactitud": acc, "kappa": float(kappa),
                "matriz_confusion": cm, "por_clase": pd.DataFrame(per_class)}


def training_eligible(label: str, status: str, label_source: str) -> bool:
    """Muestra válida: objeto aceptado, con categoría real y confirmada por el usuario."""
    return status == "aceptado" and label not in UNLABELED and "sin confirmar" not in str(label_source)


# --------------------------------------------------------------------------- coberturas
def mosaic_pixel_area_m2(transform_3857, width: int, height: int) -> Tuple[float, float, str]:
    """(área media del píxel en m², área total del mosaico en m², CRS métrico usado)."""
    w, n = transform_3857 * (0, 0)
    e, s = transform_3857 * (width, height)
    g = gpd.GeoSeries([box(w, s, e, n)], crs="EPSG:3857").to_crs("EPSG:4326")
    c = g.iloc[0].centroid
    crs = select_metric_crs(c.x, c.y)
    total = float(g.to_crs(crs).area.iloc[0])
    return total / (width * height), total, crs


def coverage_raster(items: Iterable[Tuple[np.ndarray, str]], shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict[int, str]]:
    """Ráster uint8 de coberturas: 0 = sin clasificar; cada categoría con su código.

    `items` en orden de prioridad creciente (lo último pinta encima)."""
    out = np.zeros(shape, dtype=np.uint8)
    codes: Dict[str, int] = {}
    for mask, cat in items:
        if cat in UNLABELED:
            continue
        if cat not in codes:
            codes[cat] = len(codes) + 1
        out[np.asarray(mask, dtype=bool)] = codes[cat]
    return out, {v: k for k, v in codes.items()}


def coverage_stats(raster: np.ndarray, legend: Dict[int, str], pixel_area_m2: float) -> pd.DataFrame:
    vals, counts = np.unique(raster, return_counts=True)
    total = raster.size
    rows = []
    for v, c in zip(vals, counts):
        name = legend.get(int(v), "Sin clasificar")
        rows.append({"categoria": name, "pixeles": int(c), "area_m2": round(c * pixel_area_m2, 2),
                     "porcentaje": round(100 * c / total, 2)})
    return pd.DataFrame(rows).sort_values("area_m2", ascending=False).reset_index(drop=True)


def coverage_rgb(raster: np.ndarray, legend: Dict[int, str]) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[0] = (245, 245, 245)
    for code, name in legend.items():
        lut[code] = category_color(name)
    return lut[raster]


# --------------------------------------------------------------------------- coordenadas
def vertex_table(geom_4326, metric_crs: Optional[str] = None, tolerance_m: float = 0.5) -> pd.DataFrame:
    """Cuadro de coordenadas: vértices en `metric_crs`, horario desde el vértice más al norte.

    Cada fila: vértice, Este (X), Norte (Y), lindero hacia el siguiente vértice, distancia (m) y
    acimut (grados desde el norte de la cuadrícula, sentido horario)."""
    if metric_crs is None:
        c = geom_4326.representative_point()
        metric_crs = select_metric_crs(c.x, c.y)
    g = gpd.GeoSeries([geom_4326], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
    if tolerance_m > 0:
        g = g.simplify(tolerance_m, preserve_topology=True)
    parts = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
    rows = []
    for pi, part in enumerate(sorted(parts, key=lambda p: -p.area), start=1):
        part = orient(part, sign=-1.0)  # exterior horario
        rings = [("", part.exterior)] + [(f"H{hi}-", r) for hi, r in enumerate(part.interiors, start=1)]
        for prefix, ring in rings:
            coords = list(ring.coords)[:-1]
            if not coords:
                continue
            start = max(range(len(coords)), key=lambda i: (coords[i][1], -coords[i][0]))
            coords = coords[start:] + coords[:start]
            tag = (f"P{pi}-" if len(parts) > 1 else "") + prefix
            nv = len(coords)
            for i, (x, y) in enumerate(coords):
                x2, y2 = coords[(i + 1) % nv]
                dist = math.hypot(x2 - x, y2 - y)
                az = round(math.degrees(math.atan2(x2 - x, y2 - y)) % 360, 4) % 360
                rows.append({"vertice": f"{tag}V{i + 1}", "este_x": round(x, 3), "norte_y": round(y, 3),
                             "lindero": f"{tag}V{i + 1}–{tag}V{(i + 1) % nv + 1}",
                             "distancia_m": round(dist, 3), "acimut_deg": az})
    df = pd.DataFrame(rows, columns=["vertice", "este_x", "norte_y", "lindero", "distancia_m", "acimut_deg"])
    df.attrs.update({"crs": metric_crs, "area_m2": float(g.area), "perimetro_m": float(g.length),
                     "tolerancia_m": tolerance_m})
    return df


def azimuth_to_dms(az: float) -> str:
    d = int(az)
    m_float = (az - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    return f"{d}°{m:02d}'{s:05.2f}\""


# --------------------------------------------------------------------------- indicadores
def zone_indicators(gdf: gpd.GeoDataFrame, zone_area_m2: float) -> Dict[str, float]:
    """Indicadores sobre la zona de estudio (`gdf` en EPSG:4326 con `label` y `area_m2`)."""
    if gdf.empty:
        return {"n_objetos": 0}
    built = gdf[gdf["label"].map(is_built)]
    green = gdf[gdf["label"].isin(GREEN_CATEGORIES)]
    trees = gdf[gdf["label"] == "Arbolado"]
    ha = zone_area_m2 / 10_000
    a_built = float(built["area_m2"].sum())
    return {
        "n_objetos": int(len(gdf)),
        "n_edificaciones": int(len(built)),
        "area_construida_m2": a_built,
        "indice_ocupacion": a_built / zone_area_m2 if zone_area_m2 else 0.0,
        "densidad_edificaciones_ha": len(built) / ha if ha else 0.0,
        "area_media_edificacion_m2": float(built["area_m2"].mean()) if len(built) else 0.0,
        "area_verde_m2": float(green["area_m2"].sum()),
        "cobertura_verde_pct": 100 * float(green["area_m2"].sum()) / zone_area_m2 if zone_area_m2 else 0.0,
        "n_grupos_arbolado": int(len(trees)),
        "area_zona_m2": zone_area_m2,
    }


def units_indicators(objects: gpd.GeoDataFrame, units: gpd.GeoDataFrame, id_field: str) -> gpd.GeoDataFrame:
    """Indicadores por unidad territorial (predio, manzana, lote) cargada por el usuario.

    Áreas por intersección en un CRS métrico común; un objeto que cruza dos unidades reparte su
    área. Devuelve las unidades (EPSG:4326) con: area_unidad_m2, area_construida_m2,
    indice_ocupacion, n_edificaciones, area_verde_m2, pct_verde."""
    if units.crs is None:
        units = units.set_crs("EPSG:4326")
    u4326 = units.to_crs("EPSG:4326")
    c = u4326.union_all().representative_point()
    crs = select_metric_crs(c.x, c.y)
    um = u4326.to_crs(crs)[[id_field, "geometry"]].copy()
    um["area_unidad_m2"] = um.area
    out = um.copy()
    for col in ("area_construida_m2", "n_edificaciones", "area_verde_m2"):
        out[col] = 0.0
    if not objects.empty:
        om = objects.to_crs(crs)[["object_id", "label", "geometry"]]
        inter = gpd.overlay(um[[id_field, "geometry"]], om, how="intersection", keep_geom_type=True)
        inter["a"] = inter.area
        b = inter[inter["label"].map(is_built)]
        g = inter[inter["label"].isin(GREEN_CATEGORIES)]
        built_area = b.groupby(id_field)["a"].sum()
        built_n = b.groupby(id_field)["object_id"].nunique()
        green_area = g.groupby(id_field)["a"].sum()
        out["area_construida_m2"] = out[id_field].map(built_area).fillna(0.0)
        out["n_edificaciones"] = out[id_field].map(built_n).fillna(0).astype(int)
        out["area_verde_m2"] = out[id_field].map(green_area).fillna(0.0)
    out["indice_ocupacion"] = (out["area_construida_m2"] / out["area_unidad_m2"]).round(4)
    out["pct_verde"] = (100 * out["area_verde_m2"] / out["area_unidad_m2"]).round(2)
    for col in ("area_unidad_m2", "area_construida_m2", "area_verde_m2"):
        out[col] = out[col].round(2)
    out["crs_metrico"] = crs
    return out.to_crs("EPSG:4326")


# --------------------------------------------------------------------------- informe técnico
def technical_report_html(meta: Dict, indicators: Dict, coverage: Optional[pd.DataFrame],
                          objects: pd.DataFrame, validation: Optional[Dict] = None) -> str:
    """Informe técnico autocontenido (HTML) de la zona de estudio, listo para imprimir a PDF."""
    import html as _h

    def table(df: pd.DataFrame) -> str:
        return df.to_html(index=False, border=0, classes="t", float_format=lambda v: f"{v:,.2f}")

    ind_rows = [
        ("Área de la zona analizada", f"{indicators.get('area_zona_m2', 0):,.1f} m²"),
        ("Objetos considerados", f"{indicators.get('n_objetos', 0)}"),
        ("Edificaciones (huellas)", f"{indicators.get('n_edificaciones', 0)}"),
        ("Área construida en planta", f"{indicators.get('area_construida_m2', 0):,.1f} m²"),
        ("Índice de ocupación (IO)", f"{indicators.get('indice_ocupacion', 0):.3f}"),
        ("Densidad de edificaciones", f"{indicators.get('densidad_edificaciones_ha', 0):.2f} por ha"),
        ("Área media por edificación", f"{indicators.get('area_media_edificacion_m2', 0):,.1f} m²"),
        ("Cobertura verde (zona verde + arbolado)", f"{indicators.get('cobertura_verde_pct', 0):.2f} %"),
    ]
    val_html = ""
    if validation:
        val_html = (f"<h2>Validación de la clasificación supervisada</h2><p>{_h.escape(validation['metodo'])}: "
                    f"exactitud global <b>{validation['exactitud']:.1%}</b>, índice kappa <b>{validation['kappa']:.3f}</b> "
                    f"(n = {validation['n']}).</p>" + table(validation["por_clase"]) +
                    "<h3>Matriz de confusión</h3>" + validation["matriz_confusion"].to_html(border=0, classes="t"))
    cov_html = ("<h2>Coberturas</h2>" + table(coverage)) if coverage is not None and len(coverage) else ""
    meta_rows = "".join(f"<tr><th>{_h.escape(str(k))}</th><td>{_h.escape(str(v))}</td></tr>" for k, v in meta.items())
    ind_html = "".join(f"<tr><th>{_h.escape(k)}</th><td>{_h.escape(v)}</td></tr>" for k, v in ind_rows)
    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Informe técnico — SAM 2 GeoExtractor</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:960px;margin:32px auto;color:#1f2937;line-height:1.45}}
h1{{font-size:24px;border-bottom:3px solid #0e7490;padding-bottom:6px}}h2{{font-size:18px;color:#0e7490;margin-top:28px}}
table.t,table.m{{border-collapse:collapse;width:100%;font-size:13px}}table.t th,table.t td,table.m th,table.m td{{border-bottom:1px solid #e5e7eb;padding:5px 8px;text-align:left}}
table.m th{{width:38%;color:#475569;font-weight:600}}.note{{font-size:12px;color:#64748b;margin-top:28px}}</style></head><body>
<h1>Informe técnico de la zona de estudio</h1>
<h2>Metadatos</h2><table class="m">{meta_rows}</table>
<h2>Indicadores urbanos</h2><table class="m">{ind_html}</table>
{cov_html}
{val_html}
<h2>Inventario de objetos</h2>{table(objects)}
<p class="note">Segmentación con SAM 2 (agnóstica de clase), validada por el usuario. Categorías asignadas por el usuario o
por clasificación supervisada/sugerida según la columna de fuente. Áreas planas en el CRS métrico indicado (EPSG:9377 en Colombia).
Insumo preliminar derivado de ortoimagen: no sustituye levantamientos topográficos ni información catastral oficial; sin datos de altura
no se calculan pisos ni índice de construcción.</p></body></html>"""
