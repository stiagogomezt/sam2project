"""
Clasificación automática de coberturas para las máscaras de SAM 2.

SAM 2 delimita objetos pero no les pone nombre. Este módulo asigna a cada máscara una clase GLOBAL de
cobertura, alineada con el nivel 1–2 de la leyenda CORINE Land Cover adaptada para Colombia (IDEAM):

* Con Sentinel-2 (hay NIR y SWIR): reglas espectrales sobre la media de cada objeto (NDVI, MNDWI, NDMI,
  rojo). Son transparentes y reproducibles; la clase se decide por objeto, no por píxel.
* Con ortoimagen RGB: RemoteCLIP (zero-shot) con descripciones de clases globales, más un índice de
  verdor RGB (ExG) que impide llamar «edificación» a un objeto claramente vegetal y viceversa.

La salida es una SUGERENCIA con su confianza y su método; el usuario puede corregirla en el catálogo.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Clase → (color RGB, código CLC de referencia)
GLOBAL_CLASSES: Dict[str, Tuple[Tuple[int, int, int], str]] = {
    "Edificaciones": ((228, 60, 40), "1.1"),
    "Vías y zonas duras": ((140, 140, 150), "1.2"),
    "Suelo desnudo o construido": ((205, 133, 63), "1 / 3.3"),
    "Cultivos e invernaderos": ((186, 104, 200), "2.1–2.4"),
    "Pastos y herbazales": ((240, 190, 80), "2.3 / 3.2"),
    "Bosque y arbolado": ((46, 125, 50), "3.1"),
    "Suelo desnudo": ((215, 170, 120), "3.3"),
    "Agua": ((30, 136, 229), "5.1"),
}

# Descripciones para RemoteCLIP (ortoimagen RGB). Varias por clase; la probabilidad de la clase suma
# las de sus descripciones.
RGB_PROMPTS: Dict[str, List[str]] = {
    "Edificaciones": ["rooftops of houses", "buildings", "residential buildings", "industrial building roof"],
    "Vías y zonas duras": ["road", "paved street", "parking lot", "concrete plaza"],
    "Cultivos e invernaderos": ["greenhouses", "plastic greenhouse roofs", "crop field rows"],
    "Pastos y herbazales": ["grass", "pasture", "meadow", "lawn", "sports field with grass"],
    "Bosque y arbolado": ["trees", "forest", "tree canopy"],
    "Suelo desnudo": ["bare soil", "dirt ground", "sand"],
    "Agua": ["water", "river", "lake", "swimming pool"],
}

VEGETATION = {"Pastos y herbazales", "Bosque y arbolado", "Cultivos e invernaderos"}


def class_color(name: str) -> Tuple[int, int, int]:
    return GLOBAL_CLASSES.get(name, ((200, 200, 200), ""))[0]


# --------------------------------------------------------------------------- Sentinel-2
def spectral_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    eps = 1e-6
    b = {k: np.asarray(v, dtype="float32") for k, v in bands.items() if k != "scl"}
    return {
        "ndvi": (b["nir"] - b["red"]) / (b["nir"] + b["red"] + eps),
        "mndwi": (b["green"] - b["swir16"]) / (b["green"] + b["swir16"] + eps),
        "ndmi": (b["nir"] - b["swir16"]) / (b["nir"] + b["swir16"] + eps),
        "red": b["red"],
    }


def spectral_rule(ndvi: float, mndwi: float, ndmi: float, red: float) -> Tuple[str, float]:
    """Clase global a partir de los índices MEDIOS de un objeto (reflectancias 0–1).

    Umbrales de la literatura de teledetección; el de bosque (NDVI ≥ 0,6, rojo < 0,045, NDMI > 0,15) se ajustó
    contra ESA WorldCover 2021 en Tinigua y Barrancabermeja (bosque frente a pastos). La confianza
    crece con la distancia al umbral que decidió la clase."""
    if not np.isfinite(ndvi):
        return "Sin datos", 0.0
    if mndwi > 0.05 and ndvi < 0.25:
        return "Agua", float(min(1.0, 0.6 + (mndwi - 0.05) * 2))
    if ndvi >= 0.6 and red < 0.045 and ndmi > 0.15:
        return "Bosque y arbolado", float(min(1.0, 0.6 + (ndvi - 0.6) * 2 + (0.045 - red) * 5))
    if ndvi >= 0.3:
        return "Pastos y herbazales", float(min(1.0, 0.55 + abs(ndvi - 0.5)))
    if ndvi >= 0.15:
        return "Suelo desnudo o construido", 0.5
    return "Suelo desnudo o construido", float(min(1.0, 0.6 + (0.15 - ndvi) * 2))


def classify_spectral(masks: Sequence[np.ndarray], bands: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
    idx = spectral_indices(bands)
    out = []
    for m in masks:
        m = np.asarray(m, dtype=bool)
        if not m.any():
            out.append(("Sin datos", 0.0))
            continue
        vals = {k: float(np.nanmean(v[m])) for k, v in idx.items()}
        out.append(spectral_rule(vals["ndvi"], vals["mndwi"], vals["ndmi"], vals["red"]))
    return out


FONDO_FRAC = 0.15  # una «máscara» que cubre más del 15 % de la escena es el fondo, no un objeto


def is_background(mask: np.ndarray, frac: float = FONDO_FRAC) -> bool:
    m = np.asarray(mask, dtype=bool)
    return bool(m.mean() > frac)


def pixel_classes(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Las mismas reglas de `spectral_rule`, aplicadas píxel a píxel (códigos de `paint`; 0 = sin dato)."""
    idx = spectral_indices(bands)
    ndvi, mndwi, ndmi, red = idx["ndvi"], idx["mndwi"], idx["ndmi"], idx["red"]
    code = {n: i + 1 for i, n in enumerate(GLOBAL_CLASSES)}
    out = np.full(ndvi.shape, code["Suelo desnudo o construido"], dtype=np.uint8)
    out[ndvi >= 0.3] = code["Pastos y herbazales"]
    out[(ndvi >= 0.6) & (red < 0.045) & (ndmi > 0.15)] = code["Bosque y arbolado"]
    out[(mndwi > 0.05) & (ndvi < 0.25)] = code["Agua"]
    out[~np.isfinite(ndvi)] = 0
    return out


def majority(codes: np.ndarray, size: int = 5) -> np.ndarray:
    """Filtro de mayoría (moda en una ventana `size`×`size`) para quitar el ruido de sal y pimienta."""
    from scipy import ndimage as ndi

    classes = [c for c in np.unique(codes) if c > 0]
    if not classes:
        return codes
    votes = np.stack([ndi.uniform_filter((codes == c).astype("float32"), size=size) for c in classes])
    out = np.asarray(classes, dtype=np.uint8)[votes.argmax(axis=0)]
    out[codes == 0] = 0
    return out


def spectral_map(masks: Sequence[np.ndarray], bands: Dict[str, np.ndarray], max_frac: float = FONDO_FRAC):
    """Mapa de coberturas para Sentinel-2.

    Cada objeto de SAM 2 recibe la clase de sus índices MEDIOS. Lo que no es objeto (el fondo, es decir las
    máscaras de más de `max_frac` de la escena, y los huecos sin segmentar) se clasifica píxel a píxel con las
    mismas reglas y un filtro de mayoría. Devuelve (ráster, leyenda, etiquetas por máscara, info)."""
    shape = next(iter(v for k, v in bands.items() if k != "scl")).shape
    base = majority(pixel_classes(bands))
    labels: List[Tuple[str, float]] = []
    obj_masks, obj_labels = [], []
    for m, lab in zip(masks, classify_spectral(masks, bands)):
        if is_background(m, max_frac):
            labels.append(("Fondo", 0.0))
        else:
            labels.append(lab)
            obj_masks.append(m)
            obj_labels.append(lab[0])
    raster, legend = paint(obj_masks, obj_labels, shape)
    by_obj = raster > 0
    raster[~by_obj] = base[~by_obj]
    info = {"frac_objetos": float(by_obj.mean()), "frac_pixel": float((~by_obj & (base > 0)).mean())}
    return raster, legend, labels, info


# --------------------------------------------------------------------------- Ortoimagen RGB
def excess_green(img: np.ndarray, mask: np.ndarray) -> float:
    """ExG = 2g − r − b con cromaticidades normalizadas, promedio del objeto (≈0 gris, >0,05 verde)."""
    px = img[np.asarray(mask, dtype=bool)].astype("float32")
    if px.size == 0:
        return 0.0
    s = px.sum(axis=1) + 1e-6
    r, g, b = px[:, 0] / s, px[:, 1] / s, px[:, 2] / s
    return float(np.mean(2 * g - r - b))


def rgb_features(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Rasgos simples del objeto: verdor, brillo, azul relativo y alargamiento."""
    m = np.asarray(mask, dtype=bool)
    px = img[m].astype("float32")
    if px.size == 0:
        return {"exg": 0.0, "bright": 0.0, "blue": 0.0, "elong": 1.0, "area_px": 0}
    s = px.sum(axis=1) + 1e-6
    ys, xs = np.nonzero(m)
    # alargamiento por momentos de segundo orden (razón de ejes de la elipse equivalente)
    cov = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))
    elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    return {
        "exg": float(np.mean(2 * px[:, 1] / s - px[:, 0] / s - px[:, 2] / s)),
        "bright": float(px.mean()),
        "blue": float(np.mean(px[:, 2] / s - (px[:, 0] + px[:, 1]) / (2 * s))),
        "elong": elong,
        "area_px": int(m.sum()),
    }


def texture_map(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Rugosidad local: |gris − media en ventana `size`|. Las copas de árboles son rugosas; un cultivo o un
    potrero, lisos."""
    from scipy import ndimage as ndi

    g = np.asarray(img, dtype="float32").mean(axis=2)
    return np.abs(g - ndi.uniform_filter(g, size))


def combine_rgb(probs: Dict[str, float], exg: float, feats: Optional[Dict[str, float]] = None,
                px_m2: Optional[float] = None) -> Tuple[str, float]:
    """Corrige la sugerencia de RemoteCLIP con reglas físicas simples del objeto.

    * Verde (ExG > 0,13): solo coberturas vegetales. Sin verdor (ExG < 0,07): ninguna vegetal.
      (Umbrales medidos en Usme: techos ExG mediana 0,06 y p90 0,10; vegetación mediana 0,22.)
    * Agua: solo si el objeto es oscuro (los techos azules son claros).
    * Muy alargado y sin verdor: vía.
    * Invernaderos: solo bloques grandes (> 1500 m²); un techo de casa no lo es.
    * Bosque: solo si la textura es rugosa (> 4 niveles de gris); los cultivos oscuros y lisos no lo son
      (medido contra ESA WorldCover en Usme y Facatativá)."""
    f = feats or {"exg": exg}
    p = dict(probs)
    damp = 0.05
    if f["exg"] > 0.13:
        for k in ("Edificaciones", "Vías y zonas duras", "Suelo desnudo", "Agua"):
            p[k] = p.get(k, 0.0) * damp
    elif f["exg"] < 0.07:
        for k in ("Pastos y herbazales", "Bosque y arbolado"):
            p[k] = p.get(k, 0.0) * damp
    if "bright" in f and f["bright"] > 95:
        p["Agua"] = p.get("Agua", 0.0) * damp
    if "elong" in f and f["elong"] > 5 and f["exg"] < 0.07:
        p["Vías y zonas duras"] = p.get("Vías y zonas duras", 0.0) + 0.5
    if "tex" in f and f["tex"] < 4.0:
        p["Bosque y arbolado"] = p.get("Bosque y arbolado", 0.0) * damp
    if px_m2 and "area_px" in f and f["area_px"] * px_m2 < 1500:
        p["Cultivos e invernaderos"] = p.get("Cultivos e invernaderos", 0.0) * damp
    tot = sum(p.values()) or 1.0
    name, val = max(p.items(), key=lambda kv: kv[1])
    return name, float(val / tot)


def classify_rgb(img: np.ndarray, masks: Sequence[np.ndarray], bboxes: Sequence[Sequence[int]], clip,
                 px_m2: Optional[float] = None) -> List[Tuple[str, float]]:
    """`clip` es un `semantic.SemanticClassifier` con el vocabulario «global» (`RGB_PROMPTS`), o creado
    directamente con `categories=RGB_PROMPTS`. `px_m2` = área real de un píxel."""
    from src.semantic import object_crop

    vocab = "global" if "global" in getattr(clip, "_vocab", {}) else "default"
    tex = texture_map(img)
    out = []
    for m, bb in zip(masks, bboxes):
        crop = object_crop(img, bb, pad_ratio=0.05)
        ranked = clip.classify(crop, top_k=len(RGB_PROMPTS), vocab=vocab)
        feats = rgb_features(img, m)
        feats["tex"] = float(tex[np.asarray(m, dtype=bool)].mean()) if np.asarray(m).any() else 0.0
        out.append(combine_rgb(dict(ranked), feats["exg"], feats, px_m2))
    return out


# --------------------------------------------------------------------------- mapa
def paint(masks: Sequence[np.ndarray], labels: Sequence[str], shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict[int, str]]:
    """Ráster de coberturas: los objetos grandes primero y los pequeños encima (no los tapan)."""
    names = [n for n in GLOBAL_CLASSES]
    code = {n: i + 1 for i, n in enumerate(names)}
    out = np.zeros(shape, dtype=np.uint8)
    order = np.argsort([-np.asarray(m).sum() for m in masks])
    for i in order:
        if labels[i] in code:
            out[np.asarray(masks[i], dtype=bool)] = code[labels[i]]
    return out, {v: k for k, v in code.items()}


def to_rgb(raster: np.ndarray, legend: Dict[int, str]) -> np.ndarray:
    lut = np.full((256, 3), 245, dtype=np.uint8)
    for c, n in legend.items():
        lut[c] = class_color(n)
    return lut[raster]


def area_table(raster: np.ndarray, legend: Dict[int, str], pixel_area_m2: float):
    import pandas as pd

    vals, counts = np.unique(raster, return_counts=True)
    rows = []
    for v, c in zip(vals, counts):
        name = legend.get(int(v), "Sin segmentar")
        rows.append({"cobertura": name, "clc": GLOBAL_CLASSES.get(name, (None, "—"))[1],
                     "hectareas": round(c * pixel_area_m2 / 1e4, 2), "porcentaje": round(100 * c / raster.size, 1)})
    return pd.DataFrame(rows).sort_values("hectareas", ascending=False).reset_index(drop=True)
