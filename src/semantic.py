"""
Clasificación semántica COMPLEMENTARIA (zero-shot) con RemoteCLIP.

SAM 2 segmenta sin clase (DEC-0008). Este módulo, separado del motor de segmentación, sugiere una
categoría para un objeto ya segmentado comparando su recorte con descripciones de texto mediante
RemoteCLIP (CLIP ViT-B/32 ajustado a teledetección; Liu et al., 2024, licencia Apache-2.0).

Es una SUGERENCIA: devuelve las categorías más probables y el usuario confirma. Medición en la zona
de prueba (Bogotá, Esri z19): la categoría correcta fue la primera en ≈la mitad de los recortes y
estuvo entre las 3 primeras en 8 de 9; confusiones típicas: techos azules ↔ agua, canchas ↔ césped.

Cada categoría puede tener varias descripciones: su probabilidad es la suma de las de sus textos.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import BASE_DIR

REMOTECLIP_INFO = {
    "checkpoint": str(Path(os.environ.get("REMOTECLIP_DIR", str(BASE_DIR))) / "RemoteCLIP-ViT-B-32.pt"),
    "url": "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt",
    "arch": "ViT-B-32",
    "size_mb": 605,
}

# Categoría (español, catastro / urbano) → descripciones en inglés para el codificador de texto
CATEGORY_PROMPTS: Dict[str, List[str]] = {
    "Edificación religiosa": ["church", "cathedral", "convent with cloister"],
    "Edificación institucional": ["large institutional building with courtyard", "school building", "university campus building"],
    "Edificación residencial": ["residential houses", "dense residential buildings"],
    "Edificación comercial / industrial": ["industrial warehouse with metal roof", "factory", "shopping mall"],
    "Edificio en altura": ["tall building", "skyscraper"],
    "Cancha deportiva": ["soccer field", "basketball court", "tennis court", "sports field"],
    "Zona verde": ["grass meadow", "park with lawn"],
    "Arbolado": ["trees forest", "group of trees"],
    "Vía": ["road", "street with cars"],
    "Vía férrea": ["railway", "train tracks"],
    "Parqueadero": ["parking lot with cars"],
    "Cuerpo de agua": ["river", "lake", "swimming pool"],
}

# Lista que ofrece la interfaz (incluye categorías que el clasificador no sugiere)
CATEGORIES: List[str] = list(CATEGORY_PROMPTS) + ["Cubierta", "Patio / claustro", "Andén / zona dura", "Otro"]


def object_crop(img: np.ndarray, bbox_px: Sequence[int], pad_ratio: float = 0.1, min_side: int = 48) -> np.ndarray:
    """Recorte del objeto con un margen de contexto (CLIP clasifica mejor con algo de entorno)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox_px]
    pad = int(max(x1 - x0, y1 - y0) * pad_ratio)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max((x1 - x0) / 2 + pad, (y1 - y0) / 2 + pad, min_side / 2)
    xa, xb = int(max(0, cx - half)), int(min(w, cx + half))
    ya, yb = int(max(0, cy - half)), int(min(h, cy + half))
    return img[ya:yb, xa:xb]


class SemanticClassifier:
    """RemoteCLIP zero-shot. Carga perezosa; seguro para compartir entre sesiones (solo lectura)."""

    def __init__(self, checkpoint: Optional[str] = None, categories: Optional[Dict[str, List[str]]] = None,
                 vocabularies: Optional[Dict[str, Dict[str, List[str]]]] = None):
        """`categories` es el vocabulario por defecto; `vocabularies` añade otros con nombre (p. ej.
        «global» para coberturas). Todos se codifican al crear el objeto, antes de liberar la torre de texto."""
        import torch
        import open_clip

        self.categories = categories or CATEGORY_PROMPTS
        ckpt = checkpoint or REMOTECLIP_INFO["checkpoint"]
        # Memoria: los pesos del archivo se mapean (mmap) y se ASIGNAN a los parámetros (assign=True),
        # de modo que los tensores aleatorios iniciales se liberan en lugar de copiar encima de ellos.
        # La arquitectura se crea en el dispositivo «meta» (sin reservar memoria para los pesos aleatorios):
        # así el pico de RAM es el de los pesos reales y no el doble, lo que evita que Streamlit Cloud
        # cierre el proceso por falta de memoria.
        import logging
        logging.getLogger("open_clip.factory").setLevel(logging.ERROR)  # el aviso «sin pesos» es esperado
        with torch.device("meta"):
            model = open_clip.create_model(REMOTECLIP_INFO["arch"], device="meta")
        preprocess = open_clip.image_transform(model.visual.image_size, is_train=False,
                                               mean=getattr(model.visual, "image_mean", None),
                                               std=getattr(model.visual, "image_std", None))
        state = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
        missing = model.load_state_dict(state, assign=True)
        if getattr(missing, "missing_keys", None):
            raise RuntimeError(f"Checkpoint RemoteCLIP incompleto: {missing.missing_keys[:5]}")
        del state
        # Los búferes no persistentes (máscara causal del texto) no vienen en el archivo: se rehacen en CPU.
        for mod in model.modules():
            for name, buf in list(mod.named_buffers(recurse=False)):
                if buf.is_meta:
                    if name != "attn_mask":
                        raise RuntimeError(f"Búfer sin inicializar en RemoteCLIP: {name}")
                    n = buf.shape[-1]
                    mod.register_buffer(name, torch.full((n, n), float("-inf")).triu_(1), persistent=False)
        model.eval()
        self._torch = torch
        self.model = model
        self.preprocess = preprocess
        tokenizer = open_clip.get_tokenizer(REMOTECLIP_INFO["arch"])
        self._vocab: Dict[str, Tuple[List[str], "torch.Tensor"]] = {}
        for vname, cats in [("default", self.categories)] + list((vocabularies or {}).items()):
            owners: List[str] = []
            texts: List[str] = []
            for cat, prompts in cats.items():
                for p in prompts:
                    texts.append(f"a satellite photo of {p}.")
                    owners.append(cat)
            with torch.no_grad():
                te = model.encode_text(tokenizer(texts))
            self._vocab[vname] = (owners, te / te.norm(dim=-1, keepdim=True))
        self._owners, self._text = self._vocab["default"]
        # Las descripciones ya están codificadas: la torre de texto (~40 % de los pesos) no se vuelve
        # a usar y se libera.
        for attr in ("transformer", "token_embedding", "ln_final"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        model.text_projection = None
        model.positional_embedding = None
        import gc
        gc.collect()

    def embed(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Embedding visual normalizado (512 valores) del recorte: descriptor para la
        clasificación supervisada."""
        from PIL import Image

        x = self.preprocess(Image.fromarray(np.ascontiguousarray(crop_rgb))).unsqueeze(0)
        with self._torch.no_grad():
            ie = self.model.encode_image(x)
            ie = ie / ie.norm(dim=-1, keepdim=True)
        return ie[0].numpy().astype(np.float32)

    def classify(self, crop_rgb: np.ndarray, top_k: int = 3, vocab: str = "default") -> List[Tuple[str, float]]:
        owners, text = self._vocab[vocab]
        ie = self._torch.from_numpy(self.embed(crop_rgb))[None]
        with self._torch.no_grad():
            p = (100.0 * ie @ text.T).softmax(dim=-1)[0].numpy()
        scores: Dict[str, float] = {}
        for owner, prob in zip(owners, p):
            scores[owner] = scores.get(owner, 0.0) + float(prob)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:top_k]


def classifier_available() -> Tuple[bool, str]:
    """(disponible, motivo). Se puede desactivar con GEOSAM_DISABLE_SEMANTIC=1."""
    if os.environ.get("GEOSAM_DISABLE_SEMANTIC", "0") == "1":
        return False, "desactivado por GEOSAM_DISABLE_SEMANTIC"
    try:
        import open_clip  # noqa: F401
    except ImportError:
        return False, "falta el paquete open_clip_torch"
    return True, ""
