"""
Motor de IA basado en Segment Anything Model 2 (SAM 2) para la aplicación Streamlit.

Diseño (v2):
- Los PESOS del modelo se comparten entre sesiones (`st.cache_resource`, máx. 1 modelo en RAM).
- La IMAGEN activa NO se comparte: `embed()` devuelve un `ImageEmbedding` que cada sesión guarda
  en `st.session_state` y entrega en cada `predict()`. El backend restaura ese estado bajo un lock,
  por lo que dos usuarios concurrentes no se sobrescriben los embeddings.
- La inferencia se delega en `geosam.backends.sam2.SAM2Backend` (o `MockBackend` si torch/sam2 no
  están instalados). El modo simulado queda señalado en `is_mock` y `mock_reason`.
"""
from __future__ import annotations

import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

from config import SAM2_MODELS, DEFAULT_SAM2_MODEL_KEY
from geosam.backends.protocol import ImageEmbedding, MaskCandidate

try:
    import torch
    TORCH_AVAILABLE = True
    _TORCH_ERROR = None
except ImportError as _e:  # pragma: no cover - depende del entorno
    torch = None
    TORCH_AVAILABLE = False
    _TORCH_ERROR = str(_e)

try:
    import sam2  # noqa: F401
    SAM2_AVAILABLE = True
    _SAM2_ERROR = None
except ImportError as _e:  # pragma: no cover - depende del entorno
    SAM2_AVAILABLE = False
    _SAM2_ERROR = str(_e)


def ensure_checkpoint(model_info: Dict[str, Any]) -> str:
    """Garantiza que el checkpoint exista localmente; lo descarga de forma atómica si falta.

    La descarga se escribe en `<ruta>.part` y solo se renombra si el tamaño coincide con el
    `Content-Length` del servidor, para que un corte de red no deje un `.pt` truncado.
    """
    path = model_info["checkpoint"]
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".part"
    print(f"Descargando pesos SAM 2 desde {model_info['url']} ...")
    with urllib.request.urlopen(model_info["url"], timeout=60) as resp, open(tmp_path, "wb") as fh:
        expected = int(resp.headers.get("Content-Length") or 0)
        written = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            written += len(chunk)
    if expected and written != expected:
        os.remove(tmp_path)
        raise IOError(f"Descarga incompleta del checkpoint ({written} de {expected} bytes).")
    os.replace(tmp_path, path)
    return path


class SAM2Engine:
    """Envoltorio de alto nivel del backend de segmentación usado por la interfaz."""

    def __init__(self, model_key: str = DEFAULT_SAM2_MODEL_KEY, allow_mock: Optional[bool] = None):
        if model_key not in SAM2_MODELS:
            model_key = DEFAULT_SAM2_MODEL_KEY
        if allow_mock is None:
            allow_mock = os.environ.get("GEOSAM_DISABLE_MOCK", "0") != "1"

        self.model_key = model_key
        self.model_info = SAM2_MODELS[model_key]
        self._last_emb: Optional[ImageEmbedding] = None  # solo para la API heredada
        self.current_image_id: Optional[str] = None

        if not (TORCH_AVAILABLE and SAM2_AVAILABLE):
            reason = _TORCH_ERROR or _SAM2_ERROR or "dependencias no disponibles"
            if not allow_mock:
                raise ImportError(f"SAM 2 no está disponible: {reason}")
            from geosam.backends.mock import MockBackend
            self.is_mock = True
            self.mock_reason = reason
            self.device = "cpu"
            self.device_name = "CPU (MODO SIMULADO — sin SAM 2)"
            self.backend = MockBackend(model_id=model_key)
            self.backend.load("cpu")
            return

        from geosam.backends.sam2 import SAM2Backend
        self.is_mock = False
        self.mock_reason = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"

        checkpoint_path = ensure_checkpoint(self.model_info)
        print(f"Cargando SAM 2 ({model_key}) en {self.device}...")
        self.backend = SAM2Backend(
            checkpoint_path=checkpoint_path,
            config_name=self.model_info["config"],
            model_id=model_key,
            max_cache_entries=4,
        )
        self.backend.load(self.device)
        print(f"SAM 2 inicializado en {self.device_name}.")

    @classmethod
    def from_backend(cls, backend, model_key: str, device_name: str = "CPU") -> "SAM2Engine":
        """Construye un motor sobre un backend ya cargado (inyección para pruebas o CLI)."""
        eng = cls.__new__(cls)
        eng.model_key = model_key
        eng.model_info = SAM2_MODELS.get(model_key, {})
        eng._last_emb = None
        eng.current_image_id = None
        eng.backend = backend
        eng.is_mock = type(backend).__name__ == "MockBackend"
        eng.mock_reason = "backend simulado inyectado" if eng.is_mock else None
        eng.device = getattr(backend, "device", "cpu")
        eng.device_name = device_name
        return eng

    # ------------------------------------------------------------------ API por sesión
    def embed(self, image_array: np.ndarray) -> tuple[ImageEmbedding, float]:
        """Calcula (o recupera de caché) el embedding de la imagen. Devuelve (embedding, segundos)."""
        if image_array.ndim != 3 or image_array.shape[2] != 3 or image_array.dtype != np.uint8:
            raise ValueError("Se esperaba una imagen RGB uint8 de forma (H, W, 3).")
        start = time.time()
        emb = self.backend.embed(image_array)
        return emb, time.time() - start

    def is_compatible(self, emb: Optional[ImageEmbedding]) -> bool:
        """True si el embedding fue generado por este mismo modelo."""
        return emb is not None and emb.model_id == self.model_key

    def predict(
        self,
        emb: ImageEmbedding,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_logits: Optional[np.ndarray] = None,
        multimask: bool = True,
    ) -> Dict[str, Any]:
        """Ejecuta el decodificador de máscaras con cualquier combinación de prompts.

        `points` en píxeles (x=col, y=row). Devuelve las candidatas ordenadas tal como las produce
        SAM 2 y el índice de la de mayor IoU predicho.
        """
        if not self.is_compatible(emb):
            raise ValueError("El embedding no corresponde al modelo cargado; recalcúlelo.")
        start = time.time()
        candidates: List[MaskCandidate] = self.backend.predict(
            emb, pts_xy=points, labels=labels, box_xyxy=box, mask_logits=mask_logits, multimask=multimask
        )
        scores = [float(c.score) for c in candidates]
        return {
            "candidates": candidates,
            "scores": scores,
            "best_idx": int(np.argmax(scores)) if scores else 0,
            "inference_time": time.time() - start,
            "device": self.device_name,
            "multimask": multimask,
        }

    def segment_everything(self, image_array: np.ndarray, **kwargs: Any) -> tuple[List[MaskCandidate], float]:
        """Generación automática de máscaras (SAM2AutomaticMaskGenerator). Costosa en CPU."""
        start = time.time()
        res = self.backend.segment_everything(image_array, **kwargs)
        return res, time.time() - start

    # ------------------------------------------------------------------ API heredada (v1)
    def set_image(self, image_array: np.ndarray, image_id: Optional[str] = None) -> float:
        """Compatibilidad v1: guarda el embedding en la instancia (no usar en multiusuario)."""
        self._last_emb, elapsed = self.embed(image_array)
        self.current_image_id = image_id
        return elapsed

    def predict_point(self, col: int, row: int, multimask_output: bool = True) -> Dict[str, Any]:
        """Compatibilidad v1: un punto positivo sobre la última imagen pasada a `set_image`."""
        if self._last_emb is None:
            raise RuntimeError("Debe llamar a set_image() antes de predict_point().")
        res = self.predict(self._last_emb, np.array([[col, row]]), np.array([1]), multimask=multimask_output)
        cands = res["candidates"]
        best = res["best_idx"]
        return {
            "best_mask": cands[best].mask,
            "all_masks": np.stack([c.mask for c in cands]),
            "scores": res["scores"],
            "best_score": res["scores"][best],
            "best_idx": best,
            "inference_time": res["inference_time"],
            "device": self.device_name,
            "clicked_point": (col, row),
        }


@st.cache_resource(show_spinner="Inicializando modelo SAM 2 en memoria...", max_entries=1)
def get_sam2_engine(model_key: str = DEFAULT_SAM2_MODEL_KEY) -> SAM2Engine:
    """Carga y comparte los pesos del modelo. `max_entries=1` libera el modelo anterior al cambiar."""
    return SAM2Engine(model_key=model_key)
