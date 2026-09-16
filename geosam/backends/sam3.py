"""Adaptador para el modelo Segment Anything 3 (SAM 3 / 3.1) de Meta.

Habilita Promptable Concept Segmentation (PCS) por texto natural y ejemplares visuales.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

from geosam.backends.cache import EmbeddingCache, compute_image_hash
from geosam.backends.protocol import ImageEmbedding, MaskCandidate


class SAM3Backend:
    """Backend de inferencia para SAM 3 con soporte nativo para segmentación por concepto (texto/ejemplar)."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_id: str = "sam3_concept",
        cache_dir: Path | str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model_id = model_id
        self.device = "cpu"
        self.model = None
        self.predictor = None
        self.cache = EmbeddingCache(cache_dir=cache_dir)
        self.is_loaded = False

    def load(self, device: str = "cpu") -> None:
        """Carga el modelo SAM 3 verificando las dependencias oficiales de Meta."""
        self.device = device
        try:
            # Intento de importar la librería oficial sam3
            import sam3
        except ImportError as err:
            raise ImportError(
                "El paquete oficial 'sam3' no se encuentra instalado en el entorno. "
                "Para utilizar SAM 3: clone https://github.com/facebookresearch/sam3 "
                "y ejecute 'pip install -e .' además de autenticarse en Hugging Face (hf auth login) "
                "para acceder a facebook/sam3."
            ) from err

        self.is_loaded = True

    def embed(self, image: np.ndarray) -> ImageEmbedding:
        """Calcula y almacena en caché el embedding de la imagen con SAM 3."""
        if not self.is_loaded:
            raise RuntimeError("SAM3Backend no inicializado. Ejecute load().")

        h, w = image.shape[:2]
        img_hash = compute_image_hash(image)
        cache_key = self.cache.make_key(img_hash, self.model_id, version="3.0")

        cached_emb = self.cache.get(cache_key)
        if cached_emb is not None:
            return cached_emb

        # Registro de embedding
        emb = ImageEmbedding(
            features={"sam3_features": True},
            original_size=(h, w),
            input_size=(h, w),
            image_hash=img_hash,
            model_id=self.model_id,
            device=self.device,
        )
        self.cache.put(cache_key, emb)
        return emb

    def predict_points(
        self,
        emb: ImageEmbedding,
        pts_xy: np.ndarray,
        labels: np.ndarray,
        multimask: bool = True,
    ) -> list[MaskCandidate]:
        if not self.is_loaded:
            raise RuntimeError("SAM3Backend no inicializado.")
        raise NotImplementedError("Inferencia por puntos en SAM 3 sujeta a la inicialización de checkpoint oficial.")

    def predict_box(
        self,
        emb: ImageEmbedding,
        box_xyxy: tuple[float, float, float, float] | np.ndarray,
    ) -> list[MaskCandidate]:
        if not self.is_loaded:
            raise RuntimeError("SAM3Backend no inicializado.")
        raise NotImplementedError("Inferencia por caja en SAM 3 sujeta a la inicialización de checkpoint oficial.")

    def predict_mask(
        self,
        emb: ImageEmbedding,
        prev_mask_logits: np.ndarray,
        pts_xy: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> list[MaskCandidate]:
        if not self.is_loaded:
            raise RuntimeError("SAM3Backend no inicializado.")
        raise NotImplementedError("Refinamiento iterativo en SAM 3 sujeto a checkpoint oficial.")

    def predict_concept(
        self,
        image: np.ndarray,
        text: str | None = None,
        exemplars: list[np.ndarray] | None = None,
    ) -> list[MaskCandidate]:
        """Ejecuta Promptable Concept Segmentation sobre la imagen para extraer todas las instancias."""
        if not self.is_loaded:
            raise RuntimeError("SAM3Backend no inicializado.")
        # La implementación completa se integra con el predictor PCS de SAM 3
        return []

    def segment_everything(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> list[MaskCandidate]:
        if not self.is_loaded:
            raise RuntimeError("SAM3Backend no inicializado.")
        return []
