"""Protocolo formal y estructuras de datos para backends de segmentación (SAM 2, SAM 3, Mock).

Desacopla el núcleo analítico de cualquier framework de aprendizaje profundo específico.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any, Protocol, runtime_checkable
import numpy as np


@dataclass(frozen=True)
class MaskCandidate:
    """Representa una máscara predicha por un backend de segmentación con sus metadatos asociados."""
    mask: np.ndarray  # bool [H, W]
    logits: np.ndarray | None  # float32 [H, W] o None
    score: float  # Confianza predicha por el modelo en el rango [0.0, 1.0]
    bbox_px: tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax) en coordenadas de pixel
    backend_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError(f"La máscara debe ser bidimensional (H, W), se recibió {self.mask.ndim}")
        if self.mask.dtype != bool:
            # Convertir a booleano si es necesario
            object.__setattr__(self, "mask", self.mask.astype(bool))
        if len(self.bbox_px) != 4:
            raise ValueError(f"bbox_px debe tener 4 elementos (xmin, ymin, xmax, ymax), se recibió {self.bbox_px}")


@dataclass
class ImageEmbedding:
    """Representa el embedding computado por el encoder del modelo de visión."""
    features: Any  # Tensor o estructura de características interna
    original_size: tuple[int, int]  # (H, W) dimensiones originales de la imagen
    input_size: tuple[int, int]  # Dimensiones tras preprocesamiento/transformación del modelo
    image_hash: str  # Hash SHA256 único de los bytes de la imagen
    model_id: str  # Identificador del modelo (ej. 'sam2_hiera_tiny', 'sam3_concept', 'mock')
    device: str = "cpu"
    created_at: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SegmentationBackend(Protocol):
    """Interfaz abstracta que debe cumplir cualquier backend de segmentación en GEO-SAM."""

    def load(self, device: str = "cpu") -> None:
        """Inicializa o carga los pesos del modelo en el dispositivo indicado ('cpu', 'cuda')."""
        ...

    def embed(self, image: np.ndarray) -> ImageEmbedding:
        """Calcula el embedding de imagen (operación costosa, efectuada una sola vez por imagen).

        Retorna un objeto ImageEmbedding con hash SHA256 para permitir caché en memoria y disco.
        """
        ...

    def predict_points(
        self,
        emb: ImageEmbedding,
        pts_xy: np.ndarray,
        labels: np.ndarray,
        multimask: bool = True
    ) -> list[MaskCandidate]:
        """Genera candidatos a máscara a partir de coordenadas de puntos y etiquetas binarias (1=positivo, 0=negativo)."""
        ...

    def predict_box(
        self,
        emb: ImageEmbedding,
        box_xyxy: tuple[float, float, float, float] | np.ndarray
    ) -> list[MaskCandidate]:
        """Genera candidatos a máscara a partir de una caja delimitadora (xmin, ymin, xmax, ymax)."""
        ...

    def predict_mask(
        self,
        emb: ImageEmbedding,
        prev_mask_logits: np.ndarray,
        pts_xy: np.ndarray | None = None,
        labels: np.ndarray | None = None
    ) -> list[MaskCandidate]:
        """Refina iterativamente una máscara previa a partir de sus logits y puntos correctivos adicionales."""
        ...

    def predict_concept(
        self,
        image: np.ndarray,
        text: str | None = None,
        exemplars: list[np.ndarray] | None = None
    ) -> list[MaskCandidate]:
        """Promptable Concept Segmentation (SAM 3): Segmenta todas las instancias del concepto descripto por texto o imagen-ejemplar."""
        ...

    def segment_everything(
        self,
        image: np.ndarray,
        **kwargs: Any
    ) -> list[MaskCandidate]:
        """Segmenta automáticamente todos los objetos detectables en la imagen."""
        ...
