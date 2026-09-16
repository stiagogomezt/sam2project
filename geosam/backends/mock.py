"""Backend simulado (MockBackend) determinista para pruebas unitarias e integración sin GPU ni pesos.

Cumple estrictamente con el protocolo SegmentationBackend.
"""

from __future__ import annotations
from typing import Any
import numpy as np

from geosam.backends.cache import compute_image_hash
from geosam.backends.protocol import ImageEmbedding, MaskCandidate, SegmentationBackend


class MockBackend:
    """Implementación simulada y determinista de SegmentationBackend para tests y desarrollo sin GPU."""

    def __init__(self, model_id: str = "mock_model", version: str = "1.0") -> None:
        self.model_id = model_id
        self.version = version
        self.is_loaded = False
        self.device = "cpu"

    def load(self, device: str = "cpu") -> None:
        """Marca el backend como cargado en el dispositivo indicado."""
        self.is_loaded = True
        self.device = device

    def embed(self, image: np.ndarray) -> ImageEmbedding:
        """Genera un embedding sintético determinista a partir de las dimensiones y el hash de la imagen."""
        if not self.is_loaded:
            self.load()

        if image.ndim < 2:
            raise ValueError(f"La imagen debe tener al menos 2 dimensiones, se recibió {image.shape}")

        h, w = image.shape[:2]
        img_hash = compute_image_hash(image)

        # Características simuladas (tensor ficticio en float32)
        mock_features = np.ones((1, 16, 16), dtype=np.float32)

        return ImageEmbedding(
            features=mock_features,
            original_size=(h, w),
            input_size=(h, w),
            image_hash=img_hash,
            model_id=self.model_id,
            device=self.device,
            extra={"mock": True, "version": self.version},
        )

    def _make_candidate(
        self,
        mask: np.ndarray,
        score: float,
        candidate_idx: int = 0
    ) -> MaskCandidate:
        h, w = mask.shape
        # Calcular bounding box como coordenadas de extensión (ancho = xmax - xmin)
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            ymin, ymax = int(y_indices.min()), int(y_indices.max()) + 1
            xmin, xmax = int(x_indices.min()), int(x_indices.max()) + 1
        else:
            ymin, ymax, xmin, xmax = 0, 0, 0, 0

        # Logits simulados: +10.0 en máscara, -10.0 fuera
        logits = np.where(mask, 10.0, -10.0).astype(np.float32)

        return MaskCandidate(
            mask=mask,
            logits=logits,
            score=score,
            bbox_px=(xmin, ymin, xmax, ymax),
            backend_meta={
                "candidate_index": candidate_idx,
                "backend": "MockBackend",
                "model_id": self.model_id,
            },
        )

    def predict_points(
        self,
        emb: ImageEmbedding,
        pts_xy: np.ndarray,
        labels: np.ndarray,
        multimask: bool = True,
    ) -> list[MaskCandidate]:
        """Genera máscaras sintéticas centradas en los puntos positivos con recortes de puntos negativos."""
        h, w = emb.original_size
        pts = np.asarray(pts_xy, dtype=float)
        lbls = np.asarray(labels, dtype=int)

        if len(pts) == 0:
            empty_mask = np.zeros((h, w), dtype=bool)
            return [self._make_candidate(empty_mask, score=0.0)]

        radii = [25, 35, 15] if multimask else [25]
        scores = [0.95, 0.88, 0.78] if multimask else [0.95]

        # Crear cuadrícula de coordenadas
        yy, xx = np.ogrid[:h, :w]
        candidates = []

        for idx, (radius, score) in enumerate(zip(radii, scores)):
            base_mask = np.zeros((h, w), dtype=bool)

            # Sumar influencia de puntos positivos
            for pt, lbl in zip(pts, lbls):
                px, py = pt[0], pt[1]
                dist_sq = (xx - px) ** 2 + (yy - py) ** 2
                if lbl == 1:
                    base_mask |= (dist_sq <= radius ** 2)

            # Restar influencia de puntos negativos
            for pt, lbl in zip(pts, lbls):
                px, py = pt[0], pt[1]
                dist_sq = (xx - px) ** 2 + (yy - py) ** 2
                if lbl == 0:
                    base_mask &= ~(dist_sq <= (radius * 0.8) ** 2)

            candidates.append(self._make_candidate(base_mask, score=score, candidate_idx=idx))

        return candidates

    def predict_box(
        self,
        emb: ImageEmbedding,
        box_xyxy: tuple[float, float, float, float] | np.ndarray,
    ) -> list[MaskCandidate]:
        """Genera una máscara que llena la caja delimitadora (con márgenes suaves)."""
        h, w = emb.original_size
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))

        mask = np.zeros((h, w), dtype=bool)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True

        return [self._make_candidate(mask, score=0.96, candidate_idx=0)]

    def predict_mask(
        self,
        emb: ImageEmbedding,
        prev_mask_logits: np.ndarray,
        pts_xy: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> list[MaskCandidate]:
        """Refina la máscara a partir de logits previos más puntos correctivos."""
        h, w = emb.original_size
        current_mask = (prev_mask_logits > 0)

        if pts_xy is not None and labels is not None and len(pts_xy) > 0:
            yy, xx = np.ogrid[:h, :w]
            for pt, lbl in zip(pts_xy, labels):
                px, py = pt[0], pt[1]
                dist_sq = (xx - px) ** 2 + (yy - py) ** 2
                if lbl == 1:
                    current_mask |= (dist_sq <= 15 ** 2)
                elif lbl == 0:
                    current_mask &= ~(dist_sq <= 15 ** 2)

        return [self._make_candidate(current_mask, score=0.98, candidate_idx=0)]

    def predict_concept(
        self,
        image: np.ndarray,
        text: str | None = None,
        exemplars: list[np.ndarray] | None = None,
    ) -> list[MaskCandidate]:
        """Genera candidatos de concepto simulados para pruebas de SAM 3."""
        h, w = image.shape[:2]
        # Generar dos polígonos sintéticos de prueba
        mask1 = np.zeros((h, w), dtype=bool)
        mask2 = np.zeros((h, w), dtype=bool)

        # Región 1
        s1 = min(h, w) // 5
        mask1[s1:s1*2, s1:s1*2] = True

        # Región 2
        s2 = min(h, w) // 2
        mask2[s2:s2+s1, s2:s2+s1] = True

        c1 = self._make_candidate(mask1, score=0.91, candidate_idx=0)
        c2 = self._make_candidate(mask2, score=0.87, candidate_idx=1)
        return [c1, c2]

    def segment_everything(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> list[MaskCandidate]:
        """Genera una cuadrícula de máscaras simuladas."""
        h, w = image.shape[:2]
        candidates = []
        step_y = max(10, h // 4)
        step_x = max(10, w // 4)
        idx = 0
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                m = np.zeros((h, w), dtype=bool)
                m[y:min(h, y + step_y // 2), x:min(w, x + step_x // 2)] = True
                candidates.append(self._make_candidate(m, score=0.80, candidate_idx=idx))
                idx += 1
        return candidates
