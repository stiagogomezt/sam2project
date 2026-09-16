"""Adaptador para la familia de modelos Segment Anything 2 (SAM 2 / SAM 2.1).

Implementa el protocolo SegmentationBackend utilizando la API oficial de Meta:
`sam2.build_sam.build_sam2` y `sam2.sam2_image_predictor.SAM2ImagePredictor`.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import numpy as np

from geosam.backends.cache import EmbeddingCache, compute_image_hash
from geosam.backends.protocol import ImageEmbedding, MaskCandidate


class SAM2Backend:
    """Backend de inferencia para SAM 2 / SAM 2.1 con soporte para puntos, cajas y refinamiento."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_name: str = "sam2_hiera_t.yaml",
        model_id: str = "sam2_hiera_tiny",
        cache_dir: Path | str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.config_name = config_name
        self.model_id = model_id
        self.device = "cpu"
        self.model = None
        self.predictor = None
        self.cache = EmbeddingCache(cache_dir=cache_dir)
        self.is_loaded = False

    def load(self, device: str = "cpu") -> None:
        """Carga el modelo SAM 2 usando la librería oficial sam2."""
        self.device = device
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as err:
            raise ImportError(
                "No se pudo importar 'sam2' o 'torch'. Asegúrese de instalar el paquete SAM-2 oficial: "
                "pip install -e . sobre el repositorio facebookresearch/sam2"
            ) from err

        ckpt_str = str(self.checkpoint_path) if self.checkpoint_path else None
        if ckpt_str and not os.path.exists(ckpt_str):
            raise FileNotFoundError(f"No se encontró el checkpoint de SAM 2 en: {ckpt_str}")

        # Inicializar modelo SAM 2
        self.model = build_sam2(self.config_name, ckpt_str, device=device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.is_loaded = True

    def embed(self, image: np.ndarray) -> ImageEmbedding:
        """Calcula y almacena en caché el embedding de la imagen.

        Operación costosa que se ejecuta una única vez por imagen.
        """
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError("El backend SAM 2 debe ser cargado con backend.load() antes de procesar imágenes.")

        img_hash = compute_image_hash(image)
        cache_key = self.cache.make_key(img_hash, self.model_id, version="2.1")
        cached_emb = self.cache.get(cache_key)
        if cached_emb is not None:
            # Restaurar el estado en el predictor
            self.predictor._features = cached_emb.features.get("_features")
            self.predictor._is_image_set = True
            self.predictor._orig_hw = [cached_emb.original_size]
            self.predictor._is_batch = False
            return cached_emb

        h, w = image.shape[:2]
        # set_image en SAM 2 computa el encoder de visión
        self.predictor.set_image(image)

        features = {
            "_features": getattr(self.predictor, "_features", None),
        }

        emb = ImageEmbedding(
            features=features,
            original_size=(h, w),
            input_size=getattr(self.predictor, "_input_size", (h, w)),
            image_hash=img_hash,
            model_id=self.model_id,
            device=self.device,
        )

        self.cache.put(cache_key, emb)
        return emb

    def _format_candidates(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        logits: np.ndarray | None,
        h: int,
        w: int,
    ) -> list[MaskCandidate]:
        candidates = []
        # Normalizar dimensiones: masks suele ser [N, H, W]
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        if scores.ndim == 0:
            scores = np.array([scores])
        if logits is not None and logits.ndim == 2:
            logits = logits[np.newaxis, ...]

        for i in range(len(masks)):
            m = masks[i].astype(bool)
            sc = float(scores[i])
            lg = logits[i].astype(np.float32) if logits is not None else None

            # Extraer bounding box como extensión espacial (ancho = xmax - xmin)
            y_indices, x_indices = np.where(m)
            if len(y_indices) > 0:
                ymin, ymax = int(y_indices.min()), int(y_indices.max()) + 1
                xmin, xmax = int(x_indices.min()), int(x_indices.max()) + 1
            else:
                ymin, ymax, xmin, xmax = 0, 0, 0, 0

            candidates.append(
                MaskCandidate(
                    mask=m,
                    logits=lg,
                    score=sc,
                    bbox_px=(xmin, ymin, xmax, ymax),
                    backend_meta={
                        "candidate_index": i,
                        "backend": "SAM2Backend",
                        "model_id": self.model_id,
                    },
                )
            )
        return candidates

    def predict_points(
        self,
        emb: ImageEmbedding,
        pts_xy: np.ndarray,
        labels: np.ndarray,
        multimask: bool = True,
    ) -> list[MaskCandidate]:
        """Genera máscaras a partir de prompts de puntos positivos (1) y negativos (0)."""
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError("Backend SAM 2 no inicializado. Ejecute load().")

        h, w = emb.original_size
        point_coords = np.asarray(pts_xy, dtype=np.float32)
        point_labels = np.asarray(labels, dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            mask_input=None,
            multimask_output=multimask,
            return_logits=True,
        )

        return self._format_candidates(masks, scores, logits, h, w)

    def predict_box(
        self,
        emb: ImageEmbedding,
        box_xyxy: tuple[float, float, float, float] | np.ndarray,
    ) -> list[MaskCandidate]:
        """Genera máscaras a partir de una caja delimitadora (xmin, ymin, xmax, ymax)."""
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError("Backend SAM 2 no inicializado. Ejecute load().")

        h, w = emb.original_size
        box = np.asarray(box_xyxy, dtype=np.float32)

        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            mask_input=None,
            multimask_output=False,
            return_logits=True,
        )

        return self._format_candidates(masks, scores, logits, h, w)

    def predict_mask(
        self,
        emb: ImageEmbedding,
        prev_mask_logits: np.ndarray,
        pts_xy: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> list[MaskCandidate]:
        """Refina una máscara previa usando sus logits y clics adicionales."""
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError("Backend SAM 2 no inicializado. Ejecute load().")

        h, w = emb.original_size
        mask_input = prev_mask_logits[np.newaxis, ...] if prev_mask_logits.ndim == 2 else prev_mask_logits

        point_coords = np.asarray(pts_xy, dtype=np.float32) if pts_xy is not None else None
        point_labels = np.asarray(labels, dtype=np.int32) if labels is not None else None

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            mask_input=mask_input,
            multimask_output=False,
            return_logits=True,
        )

        return self._format_candidates(masks, scores, logits, h, w)

    def predict_concept(
        self,
        image: np.ndarray,
        text: str | None = None,
        exemplars: list[np.ndarray] | None = None,
    ) -> list[MaskCandidate]:
        """SAM 2 no soporta segmentación por concepto en lenguaje natural de forma nativa."""
        raise NotImplementedError(
            "SAM 2 no soporta 'predict_concept' (segmentación por texto/ejemplares). "
            "Utilice SAM3Backend para Promptable Concept Segmentation (PCS)."
        )

    def segment_everything(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> list[MaskCandidate]:
        """Ejecuta segmentación automática sobre toda la imagen usando SAM2AutomaticMaskGenerator."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Backend SAM 2 no inicializado.")

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        generator = SAM2AutomaticMaskGenerator(self.model, **kwargs)
        results = generator.generate(image)

        candidates = []
        for i, item in enumerate(results):
            m = item["segmentation"].astype(bool)
            sc = float(item.get("predicted_iou", item.get("stability_score", 0.9)))
            bbox = item.get("bbox", [0, 0, 0, 0])  # [x, y, w, h] en formato generator
            xmin, ymin, bw, bh = bbox
            xmax, ymax = xmin + bw, ymin + bh

            candidates.append(
                MaskCandidate(
                    mask=m,
                    logits=None,
                    score=sc,
                    bbox_px=(int(xmin), int(ymin), int(xmax), int(ymax)),
                    backend_meta={
                        "candidate_index": i,
                        "backend": "SAM2Backend",
                        "area": item.get("area", int(m.sum())),
                    },
                )
            )
        return candidates
