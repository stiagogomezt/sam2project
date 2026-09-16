"""Pruebas unitarias de cumplimiento de protocolo e interfaces para backends de segmentación."""

import numpy as np
import pytest

from geosam.backends.protocol import (
    ImageEmbedding,
    MaskCandidate,
    SegmentationBackend,
)
from geosam.backends.mock import MockBackend
from geosam.backends.sam2 import SAM2Backend
from geosam.backends.sam3 import SAM3Backend


def test_mock_backend_implements_protocol() -> None:
    backend = MockBackend()
    assert isinstance(backend, SegmentationBackend)


def test_sam2_backend_implements_protocol() -> None:
    backend = SAM2Backend()
    assert isinstance(backend, SegmentationBackend)


def test_sam3_backend_implements_protocol() -> None:
    backend = SAM3Backend()
    assert isinstance(backend, SegmentationBackend)


def test_mask_candidate_validation() -> None:
    # Máscara 2D válida
    valid_mask = np.zeros((100, 100), dtype=bool)
    candidate = MaskCandidate(
        mask=valid_mask,
        logits=np.zeros((100, 100), dtype=np.float32),
        score=0.95,
        bbox_px=(10, 10, 50, 50),
    )
    assert candidate.score == 0.95
    assert candidate.bbox_px == (10, 10, 50, 50)
    assert candidate.mask.shape == (100, 100)

    # Máscara no 2D debe fallar
    with pytest.raises(ValueError, match="La máscara debe ser bidimensional"):
        MaskCandidate(
            mask=np.zeros((10, 10, 3), dtype=bool),
            logits=None,
            score=0.5,
            bbox_px=(0, 0, 10, 10),
        )

    # Bbox con longitud incorrecta debe fallar
    with pytest.raises(ValueError, match="bbox_px debe tener 4 elementos"):
        MaskCandidate(
            mask=valid_mask,
            logits=None,
            score=0.5,
            bbox_px=(0, 0, 10),  # type: ignore
        )
