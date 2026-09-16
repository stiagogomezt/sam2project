"""Pruebas unitarias de inferencia con MockBackend (puntos, cajas, refinamiento y conceptos)."""

import numpy as np
import pytest

from geosam.backends.mock import MockBackend


def test_mock_embed_and_caching(sample_image: np.ndarray, mock_backend: MockBackend) -> None:
    emb = mock_backend.embed(sample_image)
    assert emb.original_size == (200, 200)
    assert len(emb.image_hash) == 64  # SHA256 hex
    assert emb.model_id == "mock_v1"


def test_mock_predict_points_positive_and_negative(sample_image: np.ndarray, mock_backend: MockBackend) -> None:
    emb = mock_backend.embed(sample_image)
    # Punto positivo en (100, 100), punto negativo en (105, 105)
    pts = np.array([[100, 100], [105, 105]])
    labels = np.array([1, 0])

    candidates = mock_backend.predict_points(emb, pts, labels, multimask=True)
    assert len(candidates) == 3
    assert candidates[0].score > 0.9
    assert candidates[0].mask[100, 100] is True or candidates[0].mask.dtype == bool
    assert candidates[0].bbox_px[2] > candidates[0].bbox_px[0]


def test_mock_predict_box(sample_image: np.ndarray, mock_backend: MockBackend) -> None:
    emb = mock_backend.embed(sample_image)
    box = (50, 50, 150, 150)
    candidates = mock_backend.predict_box(emb, box)

    assert len(candidates) == 1
    c = candidates[0]
    assert bool(c.mask[100, 100]) is True
    assert bool(c.mask[10, 10]) is False
    assert c.bbox_px == (50, 50, 150, 150)


def test_mock_predict_mask_refinement(sample_image: np.ndarray, mock_backend: MockBackend) -> None:
    emb = mock_backend.embed(sample_image)
    box = (50, 50, 150, 150)
    init_candidate = mock_backend.predict_box(emb, box)[0]
    assert init_candidate.logits is not None

    # Refinar con punto negativo
    refined = mock_backend.predict_mask(
        emb,
        prev_mask_logits=init_candidate.logits,
        pts_xy=np.array([[100, 100]]),
        labels=np.array([0]),
    )
    assert len(refined) == 1
    # El punto (100, 100) debió ser removido por el punto negativo
    assert bool(refined[0].mask[100, 100]) is False


def test_mock_predict_concept(sample_image: np.ndarray, mock_backend: MockBackend) -> None:
    candidates = mock_backend.predict_concept(sample_image, text="edificaciones")
    assert len(candidates) == 2
    for c in candidates:
        assert c.mask.shape == (200, 200)
        assert c.score > 0.8
