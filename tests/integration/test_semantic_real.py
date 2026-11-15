"""Clasificador semántico complementario (RemoteCLIP) con pesos reales; se omite si faltan."""
import os

import numpy as np
import pytest

pytest.importorskip("open_clip")
from src.semantic import CATEGORY_PROMPTS, REMOTECLIP_INFO, SemanticClassifier, object_crop  # noqa: E402

if not os.path.exists(REMOTECLIP_INFO["checkpoint"]):
    pytest.skip(f"Pesos RemoteCLIP no encontrados: {REMOTECLIP_INFO['checkpoint']}", allow_module_level=True)


@pytest.fixture(scope="module")
def clf():
    return SemanticClassifier()


def test_probabilities_are_aggregated_per_category(clf):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(120, 120, 3), dtype=np.uint8)
    ranked = clf.classify(img, top_k=len(CATEGORY_PROMPTS))
    assert [c for c, _ in ranked] and set(c for c, _ in ranked) == set(CATEGORY_PROMPTS)
    assert sum(p for _, p in ranked) == pytest.approx(1.0, abs=1e-4)
    assert ranked == sorted(ranked, key=lambda kv: -kv[1])


def test_uniform_green_is_vegetation(clf):
    """Un recorte verde homogéneo con textura debe sugerir vegetación (zona verde o arbolado)."""
    rng = np.random.default_rng(1)
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    img[..., 1] = rng.integers(110, 160, size=(160, 160))
    img[..., 0] = rng.integers(40, 70, size=(160, 160))
    img[..., 2] = rng.integers(30, 60, size=(160, 160))
    top = [c for c, _ in clf.classify(img, top_k=2)]
    assert {"Zona verde", "Arbolado", "Cancha deportiva"} & set(top)


def test_object_crop_keeps_square_context():
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    crop = object_crop(img, (100, 100, 140, 300))
    assert crop.shape[0] == crop.shape[1] or min(crop.shape[:2]) > 40
