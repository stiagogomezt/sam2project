"""Pruebas de integración con el modelo SAM 2 REAL (no simulado).

Se omiten automáticamente si torch/sam2 no están instalados o si no se encuentra el checkpoint.
Para ejecutarlas:
    pytest tests/integration -v          # usa el checkpoint por defecto de config.py
    SAM2_TEST_MODEL=sam2_hiera_tiny pytest tests/integration -v
"""
import os
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sam2")

from config import SAM2_MODELS, DEFAULT_SAM2_MODEL_KEY  # noqa: E402

MODEL_KEY = os.environ.get("SAM2_TEST_MODEL", DEFAULT_SAM2_MODEL_KEY)
CKPT = SAM2_MODELS[MODEL_KEY]["checkpoint"]
if not os.path.exists(CKPT):
    pytest.skip(f"Checkpoint no encontrado: {CKPT}", allow_module_level=True)

from src.sam_engine import SAM2Engine  # noqa: E402


def _synthetic_scene(seed: int, size: int = 512) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Imagen con fondo texturado y un rectángulo uniforme bien contrastado (objeto conocido)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 110, size=(size, size, 3), dtype=np.uint8)
    x0, y0 = 120 + seed * 20, 140
    x1, y1 = x0 + 180, y0 + 120
    img[y0:y1, x0:x1] = (230, 40, 40)
    return img, (x0, y0, x1, y1)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


@pytest.fixture(scope="module")
def engine() -> SAM2Engine:
    eng = SAM2Engine(MODEL_KEY, allow_mock=False)
    assert not eng.is_mock
    return eng


def _gt(shape, box):
    m = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def test_point_prompt_segments_known_object(engine):
    img, box = _synthetic_scene(0)
    emb, t = engine.embed(img)
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    res = engine.predict(emb, np.array([[cx, cy]]), np.array([1]), multimask=True)
    assert len(res["candidates"]) == 3
    best = res["candidates"][res["best_idx"]]
    gt = _gt(img.shape[:2], box)
    assert best.mask.dtype == bool and best.mask.shape == gt.shape
    # Regresión: antes del fix las máscaras cubrían el 100 % de la imagen.
    assert best.mask.mean() < 0.5
    assert _iou(best.mask, gt) > 0.85
    assert best.logits.shape == (256, 256)


def test_box_prompt_and_negative_point(engine):
    img, box = _synthetic_scene(1)
    emb, _ = engine.embed(img)
    gt = _gt(img.shape[:2], box)
    res = engine.predict(emb, box=np.array(box, dtype=float), multimask=False)
    assert len(res["candidates"]) == 1
    assert _iou(res["candidates"][0].mask, gt) > 0.85

    # Punto negativo sobre el fondo junto a un positivo: no debe empeorar la máscara.
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    pts = np.array([[cx, cy], [box[0] - 30, cy]])
    res2 = engine.predict(emb, pts, np.array([1, 0]), multimask=False)
    m2 = res2["candidates"][0].mask
    assert not m2[cy, box[0] - 30]
    assert _iou(m2, gt) > 0.8


def test_iterative_refinement_with_logits(engine):
    img, box = _synthetic_scene(2)
    emb, _ = engine.embed(img)
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    first = engine.predict(emb, np.array([[cx, cy]]), np.array([1]), multimask=True)
    logits = first["candidates"][first["best_idx"]].logits
    refined = engine.predict(
        emb, np.array([[cx, cy], [cx + 40, cy]]), np.array([1, 1]), mask_logits=logits, multimask=False
    )
    gt = _gt(img.shape[:2], box)
    assert _iou(refined["candidates"][0].mask, gt) > 0.85


def test_sessions_do_not_overwrite_each_other(engine):
    """Dos 'usuarios' con imágenes distintas intercalan embed/predict en hilos distintos."""
    img_a, box_a = _synthetic_scene(0)
    img_b, box_b = _synthetic_scene(3)
    # Desplazar el objeto de B para que un cruce de embeddings sea detectable.
    img_b = np.ascontiguousarray(np.flipud(img_b))
    h = img_b.shape[0]
    box_b = (box_b[0], h - box_b[3], box_b[2], h - box_b[1])

    emb_a, _ = engine.embed(img_a)
    emb_b, _ = engine.embed(img_b)  # el predictor queda con la imagen B "activa"
    pa = np.array([[(box_a[0] + box_a[2]) // 2, (box_a[1] + box_a[3]) // 2]])
    pb = np.array([[(box_b[0] + box_b[2]) // 2, (box_b[1] + box_b[3]) // 2]])

    results: dict[str, list[float]] = {"a": [], "b": []}

    def worker(name, emb, pts, gt):
        for _ in range(4):
            r = engine.predict(emb, pts, np.array([1]), multimask=True)
            results[name].append(_iou(r["candidates"][r["best_idx"]].mask, gt))

    ta = threading.Thread(target=worker, args=("a", emb_a, pa, _gt(img_a.shape[:2], box_a)))
    tb = threading.Thread(target=worker, args=("b", emb_b, pb, _gt(img_b.shape[:2], box_b)))
    ta.start(); tb.start(); ta.join(); tb.join()
    assert min(results["a"]) > 0.85, results
    assert min(results["b"]) > 0.85, results


def test_embedding_from_other_model_is_rejected(engine):
    img, _ = _synthetic_scene(0)
    emb, _ = engine.embed(img)
    emb.model_id = "otro_modelo"
    with pytest.raises(ValueError):
        engine.predict(emb, np.array([[10, 10]]), np.array([1]))
    emb.model_id = MODEL_KEY
