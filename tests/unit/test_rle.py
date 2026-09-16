"""Pruebas unitarias de compresión/descompresión RLE estándar COCO."""

import numpy as np
import pytest

from geosam.utils.rle import mask_to_rle, rle_to_mask


def test_rle_roundtrip_random_mask() -> None:
    rng = np.random.default_rng(42)
    original_mask = rng.choice([True, False], size=(128, 128), p=[0.2, 0.8])

    rle = mask_to_rle(original_mask)
    assert "size" in rle
    assert "counts" in rle
    assert rle["size"] == [128, 128]
    assert sum(rle["counts"]) == 128 * 128

    recovered_mask = rle_to_mask(rle)
    assert np.array_equal(original_mask, recovered_mask)


def test_rle_empty_mask() -> None:
    empty = np.zeros((50, 50), dtype=bool)
    rle = mask_to_rle(empty)
    assert rle["counts"] == [2500]

    recovered = rle_to_mask(rle)
    assert np.array_equal(empty, recovered)


def test_rle_all_true_mask() -> None:
    all_true = np.ones((40, 60), dtype=bool)
    rle = mask_to_rle(all_true)
    # COCO format starts with 0 for background count
    assert rle["counts"][0] == 0
    assert rle["counts"][1] == 2400

    recovered = rle_to_mask(rle)
    assert np.array_equal(all_true, recovered)


def test_rle_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="La máscara debe ser bidimensional"):
        mask_to_rle(np.zeros((10, 10, 3), dtype=bool))
