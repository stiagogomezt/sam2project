"""Utilidades de codificación Run-Length (RLE) compatibles con estándar COCO.

Almacena máscaras binarias compactamente evitando guardar imágenes completas en disco.
"""

from typing import Any
import numpy as np


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Convierte una máscara booleana 2D [H, W] a representación RLE (orden Fortran / columna primero).

    counts alterna longitudes de ceros (fondo) y unos (objeto).
    """
    if mask.ndim != 2:
        raise ValueError(f"La máscara debe ser bidimensional (H, W), se recibió dimensión {mask.ndim}")

    h, w = mask.shape
    # Aplanar en orden columna (Fortran) según la convención estándar de COCO
    pixels = mask.ravel(order="F").astype(np.uint8)

    if pixels.size == 0:
        return {"size": [h, w], "counts": []}

    # Encontrar índices donde cambia el valor
    changes = np.diff(pixels)
    change_indices = np.where(changes != 0)[0] + 1

    # Iniciar los puntos de corte incluyendo el inicio y el final
    split_indices = np.concatenate(([0], change_indices, [pixels.size]))
    run_lengths = np.diff(split_indices).tolist()

    # Si el primer pixel es 1, COCO requiere que la lista de conteos empiece con la cuenta de 0s (que es 0)
    if pixels[0] == 1:
        run_lengths = [0] + run_lengths

    return {
        "size": [int(h), int(w)],
        "counts": [int(x) for x in run_lengths]
    }


def rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    """Reconstruye la máscara booleana 2D [H, W] desde el formato RLE."""
    size = rle.get("size")
    counts = rle.get("counts", [])

    if not size or len(size) != 2:
        raise ValueError("El RLE debe contener 'size' como [H, W]")

    h, w = int(size[0]), int(size[1])
    total_pixels = h * w

    if total_pixels == 0:
        return np.zeros((h, w), dtype=bool)

    if not counts:
        return np.zeros((h, w), dtype=bool)

    # Reconstruir array 1D alternando 0 y 1
    # counts[0] = número de ceros iniciales
    # counts[1] = número de unos siguientes...
    runs = []
    val = 0
    for count in counts:
        runs.append(np.full(count, val, dtype=bool))
        val = 1 - val

    if runs:
        flat_mask = np.concatenate(runs)
    else:
        flat_mask = np.zeros(total_pixels, dtype=bool)

    if flat_mask.size != total_pixels:
        raise ValueError(
            f"Suma de counts ({flat_mask.size}) no coincide con H*W ({total_pixels})"
        )

    return flat_mask.reshape((h, w), order="F")
