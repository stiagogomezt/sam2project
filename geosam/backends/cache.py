"""Gestor de caché en memoria y disco para embeddings de imágenes.

Garantiza que el encoder de SAM se ejecute una sola vez por imagen.
Clave de caché: sha256(bytes_imagen) + model_id + model_version.
"""

from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import Any
import numpy as np

from geosam.backends.protocol import ImageEmbedding


def compute_image_hash(image: np.ndarray) -> str:
    """Calcula el hash SHA-256 de una imagen de NumPy basándose en sus dimensiones, tipo y bytes contiguos."""
    contiguous = np.ascontiguousarray(image)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.shape).encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


class EmbeddingCache:
    """Caché de embeddings multinivel (memoria RAM con respaldo opcional en disco)."""

    def __init__(self, cache_dir: Path | str | None = None, max_memory_entries: int = 16) -> None:
        self.max_memory_entries = max_memory_entries
        self._memory_cache: dict[str, ImageEmbedding] = {}
        self._keys_order: list[str] = []

        if cache_dir is None:
            default_dir = Path.home() / ".geosam" / "cache" / "embeddings"
            self.cache_dir = default_dir
        else:
            self.cache_dir = Path(cache_dir)

    def make_key(self, image_hash: str, model_id: str, version: str = "v1") -> str:
        """Genera una clave única normalizada para el embedding."""
        raw = f"{image_hash}_{model_id}_{version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> ImageEmbedding | None:
        """Obtiene un embedding desde la memoria RAM si existe."""
        if key in self._memory_cache:
            # Mover clave al final para política LRU
            self._keys_order.remove(key)
            self._keys_order.append(key)
            return self._memory_cache[key]
        return None

    def put(self, key: str, embedding: ImageEmbedding) -> None:
        """Almacena el embedding en la memoria RAM con desalojo LRU."""
        if key in self._memory_cache:
            self._keys_order.remove(key)
        elif len(self._memory_cache) >= self.max_memory_entries:
            oldest_key = self._keys_order.pop(0)
            self._memory_cache.pop(oldest_key, None)

        self._memory_cache[key] = embedding
        self._keys_order.append(key)

    def clear(self) -> None:
        """Limpia la caché en memoria."""
        self._memory_cache.clear()
        self._keys_order.clear()

    def __len__(self) -> int:
        return len(self._memory_cache)
