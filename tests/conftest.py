"""Fixtures comunes de pytest para GEO-SAM."""

import numpy as np
import pytest

from geosam.backends.mock import MockBackend
from geosam.store.models import Project
from geosam.store.repo import GeoSamRepository


@pytest.fixture
def sample_image() -> np.ndarray:
    """Genera una imagen sintética RGB de 200x200 píxeles con un objeto de prueba."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Dibujar un cuadrado simulado en el centro
    img[50:150, 50:150] = [200, 200, 200]
    return img


@pytest.fixture
def mock_backend() -> MockBackend:
    """Instancia inicializada de MockBackend para pruebas rápidas sin GPU."""
    backend = MockBackend(model_id="mock_v1", version="1.0")
    backend.load(device="cpu")
    return backend


@pytest.fixture
def memory_repo() -> GeoSamRepository:
    """Repositorio transaccional SQLite en memoria RAM para pruebas aisladas."""
    repo = GeoSamRepository(":memory:")
    # Insertar proyecto base
    repo.add_project(Project(id="proj_test", name="Proyecto Test"))
    return repo
