"""Pruebas unitarias de persistencia relacional e invariantes de datos de la Sección 6."""

import json
import pytest

from geosam.store.models import (
    GeoObjectRecord,
    ImageRecord,
    MaskVersionRecord,
    ObjectMetricRecord,
    Project,
    PromptRecord,
    SessionRecord,
)
from geosam.store.repo import GeoSamRepository


def test_project_and_image_creation(memory_repo: GeoSamRepository) -> None:
    img = ImageRecord(
        id="img_01",
        project_id="proj_test",
        uri="data/test.tif",
        sensor="Sentinel-2",
        gsd_m=10.0,
        crs="EPSG:32618",
        width=1000,
        height=1000,
        checksum="abcd1234hash",
    )
    memory_repo.add_image(img)

    retrieved = memory_repo.get_image("img_01")
    assert retrieved is not None
    assert retrieved.sensor == "Sentinel-2"
    assert retrieved.gsd_m == 10.0
    assert retrieved.checksum == "abcd1234hash"


def test_immutable_mask_versioning_invariant(memory_repo: GeoSamRepository) -> None:
    """Invariante: Las máscaras nunca se sobrescriben, se registran como versiones vinculadas."""
    # 1. Crear imagen y objeto
    img = ImageRecord(id="img_02", project_id="proj_test", uri="test.tif", width=200, height=200, checksum="csum")
    memory_repo.add_image(img)

    obj = GeoObjectRecord(id="obj_01", project_id="proj_test", image_id="img_02", status="borrador")
    memory_repo.add_object(obj)

    # 2. Agregar versión V1
    v1 = MaskVersionRecord(
        id="mv_01",
        object_id="obj_01",
        version_no=1,
        rle_json=json.dumps({"size": [200, 200], "counts": [40000]}),
        score=0.85,
        parent_version_id=None,
    )
    memory_repo.add_mask_version(v1)

    # 3. Agregar versión V2 vinculada a V1
    v2 = MaskVersionRecord(
        id="mv_02",
        object_id="obj_01",
        version_no=2,
        rle_json=json.dumps({"size": [200, 200], "counts": [100, 200, 39700]}),
        score=0.92,
        parent_version_id="mv_01",
    )
    memory_repo.add_mask_version(v2)

    # 4. Verificar historial
    history = memory_repo.get_mask_versions("obj_01")
    assert len(history) == 2
    assert history[0].version_no == 1
    assert history[1].version_no == 2
    assert history[1].parent_version_id == "mv_01"

    # Verificar que el objeto actualizó su versión activa a V2
    updated_obj = memory_repo.get_object("obj_01")
    assert updated_obj is not None
    assert updated_obj.active_version_id == "mv_02"


def test_soft_delete_invariant(memory_repo: GeoSamRepository) -> None:
    """Invariante: Borrar un objeto es status='rechazado', sin pérdida física en la BD."""
    img = ImageRecord(id="img_03", project_id="proj_test", uri="test.tif", width=200, height=200, checksum="csum")
    memory_repo.add_image(img)

    obj = GeoObjectRecord(id="obj_delete_me", project_id="proj_test", image_id="img_03", status="borrador")
    memory_repo.add_object(obj)

    # Ejecutar borrado lógico
    memory_repo.soft_delete_object("obj_delete_me")

    # El objeto sigue existiendo en la base de datos
    retrieved = memory_repo.get_object("obj_delete_me")
    assert retrieved is not None
    assert retrieved.status == "rechazado"

    # Al listar solo objetos activos (excluyendo rechazados), no aparece
    active_objects = memory_repo.list_objects("proj_test", status="aceptado")
    assert not any(o.id == "obj_delete_me" for o in active_objects)


def test_metrics_persistence(memory_repo: GeoSamRepository) -> None:
    img = ImageRecord(id="img_04", project_id="proj_test", uri="test.tif", width=200, height=200, checksum="csum")
    memory_repo.add_image(img)

    obj = GeoObjectRecord(id="obj_metric_test", project_id="proj_test", image_id="img_04")
    memory_repo.add_object(obj)

    metric = ObjectMetricRecord(
        object_id="obj_metric_test",
        crs_metric="EPSG:9377",
        area_m2=2500.0,
        perimeter_m=200.0,
        compactness=0.785,
        rectangularity=1.0,
        elongation=0.0,
        orientation_deg=0.0,
        solidity=1.0,
        n_components=1,
        n_holes=0,
    )
    memory_repo.add_object_metric(metric)

    retrieved_metric = memory_repo.get_object_metric("obj_metric_test")
    assert retrieved_metric is not None
    assert retrieved_metric.area_m2 == 2500.0
    assert retrieved_metric.crs_metric == "EPSG:9377"
