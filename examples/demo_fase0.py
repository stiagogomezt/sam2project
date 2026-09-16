"""Demostración reproducible de la Fase 0 de GEO-SAM.

Ejecuta el flujo completo de:
1. Creación de proyecto en SQLite.
2. Ingesta de imagen simulada.
3. Generación de embedding con MockBackend.
4. Inferencia interactiva de puntos y cajas.
5. Serialización en RLE e inserción en repositorio con versionado inmutable.
6. Recuperación e impresión de trazabilidad.
"""

import json
import sys
from pathlib import Path
import numpy as np

# Asegurar que el directorio raíz del proyecto esté en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geosam.backends.mock import MockBackend
from geosam.store.models import (
    GeoObjectRecord,
    ImageRecord,
    MaskVersionRecord,
    Project,
    PromptRecord,
    SessionRecord,
)
from geosam.store.repo import GeoSamRepository
from geosam.utils.rle import mask_to_rle


def run_demo() -> None:
    print("=== GEO-SAM: Demostración de Punta a Punta (Fase 0) ===")

    # 1. Inicializar Repositorio
    db_file = Path("demo_geosam.sqlite")
    if db_file.exists():
        db_file.unlink()

    repo = GeoSamRepository(db_file)
    print("[OK] Repositorio inicializado en:", db_file)

    # 2. Registrar Proyecto e Imagen
    proj = Project(id="proj_valle", name="Extracción Catastral Piloto", crs_metric="EPSG:9377")
    repo.add_project(proj)

    img_rec = ImageRecord(
        id="img_ortofoto_01",
        project_id="proj_valle",
        uri="data/ortofoto_cali_2026.tif",
        sensor="Drone",
        gsd_m=0.1,
        crs="EPSG:9377",
        width=512,
        height=512,
        checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        georef_method="G1_Nativo",
    )
    repo.add_image(img_rec)
    print("[OK] Proyecto e Imagen registrados.")

    # 3. Inicializar Backend y Calcular Embedding
    backend = MockBackend(model_id="mock_fast_v1")
    backend.load(device="cpu")

    # Crear imagen sintética
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    test_image[100:300, 100:300] = [180, 180, 180]  # Edificio simulado

    emb = backend.embed(test_image)
    print(f"[OK] Embedding generado exitosamente (Hash: {emb.image_hash[:16]}...)")

    session = SessionRecord(
        id="sess_01",
        image_id="img_ortofoto_01",
        backend="MockBackend",
        model_version="1.0",
        device="cpu",
    )
    repo.add_session(session)

    # 4. Inferencia con Clic / Punto
    click_xy = np.array([[200, 200]])
    labels = np.array([1])
    candidates = backend.predict_points(emb, click_xy, labels, multimask=True)
    best_candidate = candidates[0]
    print(f"[OK] Inferencia de puntos: {len(candidates)} candidatos generados (Score mejor candidato: {best_candidate.score:.2f})")

    # 5. Guardar en Repositorio con Formato RLE y Versionado
    rle = mask_to_rle(best_candidate.mask)
    obj = GeoObjectRecord(
        id="obj_bldg_001",
        project_id="proj_valle",
        image_id="img_ortofoto_01",
        label="Edificación",
        class_source="manual",
        status="aceptado",
    )
    repo.add_object(obj)

    prompt = PromptRecord(
        id="prmpt_01",
        session_id="sess_01",
        object_id="obj_bldg_001",
        type="point",
        payload_json=json.dumps({"pts": [[200, 200]], "labels": [1]}),
        latency_ms=12.5,
    )
    repo.add_prompt(prompt)

    v1 = MaskVersionRecord(
        id="mv_v1_001",
        object_id="obj_bldg_001",
        version_no=1,
        rle_json=json.dumps(rle),
        score=best_candidate.score,
        prompt_id="prmpt_01",
        accepted=True,
    )
    repo.add_mask_version(v1)

    # 6. Consultar y verificar trazabilidad
    stored_obj = repo.get_object("obj_bldg_001")
    history = repo.get_mask_versions("obj_bldg_001")
    print(f"[OK] Trazabilidad verificada: Objeto {stored_obj.id} con {len(history)} versión(es) activa(s) (Versión actual: {stored_obj.active_version_id})")
    print(f"[OK] Tamaño de compresión RLE: {len(rle['counts'])} tramos frente a {512*512} píxeles crudos.")

    repo.close()
    if db_file.exists():
        db_file.unlink()
    print("=== Demo Fase 0 Completada con Éxito ===")


if __name__ == "__main__":
    run_demo()
