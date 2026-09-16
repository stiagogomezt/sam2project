"""Repositorio de persistencia transaccional SQLite / GeoPackage para GEO-SAM.

Implementa las invariantes de datos de la Sección 6:
1. Versionado inmutable de máscaras (parent_version_id, sin sobrescritura).
2. Borrado lógico de objetos (status='rechazado', papelera).
3. Almacenamiento eficiente en RLE (JSON).
"""

from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Any

from geosam.store.models import (
    ChangeLinkRecord,
    ExportLogRecord,
    GeoObjectRecord,
    ImageRecord,
    MaskVersionRecord,
    ObjectMetricRecord,
    Project,
    PromptRecord,
    SessionRecord,
    ValidationRecord,
)
from geosam.store.schema import init_db


class GeoSamRepository:
    """Repositorio transaccional de acceso a datos para proyectos GEO-SAM."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        # Activar soporte de llaves foráneas en SQLite
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.row_factory = sqlite3.Row
        init_db(self._conn)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    # --- PROJECT ---
    def add_project(self, project: Project) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO project (id, name, crs_storage, crs_metric, created_at, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (project.id, project.name, project.crs_storage, project.crs_metric, project.created_at, project.notes),
            )

    def get_project(self, project_id: str) -> Project | None:
        cursor = self._conn.execute("SELECT * FROM project WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return Project(
            id=row["id"],
            name=row["name"],
            crs_storage=row["crs_storage"],
            crs_metric=row["crs_metric"],
            created_at=row["created_at"],
            notes=row["notes"],
        )

    # --- IMAGE ---
    def add_image(self, image: ImageRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO image (id, project_id, uri, source_id, acquired_at, sensor,
                                   gsd_m, crs, transform_json, width, height, checksum,
                                   georef_method, georef_rmse, provenance_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image.id,
                    image.project_id,
                    image.uri,
                    image.source_id,
                    image.acquired_at,
                    image.sensor,
                    image.gsd_m,
                    image.crs,
                    image.transform_json,
                    image.width,
                    image.height,
                    image.checksum,
                    image.georef_method,
                    image.georef_rmse,
                    image.provenance_json,
                ),
            )

    def get_image(self, image_id: str) -> ImageRecord | None:
        cursor = self._conn.execute("SELECT * FROM image WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return ImageRecord(
            id=row["id"],
            project_id=row["project_id"],
            uri=row["uri"],
            source_id=row["source_id"],
            acquired_at=row["acquired_at"],
            sensor=row["sensor"],
            gsd_m=row["gsd_m"],
            crs=row["crs"],
            transform_json=row["transform_json"],
            width=row["width"],
            height=row["height"],
            checksum=row["checksum"],
            georef_method=row["georef_method"],
            georef_rmse=row["georef_rmse"],
            provenance_json=row["provenance_json"],
        )

    # --- SESSION ---
    def add_session(self, session: SessionRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO session (id, image_id, backend, model_version, checkpoint, device, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.image_id,
                    session.backend,
                    session.model_version,
                    session.checkpoint,
                    session.device,
                    session.started_at,
                ),
            )

    # --- PROMPT ---
    def add_prompt(self, prompt: PromptRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO prompt (id, session_id, object_id, type, payload_json, created_at, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prompt.id,
                    prompt.session_id,
                    prompt.object_id,
                    prompt.type,
                    prompt.payload_json,
                    prompt.created_at,
                    prompt.latency_ms,
                ),
            )

    # --- MASK VERSION (INVARIANTE 1: INMUTABLE) ---
    def add_mask_version(self, mask_version: MaskVersionRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO mask_version (id, object_id, version_no, rle_json, score,
                                          iou_pred, parent_version_id, prompt_id, created_at, accepted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mask_version.id,
                    mask_version.object_id,
                    mask_version.version_no,
                    mask_version.rle_json,
                    mask_version.score,
                    mask_version.iou_pred,
                    mask_version.parent_version_id,
                    mask_version.prompt_id,
                    mask_version.created_at,
                    1 if mask_version.accepted else 0,
                ),
            )
            # Actualizar active_version_id en el objeto
            self._conn.execute(
                "UPDATE object SET active_version_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (mask_version.id, mask_version.object_id),
            )

    def get_mask_versions(self, object_id: str) -> list[MaskVersionRecord]:
        cursor = self._conn.execute(
            "SELECT * FROM mask_version WHERE object_id = ? ORDER BY version_no ASC",
            (object_id,),
        )
        results = []
        for row in cursor.fetchall():
            results.append(
                MaskVersionRecord(
                    id=row["id"],
                    object_id=row["object_id"],
                    version_no=row["version_no"],
                    rle_json=row["rle_json"],
                    score=row["score"],
                    iou_pred=row["iou_pred"],
                    parent_version_id=row["parent_version_id"],
                    prompt_id=row["prompt_id"],
                    created_at=row["created_at"],
                    accepted=bool(row["accepted"]),
                )
            )
        return results

    # --- OBJECT (INVARIANTE 2: BORRADO LÓGICO) ---
    def add_object(self, obj: GeoObjectRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO object (id, project_id, image_id, label, class_source, status,
                                    geom_4326_geojson, geom_4326_wkt, active_version_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obj.id,
                    obj.project_id,
                    obj.image_id,
                    obj.label,
                    obj.class_source,
                    obj.status,
                    obj.geom_4326_geojson,
                    obj.geom_4326_wkt,
                    obj.active_version_id,
                    obj.created_at,
                    obj.updated_at,
                ),
            )

    def get_object(self, object_id: str) -> GeoObjectRecord | None:
        cursor = self._conn.execute("SELECT * FROM object WHERE id = ?", (object_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return GeoObjectRecord(
            id=row["id"],
            project_id=row["project_id"],
            image_id=row["image_id"],
            label=row["label"],
            class_source=row["class_source"],
            status=row["status"],
            geom_4326_geojson=row["geom_4326_geojson"],
            geom_4326_wkt=row["geom_4326_wkt"],
            active_version_id=row["active_version_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def soft_delete_object(self, object_id: str) -> None:
        """Borrado lógico: marca el objeto con status='rechazado' sin eliminar la fila ni su historial."""
        with self._conn:
            self._conn.execute(
                "UPDATE object SET status = 'rechazado', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (object_id,),
            )

    def list_objects(self, project_id: str, status: str | None = None) -> list[GeoObjectRecord]:
        """Lista objetos de un proyecto, opcionalmente filtrados por estado (ej. excluyendo 'rechazado')."""
        if status:
            cursor = self._conn.execute(
                "SELECT * FROM object WHERE project_id = ? AND status = ? ORDER BY created_at ASC",
                (project_id, status),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM object WHERE project_id = ? ORDER BY created_at ASC",
                (project_id,),
            )

        objects = []
        for row in cursor.fetchall():
            objects.append(
                GeoObjectRecord(
                    id=row["id"],
                    project_id=row["project_id"],
                    image_id=row["image_id"],
                    label=row["label"],
                    class_source=row["class_source"],
                    status=row["status"],
                    geom_4326_geojson=row["geom_4326_geojson"],
                    geom_4326_wkt=row["geom_4326_wkt"],
                    active_version_id=row["active_version_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return objects

    # --- METRICS ---
    def add_object_metric(self, metric: ObjectMetricRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO object_metric (
                    object_id, crs_metric, area_m2, perimeter_m, compactness,
                    rectangularity, elongation, orientation_deg, solidity,
                    n_components, n_holes, computed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.object_id,
                    metric.crs_metric,
                    metric.area_m2,
                    metric.perimeter_m,
                    metric.compactness,
                    metric.rectangularity,
                    metric.elongation,
                    metric.orientation_deg,
                    metric.solidity,
                    metric.n_components,
                    metric.n_holes,
                    metric.computed_at,
                ),
            )

    def get_object_metric(self, object_id: str) -> ObjectMetricRecord | None:
        cursor = self._conn.execute("SELECT * FROM object_metric WHERE object_id = ?", (object_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return ObjectMetricRecord(
            object_id=row["object_id"],
            crs_metric=row["crs_metric"],
            area_m2=row["area_m2"],
            perimeter_m=row["perimeter_m"],
            compactness=row["compactness"],
            rectangularity=row["rectangularity"],
            elongation=row["elongation"],
            orientation_deg=row["orientation_deg"],
            solidity=row["solidity"],
            n_components=row["n_components"],
            n_holes=row["n_holes"],
            computed_at=row["computed_at"],
        )

    # --- VALIDATION ---
    def add_validation(self, val: ValidationRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO validation (id, object_id, reference_layer, ref_feature_id, iou, boundary_f1, area_ratio, method, at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    val.id,
                    val.object_id,
                    val.reference_layer,
                    val.ref_feature_id,
                    val.iou,
                    val.boundary_f1,
                    val.area_ratio,
                    val.method,
                    val.at,
                ),
            )

    # --- CHANGE LINK ---
    def add_change_link(self, link: ChangeLinkRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO change_link (id, object_a, object_b, delta_area_m2, pct_change, centroid_shift_m, method, at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link.id,
                    link.object_a,
                    link.object_b,
                    link.delta_area_m2,
                    link.pct_change,
                    link.centroid_shift_m,
                    link.method,
                    link.at,
                ),
            )

    # --- EXPORT LOG ---
    def add_export_log(self, log: ExportLogRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO export_log (id, project_id, format, path, n_objects, filters_json, at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log.id,
                    log.project_id,
                    log.format,
                    log.path,
                    log.n_objects,
                    log.filters_json,
                    log.at,
                ),
            )
