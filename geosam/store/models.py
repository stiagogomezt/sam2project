"""Modelos de dominio tipados para el repositorio de datos de GEO-SAM."""

from __future__ import annotations
from dataclasses import dataclass, field
import datetime
from typing import Any


@dataclass
class Project:
    id: str
    name: str
    crs_storage: str = "EPSG:4326"
    crs_metric: str = "EPSG:9377"
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    notes: str | None = None


@dataclass
class ImageRecord:
    id: str
    project_id: str
    uri: str
    source_id: str | None = None
    acquired_at: str | None = None
    sensor: str | None = None
    gsd_m: float | None = None
    crs: str | None = None
    transform_json: str | None = None
    width: int = 0
    height: int = 0
    checksum: str = ""
    georef_method: str | None = None
    georef_rmse: float | None = None
    provenance_json: str | None = None


@dataclass
class SessionRecord:
    id: str
    image_id: str
    backend: str
    model_version: str
    device: str
    checkpoint: str | None = None
    started_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


@dataclass
class PromptRecord:
    id: str
    session_id: str
    type: str  # 'point', 'box', 'mask', 'text', 'exemplar', 'auto'
    payload_json: str
    object_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    latency_ms: float | None = None


@dataclass
class MaskVersionRecord:
    id: str
    object_id: str
    version_no: int
    rle_json: str
    score: float | None = None
    iou_pred: float | None = None
    parent_version_id: str | None = None
    prompt_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    accepted: bool = False


@dataclass
class GeoObjectRecord:
    id: str
    project_id: str
    image_id: str
    label: str | None = None
    class_source: str | None = None  # 'manual', 'text_prompt', 'rule', None
    status: str = "borrador"  # 'borrador', 'aceptado', 'revisar', 'rechazado'
    geom_4326_geojson: str | None = None
    geom_4326_wkt: str | None = None
    active_version_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


@dataclass
class ObjectMetricRecord:
    object_id: str
    crs_metric: str
    area_m2: float | None = None
    perimeter_m: float | None = None
    compactness: float | None = None
    rectangularity: float | None = None
    elongation: float | None = None
    orientation_deg: float | None = None
    solidity: float | None = None
    n_components: int = 1
    n_holes: int = 0
    computed_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


@dataclass
class ValidationRecord:
    id: str
    object_id: str
    reference_layer: str
    ref_feature_id: str | None = None
    iou: float | None = None
    boundary_f1: float | None = None
    area_ratio: float | None = None
    method: str | None = None
    at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


@dataclass
class ChangeLinkRecord:
    id: str
    object_a: str
    object_b: str
    delta_area_m2: float | None = None
    pct_change: float | None = None
    centroid_shift_m: float | None = None
    method: str | None = None
    at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


@dataclass
class ExportLogRecord:
    id: str
    project_id: str
    format: str
    path: str
    n_objects: int
    filters_json: str | None = None
    at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
