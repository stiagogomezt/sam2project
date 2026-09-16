"""Definición del esquema relacional DDL para persistencia en SQLite / GeoPackage.

Contrato formal según la sección 6 de la especificación técnica.
"""

DDL_STATEMENTS = """
CREATE TABLE IF NOT EXISTS project (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    crs_storage TEXT NOT NULL DEFAULT 'EPSG:4326',
    crs_metric TEXT NOT NULL DEFAULT 'EPSG:9377',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS image (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    uri TEXT NOT NULL,
    source_id TEXT,
    acquired_at TIMESTAMP,
    sensor TEXT,
    gsd_m REAL,
    crs TEXT,
    transform_json TEXT,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    georef_method TEXT,
    georef_rmse REAL,
    provenance_json TEXT
);

CREATE TABLE IF NOT EXISTS session (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES image(id) ON DELETE CASCADE,
    backend TEXT NOT NULL,
    model_version TEXT NOT NULL,
    checkpoint TEXT,
    device TEXT NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prompt (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    object_id TEXT,
    type TEXT NOT NULL CHECK(type IN ('point', 'box', 'mask', 'text', 'exemplar', 'auto')),
    payload_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    latency_ms REAL
);

CREATE TABLE IF NOT EXISTS mask_version (
    id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL,
    version_no INTEGER NOT NULL,
    rle_json TEXT NOT NULL,
    score REAL,
    iou_pred REAL,
    parent_version_id TEXT REFERENCES mask_version(id),
    prompt_id TEXT REFERENCES prompt(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accepted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS object (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    image_id TEXT NOT NULL REFERENCES image(id) ON DELETE CASCADE,
    label TEXT,
    class_source TEXT CHECK(class_source IN ('manual', 'text_prompt', 'rule', NULL)),
    status TEXT NOT NULL DEFAULT 'borrador' CHECK(status IN ('borrador', 'aceptado', 'revisar', 'rechazado')),
    geom_4326_geojson TEXT,
    geom_4326_wkt TEXT,
    active_version_id TEXT REFERENCES mask_version(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS object_metric (
    object_id TEXT PRIMARY KEY REFERENCES object(id) ON DELETE CASCADE,
    crs_metric TEXT NOT NULL,
    area_m2 REAL,
    perimeter_m REAL,
    compactness REAL,
    rectangularity REAL,
    elongation REAL,
    orientation_deg REAL,
    solidity REAL,
    n_components INTEGER,
    n_holes INTEGER,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS validation (
    id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL REFERENCES object(id) ON DELETE CASCADE,
    reference_layer TEXT NOT NULL,
    ref_feature_id TEXT,
    iou REAL,
    boundary_f1 REAL,
    area_ratio REAL,
    method TEXT,
    at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS change_link (
    id TEXT PRIMARY KEY,
    object_a TEXT NOT NULL REFERENCES object(id) ON DELETE CASCADE,
    object_b TEXT NOT NULL REFERENCES object(id) ON DELETE CASCADE,
    delta_area_m2 REAL,
    pct_change REAL,
    centroid_shift_m REAL,
    method TEXT,
    at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS export_log (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    format TEXT NOT NULL,
    path TEXT NOT NULL,
    n_objects INTEGER NOT NULL,
    filters_json TEXT,
    at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(connection) -> None:
    """Ejecuta los statements DDL en la conexión SQLite / GeoPackage dada."""
    cursor = connection.cursor()
    cursor.executescript(DDL_STATEMENTS)
    connection.commit()
