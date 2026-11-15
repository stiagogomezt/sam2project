"""
Configuración central para SAM2_STREAMLIT_APP.
Constantes, límites, endpoints de teselas y configuración de modelos SAM 2.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Carpeta de pesos SAM 2 (configurable para despliegues con almacenamiento persistente).
CHECKPOINT_DIR = Path(os.environ.get("SAM2_CHECKPOINT_DIR", str(BASE_DIR)))

# ==========================================
# PARÁMETROS ESPACIALES Y TESELAS
# ==========================================
DEFAULT_BBOX = {
    "min_lon": -74.08300,
    "min_lat": 4.60800,
    "max_lon": -74.08000,
    "max_lat": 4.61100,
}

DEFAULT_ZOOM = 18
MIN_ZOOM = 14
MAX_ZOOM = 19

# Límites de seguridad para evitar saturación de RAM
MAX_TILES_LIMIT = 100
WARNING_TILES_LIMIT = 40
AVG_TILE_SIZE_KB = 250 # Tamaño promedio en KB de una tesela JPEG/PNG

# Proveedor de imágenes de alta resolución
ESRI_IMAGERY_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# ==========================================
# CONFIGURACIÓN SAM 2
# ==========================================
SAM2_MODELS = {
    # SAM 2.1 (092824): versión oficial mejorada de Meta; misma API y mismo coste que SAM 2.
    "sam2.1_hiera_tiny": {
        "name": "SAM 2.1 Tiny (Rápido - Recomendado CPU)",
        "checkpoint": str(CHECKPOINT_DIR / "sam2.1_hiera_tiny.pt"),
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    },
    "sam2.1_hiera_small": {
        "name": "SAM 2.1 Small (Equilibrado)",
        "checkpoint": str(CHECKPOINT_DIR / "sam2.1_hiera_small.pt"),
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    },
    "sam2.1_hiera_large": {
        "name": "SAM 2.1 Large (Máxima precisión - Requiere GPU)",
        "checkpoint": str(CHECKPOINT_DIR / "sam2.1_hiera_large.pt"),
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    },
    # SAM 2 (072824): versión original, conservada por compatibilidad.
    "sam2_hiera_tiny": {
        "name": "SAM 2 Tiny (versión 072824)",
        "checkpoint": str(CHECKPOINT_DIR / "sam2_hiera_tiny.pt"),
        "config": "configs/sam2/sam2_hiera_t.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    },
    "sam2_hiera_small": {
        "name": "SAM 2 Small (versión 072824)",
        "checkpoint": str(CHECKPOINT_DIR / "sam2_hiera_small.pt"),
        "config": "configs/sam2/sam2_hiera_s.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    },
    "sam2_hiera_large": {
        "name": "SAM 2 Large (versión 072824)",
        "checkpoint": str(CHECKPOINT_DIR / "sam2_hiera_large.pt"),
        "config": "configs/sam2/sam2_hiera_l.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    },
}

DEFAULT_SAM2_MODEL_KEY = "sam2.1_hiera_tiny"

# ==========================================
# SISTEMAS DE COORDENADAS DE REFERENCIA (CRS)
# ==========================================
CRS_GEOGRAPHIC = "EPSG:4326"
CRS_WEB_MERCATOR = "EPSG:3857"
# EPSG:9377 = MAGNA-SIRGAS 2018 / Origen-Nacional (proyección oficial única para Colombia, IGAC).
# EPSG:3116 = MAGNA-SIRGAS / Colombia Bogotá zone (zona antigua; se conserva como referencia).
CRS_COLOMBIA_METRIC = "EPSG:9377"
CRS_COLOMBIA_BOGOTA_ZONE = "EPSG:3116"

# Resolución de entrada del image encoder de SAM 2: toda imagen se redimensiona a 1024 x 1024 px.
SAM2_INPUT_SIZE = 1024
