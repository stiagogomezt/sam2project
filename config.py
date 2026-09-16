"""
Configuración central para SAM2_STREAMLIT_APP.
Constantes, límites, endpoints de teselas y configuración de modelos SAM 2.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

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
    "sam2_hiera_tiny": {
        "name": "SAM 2 Tiny (Rápido - Recomendado CPU)",
        "checkpoint": str(BASE_DIR / "sam2_hiera_tiny.pt"),
        "config": "configs/sam2/sam2_hiera_t.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    },
    "sam2_hiera_small": {
        "name": "SAM 2 Small (Equilibrado)",
        "checkpoint": str(BASE_DIR / "sam2_hiera_small.pt"),
        "config": "configs/sam2/sam2_hiera_s.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    },
    "sam2_hiera_large": {
        "name": "SAM 2 Large (Máxima precisión - Requiere GPU)",
        "checkpoint": str(BASE_DIR / "sam2_hiera_large.pt"),
        "config": "configs/sam2/sam2_hiera_l.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    },
}

DEFAULT_SAM2_MODEL_KEY = "sam2_hiera_tiny"

# ==========================================
# SISTEMAS DE COORDENADAS DE REFERENCIA (CRS)
# ==========================================
CRS_GEOGRAPHIC = "EPSG:4326"
CRS_WEB_MERCATOR = "EPSG:3857"
CRS_COLOMBIA_METRIC = "EPSG:3116" # MAGNA-SIRGAS Origen Nacional
