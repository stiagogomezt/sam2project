"""Zonas de ejemplo en Colombia para demostraciones (ortoimagen y Sentinel-2)."""
from typing import Dict

EXAMPLES: Dict[str, Dict] = {
    "Usme · borde urbano-rural (Bogotá)": {
        "source": "Ortoimagen", "zoom": 18, "bbox": (-74.1262, 4.4728, -74.1207, 4.4783),
        "nota": "Construcciones, vías, pastos y bosque en el borde sur de Bogotá.",
    },
    "Facatativá · floricultura (Cundinamarca)": {
        "source": "Ortoimagen", "zoom": 17, "bbox": (-74.322, 4.808, -74.312, 4.818),
        "nota": "Invernaderos, pastos, cultivos y relictos de bosque en la Sabana.",
    },
    "Madrid · expansión urbana (Cundinamarca)": {
        "source": "Ortoimagen", "zoom": 17, "bbox": (-74.272, 4.735, -74.262, 4.745),
        "nota": "Vivienda nueva, industria, humedales y suelo agrícola.",
    },
    "PNN Tinigua · deforestación (Meta)": {
        "source": "Sentinel-2", "bbox": (-74.11, 2.38, -74.01, 2.48), "start": "2025-01-01", "end": "2025-02-28",
        "nota": "Potreros abiertos en la selva dentro del parque nacional.",
    },
    "Embalse de Tominé (Cundinamarca)": {
        "source": "Sentinel-2", "bbox": (-73.89, 4.91, -73.79, 5.01), "start": "2025-01-01", "end": "2025-02-28",
        "nota": "Agua, bosque, pastos y el casco urbano de Guatavita.",
    },
    "Río Magdalena · Barrancabermeja (Santander)": {
        "source": "Sentinel-2", "bbox": (-73.90, 7.02, -73.80, 7.12), "start": "2025-01-01", "end": "2025-03-31",
        "nota": "Río, ciénagas, ciudad y refinería.",
    },
}
