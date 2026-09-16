# Registro de Licencias y Atribución — GEO-SAM

En cumplimiento con la **Regla de Trabajo 7 (§0)** y la Sección 5.4 de la especificación técnica, este documento registra las licencias de software, modelos fundacionales y datos geoespaciales empleados en la plataforma.

---

## 1. Modelos Fundacionales de Segmentación

### Meta Segment Anything 2 (SAM 2 / SAM 2.1)
- **Autor:** Meta AI Research
- **Repositorio:** `facebookresearch/sam2`
- **Código:** Apache License 2.0
- **Pesos del modelo:** Meta SAM License / Non-commercial research.
- **Atribución:** Ravi et al., "SAM 2: Segment Anything in Images and Videos", Meta AI, 2024.

### Meta Segment Anything 3 (SAM 3 / 3.1)
- **Autor:** Meta AI Research
- **Repositorio:** `facebookresearch/sam3` / Hugging Face `facebook/sam3`
- **Código:** Apache License 2.0 / SAM 3 License
- **Pesos del modelo:** Sujeto a solicitud de acceso y términos de Meta SAM 3 License (Hugging Face Auth requerido).
- **Atribución:** Meta AI, "Segment Anything 3: Promptable Concept Segmentation", 2025/2026.

---

## 2. Librerías de Código Abierto (Core)

| Componente | Licencia | Uso en GEO-SAM |
|---|---|---|
| `rasterio` | BSD 3-Clause | Lectura y manipulación de rásteres geoespaciales y transformadas afines |
| `shapely` | BSD 3-Clause | Operaciones geométricas vectoriales y validación topológica |
| `geopandas` | BSD 3-Clause | Gestión de geodataframes y persistencia GIS |
| `pyproj` | MIT | Reproyección geodésica y selección de CRS métricos |
| `pystac-client` | Apache 2.0 | Búsqueda y consulta en catálogos STAC |
| `click` | BSD 3-Clause | Interfaz de línea de comandos CLI |
| `pytest` | MIT | Marco de pruebas automatizadas |

---

## 3. Fuentes de Datos Geoespaciales Abiertos (Provisión Fase 5–6)

- **Copernicus Data Space Ecosystem (Sentinel-1 / Sentinel-2):**
  - Licencia: Términos de uso de datos abiertos de Copernicus (acceso libre y gratuito con atribución a la Unión Europea).
  - Atribución obligatoria: *"Contains modified Copernicus Sentinel data [año]"*.
- **Overture Maps Foundation (Buildings):**
  - Licencia: Community Data License Agreement – Permissive (CDLA-Permissive 2.0).
- **OpenStreetMap (Overpass API):**
  - Licencia: Open Database License (ODbL) 1.0. Atribución obligatoria: *"© OpenStreetMap contributors"*.
- **Datos Abiertos Colombia (IGAC, IDECA, DANE, datos.gov.co):**
  - Licencia: Ley 1712 de 2014 / Datos Abiertos del Estado Colombiano (reutilización libre con citación de la entidad emisora).
