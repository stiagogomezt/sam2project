# Registro de Decisiones de Arquitectura e Ingeniería (ADR) — GEO-SAM

## DEC-0001: Desacoplamiento del Núcleo Analítico y Decisión de UI
- **Fecha:** 2026-09-16
- **Estado:** Aceptado
- **Contexto:**
  El documento maestro advierte sobre las limitaciones intrínsecas de Streamlit al trabajar con lienzos geoespaciales interactivos (re-ejecución del script en cada interacción, latencia de renderizado, y dificultad para gestionar pilas de deshacer/rehacer y multimask complejas).
- **Decisión:**
  1. El paquete `geosam/` es 100% independiente de cualquier framework web o UI. Todas las operaciones de cálculo, persistencia, vectorización y métricas son ejecutables por CLI o scripts de Python.
  2. Como destino de interfaz interactiva, se adopta formalmente la **Opción A: Arquitectura Desacoplada (FastAPI + Cliente Web MapLibre GL / Leaflet)**. Esto permite interacción de baja latencia (<200 ms), manipulación fluida de puntos y cajas, y control de historial `Ctrl+Z` en memoria de cliente.
  3. Se mantiene compatibilidad transitoria para pruebas rápidas en notebooks o scripts aislados.

---

## DEC-0002: Inferencia Desacoplada mediante Protocolo `SegmentationBackend`
- **Fecha:** 2026-09-16
- **Estado:** Aceptado
- **Contexto:**
  El sistema necesita soportar SAM 2 / 2.1 (prompts geométricos finos de puntos/cajas), SAM 3 (Promptable Concept Segmentation por texto natural y ejemplares visuales), y ejecución en entornos CI sin GPU ni descarga de pesos gigamétricos.
- **Decisión:**
  Se define el protocolo formal `@runtime_checkable` `SegmentationBackend` en `geosam/backends/protocol.py`. Ningún módulo de vectorización, métricas o interfaz importa directamente librerías de `torch` o `sam2`/`sam3`. Se implementa `MockBackend` determinista para cubrir el 100% de los flujos de pruebas unitarias en CPU.

---

## DEC-0003: Almacenamiento Inmutable de Máscaras en RLE (COCO)
- **Fecha:** 2026-09-16
- **Estado:** Aceptado
- **Contexto:**
  Guardar máscaras como archivos PNG individuales consume gigabytes de disco para pocos cientos de objetos y dificulta la trazabilidad de versiones iterativas.
- **Decisión:**
  Se implementa codificación Run-Length (RLE) en orden columna (Fortran) compatible con COCO en `geosam/utils/rle.py`. Las máscaras se almacenan como cadenas JSON en la tabla `mask_version` junto con su `parent_version_id`, asegurando que ninguna versión de máscara se sobrescriba (Invariante de inmutabilidad).

---

## DEC-0004: Prohibición de Cálculo Métrico en Coordenadas Geográficas
- **Fecha:** 2026-09-16
- **Estado:** Aceptado
- **Contexto:**
  Calcular áreas o longitudes con `shapely` en grados (EPSG:4326) produce valores físicamente absurdos.
- **Decisión:**
  Toda geometría debe ser proyectada a un CRS métrico antes de calcular métricas (por defecto EPSG:9377 MAGNA-SIRGAS Origen Nacional para Colombia o UTM zonal para otras regiones). Se codificarán pruebas que fallen explícitamente si se detecta cálculo en grados.
