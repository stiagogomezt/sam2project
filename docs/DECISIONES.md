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

---

## DEC-0005: Pesos compartidos, imagen por sesión
- **Fecha:** 2026-11-15
- **Estado:** Aceptado
- **Contexto:** `SAM2ImagePredictor` guarda la imagen activa en atributos privados (`_features`, `_orig_hw`). Con `st.cache_resource` el predictor se comparte entre sesiones y una sesión sobrescribía el embedding de otra.
- **Decisión:** `st.cache_resource(max_entries=1)` comparte solo los pesos. Cada sesión guarda su `ImageEmbedding` en `st.session_state`, y `SAM2Backend.predict` lo restaura bajo un `RLock` antes de llamar al decodificador. La app usa ahora el backend de `geosam/` (DEC-0002) en lugar de duplicar la inferencia en `src/`.
- **Verificación:** `tests/integration/test_sam2_real.py::test_sessions_do_not_overwrite_each_other`.

## DEC-0006: EPSG:9377 por área de uso, más medidas geodésicas
- **Fecha:** 2026-11-15
- **Estado:** Aceptado (sustituye el uso de EPSG:3116 en `src/geospatial.py`)
- **Decisión:** El CRS métrico es EPSG:9377 si el punto representativo del objeto cae en su área de uso oficial (pyproj), o la zona UTM correspondiente en otro caso. Además se reportan área y perímetro geodésicos (WGS84, `pyproj.Geod`) como referencia independiente de la proyección. En Bogotá, 9377 da ≈0,12 % menos área que la geodésica por su k₀ = 0,9992.

## DEC-0007: Simplificación de contorno de 1 píxel por defecto
- **Fecha:** 2026-11-15
- **Estado:** Aceptado
- **Contexto:** El polígono ráster en escalera sobrestima el perímetro ≈28 % (medido en círculos de radio 10 a 150 px), lo que distorsiona perímetro y compacidad.
- **Decisión:** Douglas-Peucker con tolerancia de 1 px (configurable de 0 a 3) antes de medir. Error de perímetro ≈1 % y de área < 1 %. `pixel_area_m2` conserva el área exacta de la máscara.

## DEC-0008: Etiquetas del usuario, no clases del modelo
- **Fecha:** 2026-11-15
- **Estado:** Aceptado
- **Decisión:** SAM 2 es agnóstico de clase. La etiqueta de cada objeto la escribe el usuario y se exporta como categoría COCO. El score se expone como `pred_iou`. Cualquier clasificación automática futura será un componente separado y documentado.
