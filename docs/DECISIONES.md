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

## DEC-0009: Clasificación complementaria como sugerencia (RemoteCLIP zero-shot)
- **Fecha:** 2026-11-15
- **Estado:** Aceptado
- **Contexto:** El usuario necesita categorizar objetos (p. ej. "Edificación religiosa"). SAM 2 no
  clasifica (DEC-0008).
- **Decisión:** Un componente separado (`src/semantic.py`) usa RemoteCLIP ViT-B/32 en modo zero-shot
  sobre el recorte del objeto. Devuelve las 3 categorías más probables, sumando varias descripciones
  por categoría. El usuario confirma; la fuente queda registrada en `label_source`.
- **Evidencia medida (Bogotá, Esri z19, 10 recortes):** primera opción correcta en ≈la mitad; correcta
  entre las 3 primeras en 8 de 9 evaluables. Confusiones: techo azul ↔ agua, cancha ↔ césped.
- **Coste:** pesos de 605 MB (descarga única), ≈80 ms por objeto en CPU y ≈+600 MB de RSS, casi todo
  páginas del archivo mapeado (pesos con mmap + assign; la torre de texto se libera tras codificar
  las descripciones).
- **Alternativa futura:** un clasificador supervisado entrenado con el dataset COCO exportado.

## DEC-0010: Huella de edificaciones compuestas
- **Fecha:** 2026-11-15
- **Estado:** Aceptado
- **Contexto:** Un clic segmenta una parte homogénea (techo, patio). Las edificaciones compuestas
  necesitan caja + puntos, y sus patios inflan el perímetro.
- **Decisión:** Post-proceso opcional y explícito (rellenar huecos, cierre morfológico, parte
  principal) aplicado a la máscara antes de vectorizar. Se reportan el perímetro exterior y el total.
- **Evidencia:** convento completo con SAM 2.1 Large, caja + 5 puntos + patios rellenados → una sola
  parte, IoU predicho 0,936, ≈15 300 m², perímetro exterior ≈700 m (en la app, E2E).

## DEC-0011: Clasificador supervisado ligero (GEOBIA) sin dependencias nuevas
- **Fecha:** 2026-11-15
- **Estado:** Aceptado
- **Decisión:** Regresión logística multinomial con L2 y clases balanceadas, entrenada con L-BFGS
  en PyTorch (ya instalado; se evita scikit-learn). Descriptores: 6 de forma (log área, compacidad,
  solidez, rectangularidad, elongación, log P/√A) y 10 de color (medias y desviaciones RGB, tono
  ponderado por saturación, saturación, brillo). RemoteCLIP (512) es opcional.
- **Motivo:** con decenas de muestras un modelo lineal regularizado generaliza mejor que modelos
  más complejos. Medido en 71 objetos reales: forma + color 83 % (kappa 0,77) frente a +RemoteCLIP
  82 %; RemoteCLIP solo, 79 %.
- **Reglas:** solo cuentan como muestras los objetos aceptados con categoría confirmada. Mínimo
  2 clases con 3 o más muestras (recomendado 10 o más). Las predicciones quedan "sin confirmar".
