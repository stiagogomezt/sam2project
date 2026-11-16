# Registro de cambios

## [2.1.0] — 2026-11-15 · Edificaciones compuestas, RemoteCLIP, rediseño GIS y carga de modelos

Motivo: al segmentar un convento con un clic, SAM 2 tomaba solo una parte (nave o claustro) y el
perímetro incluía los bordes de los patios. Además, la interfaz requería jerarquía profesional SIG
y control explícito con progreso real al descargar o cambiar modelos SAM 2 / SAM 2.1.

### Añadido
- `src/ui.py` y `.streamlit/config.toml` (designoreado en `.gitignore`, ignorando solo `secrets.toml`): sistema de diseño GIS (petróleo/slate claro y oscuro), cabecera compacta, *stepper* horizontal de 5 etapas, ficha técnica de modelos, cuadrícula de 3 mini-indicadores y exportación agrupada en *Vectorial*, *Ráster* y *Dataset*.
- Progreso y control de modelos (`src/sam_engine.py`, `config.py`): `ensure_checkpoint(..., progress_callback)` reporta MB descargados/totales; `GEOSAM_MAX_MODELS` (defecto `1`) controla `max_entries` en `@st.cache_resource`; protección en CPU para variantes *Large* (habilitable con GPU CUDA o `GEOSAM_ALLOW_LARGE_CPU=1`); medición de tiempos reales con `time.perf_counter()`.
- `src/postprocess.py`: forma de la huella (rellenar patios, unir partes separadas, solo la parte
  principal, fragmentos mínimos) aplicada a la máscara, de modo que vector, máscara y exportaciones
  sigan siendo coherentes.
- Métrica `perimeter_ext_m` (perímetro exterior, sin patios), que el panel muestra como métrica principal.
- Aviso cuando el objeto toca el borde de la ortofoto (posible recorte), visible en el panel y en el
  detalle; nuevo atributo `touches_edge`.
- Fusión de objetos del catálogo (p. ej. nave + claustro + alas) con recálculo de métricas.
- Categorías: lista catastral/urbana con posibilidad de escribir una nueva, más **sugerencia
  zero-shot con RemoteCLIP** (`src/semantic.py`) y sugerencia en lote para objetos sin categoría.
  La fuente se registra en `label_source` ("usuario" o "RemoteCLIP (confirmada por el usuario)").
- Consejo en el panel para edificaciones compuestas: caja + puntos.
- Pruebas: `tests/unit/test_postprocess_and_merge.py`, `tests/unit/test_sam_engine_model_ux.py`,
  `tests/integration/test_semantic_real.py` y E2E `test_composite_building_shape_and_category`.

### Cambiado
- El cambio de modelo en el selector muestra su ficha técnica sin recargar el motor hasta pulsar **Cargar modelo** (`st.status` en lugar del bloqueo gris global de página).
- Íconos Material (`:material/...:`) consistentes en botones, barra lateral y pestañas; miniaturas cuadradas uniformes de candidatas.
- La etiqueta libre pasa a ser un selector "Categoría (la asigna usted; SAM 2 no clasifica)".
- `requirements.txt`: se añade `open_clip_torch`.

## [2.0.0] — 2026-11-15 · Estabilización, multiobjeto y análisis geoespacial

### Corregido
- **`geosam/backends/sam2.py`**: con `return_logits=True` los logits se convertían a `bool` sin umbral y las tres máscaras cubrían el 100 % de la imagen. Ahora se umbraliza con `mask_threshold` (0,0).
- **Embeddings compartidos entre usuarios** (`src/sam_engine.py`): el predictor guardaba la imagen activa en un singleton global. Ahora cada sesión conserva su `ImageEmbedding` y el backend lo restaura bajo un lock.
- **Cambio de modelo con una imagen cargada**: el motor nuevo no tenía imagen y la inferencia fallaba. Ahora el embedding se recalcula y solo queda un modelo en RAM.
- **Re-disparo del clic al descartar** y **pérdida de zoom** en cada clic (`app.py`, `src/visualization.py`).
- **Descargas que fallaban al pulsar varios botones seguidos** (identificadores diferidos revocados por el rerun): ahora usan `on_click="ignore"`.
- `min_pixel_area` no se usaba; `datetime.utcnow()` obsoleto; `subdomains` sin efecto en la capa Esri.
- Shapefile: GDAL renombraba `centroid_lat` / `centroid_lon` a `centroid_l` / `centroid_1`. Ahora hay nombres DBF explícitos y un archivo `campos.csv`.
- El centroide se calculaba en grados; ahora se calcula en el CRS métrico.
- El comentario de `EPSG:3116` decía "Origen Nacional"; corresponde a la zona Bogotá.

### Cambiado
- CRS métrico de Colombia: **EPSG:9377** (Origen-Nacional, oficial), seleccionado por su área de uso. UTM fuera de Colombia.
- El score `confidence` pasa a llamarse **`pred_iou`** (es IoU predicho, no probabilidad).
- Modelo por defecto: **SAM 2.1 Tiny**. Los modelos SAM 2 (072824) se conservan.
- `requirements.txt`: versiones mínimas de `streamlit` (1.67) y `streamlit-folium` (0.27) por las APIs usadas. Nuevo `requirements-dev.txt`.

### Añadido
- Prompts: puntos ➕/➖, caja con 2 clics, refinamiento con logits, selector de 3 candidatas y deshacer (`src/interactive.py`).
- Catálogo de objetos con estados, visibilidad, edición en tabla, refinado con versiones y RLE (`src/objects.py`).
- Métricas: área y perímetro geodésicos, convex hull, solidez, compacidad, rectangularidad, elongación, relación de aspecto, orientación, partes, huecos y GSD. Simplificación de contorno configurable.
- Relaciones espaciales (vecino más cercano, contactos y pares DE-9IM) y estadísticas por etiqueta.
- Exportación: GeoPackage multicapa, GeoTIFF de máscaras, dataset COCO y CSV conectado a la UI.
- Segmentación automática (`SAM2AutomaticMaskGenerator`) con resultados como objetos *propuestos*.
- Informe de teselas fallidas y resolución efectiva de SAM 2 en la barra lateral.
- Pruebas: `tests/unit/test_geometry_metrics.py`, `tests/unit/test_interactive_and_catalog.py`, `tests/integration/test_sam2_real.py` (modelo real) y `tests/e2e/test_ui_e2e.py` (Playwright).
- Documentación: `docs/ESTADO.md`, nuevas decisiones en `docs/DECISIONES.md` y README actualizado.

## [1.0.0] — 2cf0231
Prototipo de un objeto por clic con exportación GeoJSON, GeoTIFF, PNG y Shapefile.
