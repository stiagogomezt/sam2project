# Estado de funcionalidades — SAM 2 GeoExtractor

Fecha: 2026-11-15 · Rama: `mejoras/estabilizacion-y-multiobjeto`

Leyenda de evidencia: **U** prueba unitaria · **I** integración con SAM 2 real · **E** prueba E2E en navegador real · **M** medición manual documentada.

## Terminado y verificado

### Fase 1 — Estabilización
| Funcionalidad | Archivo | Evidencia |
|---|---|---|
| Embeddings por sesión (sin cruce entre usuarios) + lock | `src/sam_engine.py`, `geosam/backends/sam2.py` | I `test_sessions_do_not_overwrite_each_other` (falla con el código anterior, pasa con el nuevo) |
| `SAM2Backend` corregido (antes devolvía máscaras del 100 % de la imagen) | `geosam/backends/sam2.py` | I `test_point_prompt_segments_known_object` |
| Cambio de modelo con imagen cargada (recalcula el embedding; antes fallaba) | `app.py`, `src/sam_engine.py` | I `test_embedding_from_other_model_is_rejected`, E `test_model_switch_and_automatic` |
| Un solo modelo en RAM por defecto (`GEOSAM_MAX_MODELS`, default `1`) + bloqueo `Large` en CPU (`GEOSAM_ALLOW_LARGE_CPU`) | `src/sam_engine.py`, `app.py` | U (`test_sam_engine_model_ux.py`), M |
| Carga explícita de modelo (`Cargar modelo`), ficha técnica, progreso por pasos (`st.status`) y rediseño GIS (`src/ui.py`) | `app.py`, `src/sam_engine.py`, `src/ui.py` | U, E (`test_model_switch_and_automatic`) |
| Aviso rojo permanente en modo simulado + `backend` registrado por objeto | `app.py`, `src/objects.py` | U |
| Descarga atómica de pesos con verificación de tamaño | `src/sam_engine.py` | M (descarga real de 2 checkpoints) |
| SAM 2.1 (Tiny/Small/Large) además de SAM 2 | `config.py` | I (ambas versiones Tiny) |
| Visor = imagen procesada (`max_native_zoom` = zoom de descarga) | `src/visualization.py` | E |
| Zoom y encuadre conservados tras cada clic | `src/visualization.py`, `app.py` | E |
| Descartar no vuelve a disparar el clic | `app.py` | E |
| Teselas fallidas detectadas y avisadas | `src/imagery_loader.py` | M |
| Resolución efectiva de SAM 2 (reducción a 1024 px) informada | `app.py` | M |
| Clic → píxel: convención floor + 0,5 interno de SAM verificada | `src/segmentation.py` | M (`rowcol` usa floor) |

### Fase 2 — Explotación de SAM 2
| Funcionalidad | Origen | Evidencia |
|---|---|---|
| Puntos positivos y negativos acumulables | SAM 2 (oficial) | I, E |
| Caja con 2 clics | prompt oficial, interacción propia | I, E (5/5 repeticiones) |
| Refinamiento iterativo con logits (`mask_input`) | SAM 2 (oficial) | I `test_iterative_refinement_with_logits` |
| 3 candidatas + selector visual con IoU predicho | oficial + UI propia | E |
| Deshacer (reproducción determinista) | propio | U, E |
| Segmentación automática → objetos *propuestos* | `SAM2AutomaticMaskGenerator` (oficial) | E (8×8: 8 objetos en ≈16 s) |

### Fase 3 — Administración de objetos
ID único, etiqueta editable, estados aceptado/propuesto/rechazado, visibilidad, centrar, refinar (con versión), eliminar, eliminar rechazados y máscaras en RLE COCO. Evidencia: U `test_catalog_lifecycle_and_exports`; E edición de visibilidad en la tabla, refinado a v2.

### Fase 4 — Análisis geoespacial
Vectorización con filtro de área mínima (antes el parámetro no se usaba), simplificación de 1 px contra el efecto escalera, EPSG:9377 o UTM, área y perímetro planares y geodésicos, centroide calculado en el CRS métrico, bbox, convex hull, solidez, compacidad, rectangularidad, elongación, relación de aspecto, orientación, partes y huecos, GSD real. También relaciones espaciales (vecino más cercano, contactos y pares con relación DE-9IM). Evidencia: U `tests/unit/test_geometry_metrics.py` (rectángulo, rectángulo rotado, círculo, forma en L, error de perímetro, área frente a pyproj.Geod).

### Fase 5 — Explotación
GeoJSON, GeoPackage multicapa, Shapefile con nombres legibles, CSV, GeoTIFF de la ortofoto y de las máscaras, dataset COCO y PNG. También estadísticas por estado y por etiqueta, con gráfico de áreas. Evidencia: U y E (los 7 archivos descargados desde el navegador y validados con geopandas, pyogrio y rasterio).

### v2.1 — Edificaciones compuestas y categorías
| Funcionalidad | Evidencia |
|---|---|
| Forma de la huella (patios, unión, parte principal) aplicada a la máscara | U `test_postprocess_and_merge.py` |
| Perímetro exterior (`perimeter_ext_m`) | U |
| Aviso de objeto recortado por el borde | U, M (convento en la zona por defecto) |
| Fusión de objetos | U `test_catalog_merge_with_postprocess` |
| Categorías + sugerencia RemoteCLIP (individual y en lote) | I `test_semantic_real.py`, E `test_composite_building_shape_and_category` |
| Convento completo en la app (caja + 5 puntos + patios, Large) | E manual: 1 parte, IoU pred. 0,936, ≈15 340 m², perím. ext. ≈698 m |

## Pendiente

| Elemento | Situación | Siguiente paso |
|---|---|---|
| Persistencia entre sesiones | El catálogo vive en `st.session_state` y se pierde al cerrar la pestaña. `geosam/store` (SQLite + versiones) existe pero no está conectado | Guardar y cargar el proyecto `.gpkg`/SQLite usando `GeoSamRepository` |
| Ventanas de trabajo para mosaicos grandes | Por encima de 1024 px, SAM 2 pierde resolución efectiva (se avisa) | Embedding por ventana de 1024 px centrada en el primer prompt |
| CLI `cli/main.py` | Comandos declarados que solo imprimen texto | Conectar con `src/` y `geosam/` |
| `src/classifier.py` | No conectado: necesita banda NIR (4 bandas) y Esri entrega RGB. La clasificación sobre RGB la cubre ahora `src/semantic.py` (sugerencias) | Clasificador supervisado entrenado con el dataset COCO exportado |
| SAM 3 (`geosam/backends/sam3.py`) | Esqueleto | Integrar cuando haya checkpoint y licencia verificados |
| Fase 6: multitemporal, detección de cambios, propagación/tracking, fine-tuning | No implementado | Ver la propuesta de diseño abajo |
| Caché de embeddings en disco | `EmbeddingCache.cache_dir` declarado pero sin implementar | Serializar `_features` con `torch.save` por hash |

## Fase 6 — Diseño preparado (no implementado)

- **Multitemporal:** cargar dos mosaicos del mismo BBOX (p. ej. Esri Wayback u ortofotos propias) con la misma transformación afín. Los objetos se emparejan por solape (IoU geográfico), y la tabla `change_link` de `geosam/store/schema.py` ya modela el vínculo.
- **Propagación entre fechas:** el `SAM2VideoPredictor` oficial trata una pila de imágenes coregistradas como fotogramas y propaga la máscara de la fecha 1 a la fecha 2. Solo tiene sentido con coregistro preciso y cambios moderados; hay que validarlo antes de exponerlo.
- **Clasificación semántica:** debe ser un componente aparte (SAM 2 no clasifica). El dataset COCO exportado, con etiquetas del usuario, sirve de entrenamiento.
- **Fine-tuning:** el dataset COCO (imagen + RLE + georreferencia) es la entrada; el entrenamiento requiere GPU fuera de Streamlit Cloud.

## Limitaciones conocidas

- Un clic hecho mientras la app aún se reejecuta puede perderse (interacción Streamlit ↔ Folium). Los contadores ➕ ➖ ▭ del panel confirman cada prompt registrado.
- La orientación es acimut de cuadrícula del CRS, no norte verdadero (la diferencia por convergencia de meridianos es despreciable a escala de objeto).
- Esri World Imagery puede devolver teselas de "sin datos" con código HTTP 200; esas no se detectan como fallidas.
