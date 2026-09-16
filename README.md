# 🛰️ SAM 2 GeoExtractor (SAM2_STREAMLIT_APP)

> **Herramienta SIG para la Extracción Inteligente de Objetos Geoespaciales a partir de Imágenes Aéreas y Satelitales de Alta Resolución mediante Segment Anything Model 2 (SAM 2).**

---

## 📌 Descripción General

**SAM2_STREAMLIT_APP** es una plataforma de análisis geoespacial interactiva desarrollada en **Python** y **Streamlit**. Permite a analistas SIG, ingenieros de teledetección y científicos de datos:

1. **Definir Zonas de Estudio**: Configuración y validación matemática de Bounding Box (WGS84).
2. **Descarga Continua de Ortofotos**: Descarga multihilo de teselas de alta resolución (Esri World Imagery) y ensamblado de mosaico georreferenciado sin desfase espacial.
3. **Mapeo Sub-Píxel de Clics**: Conversión matemática rigurosa entre Leaflet/Folium y la matriz ráster nativa (**EPSG:3857**).
4. **Segmentación con IA (SAM 2)**: Inferencia interactiva con pre-cómputo de embeddings para segmentación instantánea al clic.
5. **Control de Calidad (Aceptar/Reintentar)**: Validación interactiva del objeto antes de su persistencia.
6. **Extracción y Cálculo Espacial**: Recorte con canal alfa (RGBA) y cálculo de superficie ($m^2$) y perímetro ($m$) en proyecciones métricas oficiales (**EPSG:3116** / UTM).
7. **Exportación Multiformato**: GeoJSON vectorial con metadatos enriquecidos, GeoTIFF georreferenciado, PNG transparente, Shapefile (.zip) y CSV.

---

## 🏗️ Arquitectura Modular

```
sam2_streamlit_app/
├── config.py                 # Parámetros espaciales, BBOX por defecto y configs de SAM 2
├── app.py                    # Interfaz web interactiva en Streamlit
├── test_pipeline.py          # Suite de pruebas automatizadas
├── requirements.txt          # Dependencias del proyecto
└── src/
    ├── tile_manager.py       # Cálculo de teselas y validación de BBOX
    ├── imagery_loader.py     # Descarga concurrente con reintentos HTTP
    ├── mosaic_builder.py     # Ensamblado de mosaico y georreferenciación Afín en EPSG:3857
    ├── highres_loader.py     # Adaptador retrocompatible
    ├── sam_engine.py         # Inferencia SAM 2, detección CUDA/CPU y cache
    ├── segmentation.py       # Mapeo de coordenadas y recorte con canal alfa
    ├── geospatial.py         # Vectorización de máscaras y cálculo métrico (m², perímetro)
    ├── exporter.py           # Generación de GeoJSON, GeoTIFF, PNG, Shapefile y CSV
    └── visualization.py      # Mapas interactivos con Folium
```

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el Repositorio
```bash
git clone https://github.com/stiagogomezt/sam2project.git
cd sam2project
```

### 2. Entorno Virtual y Dependencias
```bash
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Pesos del Modelo SAM 2
La aplicación descarga automáticamente los pesos del modelo configurado (`sam2_hiera_tiny.pt`), o puedes descargarlos manualmente:
- [sam2_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt)
- [sam2_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
- [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

### 4. Ejecutar la Aplicación
```bash
streamlit run app.py
```

---

## 🧪 Pruebas Automatizadas

Ejecutar la suite de pruebas geoespaciales:
```bash
python test_pipeline.py
```

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
