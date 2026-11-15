"""
Módulo de exportación multiformato para objetos geoespaciales y datos derivados.
Formatos: GeoJSON, CSV, Shapefile (.zip), GeoPackage (multicapa), GeoTIFF (ortofoto RGB/RGBA y
ráster de etiquetas), PNG transparente y dataset COCO (imagen + anotaciones RLE + georreferencia).
"""
import io
import json
import os
import zipfile
import tempfile
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence
import numpy as np
from PIL import Image
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from config import CRS_GEOGRAPHIC, CRS_WEB_MERCATOR

def export_geojson(gdf: gpd.GeoDataFrame) -> str:
    """Exporta el GeoDataFrame como cadena formateada en formato GeoJSON."""
    return gdf.to_json(indent=2)

def export_csv(gdf: gpd.GeoDataFrame) -> str:
    """Exporta los atributos del GeoDataFrame (sin la columna geometría) en formato CSV."""
    df_no_geom = gdf.drop(columns=["geometry"]) if "geometry" in gdf.columns else gdf
    return df_no_geom.to_csv(index=False)

def export_png(pil_image: Image.Image) -> bytes:
    """Exporta una imagen PIL a bytes en formato PNG."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()

def export_geotiff(
    img_array: np.ndarray,
    transform: rasterio.Affine,
    crs_str: str = CRS_WEB_MERCATOR,
    mask: Optional[np.ndarray] = None
) -> bytes:
    """
    Genera un GeoTIFF georreferenciado en memoria.
    Si se proporciona una máscara, se incluye como cuarto canal (Alfa) para preservar la extracción.
    """
    h, w = img_array.shape[:2]
    
    if mask is not None:
        # 4 canales: RGB + Alpha
        alpha = (mask.astype(bool) * 255).astype(np.uint8)
        rgba = np.dstack((img_array, alpha))
        count = 4
        data = np.moveaxis(rgba, -1, 0) # (4, H, W)
    else:
        # 3 canales: RGB
        count = 3
        data = np.moveaxis(img_array, -1, 0) # (3, H, W)
        
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=h,
        width=w,
        count=count,
        dtype=np.uint8,
        crs=crs_str,
        transform=transform,
        compress="deflate"
    ) as dst:
        dst.write(data)
        
    return memfile.read()

# Nombres DBF (máx. 10 caracteres) explícitos y legibles; evita el "lavado" automático de GDAL
# (centroid_lat → centroid_l, centroid_lon → centroid_1).
SHAPEFILE_FIELD_MAP = {
    "object_id": "obj_id", "label": "label", "status": "status", "pred_iou": "pred_iou",
    "area_m2": "area_m2", "perimeter_m": "perim_m", "perimeter_ext_m": "perim_ext", "area_geod_m2": "area_geod",
    "perimeter_geod_m": "perim_geod", "pixel_area_m2": "px_area_m2",
    "pixel_count": "px_count", "hull_area_m2": "hull_m2", "solidity": "solidity",
    "compactness": "compact", "rectangularity": "rectang", "elongation": "elong",
    "aspect_ratio": "aspect", "orientation_deg": "orient_deg", "mrr_length_m": "mrr_len_m",
    "mrr_width_m": "mrr_wid_m", "n_parts": "n_parts", "n_holes": "n_holes",
    "centroid_lat": "cent_lat", "centroid_lon": "cent_lon", "min_lon": "min_lon",
    "min_lat": "min_lat", "max_lon": "max_lon", "max_lat": "max_lat",
    "metric_crs": "metric_crs", "gsd_m": "gsd_m", "timestamp": "timestamp",
    "model_id": "model_id", "backend": "backend", "origin": "origin", "version": "version",
    "raster_value": "rast_val", "label_source": "label_src", "touches_edge": "edge_cut", "nn_id": "nn_id", "nn_dist_m": "nn_dist_m", "n_contacts": "n_contact",
}


def shapefile_field_names(columns: Iterable[str]) -> Dict[str, str]:
    """Mapa columna → nombre DBF único de ≤10 caracteres."""
    mapping: Dict[str, str] = {}
    used: set = set()
    for col in columns:
        if col == "geometry":
            continue
        base = SHAPEFILE_FIELD_MAP.get(col, col)[:10]
        name, k = base, 1
        while name.lower() in used:
            suffix = str(k)
            name = base[: 10 - len(suffix)] + suffix
            k += 1
        used.add(name.lower())
        mapping[col] = name
    return mapping


def export_shapefile_zip(gdf: gpd.GeoDataFrame, base_name: str = "objetos_sam2") -> bytes:
    """
    Empaqueta el GeoDataFrame como ESRI Shapefile (.shp/.shx/.dbf/.prj/.cpg) en un .zip, con
    nombres de campo cortos explícitos y un diccionario `campos.csv` (nombre DBF → nombre completo).
    """
    mapping = shapefile_field_names(gdf.columns)
    out = gdf.rename(columns=mapping)
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = os.path.join(tmpdir, f"{base_name}.shp")
        out.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        with open(os.path.join(tmpdir, "campos.csv"), "w", encoding="utf-8") as fh:
            fh.write("campo_dbf,campo_completo\n")
            for full, short in mapping.items():
                fh.write(f"{short},{full}\n")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(tmpdir):
                full_path = os.path.join(tmpdir, fname)
                zf.write(full_path, arcname=fname)
                
        buf.seek(0)
        return buf.getvalue()


def export_geopackage(layers: Dict[str, gpd.GeoDataFrame]) -> bytes:
    """GeoPackage (.gpkg) con una capa por entrada de `layers` (se omiten las capas vacías)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sam2_geoextractor.gpkg")
        written = 0
        for name, layer in layers.items():
            if layer is None or layer.empty:
                continue
            layer.to_file(path, layer=name, driver="GPKG")
            written += 1
        if written == 0:
            raise ValueError("No hay capas con datos para exportar.")
        with open(path, "rb") as fh:
            return fh.read()


def export_label_geotiff(
    masks: Sequence[np.ndarray],
    values: Sequence[int],
    transform: rasterio.Affine,
    crs_str: str = CRS_WEB_MERCATOR,
) -> bytes:
    """Ráster de etiquetas georreferenciado (uint16): 0 = fondo, `values[i]` = objeto i.

    Si dos máscaras se solapan prevalece la última de la lista.
    """
    if not masks:
        raise ValueError("No hay máscaras para exportar.")
    h, w = masks[0].shape
    label = np.zeros((h, w), dtype=np.uint16)
    for m, v in zip(masks, values):
        label[np.asarray(m, dtype=bool)] = int(v)
    memfile = MemoryFile()
    with memfile.open(driver="GTiff", height=h, width=w, count=1, dtype=np.uint16, crs=crs_str,
                      transform=transform, compress="deflate", nodata=0) as dst:
        dst.write(label, 1)
        dst.set_band_description(1, "object_raster_value")
    return memfile.read()


def export_coco_dataset(
    img_array: np.ndarray,
    objects: List[dict],
    transform: rasterio.Affine,
    crs_str: str = CRS_WEB_MERCATOR,
    image_name: str = "mosaico.png",
) -> bytes:
    """Dataset propio en formato COCO (instancias) empaquetado en .zip.

    `objects`: dicts con `object_id`, `label`, `mask_rle` (RLE COCO sin comprimir, orden Fortran),
    `bbox_px` (xmin, ymin, xmax, ymax), `pixel_count`, `pred_iou` y `area_m2`.
    Las categorías son las ETIQUETAS ASIGNADAS POR EL USUARIO (SAM 2 no clasifica).
    Incluye `georreferencia.json` (transformación afín + CRS) para reproyectar anotaciones.
    """
    h, w = img_array.shape[:2]
    labels = sorted({o["label"] for o in objects})
    cat_ids = {lab: i + 1 for i, lab in enumerate(labels)}
    annotations = []
    for i, o in enumerate(objects, start=1):
        xmin, ymin, xmax, ymax = o["bbox_px"]
        annotations.append({
            "id": i, "image_id": 1, "category_id": cat_ids[o["label"]],
            "segmentation": o["mask_rle"], "area": int(o["pixel_count"]),
            "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)], "iscrowd": 0,
            "attributes": {"object_id": o["object_id"], "pred_iou": o.get("pred_iou"),
                           "area_m2": o.get("area_m2")},
        })
    coco = {
        "info": {"description": "Dataset generado con SAM 2 GeoExtractor",
                 "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "segmentation_note": "Máscaras de SAM 2 validadas por el usuario; categorías asignadas manualmente."},
        "images": [{"id": 1, "file_name": image_name, "width": int(w), "height": int(h)}],
        "annotations": annotations,
        "categories": [{"id": cid, "name": lab, "supercategory": "objeto"} for lab, cid in cat_ids.items()],
    }
    georef = {"crs": crs_str, "affine_transform": list(transform)[:6], "width": int(w), "height": int(h),
              "pixel_convention": "rasterio/GDAL: (col, row) → (x, y) = transform * (col, row), esquina superior izquierda"}
    img_buf = io.BytesIO()
    Image.fromarray(img_array).save(img_buf, format="PNG")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"images/{image_name}", img_buf.getvalue())
        zf.writestr("annotations/instances.json", json.dumps(coco, ensure_ascii=False, indent=1))
        zf.writestr("georreferencia.json", json.dumps(georef, indent=2))
    return buf.getvalue()
