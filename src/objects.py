"""
Catálogo de objetos geoespaciales de la sesión (independiente de Streamlit).

Cada objeto conserva:
- identificador único, etiqueta (asignada por el usuario: SAM 2 no clasifica), estado
  (aceptado / propuesto / rechazado), visibilidad y versión;
- la máscara en RLE COCO (geosam.utils.rle) para no retener matrices H×W en memoria;
- su GeoDataFrame de 1 fila con geometría EPSG:4326 y métricas;
- la secuencia de prompts que lo generó, para poder refinarlo después;
- trazabilidad del modelo (model_id, backend real o simulado, origen interactivo/automático).

Rechazar no borra (queda para trazabilidad y se excluye de la exportación por defecto);
eliminar sí lo quita del catálogo.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from config import CRS_GEOGRAPHIC
from geosam.utils.rle import mask_to_rle, rle_to_mask

STATUSES = ("aceptado", "propuesto", "rechazado")
PALETTE = ["#00E5FF", "#FF4081", "#76FF03", "#FFAB00", "#D500F9", "#1DE9B6",
           "#FF6E40", "#2979FF", "#EEFF41", "#F50057", "#00B0FF", "#64DD17"]


def new_object_id() -> str:
    return f"OBJ-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class GeoObject:
    object_id: str
    label: str
    gdf: gpd.GeoDataFrame
    mask_rle: Dict[str, Any]
    bbox_px: tuple
    pred_iou: float
    model_id: str
    backend: str
    origin: str = "interactivo"
    status: str = "aceptado"
    visible: bool = True
    version: int = 1
    prompts: Optional[Dict[str, Any]] = None
    color: str = PALETTE[0]
    raster_value: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def mask(self) -> np.ndarray:
        return rle_to_mask(self.mask_rle)

    @property
    def pixel_count(self) -> int:
        counts = self.mask_rle.get("counts", [])
        return int(sum(counts[1::2]))

    def attributes(self) -> Dict[str, Any]:
        row = self.gdf.iloc[0].drop(labels=["geometry"]).to_dict()
        row.update({"object_id": self.object_id, "label": self.label, "status": self.status,
                    "origin": self.origin, "model_id": self.model_id, "backend": self.backend,
                    "version": self.version, "raster_value": self.raster_value})
        return row


class ObjectCatalog:
    def __init__(self) -> None:
        self._objects: Dict[str, GeoObject] = {}
        self.revision = 0  # cambia con cada modificación (útil para claves de widgets)
        self._next_value = 1

    # ------------------------------------------------------------------ CRUD
    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self):
        return iter(self._objects.values())

    def __contains__(self, object_id: str) -> bool:
        return object_id in self._objects

    def get(self, object_id: str) -> GeoObject:
        return self._objects[object_id]

    def ids(self) -> List[str]:
        return list(self._objects.keys())

    def _touch(self) -> None:
        self.revision += 1

    def add_from_mask(
        self,
        mask: np.ndarray,
        gdf: gpd.GeoDataFrame,
        pred_iou: float,
        model_id: str,
        backend: str,
        label: str = "objeto",
        origin: str = "interactivo",
        status: str = "aceptado",
        prompts: Optional[Dict[str, Any]] = None,
        object_id: Optional[str] = None,
    ) -> GeoObject:
        if gdf is None or gdf.empty:
            raise ValueError("La máscara no generó ningún polígono válido.")
        if status not in STATUSES:
            raise ValueError(f"Estado inválido: {status}")
        oid = object_id or str(gdf["object_id"].iloc[0]) or new_object_id()
        if oid in self._objects:
            raise KeyError(f"Ya existe el objeto {oid}")
        m = np.asarray(mask, dtype=bool)
        ys, xs = np.where(m)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else (0, 0, 0, 0)
        g = gdf.copy()
        g["object_id"] = oid
        g["label"] = label
        obj = GeoObject(
            object_id=oid, label=label, gdf=g, mask_rle=mask_to_rle(m), bbox_px=bbox,
            pred_iou=float(pred_iou), model_id=model_id, backend=backend, origin=origin,
            status=status, prompts=prompts, color=PALETTE[(self._next_value - 1) % len(PALETTE)],
            raster_value=self._next_value,
        )
        self._next_value += 1
        self._objects[oid] = obj
        self._touch()
        return obj

    def replace_geometry(self, object_id: str, mask: np.ndarray, gdf: gpd.GeoDataFrame,
                         pred_iou: float, prompts: Optional[Dict[str, Any]], model_id: str, backend: str) -> GeoObject:
        """Nueva versión de un objeto existente (refinamiento): conserva id, etiqueta y color."""
        obj = self._objects[object_id]
        if gdf is None or gdf.empty:
            raise ValueError("La máscara refinada no generó ningún polígono válido.")
        m = np.asarray(mask, dtype=bool)
        ys, xs = np.where(m)
        g = gdf.copy()
        g["object_id"] = object_id
        g["label"] = obj.label
        obj.gdf = g
        obj.mask_rle = mask_to_rle(m)
        obj.bbox_px = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        obj.pred_iou = float(pred_iou)
        obj.prompts = prompts
        obj.model_id = model_id
        obj.backend = backend
        obj.version += 1
        obj.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._touch()
        return obj

    def update(self, object_id: str, **fields: Any) -> None:
        obj = self._objects[object_id]
        changed = False
        for k, v in fields.items():
            if k == "status" and v not in STATUSES:
                raise ValueError(f"Estado inválido: {v}")
            if k not in ("label", "status", "visible"):
                raise KeyError(f"Campo no editable: {k}")
            if getattr(obj, k) != v:
                setattr(obj, k, v)
                changed = True
                if k == "label":
                    obj.gdf["label"] = v
        if changed:
            obj.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._touch()

    def remove(self, object_id: str) -> None:
        del self._objects[object_id]
        self._touch()

    def clear(self) -> None:
        self._objects.clear()
        self._next_value = 1
        self._touch()

    # ------------------------------------------------------------------ consultas
    def filter(self, statuses: Optional[Iterable[str]] = None, visible_only: bool = False) -> List[GeoObject]:
        st = set(statuses) if statuses else set(STATUSES)
        return [o for o in self._objects.values() if o.status in st and (o.visible or not visible_only)]

    def to_geodataframe(self, statuses: Optional[Iterable[str]] = None) -> gpd.GeoDataFrame:
        objs = self.filter(statuses)
        if not objs:
            return gpd.GeoDataFrame(columns=["object_id", "geometry"], geometry="geometry", crs=CRS_GEOGRAPHIC)
        rows = [{**o.attributes(), "geometry": o.gdf.geometry.iloc[0]} for o in objs]
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_GEOGRAPHIC)

    def table(self) -> pd.DataFrame:
        """Tabla resumida para la interfaz (sin geometría)."""
        rows = []
        for o in self._objects.values():
            a = o.attributes()
            rows.append({
                "ver": o.visible, "object_id": o.object_id, "label": o.label, "status": o.status,
                "area_m2": a.get("area_m2"), "perimeter_m": a.get("perimeter_m"),
                "pred_iou": a.get("pred_iou"), "compactness": a.get("compactness"),
                "origin": o.origin, "version": o.version, "color": o.color,
            })
        return pd.DataFrame(rows, columns=["ver", "object_id", "label", "status", "area_m2", "perimeter_m",
                                           "pred_iou", "compactness", "origin", "version", "color"])

    def statistics(self, statuses: Optional[Iterable[str]] = ("aceptado",)) -> Dict[str, Any]:
        gdf = self.to_geodataframe(statuses)
        if gdf.empty:
            return {"n": 0}
        a = gdf["area_m2"].astype(float)
        return {
            "n": int(len(gdf)),
            "area_total_m2": float(a.sum()),
            "area_media_m2": float(a.mean()),
            "area_mediana_m2": float(a.median()),
            "area_min_m2": float(a.min()),
            "area_max_m2": float(a.max()),
            "perimetro_total_m": float(gdf["perimeter_m"].astype(float).sum()),
            "compacidad_media": float(gdf["compactness"].astype(float).mean()),
            "por_etiqueta": gdf.groupby("label").agg(n=("object_id", "count"),
                                                     area_m2=("area_m2", "sum")).reset_index(),
        }
