"""Sesión interactiva de prompts y catálogo de objetos (con MockBackend, sin GPU)."""
import io
import json
import zipfile

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin

from geosam.backends.mock import MockBackend
from src.exporter import (
    export_coco_dataset, export_geopackage, export_label_geotiff, export_shapefile_zip,
    shapefile_field_names,
)
from src.geospatial import mask_to_geodataframe
from src.interactive import InteractiveSegmenter
from src.objects import ObjectCatalog
from src.sam_engine import SAM2Engine

TR = from_origin(-8246700.0, 513800.0, 0.5, 0.5)


@pytest.fixture
def engine_emb():
    be = MockBackend(model_id="mock")
    be.load("cpu")
    eng = SAM2Engine.from_backend(be, "mock")
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    emb, _ = eng.embed(img)
    return eng, emb


def test_single_point_gives_three_candidates_then_refines(engine_emb):
    eng, emb = engine_emb
    seg = InteractiveSegmenter()
    r = seg.add_point(eng, emb, 100, 100)
    assert len(r["candidates"]) == 3 and r["multimask"] is True
    seg.select(1)
    assert seg.selected_idx == 1
    r2 = seg.add_point(eng, emb, 110, 100, positive=False)
    assert len(r2["candidates"]) == 1 and r2["multimask"] is False
    assert seg.summary() == {"positivos": 1, "negativos": 1, "caja": 0}


def test_box_needs_two_clicks_and_undo_replays(engine_emb):
    eng, emb = engine_emb
    seg = InteractiveSegmenter()
    assert seg.click_box_corner(eng, emb, 150, 150) is None
    assert seg.pending_corner == (150, 150)
    r = seg.click_box_corner(eng, emb, 50, 60)
    assert seg.events[-1].box == (50, 60, 150, 150)  # normalizada
    assert r["candidates"][0].mask[100, 100]
    seg.add_point(eng, emb, 20, 20)
    before = seg.selected_candidate.mask.copy()
    seg.undo(eng, emb)
    assert len(seg.events) == 1
    assert not np.array_equal(before, seg.selected_candidate.mask)
    seg.undo(eng, emb)
    assert seg.is_empty and seg.result is None


def test_tiny_box_rejected(engine_emb):
    eng, emb = engine_emb
    seg = InteractiveSegmenter()
    seg.click_box_corner(eng, emb, 10, 10)
    with pytest.raises(ValueError):
        seg.click_box_corner(eng, emb, 11, 11)


def test_prompts_roundtrip(engine_emb):
    eng, emb = engine_emb
    seg = InteractiveSegmenter()
    seg.add_point(eng, emb, 100, 100)
    seg.select(2)
    seg.add_point(eng, emb, 120, 100)
    restored = InteractiveSegmenter.from_dict(seg.to_dict(), eng, emb)
    assert np.array_equal(restored.selected_candidate.mask, seg.selected_candidate.mask)
    assert restored.selections == seg.selections


def _obj_mask(x0, y0, x1, y1, shape=(200, 200)):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_catalog_lifecycle_and_exports():
    cat = ObjectCatalog()
    masks = [_obj_mask(10, 10, 50, 50), _obj_mask(60, 60, 120, 90), _obj_mask(130, 20, 180, 40)]
    for i, m in enumerate(masks):
        gdf = mask_to_geodataframe(m, TR, confidence=0.9)
        cat.add_from_mask(m, gdf, 0.9, "mock", "MockBackend", label="edificio" if i < 2 else "árbol")
    ids = cat.ids()
    assert len(set(ids)) == 3
    rev = cat.revision
    cat.update(ids[2], status="rechazado")
    cat.update(ids[1], visible=False, label="bodega")
    assert cat.revision > rev
    assert len(cat.filter(["aceptado"])) == 2
    assert len(cat.filter(visible_only=True)) == 2
    with pytest.raises(ValueError):
        cat.update(ids[0], status="otro")
    assert np.array_equal(cat.get(ids[0]).mask(), masks[0])
    assert cat.get(ids[0]).pixel_count == int(masks[0].sum())

    # Refinar crea una nueva versión con el mismo id
    m_new = _obj_mask(10, 10, 60, 50)
    cat.replace_geometry(ids[0], m_new, mask_to_geodataframe(m_new, TR), 0.95, None, "mock", "MockBackend")
    assert cat.get(ids[0]).version == 2

    stats = cat.statistics(("aceptado",))
    assert stats["n"] == 2 and stats["area_total_m2"] > 0

    gdf = cat.to_geodataframe(("aceptado",))
    # Shapefile: nombres únicos ≤ 10 caracteres y legibles
    mapping = shapefile_field_names(gdf.columns)
    assert all(len(v) <= 10 for v in mapping.values())
    assert len(set(v.lower() for v in mapping.values())) == len(mapping)
    assert mapping["centroid_lat"] == "cent_lat" and mapping["centroid_lon"] == "cent_lon"
    z = zipfile.ZipFile(io.BytesIO(export_shapefile_zip(gdf)))
    assert {"campos.csv"} <= set(z.namelist())
    shp = [n for n in z.namelist() if n.endswith(".shp")][0]
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        z.extractall(d)
        back = gpd.read_file(os.path.join(d, shp))
    assert "cent_lat" in back.columns and "perim_m" in back.columns and len(back) == 2

    # GeoPackage multicapa legible
    with tempfile.NamedTemporaryFile(suffix=".gpkg") as f:
        f.write(export_geopackage({"objetos": gdf, "vacia": gdf.iloc[0:0]}))
        f.flush()
        import pyogrio
        layers = [lyr[0] for lyr in pyogrio.list_layers(f.name)]
        assert layers == ["objetos"]
        assert len(gpd.read_file(f.name, layer="objetos")) == 2

    # Ráster de etiquetas
    objs = cat.filter(["aceptado"])
    tif = export_label_geotiff([o.mask() for o in objs], [o.raster_value for o in objs], TR)
    import rasterio
    with rasterio.MemoryFile(tif) as mf, mf.open() as ds:
        band = ds.read(1)
        assert ds.crs.to_string() == "EPSG:3857"
        assert set(np.unique(band)) == {0} | {o.raster_value for o in objs}

    # Dataset COCO
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    coco_zip = export_coco_dataset(img, [{
        "object_id": o.object_id, "label": o.label, "mask_rle": o.mask_rle, "bbox_px": o.bbox_px,
        "pixel_count": o.pixel_count, "pred_iou": o.pred_iou, "area_m2": float(o.gdf["area_m2"].iloc[0])}
        for o in objs], TR)
    zc = zipfile.ZipFile(io.BytesIO(coco_zip))
    coco = json.loads(zc.read("annotations/instances.json"))
    assert len(coco["annotations"]) == 2
    assert {c["name"] for c in coco["categories"]} == {"edificio", "bodega"}
    assert sum(coco["annotations"][0]["segmentation"]["counts"]) == 200 * 200

    cat.remove(ids[2])
    assert len(cat) == 2
