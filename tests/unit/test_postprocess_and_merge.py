"""Post-procesamiento de huellas, perímetro exterior, fusión de objetos y recorte en el borde."""
import numpy as np
import pytest
from rasterio.transform import from_origin

from src.geospatial import compute_shape_metrics, mask_to_geodataframe
from src.objects import ObjectCatalog
from src.postprocess import merge_masks, refine_mask, touches_border
from shapely.geometry import box

TR = from_origin(-8246700.0, 513800.0, 0.5, 0.5)


def _cloister(size=200):
    """Edificio cuadrado con patio central (claustro) de 40×40 px."""
    m = np.zeros((size, size), dtype=bool)
    m[50:150, 50:150] = True
    m[80:120, 80:120] = False
    return m


def test_fill_holes_gives_exterior_footprint():
    m = _cloister()
    filled = refine_mask(m, fill_holes=True)
    assert filled[100, 100] and filled.sum() == 100 * 100
    raw = mask_to_geodataframe(m, TR).iloc[0]
    assert raw["n_holes"] == 1
    # perímetro total incluye el borde del patio; el exterior no
    assert raw["perimeter_m"] > raw["perimeter_ext_m"]
    full = mask_to_geodataframe(filled, TR).iloc[0]
    assert full["n_holes"] == 0
    assert full["perimeter_ext_m"] == pytest.approx(raw["perimeter_ext_m"], rel=1e-6)


def test_closing_joins_separated_parts():
    m = np.zeros((100, 100), dtype=bool)
    m[20:80, 10:45] = True
    m[20:80, 49:90] = True  # separadas por 4 px
    assert mask_to_geodataframe(m, TR).iloc[0]["n_parts"] == 2
    joined = refine_mask(m, close_px=3)
    assert mask_to_geodataframe(joined, TR).iloc[0]["n_parts"] == 1
    assert refine_mask(m, close_px=1).sum() == m.sum()  # 1 px no alcanza a unir 4 px de separación


def test_keep_largest_and_min_part():
    m = np.zeros((100, 100), dtype=bool)
    m[10:60, 10:60] = True
    m[80:85, 80:85] = True
    assert refine_mask(m, keep_largest=True).sum() == 2500
    assert refine_mask(m, min_part_px=30).sum() == 2500
    assert refine_mask(m, min_part_px=20).sum() == 2525
    assert np.array_equal(refine_mask(m), m)  # sin opciones: idéntica


def test_touches_border():
    m = np.zeros((50, 50), dtype=bool)
    m[10:20, 10:20] = True
    assert not touches_border(m)
    m[0, 15] = True
    assert touches_border(m)


def test_exterior_perimeter_metric():
    g = box(0, 0, 30, 30).difference(box(10, 10, 20, 20))
    met = compute_shape_metrics(g)
    assert met["perimeter_ext_m"] == pytest.approx(120)
    assert met["perimeter_m"] == pytest.approx(160)


def test_catalog_merge_with_postprocess():
    cat = ObjectCatalog()
    a = np.zeros((200, 200), dtype=bool); a[0:60, 20:80] = True      # toca el borde superior
    b = np.zeros((200, 200), dtype=bool); b[62:120, 20:80] = True    # separado 2 px
    for m, lab in ((a, "nave"), (b, "claustro")):
        cat.add_from_mask(m, mask_to_geodataframe(m, TR), 0.8, "mock", "MockBackend", label=lab)
    ids = cat.ids()
    assert cat.get(ids[0]).touches_edge and not cat.get(ids[1]).touches_edge
    with pytest.raises(ValueError):
        cat.merge(ids[:1], lambda m: mask_to_geodataframe(m, TR))
    new = cat.merge(ids, lambda m: mask_to_geodataframe(m, TR), label="Edificación religiosa",
                    postprocess=lambda m: refine_mask(m, close_px=2))
    assert len(cat) == 1 and new.label == "Edificación religiosa" and new.origin == "fusión"
    assert new.gdf.iloc[0]["n_parts"] == 1
    assert new.prompts == {"merged_from": ids}
    assert new.touches_edge
    assert merge_masks([a, b]).sum() == a.sum() + b.sum()
