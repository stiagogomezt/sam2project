from config import DEFAULT_BBOX, DEFAULT_ZOOM
from src.url_params import parse_view_params


def test_bbox_y_zoom_validos():
    bbox, z, from_url = parse_view_params({"bbox": "-74.1270,4.4745,-74.1240,4.4770", "z": "19"})
    assert from_url and z == 19
    assert bbox == {"min_lon": -74.127, "min_lat": 4.4745, "max_lon": -74.124, "max_lat": 4.477}


def test_valores_invalidos_vuelven_a_los_predeterminados():
    for params in ({}, {"bbox": "a,b,c,d"}, {"bbox": "-74.1,4.5,-74.2,4.6"}, {"bbox": "-75,4,-74,5"}, {"z": "25"}):
        bbox, z, from_url = parse_view_params(params)
        assert bbox == DEFAULT_BBOX and z == DEFAULT_ZOOM and not from_url


def test_acepta_listas_como_las_de_query_params():
    bbox, z, _ = parse_view_params({"bbox": ["-74.08,4.60,-74.07,4.61"], "z": ["18"]})
    assert z == 18 and bbox["max_lat"] == 4.61
