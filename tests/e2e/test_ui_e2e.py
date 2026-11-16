"""Pruebas de interfaz de punta a punta en un navegador real (Playwright + Chromium headless).

Arrancan la app con `streamlit run`, descargan teselas reales de Esri y usan el modelo SAM 2 real.
Requisitos: red, checkpoint de SAM 2, `pip install -r requirements-dev.txt` y
`python -m playwright install chromium`.

    RUN_E2E=1 pytest tests/e2e -v
"""
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pytest

if os.environ.get("RUN_E2E") != "1":
    pytest.skip("Defina RUN_E2E=1 para ejecutar las pruebas E2E de interfaz.", allow_module_level=True)
playwright = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
LABEL = "Categoría (la asigna usted; SAM 2 no clasifica)"


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


@pytest.fixture(scope="module")
def app_url():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
         "--server.port", str(port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://localhost:{port}"
    for _ in range(60):
        try:
            socket.create_connection(("localhost", port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.5)
    yield url
    proc.terminate(); proc.wait(timeout=20)


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def map_frame(page):
    deadline = time.time() + 15
    while time.time() < deadline:
        for f in page.frames:
            try:
                if f.query_selector(".leaflet-container"):
                    return f
            except Exception:
                pass
        page.wait_for_timeout(300)
    raise AssertionError("No se encontró el mapa Leaflet")


def leaflet_zoom(page):
    return map_frame(page).evaluate(
        "() => { const k = Object.keys(window).find(k => k.startsWith('map_') && window[k] && window[k].getZoom);"
        " return k ? window[k].getZoom() : null; }")


def n_polygons(page):
    return map_frame(page).evaluate("() => document.querySelectorAll('.leaflet-overlay-pane path').length")


def click_map(page, dx=0, dy=0):
    box = map_frame(page).query_selector(".leaflet-container").bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2 + dx, box["y"] + box["height"] / 2 + dy)


def wait_idle(page, timeout=90):
    page.wait_for_timeout(800)
    end = time.time() + timeout
    while time.time() < end:
        if not page.query_selector("[data-testid='stStatusWidget']"):
            page.wait_for_timeout(600)
            if not page.query_selector("[data-testid='stStatusWidget']"):
                return
        page.wait_for_timeout(400)


def no_exceptions(page):
    ex = page.query_selector_all("[data-testid='stException']")
    assert not ex, ex[0].inner_text()[:2000]


def load_orthophoto(browser, url):
    page = browser.new_page(viewport={"width": 1600, "height": 1300}, accept_downloads=True)
    page.goto(url, wait_until="networkidle")
    page.get_by_role("button", name="Descargar y cargar ortofoto").click(timeout=120000)
    page.wait_for_selector("text=Herramienta de prompt", timeout=180000)
    wait_idle(page); page.wait_for_timeout(2000)
    no_exceptions(page)
    return page


def set_category(page, label):
    """Selectbox con accept_new_options: escribir y confirmar con Enter."""
    box = page.locator("[data-testid='stSelectbox']").filter(has_text=LABEL).first
    box.click()
    page.keyboard.type(label)
    page.keyboard.press("Enter")
    wait_idle(page)


def accept(page, label=None):
    if label:
        set_category(page, label)
    page.get_by_role("button", name="Aceptar objeto").click()
    wait_idle(page); page.wait_for_timeout(1000)
    no_exceptions(page)


def test_prompts_zoom_and_discard(browser, app_url):
    page = load_orthophoto(browser, app_url)
    f = map_frame(page)
    f.click(".leaflet-control-zoom-in"); page.wait_for_timeout(600)
    f.click(".leaflet-control-zoom-in"); page.wait_for_timeout(1000)
    zoom = leaflet_zoom(page)

    click_map(page)
    page.wait_for_selector("text=Candidatas de SAM 2", timeout=60000); wait_idle(page)
    no_exceptions(page)
    assert leaflet_zoom(page) == zoom, "El mapa perdió el zoom tras el clic"
    assert page.get_by_text(re.compile(r"^(?:✓ )?\d · IoU pred\.")).count() == 3

    page.locator("button", has_text="Excluir").first.click(); wait_idle(page)
    click_map(page, 6, 6); wait_idle(page)            # clic DENTRO de la máscara
    assert "➖ 1" in page.inner_text("body")
    page.get_by_role("button", name="Deshacer").click(); wait_idle(page)
    assert "➖ 0" in page.inner_text("body")
    accept(page, "zona verde")
    assert "Objetos (1)" in page.get_by_role("tab").first.inner_text()

    page.locator("button", has_text="Incluir").first.click(); wait_idle(page)
    click_map(page, -120, -80); wait_idle(page); page.wait_for_timeout(800)
    page.get_by_role("button", name="Descartar").click(); wait_idle(page)
    page.get_by_role("tab", name="Estadísticas").click(); wait_idle(page); page.wait_for_timeout(1000)
    body = page.inner_text("body")
    assert "➕ 0" in body and "Candidatas de SAM 2" not in body, "El clic descartado se volvió a disparar"

    page.locator("button", has_text="Caja (2 clics)").first.click(); wait_idle(page)
    click_map(page, 120, 60); wait_idle(page)
    assert page.query_selector("text=Primera esquina marcada")
    click_map(page, 220, 160); wait_idle(page); page.wait_for_timeout(800)
    assert "▭ 1" in page.inner_text("body")
    accept(page)
    assert leaflet_zoom(page) == zoom
    page.close()


def test_catalog_refine_visibility_and_exports(browser, app_url):
    page = load_orthophoto(browser, app_url)
    for dx, dy, lab in [(0, 0, "zona verde"), (150, 120, "cubierta")]:
        click_map(page, dx, dy)
        page.wait_for_selector("text=IoU predicho", timeout=60000); wait_idle(page)
        accept(page, lab)

    # Ocultar el 2.º objeto desde la tabla
    before = n_polygons(page)
    grid = page.locator("[data-testid='stDataFrame']").first
    grid.scroll_into_view_if_needed(); page.wait_for_timeout(400)
    gb = grid.bounding_box()
    page.mouse.click(gb["x"] + 50, gb["y"] + 35 + 35 + 17); wait_idle(page); page.wait_for_timeout(1000)
    if n_polygons(page) == before:  # el primer clic solo seleccionó la celda
        page.mouse.click(gb["x"] + 50, gb["y"] + 35 + 35 + 17); wait_idle(page); page.wait_for_timeout(1000)
    assert n_polygons(page) == before - 1
    no_exceptions(page)

    # Refinar el objeto seleccionado → versión 2
    page.get_by_role("button", name=re.compile(r"Refinar$")).click(); wait_idle(page)
    assert "Refinando OBJ-" in page.inner_text("body")
    page.locator("button", has_text="Incluir").first.click(); wait_idle(page)
    click_map(page, 30, 30); wait_idle(page)
    accept(page)
    page.get_by_role("tab", name="Detalle y métricas").click(); wait_idle(page)
    assert page.get_by_text(" · v2", exact=False).count() > 0

    # Descargas
    page.get_by_role("tab", name="Exportar").click(); wait_idle(page)
    out = Path(tempfile.mkdtemp())
    files = {}
    for n in ["GeoJSON", "GeoPackage", "Shapefile (.zip)", "CSV de atributos", "Ortofoto GeoTIFF",
              "Máscaras GeoTIFF", "Dataset COCO (.zip)"]:
        btn = page.locator("[data-testid='stDownloadButton'] button", has_text=n).first
        btn.scroll_into_view_if_needed()
        with page.expect_download(timeout=60000) as dl:
            btn.click()
        files[n] = out / dl.value.suggested_filename
        dl.value.save_as(files[n])
    page.close()

    import geopandas as gpd, pyogrio, rasterio, numpy as np
    g = gpd.read_file(files["GeoJSON"])
    assert len(g) == 2 and g.crs.to_epsg() == 4326 and set(g["label"]) == {"zona verde", "cubierta"}
    assert {"objetos", "convex_hull", "bbox"} <= {l[0] for l in pyogrio.list_layers(files["GeoPackage"])}
    with tempfile.TemporaryDirectory() as d:
        zipfile.ZipFile(files["Shapefile (.zip)"]).extractall(d)
        shp = gpd.read_file(next(Path(d).glob("*.shp")))
        assert {"cent_lat", "cent_lon", "perim_m"} <= set(shp.columns)
    with rasterio.open(files["Máscaras GeoTIFF"]) as ds:
        assert len(np.unique(ds.read(1))) == 3 and ds.crs.to_epsg() == 3857
    coco = json.loads(zipfile.ZipFile(files["Dataset COCO (.zip)"]).read("annotations/instances.json"))
    assert len(coco["annotations"]) == 2


def test_model_switch_and_automatic(browser, app_url):
    page = load_orthophoto(browser, app_url)
    page.locator("[data-testid='stSidebar'] [data-testid='stSelectbox']").first.click()
    page.get_by_role("option", name="SAM 2 Tiny (versión 072824)").click()
    wait_idle(page); page.wait_for_timeout(400)
    page.get_by_role("button", name="Cargar modelo").click()
    wait_idle(page, 240); page.wait_for_timeout(1500)
    no_exceptions(page)
    click_map(page, -150, 100)
    page.wait_for_selector("text=IoU predicho", timeout=60000); wait_idle(page)
    no_exceptions(page)
    page.get_by_role("button", name="Descartar").click(); wait_idle(page)

    page.get_by_role("tab", name="Segmentación automática").click(); wait_idle(page)
    page.locator("[role='tabpanel']:visible [data-testid='stSelectbox']").first.click()
    page.get_by_role("option", name="8", exact=True).click(); wait_idle(page)
    page.get_by_role("button", name="Ejecutar segmentación automática").click()
    wait_idle(page, 400); page.wait_for_timeout(1000)
    no_exceptions(page)
    n = int(page.get_by_role("tab").first.inner_text().split("(")[1].rstrip(")"))
    assert n > 0
    page.close()


def test_composite_building_shape_and_category(browser, app_url):
    """Caja + puntos sobre una edificación compuesta, patios rellenados y categoría sugerida."""
    page = load_orthophoto(browser, app_url)
    page.locator("button", has_text="Caja (2 clics)").first.click(); wait_idle(page)
    click_map(page, -160, -140); wait_idle(page)
    click_map(page, 40, 20); wait_idle(page)
    page.locator("button", has_text="Incluir").first.click(); wait_idle(page)
    click_map(page, -60, -60); wait_idle(page)
    page.wait_for_selector("text=Perímetro exterior", timeout=60000)
    box = page.locator("[data-testid='stCheckbox']").filter(has_text="Rellenar patios y huecos interiores").first
    if not box.is_visible():
        page.get_by_text("Forma de la huella").click(); wait_idle(page)
    box.scroll_into_view_if_needed()
    if not box.locator("input").is_checked():
        box.locator("label").click(); wait_idle(page)
    no_exceptions(page)
    # el rerun que provoca la casilla puede tardar: esperar a que desaparezcan los huecos
    deadline = time.time() + 20
    while time.time() < deadline:
        body = page.inner_text("body")
        if "0 hueco(s)" in body or "hueco(s)" not in body:
            break
        page.wait_for_timeout(500)
    assert "0 hueco(s)" in body or "hueco(s)" not in body
    btn = page.get_by_role("button", name="Sugerir categoría (RemoteCLIP)")
    if btn.is_enabled():
        btn.click(); wait_idle(page, 300)
        no_exceptions(page)
        sug = page.locator("button", has_text="%")
        if sug.count():
            first = sug.first.inner_text().split(" · ")[0]
            sug.first.click(); wait_idle(page)
            accept(page)
            page.get_by_role("tab", name="Detalle y métricas").click(); wait_idle(page)
            body = page.inner_text("body")
            assert first in body and "RemoteCLIP (confirmada por el usuario)" in body
            page.close()
            return
    accept(page, "Edificación religiosa")
    page.close()


def test_products_tab(browser, app_url):
    """Productos con un catálogo mínimo: sin excepciones, aviso de muestras insuficientes y descargas."""
    page = load_orthophoto(browser, app_url)
    for dx, dy, lab in [(0, 0, "Zona verde"), (150, 120, "Cubierta de teja")]:
        map_frame(page).frame_element().scroll_into_view_if_needed()
        click_map(page, dx, dy)
        page.wait_for_selector("text=IoU predicho", timeout=60000); wait_idle(page)
        accept(page, lab)
    page.get_by_role("tab", name="Productos").click(); wait_idle(page)
    no_exceptions(page)
    assert page.get_by_text("Para entrenar necesita al menos", exact=False).count() > 0
    assert page.get_by_role("button", name="🎓 Entrenar y validar").is_disabled()
    for name in ["Mapa de coberturas", "Coordenadas y linderos", "Indicadores e informe"]:
        page.locator("button", has_text=name).first.click(); wait_idle(page, 120)
        no_exceptions(page)
    out = Path(tempfile.mkdtemp())
    btn = page.locator("[data-testid='stDownloadButton'] button", has_text="Informe técnico").first
    with page.expect_download(timeout=60000) as dl:
        btn.click()
    html = dl.value.path().read_text(encoding="utf-8")
    assert "Índice de ocupación" in html and "Inventario de objetos" in html
    page.locator("button", has_text="Coordenadas y linderos").first.click(); wait_idle(page)
    btn = page.locator("[data-testid='stDownloadButton'] button", has_text="Cuadro de coordenadas").first
    with page.expect_download(timeout=60000) as dl:
        btn.click()
    first = dl.value.path().read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("vertice,este_x,norte_y,lindero,distancia_m,acimut_deg")
    page.close()
