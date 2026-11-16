"""
Sistema de diseño y componentes HTML/CSS centralizados para SAM 2 GeoExtractor.

Usa variables CSS compatibles con tema claro y oscuro de Streamlit para mantener
contraste AA, densidad informativa profesional tipo SIG y componentes consistentes.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

import streamlit as st


# Paleta y tokens compartidos (exportables a módulos de visualización si hace falta)
TOKENS = {
    "primary": "#0F5E63",
    "primary_hover": "#0B494D",
    "accent": "#0284C7",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
}

STAGES = [
    {
        "num": 1,
        "key": "aoi",
        "title": "Zona de estudio",
        "detail": "Pasos 1–2 · Definir coordenadas WGS84, zoom y verificar teselas.",
    },
    {
        "num": 2,
        "key": "imagery",
        "title": "Imagen",
        "detail": "Pasos 2–3 · Descargar ortofoto y calcular embedding de SAM 2.",
    },
    {
        "num": 3,
        "key": "segment",
        "title": "Segmentar y refinar",
        "detail": "Pasos 4–5 · Marcar puntos/caja y refinar la máscara candidata.",
    },
    {
        "num": 4,
        "key": "validate",
        "title": "Validar",
        "detail": "Pasos 6–7 · Elegir candidata (IoU), asignar etiqueta y aceptar objeto.",
    },
    {
        "num": 5,
        "key": "export",
        "title": "Analizar y exportar",
        "detail": "Pasos 8–10 · Revisar métricas, estadísticas y exportar a GIS/COCO.",
    },
]

EXPORT_DESCRIPTIONS = {
    "geojson": "Geometrías vectoriales y atributos en estándar web WGS84.",
    "gpkg": "Base espacial OGC multicapa con sistema métrico proyectado.",
    "shp": "Archivo ESRI comprimido con campos compatibles (máx. 10 car.).",
    "csv": "Tabla tabular de áreas, perímetros, centroides y estados.",
    "dxf": "Capas CAD por categoría en sistema Origen Nacional (EPSG:9377).",
    "kml": "Visualización directa de polígonos y etiquetas en Google Earth.",
    "ortho": "Mosaico satelital georreferenciado de 3 bandas RGB.",
    "masks": "Ráster de etiquetas enteras por ID de objeto validado.",
    "coco": "Dataset listo para entrenamiento de visión por computador.",
}


def inject_global_css() -> None:
    """Inyecta tokens CSS y ajustes estructurales para tema claro y oscuro."""
    st.markdown(
        """
        <style>
        :root {
            --gis-primary: #0F5E63;
            --gis-primary-soft: rgba(15, 94, 99, 0.10);
            --gis-primary-border: rgba(15, 94, 99, 0.30);
            --gis-surface: var(--secondary-background-color, #EEF3F7);
            --gis-bg: var(--background-color, #F8FAFC);
            --gis-text: var(--text-color, #0F172A);
            --gis-muted: #475569;
            --gis-border: rgba(148, 163, 184, 0.35);
            --gis-ok-bg: #ECFDF5;
            --gis-ok-fg: #065F46;
            --gis-ok-bd: #6EE7B7;
            --gis-warn-bg: #FFFBEB;
            --gis-warn-fg: #92400E;
            --gis-warn-bd: #FCD34D;
            --gis-radius: 10px;
            --gis-radius-sm: 6px;
            --gis-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }

        [data-theme="dark"], .stApp[data-theme="dark"] {
            --gis-primary: #14B8A6;
            --gis-primary-soft: rgba(20, 184, 166, 0.14);
            --gis-primary-border: rgba(20, 184, 166, 0.40);
            --gis-muted: #94A3B8;
            --gis-border: rgba(148, 163, 184, 0.25);
            --gis-ok-bg: rgba(6, 95, 70, 0.35);
            --gis-ok-fg: #A7F3D0;
            --gis-ok-bd: rgba(16, 185, 129, 0.45);
            --gis-warn-bg: rgba(146, 64, 14, 0.35);
            --gis-warn-fg: #FDE68A;
            --gis-warn-bd: rgba(245, 158, 11, 0.45);
            --gis-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --gis-muted: #94A3B8;
            }
        }

        /* Espaciado superior adaptado para no quedar cubierto por el header de Streamlit */
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-top: 4.25rem !important;
            padding-bottom: 1.75rem !important;
            max-width: 1560px;
        }

        [data-testid="stCaptionContainer"] p {
            line-height: 1.3;
        }

        /* Barra superior de producto */
        .gis-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            padding: 10px 14px;
            margin-top: 1.25rem;
            margin-bottom: 10px;
            background: var(--gis-surface);
            border: 1px solid var(--gis-border);
            border-radius: var(--gis-radius);
            box-shadow: var(--gis-shadow);
        }
        .gis-brand {
            display: flex;
            align-items: baseline;
            gap: 10px;
            flex-wrap: wrap;
        }
        h1.gis-title {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 1.25rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
            letter-spacing: -0.01em;
            color: var(--gis-text);
        }
        .gis-version {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 999px;
            background: var(--gis-primary-soft);
            color: var(--gis-primary);
            border: 1px solid var(--gis-primary-border);
        }
        .gis-subtitle {
            font-size: 0.82rem;
            color: var(--gis-muted);
        }
        .gis-status-group {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .gis-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid var(--gis-border);
            background: var(--gis-bg);
            color: var(--gis-text);
            white-space: nowrap;
        }
        .gis-pill--ok {
            background: var(--gis-ok-bg);
            color: var(--gis-ok-fg);
            border-color: var(--gis-ok-bd);
        }
        .gis-pill--warn {
            background: var(--gis-warn-bg);
            color: var(--gis-warn-fg);
            border-color: var(--gis-warn-bd);
        }
        .gis-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: currentColor;
            display: inline-block;
        }

        /* Stepper horizontal de 5 etapas (legible desde 1366 px) */
        .gis-stepper {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 14px;
        }
        .gis-step {
            position: relative;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 10px;
            border-radius: var(--gis-radius-sm);
            border: 1px solid var(--gis-border);
            background: var(--gis-bg);
            cursor: help;
            transition: border-color 0.15s ease, background 0.15s ease;
            min-width: 0;
        }
        .gis-step--done {
            background: var(--gis-ok-bg);
            border-color: var(--gis-ok-bd);
            color: var(--gis-ok-fg);
        }
        .gis-step--active {
            background: var(--gis-primary-soft);
            border-color: var(--gis-primary);
            box-shadow: inset 0 0 0 1px var(--gis-primary);
        }
        .gis-step--pending {
            opacity: 0.82;
        }
        .gis-step__badge {
            flex: 0 0 auto;
            width: 22px;
            height: 22px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.74rem;
            font-weight: 700;
            background: var(--gis-surface);
            color: var(--gis-text);
            border: 1px solid var(--gis-border);
        }
        .gis-step--done .gis-step__badge {
            background: #059669;
            color: #FFFFFF;
            border-color: #059669;
        }
        .gis-step--active .gis-step__badge {
            background: var(--gis-primary);
            color: #FFFFFF;
            border-color: var(--gis-primary);
        }
        .gis-step__text {
            min-width: 0;
            display: flex;
            flex-direction: column;
        }
        .gis-step__title {
            font-size: 0.80rem;
            font-weight: 600;
            line-height: 1.15;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .gis-step__state {
            font-size: 0.70rem;
            color: var(--gis-muted);
            line-height: 1.15;
        }
        .gis-step--done .gis-step__state {
            color: var(--gis-ok-fg);
        }
        .gis-step[data-tooltip]:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            left: 0;
            top: calc(100% + 6px);
            z-index: 999;
            min-width: 220px;
            max-width: 280px;
            padding: 7px 9px;
            border-radius: 6px;
            background: #0F172A;
            color: #F8FAFC;
            font-size: 0.74rem;
            font-weight: 400;
            line-height: 1.35;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.25);
            pointer-events: none;
        }
        @media (max-width: 980px) {
            .gis-stepper {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        /* Cuadrícula de mini-indicadores en barra lateral */
        .gis-kpi-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 6px;
            margin: 6px 0 8px 0;
        }
        .gis-kpi {
            padding: 7px 6px;
            border-radius: var(--gis-radius-sm);
            border: 1px solid var(--gis-border);
            background: var(--gis-bg);
            text-align: center;
        }
        .gis-kpi__val {
            font-size: 0.86rem;
            font-weight: 700;
            line-height: 1.2;
            color: var(--gis-text);
            font-variant-numeric: tabular-nums;
        }
        .gis-kpi__lbl {
            font-size: 0.68rem;
            color: var(--gis-muted);
            margin-top: 2px;
        }

        /* Ficha técnica del modelo seleccionado */
        .gis-model-card {
            padding: 9px 10px;
            margin: 6px 0 8px 0;
            border-radius: var(--gis-radius-sm);
            border: 1px solid var(--gis-border);
            background: var(--gis-bg);
            font-size: 0.78rem;
            line-height: 1.45;
        }
        .gis-model-card__row {
            display: flex;
            justify-content: space-between;
            gap: 8px;
        }
        .gis-model-card__k {
            color: var(--gis-muted);
        }
        .gis-model-card__v {
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }

        /* Marco y leyenda del mapa */
        .gis-map-legend {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            padding: 6px 12px;
            margin-top: -6px;
            margin-bottom: 10px;
            border: 1px solid var(--gis-border);
            border-radius: 0 0 var(--gis-radius) var(--gis-radius);
            background: var(--gis-surface);
            font-size: 0.76rem;
        }
        .gis-legend-items {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }
        .gis-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            color: var(--gis-text);
            font-weight: 500;
        }
        .gis-swatch {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            border: 1px solid rgba(15, 23, 42, 0.25);
        }
        .gis-swatch--box {
            border-radius: 2px;
        }

        /* Contadores de prompts */
        .gis-counters {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }
        .gis-counter-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 8px;
            border-radius: 999px;
            border: 1px solid var(--gis-border);
            background: var(--gis-bg);
            font-size: 0.76rem;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }

        /* Inspector vacío y tarjetas de métricas */
        .gis-inspector-empty {
            padding: 10px 12px;
            border-radius: var(--gis-radius-sm);
            background: var(--gis-primary-soft);
            border-left: 3px solid var(--gis-primary);
            font-size: 0.85rem;
            line-height: 1.45;
        }
        .gis-export-desc {
            font-size: 0.78rem;
            color: var(--gis-muted);
            margin: -4px 0 10px 0;
            line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def compute_stage_states(
    has_mosaic: bool,
    has_embedding: bool,
    has_prompts: bool,
    has_candidate: bool,
    catalog_count: int,
) -> List[str]:
    """Calcula el estado ('done' | 'active' | 'pending') de las 5 etapas del flujo."""
    done = [
        has_mosaic,
        has_mosaic and has_embedding,
        has_prompts and has_candidate,
        catalog_count > 0,
        catalog_count > 0 and not has_candidate,
    ]
    states: List[str] = []
    active_assigned = False
    for is_done in done:
        if is_done:
            states.append("done")
        elif not active_assigned:
            states.append("active")
            active_assigned = True
        else:
            states.append("pending")
    return states


def render_topbar(
    model_name: str,
    device_name: str,
    is_mock: bool,
    model_loaded: bool = True,
) -> None:
    """Renderiza cabecera compacta con un único H1 y píldora de estado del motor."""
    safe_model = html.escape(model_name)
    safe_device = html.escape(device_name)
    if is_mock:
        pill_cls = "gis-pill gis-pill--warn"
        pill_label = f"MODO SIMULADO · {safe_model} · {safe_device}"
    elif not model_loaded:
        pill_cls = "gis-pill gis-pill--warn"
        pill_label = f"Motor sin cargar · {safe_model}"
    else:
        pill_cls = "gis-pill gis-pill--ok"
        pill_label = f"{safe_model} · {safe_device}"

    st.markdown(
        f"""
        <header class="gis-topbar">
            <div class="gis-brand">
                <h1 class="gis-title">SAM 2 GeoExtractor</h1>
                <span class="gis-version">v2</span>
                <span class="gis-subtitle">Segmentación geoespacial interactiva</span>
            </div>
            <div class="gis-status-group">
                <span class="{pill_cls}">
                    <span class="gis-dot"></span>
                    {pill_label}
                </span>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(
    has_mosaic: bool,
    has_embedding: bool,
    has_prompts: bool,
    has_candidate: bool,
    catalog_count: int,
) -> None:
    """Renderiza el stepper horizontal de 5 etapas con tooltips de los 10 pasos."""
    states = compute_stage_states(
        has_mosaic=has_mosaic,
        has_embedding=has_embedding,
        has_prompts=has_prompts,
        has_candidate=has_candidate,
        catalog_count=catalog_count,
    )
    state_labels = {
        "done": "Completada",
        "active": "En curso",
        "pending": "Pendiente",
    }
    cards: List[str] = []
    for stage, state in zip(STAGES, states):
        badge = "✓" if state == "done" else str(stage["num"])
        tooltip = html.escape(stage["detail"])
        title = html.escape(stage["title"])
        cards.append(
            f'<div class="gis-step gis-step--{state}" data-tooltip="{tooltip}" title="{tooltip}">'
            f'  <span class="gis-step__badge">{badge}</span>'
            f'  <div class="gis-step__text">'
            f'    <span class="gis-step__title">{title}</span>'
            f'    <span class="gis-step__state">{state_labels[state]}</span>'
            f"  </div>"
            f"</div>"
        )
    st.markdown(f'<nav class="gis-stepper" aria-label="Flujo de trabajo">{"".join(cards)}</nav>', unsafe_allow_html=True)


def render_tiles_kpis(n_tiles: int, width_px: int, height_px: int, sam_px: int, gs_m: Optional[float] = None) -> None:
    """Cuadrícula de 3 mini-indicadores para la zona de estudio en la barra lateral."""
    res_txt = f"{gs_m:.2f} m/px" if gs_m is not None and gs_m > 0 else f"{sam_px} px"
    st.markdown(
        f"""
        <div class="gis-kpi-grid">
            <div class="gis-kpi" title="Número de teselas TMS a descargar">
                <div class="gis-kpi__val">{n_tiles}</div>
                <div class="gis-kpi__lbl">Teselas</div>
            </div>
            <div class="gis-kpi" title="Resolución total del mosaico descargado">
                <div class="gis-kpi__val">{width_px}×{height_px}</div>
                <div class="gis-kpi__lbl">Tamaño (px)</div>
            </div>
            <div class="gis-kpi" title="Entrada normalizada de SAM 2 ({sam_px}×{sam_px} px)">
                <div class="gis-kpi__val">{html.escape(res_txt)}</div>
                <div class="gis-kpi__lbl">Escala SAM 2</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(model_info: Dict[str, Any], downloaded: bool, is_active: bool) -> None:
    """Ficha compacta del modelo elegido en el selector sin cargarlo automáticamente."""
    size_mb = int(model_info.get("size_mb", 0) or 0)
    ram_gb = float(model_info.get("ram_gb", 0.0) or 0.0)
    embed_cpu = html.escape(str(model_info.get("embed_s_cpu", "—")))
    disk_txt = "En disco (listo)" if downloaded else f"Pendiente de descarga (~{size_mb} MB)"
    active_txt = " · Activo" if is_active else ""
    st.markdown(
        f"""
        <div class="gis-model-card">
            <div class="gis-model-card__row"><span class="gis-model-card__k">Checkpoint</span><span class="gis-model-card__v">~{size_mb} MB</span></div>
            <div class="gis-model-card__row"><span class="gis-model-card__k">RAM estimada</span><span class="gis-model-card__v">~{ram_gb:.1f} GB</span></div>
            <div class="gis-model-card__row"><span class="gis-model-card__k">Embedding CPU</span><span class="gis-model-card__v">{embed_cpu}</span></div>
            <div class="gis-model-card__row"><span class="gis-model-card__k">Estado</span><span class="gis-model-card__v">{html.escape(disk_txt + active_txt)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_map_legend(summary: Dict[str, Any]) -> None:
    """Leyenda compacta debajo del mapa + contadores de prompts activos."""
    pos = int(summary.get("positivos", summary.get("positive_points", 0)))
    neg = int(summary.get("negativos", summary.get("negative_points", 0)))
    has_box = int(summary.get("caja", 1 if summary.get("has_box") else 0))
    st.markdown(
        f"""
        <div class="gis-map-legend">
            <div class="gis-legend-items">
                <span class="gis-legend-item"><span class="gis-swatch" style="background:#00C853;"></span>Punto +</span>
                <span class="gis-legend-item"><span class="gis-swatch" style="background:#D50000;"></span>Punto −</span>
                <span class="gis-legend-item"><span class="gis-swatch gis-swatch--box" style="background:#FFD600;"></span>Caja</span>
                <span class="gis-legend-item"><span class="gis-swatch" style="background:#00FFFF;"></span>Candidata</span>
                <span class="gis-legend-item"><span class="gis-swatch" style="background:linear-gradient(135deg,#00E5FF 50%,#FF6D00 50%);"></span>Objetos</span>
            </div>
            <div class="gis-counters">
                <span class="gis-counter-badge" title="Puntos positivos">➕ {pos}</span>
                <span class="gis-counter-badge" title="Puntos negativos">➖ {neg}</span>
                <span class="gis-counter-badge" title="Caja delimitadora">▭ {has_box}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_inspector(has_mosaic: bool, mode: str, refining_id: Optional[str] = None) -> None:
    """Guía de 2 líneas contextual según el estado actual del flujo y herramienta."""
    if not has_mosaic:
        msg = (
            "<b>1. Cargue una ortofoto:</b> defina la zona de estudio en la barra lateral y pulse "
            "<i>Descargar y cargar ortofoto</i>.<br>"
            "<b>2. Segmente:</b> luego haga clic o trace una caja directamente sobre el mapa."
        )
    elif refining_id:
        safe_id = html.escape(refining_id)
        msg = (
            f"<b>Refinando {safe_id}:</b> agregue puntos positivos o negativos sobre el objeto.<br>"
            "Acepte el objeto al terminar para actualizar su versión en el catálogo."
        )
    elif mode == "pos":
        msg = (
            "<b>Herramienta Incluir:</b> haga clic sobre el objeto de interés para generar 3 máscaras.<br>"
            "Cambie a <i>Excluir</i> si necesita recortar sombras, patios o bordes vecinos."
        )
    elif mode == "neg":
        msg = (
            "<b>Herramienta Excluir:</b> haga clic sobre áreas que NO pertenecen al objeto.<br>"
            "Si aún no hay máscara previa, cambie a <i>Incluir</i> o <i>Caja (2 clics)</i>."
        )
    else:
        msg = (
            "<b>Herramienta Caja (2 clics):</b> marque la esquina superior izquierda y luego la opuesta.<br>"
            "Ideal para edificios, piscinas o lotes con contorno rectangular definido."
        )
    st.markdown(f'<div class="gis-inspector-empty">{msg}</div>', unsafe_allow_html=True)

