"""Punto de entrada principal para el CLI de GEO-SAM.

Subcomandos principales: ingest, segment, metrics, validate, export.
"""

from __future__ import annotations
import sys
import click

import geosam


@click.group()
@click.version_option(version=geosam.__version__, prog_name="geosam")
def cli() -> None:
    """GEO-SAM: Plataforma experimental de extracción de objetos geoespaciales.

    Asistida por modelos fundacionales de segmentación (SAM 2 / SAM 3).
    """
    pass


@cli.command()
@click.argument("raster_path", type=click.Path(exists=True))
@click.option("--project", "-p", required=True, help="Identificador o nombre del proyecto.")
@click.option("--sensor", help="Sensor satelital o aéreo (ej. Sentinel-2, PlanetScope, Drone).")
def ingest(raster_path: str, project: str, sensor: str | None) -> None:
    """Ingesta de rásteres georreferenciados (GeoTIFF/COG) o imágenes en el catálogo del proyecto."""
    click.echo(f"Ingestando ráster: {raster_path} en el proyecto '{project}' (sensor: {sensor or 'no especificado'})...")


@cli.command()
@click.option("--image-id", required=True, help="ID de la imagen a segmentar.")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["sam2", "sam3", "mock"]),
    default="mock",
    show_default=True,
    help="Backend de segmentación.",
)
@click.option("--prompt-type", type=click.Choice(["point", "box", "text", "auto"]), default="auto")
@click.option("--device", default="cpu", show_default=True, help="Dispositivo de cómputo (cpu, cuda).")
def segment(image_id: str, backend: str, prompt_type: str, device: str) -> None:
    """Ejecuta inferencia de segmentación asistida por puntos, cajas, texto o modo automático."""
    click.echo(f"Segmentando imagen {image_id} con backend={backend}, prompt={prompt_type}, device={device}...")


@cli.command()
@click.option("--project", "-p", required=True, help="Identificador del proyecto.")
@click.option("--crs-metric", default="EPSG:9377", show_default=True, help="CRS métrico proyectado para cálculos.")
def metrics(project: str, crs_metric: str) -> None:
    """Calcula métricas geométricas y topológicas (área, compacidad, elongación, etc.) sobre los objetos."""
    click.echo(f"Calculando métricas analíticas para el proyecto '{project}' usando CRS={crs_metric}...")


@cli.command()
@click.option("--project", "-p", required=True, help="Identificador del proyecto.")
@click.option("--reference", "-r", required=True, help="Capa de referencia (Overture, Catastro, OSM).")
def validate(project: str, reference: str) -> None:
    """Evalúa calidad geométrica (IoU, Boundary F1, sesgo de área) contrastando con capas de referencia."""
    click.echo(f"Validando objetos del proyecto '{project}' contra referencia '{reference}'...")


@cli.command()
@click.option("--project", "-p", required=True, help="Identificador del proyecto.")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["gpkg", "geojson", "coco", "yolo", "ladm"]),
    default="gpkg",
    show_default=True,
    help="Formato de exportación SIG o dataset.",
)
@click.option("--output", "-o", required=True, type=click.Path(), help="Ruta de destino del archivo generado.")
def export(project: str, format: str, output: str) -> None:
    """Exporta geometrías vectorizadas, métricas y metadatos a formatos SIG o datasets de entrenamiento."""
    click.echo(f"Exportando proyecto '{project}' a formato {format.upper()} en: {output}...")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
