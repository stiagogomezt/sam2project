"""Pruebas unitarias de la interfaz de línea de comandos (CLI) de GEO-SAM."""

from click.testing import CliRunner

from cli.main import cli


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "GEO-SAM: Plataforma experimental de extracción de objetos geoespaciales" in result.output
    # Verificar que los subcomandos exigidos están presentes
    assert "ingest" in result.output
    assert "segment" in result.output
    assert "metrics" in result.output
    assert "validate" in result.output
    assert "export" in result.output


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "geosam, version 0.1.0" in result.output


def test_subcommand_helps() -> None:
    runner = CliRunner()
    for cmd in ["ingest", "segment", "metrics", "validate", "export"]:
        res = runner.invoke(cli, [cmd, "--help"])
        assert res.exit_code == 0, f"Fallo al ejecutar {cmd} --help"
