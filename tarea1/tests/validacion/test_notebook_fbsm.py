"""Contratos de ejecución y figuras del reporte del Problema 3."""

from pathlib import Path

import pytest

from reporte_problema3 import generar_reporte_problema3


@pytest.fixture(scope="module")
def reporte_rapido(tmp_path_factory):
    ruta = tmp_path_factory.mktemp("reporte_problema3")
    resumen = generar_reporte_problema3(ruta, modo_rapido=True)
    return ruta, resumen


def _assert_png_valido(ruta: Path) -> None:
    assert ruta.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert ruta.stat().st_size > 10_000


def test_notebook_3a_plots_exist(reporte_rapido):
    ruta, resumen = reporte_rapido
    _assert_png_valido(ruta / "3a_fbsm_trayectorias.png")
    assert resumen["3a"]["convergio"]
    assert resumen["3a"]["iteraciones"] >= 2
    assert resumen["3a"]["costo_final"] < resumen["3a"]["costo_inicial"]


def test_notebook_3b_comparison(reporte_rapido):
    ruta, resumen = reporte_rapido
    _assert_png_valido(ruta / "3b_fbsm_vs_riccati.png")
    _assert_png_valido(ruta / "3b_error_l2_vs_h.png")
    assert resumen["3b"]["error_h_001"] < 0.01
    assert all(
        error_fino < error_grueso
        for error_grueso, error_fino in zip(
            resumen["3b"]["errores_l2"], resumen["3b"]["errores_l2"][1:]
        )
    )


def test_notebook_3c_sir_plots(reporte_rapido):
    ruta, resumen = reporte_rapido
    _assert_png_valido(ruta / "3c_sir_trayectorias.png")
    _assert_png_valido(ruta / "3c_sir_comparacion_ab.png")
    assert resumen["3c"]["convergio"]
    assert resumen["3c"]["omega"] == pytest.approx(0.2)
    assert resumen["3c"]["min_estado"] >= 0.0
    assert resumen["3c"]["control_max"] <= 0.4 + 1e-12
    assert resumen["3c"]["control_medio_alto"] > resumen["3c"]["control_medio_bajo"]


def test_notebook_documenta_secciones_y_riccati():
    notebook = Path("tarea1/notebooks/ejecucion_tarea1.py").read_text(encoding="utf-8")
    assert "## Problema 3a" in notebook
    assert "## Problema 3b" in notebook
    assert "## Problema 3c" in notebook
    assert r"-\dot{P}" in notebook
    assert "error en norma" in notebook.lower()
    assert "omega=0.2" in notebook
