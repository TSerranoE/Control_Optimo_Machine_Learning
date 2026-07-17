"""Contratos de ejecución y figuras del reporte del Problema 3."""

import ast
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pytest

from integradores import EDOSolver
import reporte_problema3 as reporte


@pytest.fixture(scope="module")
def reporte_rapido(tmp_path_factory):
    ruta = tmp_path_factory.mktemp("reporte_problema3")
    resumen = reporte.generar_reporte_problema3(ruta, modo_rapido=True)
    yield ruta, resumen
    for figura in resumen["figuras"]:
        plt.close(figura)


def _assert_png_valido(ruta: Path) -> None:
    assert ruta.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert ruta.stat().st_size > 10_000


def test_reporte_problema3_no_importa_ni_usa_solve_ivp():
    fuente = Path(reporte.__file__).read_text(encoding="utf-8")

    assert "solve_ivp" not in fuente


def test_comparar_lqr_usa_solver_inyectado_con_grilla_rk4(monkeypatch):
    solver = EDOSolver()
    solve_original = solver.solve
    llamadas = []

    def registrar_solve(*args, **kwargs):
        llamadas.append((args, kwargs))
        return solve_original(*args, **kwargs)

    problema_lqr = reporte.ProblemaLQR
    monkeypatch.setattr(solver, "solve", registrar_solve)
    monkeypatch.setattr(
        reporte,
        "ProblemaLQR",
        lambda *args, **kwargs: problema_lqr(*args, **kwargs, solver=solver),
    )

    tiempos, resultado, control_ref, error = reporte._comparar_lqr(0.1)
    campo, _, t_span, paso = llamadas[-1][0]

    assert len(llamadas) == 2
    assert llamadas[-1][1]["method"] == "rk4"
    assert t_span == (0.0, 2.0)
    assert paso == pytest.approx(0.1)
    assert campo(0.0, np.array([1.0]), None).shape == (1,)
    np.testing.assert_array_equal(tiempos, np.linspace(0.0, 2.0, 21))
    assert resultado.estado.shape == (tiempos.size, 1)
    assert resultado.control_optimo.shape == control_ref.shape == (tiempos.size, 1)
    assert np.isfinite(error)


def test_reporte_conserva_cinco_png_y_figuras_renderizables(reporte_rapido):
    ruta, resumen = reporte_rapido
    nombres = {
        "3a_fbsm_trayectorias.png",
        "3b_fbsm_vs_riccati.png",
        "3b_error_l2_vs_h.png",
        "3c_sir_trayectorias.png",
        "3c_sir_comparacion_ab.png",
    }

    assert {archivo.name for archivo in ruta.glob("*.png")} == nombres
    assert len(resumen["figuras"]) == 5
    assert all(isinstance(figura, Figure) for figura in resumen["figuras"])
    assert all(plt.fignum_exists(figura.number) for figura in resumen["figuras"])


def test_notebook_muestra_y_cierra_figuras_sin_ipython(monkeypatch):
    ruta_notebook = Path("tarea1/notebooks/ejecucion_tarea1.py")
    codigo = ruta_notebook.read_text(encoding="utf-8")
    modulo = ast.parse(codigo)
    nodos = [
        nodo
        for nodo in modulo.body
        if (
            isinstance(nodo, ast.Assign)
            and any(
                isinstance(destino, ast.Name)
                and destino.id == "figuras_problema3"
                for destino in nodo.targets
            )
        )
        or (
            isinstance(nodo, ast.If)
            and "plt.show" in ast.unparse(nodo)
        )
        or (
            isinstance(nodo, ast.For)
            and ast.unparse(nodo.iter) == "figuras_problema3"
        )
    ]
    figuras = [object() for _ in range(5)]
    llamadas_show = []
    figuras_cerradas = []

    monkeypatch.setitem(sys.modules, "ipykernel", object())
    monkeypatch.setattr(plt, "show", lambda **kwargs: llamadas_show.append(kwargs))
    monkeypatch.setattr(plt, "close", figuras_cerradas.append)
    exec(
        compile(ast.Module(body=nodos, type_ignores=[]), str(ruta_notebook), "exec"),
        {"plt": plt, "resumen_problema3": {"figuras": figuras}, "sys": sys},
    )

    assert "IPython" not in codigo
    assert llamadas_show == [{"block": False}]
    assert figuras_cerradas == figuras


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
