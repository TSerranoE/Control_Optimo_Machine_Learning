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


def test_figuras_tienen_titulos_etiquetas_y_leyendas_descriptivas(reporte_rapido):
    _, resumen = reporte_rapido
    figuras = resumen["figuras"]

    for figura in figuras:
        titulos = [eje.get_title() for eje in figura.axes]
        assert all("Problema 3" not in titulo for titulo in titulos)
        assert figura._suptitle is None or "Problema 3" not in figura._suptitle.get_text()
        assert all(eje.get_legend() is not None for eje in figura.axes if eje.lines)
        assert all(eje.get_xlabel() and eje.get_ylabel() for eje in figura.axes if eje.lines)

    for figura in figuras[:3]:
        assert all("LQR" in eje.get_title() for eje in figura.axes if eje.lines)
    for figura in figuras[3:]:
        assert all("SIR" in eje.get_title() for eje in figura.axes if eje.lines)

    assert [eje.get_ylabel() for eje in figuras[0].axes] == [
        "Estado LQR x(t)", "Control LQR u*(t)", "Valor del costo J",
    ]
    assert figuras[3].axes[0].get_ylabel() == "Proporción poblacional"
    assert figuras[3].axes[1].get_ylabel() == "Tasa de vacunación u*(t)"
    leyendas = [
        [[texto.get_text() for texto in eje.get_legend().get_texts()] for eje in figura.axes]
        for figura in figuras
    ]
    assert leyendas == [
        [["Estado LQR por FBSM"], ["Control LQR por FBSM"], ["Costo evaluado por FBSM"]],
        [["Control por FBSM", "Control por Riccati"]],
        [["Error L² FBSM-Riccati"]],
        [["Susceptibles S(t)", "Infectados I(t)"], ["Control SIR óptimo u*(t)"]],
        [["Solución SIR, A/B=10", "Solución SIR, A/B=1"]],
    ]


def test_figuras_documentan_escalas_y_parametros(reporte_rapido):
    _, resumen = reporte_rapido
    figuras = resumen["figuras"]
    textos = [" ".join(texto.get_text() for texto in figura.texts) for figura in figuras]
    lqr_requeridos = ("A=-1", "B=1", "Q=1", "R=1", "S=1", "x0=1", "T=2")
    sir_requeridos = (
        "beta=0.3", "gamma=0.1", "B=1", "u_max=0.4", "S0=0.99", "I0=0.01",
        "omega=0.2", "h=0.1", "T=8",
    )

    assert all(all(parametro in texto for parametro in lqr_requeridos) for texto in textos[:3])
    assert "h=0.01" in textos[0] and "h=0.01" in textos[1]
    assert "h={0.1, 0.05, 0.025, 0.01}" in textos[2]
    assert "A=10" in textos[3]
    assert "A/B comparados: 10 y 1" in textos[4]
    assert all(all(parametro in texto for parametro in sir_requeridos) for texto in textos[3:])

    eje_error = figuras[2].axes[0]
    assert eje_error.get_xscale() == eje_error.get_yscale() == "log"
    assert "escala logarítmica" in eje_error.get_xlabel()
    assert "escala logarítmica" in eje_error.get_ylabel()
    assert "escala log-log" in textos[2]


def test_notebook_muestra_y_cierra_figuras_sin_ipython(monkeypatch):
    ruta_notebook = Path("tarea1/notebooks/ejecucion_tarea1.py")
    codigo = ruta_notebook.read_text(encoding="utf-8")
    modulo = ast.parse(codigo)
    inicio = next(
        indice for indice, nodo in enumerate(modulo.body)
        if isinstance(nodo, ast.Assign)
        and any(isinstance(destino, ast.Name) and destino.id == "figuras_problema3" for destino in nodo.targets)
    )
    nodos = modulo.body[inicio:inicio + 3]
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


def test_notebook_4c_uses_public_report_and_documents_interpretation():
    notebook = Path("tarea1/notebooks/ejecucion_tarea1.py").read_text(encoding="utf-8")

    assert "## Problema 4c" in notebook
    assert "from src.reporte_problema4 import generar_reporte_problema4" in notebook
    assert 'RUTA_BASE / "4_gradiente_proyectado"' in notebook
    assert "generar_reporte_problema4(" in notebook
    assert "modo_rapido=MODO_RAPIDO_PROBLEMA4" in notebook
    assert "tabla_lqr.to_string(index=False)" in notebook
    assert "tabla_sir.to_string(index=False)" in notebook
    assert "not evidence of global optimality" in notebook


def test_notebook_4c_shows_and_closes_all_report_figures(monkeypatch):
    codigo = Path("tarea1/notebooks/ejecucion_tarea1.py").read_text(encoding="utf-8")
    modulo = ast.parse(codigo)
    inicio = next(
        indice for indice, nodo in enumerate(modulo.body)
        if isinstance(nodo, ast.Assign)
        and any(isinstance(destino, ast.Name) and destino.id == "figuras_problema4" for destino in nodo.targets)
    )
    nodos = modulo.body[inicio:inicio + 3]
    figuras = [object() for _ in range(3)]
    llamadas_show, figuras_cerradas = [], []

    monkeypatch.setitem(sys.modules, "ipykernel", object())
    monkeypatch.setattr(plt, "show", lambda **kwargs: llamadas_show.append(kwargs))
    monkeypatch.setattr(plt, "close", figuras_cerradas.append)
    exec(
        compile(ast.Module(body=nodos, type_ignores=[]), "notebook-4c", "exec"),
        {"plt": plt, "resultado_problema4": type("R", (), {"figuras": figuras})(), "sys": sys},
    )

    assert llamadas_show == [{"block": False}]
    assert figuras_cerradas == figuras
