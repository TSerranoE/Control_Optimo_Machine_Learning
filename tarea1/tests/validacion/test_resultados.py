"""Contratos estructurales de los resultados reportables del Problema 1."""

from pathlib import Path

import numpy as np
import pytest

from utils.resultados import (
    COLUMNAS_RESULTADOS,
    ResultadoInvalidoError,
    renderizar_latex,
    serializar_csv,
    tabla_resultados,
)

RAIZ = Path(__file__).resolve().parents[3]
NOTEBOOK = RAIZ / "tarea1/notebooks/ejecucion_tarea1.py"
ASSETS = "tarea1/informe/assets/generated/problema1"


def _resultados():
    return [
        {"metodo": "rk4", "h": 0.1, "error_inf": 0.0125, "tiempo_s": 0.25},
        {"metodo": "euler", "h": 0.05, "error_inf": np.nan, "tiempo_s": 1.5},
        {"metodo": "heun", "h": 0.025, "error_inf": np.inf, "tiempo_s": None},
        {"metodo": "cn", "h": 0.0125, "error_inf": -np.inf, "tiempo_s": 2.0},
    ]


def test_tabla_deriva_estado_formatea_y_preserva_orden():
    tabla = tabla_resultados(_resultados())

    assert tuple(tabla.columns) == COLUMNAS_RESULTADOS == (
        "metodo", "h", "estado", "error_inf", "tiempo_s"
    )
    assert tabla["metodo"].tolist() == ["rk4", "euler", "heun", "cn"]
    assert tabla["h"].tolist() == [
        "1.00000000e-01", "5.00000000e-02", "2.50000000e-02", "1.25000000e-02"
    ]
    assert tabla["estado"].tolist() == ["finito", "no_finito", "no_finito", "no_finito"]
    assert tabla["error_inf"].tolist() == ["1.25000000e-02", "NaN", "+Inf", "-Inf"]
    assert tabla["tiempo_s"].tolist() == [
        "2.50000000e-01", "1.50000000e+00", "No disponible", "2.00000000e+00"
    ]


@pytest.mark.parametrize("cambio", [
    {"estado": "finito"}, {"metodo": ""}, {"h": np.inf},
    {"error_inf": "NaN"}, {"tiempo_s": -0.1}, {"tiempo_s": np.nan},
])
def test_tabla_rechaza_estado_suministrado_y_valores_invalidos(cambio):
    with pytest.raises(ResultadoInvalidoError):
        tabla_resultados([_resultados()[0] | cambio])


def test_csv_es_determinista_y_preserva_no_finitos(tmp_path):
    csv = serializar_csv(tabla_resultados(_resultados()))
    ruta = tmp_path / "resultados.csv"
    ruta.write_bytes(csv.encode())

    assert csv.splitlines() == [
        "metodo,h,estado,error_inf,tiempo_s",
        "rk4,1.00000000e-01,finito,1.25000000e-02,2.50000000e-01",
        "euler,5.00000000e-02,no_finito,NaN,1.50000000e+00",
        "heun,2.50000000e-02,no_finito,+Inf,No disponible",
        "cn,1.25000000e-02,no_finito,-Inf,2.00000000e+00",
    ]
    assert b"\r" not in ruta.read_bytes()


def test_latex_escapa_texto_y_declara_limitaciones():
    tabla = tabla_resultados([{
        "metodo": "rk4_#%&${}~^\\", "h": 0.1,
        "error_inf": np.nan, "tiempo_s": 0.25,
    }, {"metodo": "cn", "h": 0.05, "error_inf": np.inf, "tiempo_s": None}])
    latex = renderizar_latex(tabla)

    assert r"rk4\_\#\%\&\$\{\}\textasciitilde{}\textasciicircum{}\textbackslash{}" in latex
    assert "No finito (NaN)" in latex and "No finito (+Inf)" in latex
    assert "mu=1000" in latex and r"\label{tab:van-der-pol-mu-1000}" in latex
    assert "tiempo_s is one observed execution, not a stable benchmark." in latex


def test_formatos_comparten_orden_estado_evidencia_y_tiempo():
    tabla = tabla_resultados(_resultados())
    csv, latex = serializar_csv(tabla), renderizar_latex(tabla)
    for valor in ("rk4", "euler", "finito", "NaN", "+Inf", "-Inf",
                  "2.50000000e-01", "No disponible"):
        assert valor in csv and valor in latex
    assert "no_finito" in csv and r"no\_finito" in latex


def test_import_compatible_reexporta_helper_canonico():
    from utils.visualizacion import tabla_resultados as compatible

    assert compatible is tabla_resultados
    assert tuple(compatible(_resultados()).columns) == COLUMNAS_RESULTADOS


def test_notebook_configura_paths_locales_antes_de_imports():
    fuente = NOTEBOOK.read_text(encoding="utf-8")
    for linea in ('TAREA1_DIR = Path(__file__).resolve().parents[1]',
                  'SRC_DIR = TAREA1_DIR / "src"',
                  'sys.path.insert(0, str(TAREA1_DIR))',
                  'sys.path.insert(0, str(SRC_DIR))'):
        assert linea in fuente
        assert fuente.index(linea) < fuente.index("from src.integradores import EDOSolver")


def test_notebook_declara_configuracion_y_rutas_reportables():
    fuente = NOTEBOOK.read_text(encoding="utf-8")
    for texto in ('"alpha": 1.1', '"beta": 0.4', '"delta": 0.1', '"gamma": 0.4',
                  "x0_lotka = np.array([10.0, 5.0])", "complementario",
                  "one observed execution, not a stable benchmark", "limitación"):
        assert texto.lower() in fuente.lower()
    for nombre in ("lotka_volterra_error_vs_h.png", "lotka_volterra_series_temporales.png",
                   "lotka_volterra_diagrama_fase.png", "van_der_pol_mu_1000_time_precision.csv",
                   "van_der_pol_mu_1000_time_precision.tex"):
        assert f'RUTA_ASSETS_PROBLEMA1 / "{nombre}"' in fuente


def test_gitignore_permite_solo_los_cinco_assets():
    reglas = (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()
    esperadas = [
        "tarea1/informe/assets/generated/*", f"!{ASSETS}/", f"{ASSETS}/*",
        f"!{ASSETS}/lotka_volterra_error_vs_h.png",
        f"!{ASSETS}/lotka_volterra_series_temporales.png",
        f"!{ASSETS}/lotka_volterra_diagrama_fase.png",
        f"!{ASSETS}/van_der_pol_mu_1000_time_precision.csv",
        f"!{ASSETS}/van_der_pol_mu_1000_time_precision.tex",
    ]
    assert reglas[-8:] == esperadas
    assert f"!{ASSETS}/*" not in reglas


def test_pruebas_estructurales_no_ejecutan_simulacion_stiff():
    fuente = Path(__file__).read_text(encoding="utf-8")
    assert "src." + "validacion_problema1" not in fuente
    assert "ejecutar_" + "experimento(" not in fuente
