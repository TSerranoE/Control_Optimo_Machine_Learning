from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import reporte_problema4 as reporte


class _BoxProblem:
    _m = 2
    _n = 1

    @staticmethod
    def _dl_du(t, x, u):
        return np.asarray(u, dtype=float)

    @staticmethod
    def _df_du(t, x, u):
        return np.array([[1.0, -1.0]])

    @staticmethod
    def proj(u, metodo_integracion):
        return np.clip(np.asarray(u, dtype=float), 0.0, 1.0)


def test_contract_result_is_frozen_and_copies_mutable_tables(tmp_path):
    tabla_lqr = pd.DataFrame({"method": ["FBSM"]})
    resultado = reporte.ResultadoComparacion((), tabla_lqr, pd.DataFrame(), tmp_path)
    tabla_lqr.loc[0, "method"] = "changed"

    assert resultado.figuras == ()
    assert resultado.tabla_lqr.loc[0, "method"] == "FBSM"
    assert resultado.ruta_salida == tmp_path
    with pytest.raises(FrozenInstanceError):
        resultado.ruta_salida = Path("other")


def test_contract_exposes_only_approved_public_function():
    assert reporte.__all__ == ["ResultadoComparacion", "generar_reporte_problema4"]


@pytest.mark.parametrize(
    ("ruta", "mensaje"),
    [
        (lambda root: root / "missing" / "report", "parent"),
        (lambda root: root / "output.txt", "directory"),
    ],
)
def test_validation_report_boundary_rejects_invalid_output_before_runner(
    tmp_path, ruta, mensaje
):
    destino = ruta(tmp_path)
    if destino.suffix:
        destino.write_text("not a directory")

    with pytest.raises(ValueError, match=mensaje):
        reporte.generar_reporte_problema4(destino, modo_rapido=True)


@pytest.mark.parametrize(
    ("tiempos", "h", "mensaje"),
    [
        ([0.0], 0.1, "two"),
        ([0.0, 0.1], 0.0, "positive"),
        ([0.0, 0.1, 0.25], 0.1, "aligned"),
        ([0.0, np.inf], 0.1, "finite"),
    ],
)
def test_validation_grid_rejects_invalid_values(tiempos, h, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        reporte._validar_grilla(np.array(tiempos), h)


def test_validation_trajectory_normalizes_scalar_and_accepts_vector_controls():
    tiempos = np.array([0.0, 0.5, 1.0])
    scalar = reporte._validar_trayectorias(
        tiempos, np.array([0.0, 0.5, 1.0]), np.ones((3, 1)), np.ones((3, 1))
    )
    vector = reporte._validar_trayectorias(
        tiempos, np.ones((3, 2)), np.ones((3, 1)), np.ones((3, 1))
    )

    assert scalar[0].shape == (3, 1)
    assert vector[0].shape == (3, 2)
    assert not np.shares_memory(scalar[0], np.array([0.0, 0.5, 1.0]))


@pytest.mark.parametrize(
    ("control", "estados", "adjuntos", "mensaje"),
    [
        (np.ones((2, 1)), np.ones((3, 1)), np.ones((3, 1)), "nodes"),
        (np.ones((3, 1, 1)), np.ones((3, 1)), np.ones((3, 1)), "control"),
        (np.ones((3, 1)), np.ones(3), np.ones((3, 1)), "states"),
        (np.ones((3, 1)), np.ones((3, 1)), np.full((3, 1), np.nan), "finite"),
    ],
)
def test_validation_trajectory_rejects_shapes_and_nonfinite(
    control, estados, adjuntos, mensaje
):
    with pytest.raises(ValueError, match=mensaje):
        reporte._validar_trayectorias(
            np.array([0.0, 0.5, 1.0]), control, estados, adjuntos
        )


def test_metric_common_trapezoidal_cost_triangulates_nonuniform_grid():
    assert reporte._costo_trapezoidal(
        np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 2.0]), 3.0
    ) == pytest.approx(4.0)
    assert reporte._costo_trapezoidal(
        np.array([0.0, 0.25, 1.0]), np.array([2.0, 2.0, 2.0]), 0.5
    ) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="finite"): reporte._costo_trapezoidal(np.array([0.0, 1.0]), np.array([0.0, np.nan]), 0.0)


def test_metric_projected_stationarity_handles_vector_and_scalar_controls():
    tiempos = np.array([0.0, 0.5, 1.0])
    estados = np.zeros((3, 1))
    adjuntos = np.ones((3, 1))
    vector = np.zeros((3, 2))

    residual = reporte._residuo_estacionariedad(
        _BoxProblem(), tiempos, estados, vector, adjuntos, "rk4"
    )
    assert residual == pytest.approx(1.0)

    class ScalarProblem(_BoxProblem):
        _m = 1

        @staticmethod
        def _dl_du(t, x, u):
            return np.asarray(u)

        @staticmethod
        def _df_du(t, x, u):
            return np.array([[0.0]])

    assert reporte._residuo_estacionariedad(
        ScalarProblem(), tiempos, estados, np.zeros(3), np.zeros((3, 1)), "rk4"
    ) == pytest.approx(0.0)


@pytest.mark.parametrize(("bad_cost", "bad_reference"), [(np.nan, 0.0), (np.inf, 0.0), (-np.inf, 0.0), (0.0, np.nan), (0.0, np.inf), (0.0, -np.inf)])
def test_metric_riccati_l2_and_relative_gap_include_zero_reference_guard(bad_cost, bad_reference):
    tiempos = np.array([0.0, 0.5, 1.0])
    l2, brecha = reporte._metricas_riccati(
        tiempos, np.ones(3), np.zeros((3, 1)), 3.0, 2.0
    )
    assert l2 == pytest.approx(1.0)
    assert brecha == pytest.approx(0.5)
    _, brecha_cero = reporte._metricas_riccati(
        tiempos, np.zeros(3), np.zeros(3), np.finfo(float).eps, 0.0
    )
    assert brecha_cero == pytest.approx(1.0)
    with pytest.raises(ValueError, match="finite"):
        reporte._metricas_riccati(tiempos, np.zeros(3), np.zeros(3), bad_cost, bad_reference)


def test_control_riccati_uses_reference_feedback():
    class Reference:
        control_riccati = staticmethod(lambda t, x: np.array([t + x[0]]))
        control_optimo_puntual = staticmethod(lambda *args: pytest.fail("pointwise control used"))
        _f = staticmethod(lambda t, x, u: x + u)

    control = reporte._control_riccati(Reference(), np.array([0.0, 1.0]), np.array([[2.0], [3.0]]))
    np.testing.assert_allclose(control, [[2.0], [4.0]])
    np.testing.assert_allclose(reporte._dinamica_riccati(Reference(), 1.0, np.array([3.0])), [7.0])


@pytest.mark.parametrize("tiempos", [[1.0, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, np.nan, 1.0]])
def test_validation_metric_paths_reject_invalid_time_grids(tiempos):
    datos = np.zeros((3, 1))
    with pytest.raises(ValueError, match="grid"):
        reporte._residuo_estacionariedad(
            _BoxProblem(), np.array(tiempos), datos, np.zeros((3, 2)), datos, "rk4"
        )
    with pytest.raises(ValueError, match="grid"):
        reporte._metricas_riccati(np.array(tiempos), datos, datos, 0.0, 0.0)
    with pytest.raises(ValueError, match="grid"): reporte._costo_trapezoidal(np.array(tiempos), np.zeros(3), 0.0)


def test_metric_native_labels_order_and_exploratory_solver_timing(monkeypatch):
    reloj = iter([10.0, 10.25])
    monkeypatch.setattr(reporte, "perf_counter", lambda: next(reloj))

    valor, duracion = reporte._cronometrar_solver(lambda: "solved")
    control = np.zeros((2, 1))
    filas = [
        reporte._FilaComparacion("Projected gradient", control, np.zeros((2, 1)), np.zeros((2, 1)), 1.0, True, 2, reporte._etiqueta_historia("Projected gradient", 2), duracion, 10.0),
        reporte._FilaComparacion("FBSM", np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1)), 1.0, True, 3, reporte._etiqueta_historia("FBSM", 3), None, 1.0),
    ]

    assert valor == "solved"
    assert duracion == pytest.approx(0.25)
    assert [fila.metodo for fila in reporte._ordenar_filas(filas)] == ["FBSM", "Projected gradient"]
    assert filas[0].etiqueta_historia == "Projected gradient initial control k=0; accepted controls k=1..2"
    assert filas[1].etiqueta_historia == "FBSM accepted control k=1..3"
    assert filas[0].tiempo_exploratorio_segundos == pytest.approx(0.25)
    control[0, 0] = 9.0
    assert filas[0].control[0, 0] == 0.0
    assert not filas[0].control.flags.writeable


class _RunnerProblem(_BoxProblem):
    def __init__(self):
        self.projected_calls = []

    def evaluar_costo(self, control, h, metodo_integracion):
        return float(np.sum(np.asarray(control)) + h)

    def gradiente_proyectado(self, control, **kwargs):
        self.projected_calls.append((control.copy(), kwargs))
        return SimpleNamespace(
            control=np.full_like(control, 0.25),
            estados=np.zeros((len(control), 1)),
            adjuntos=np.zeros((len(control), 1)),
            iteraciones=4,
            convergio=True,
        )


def test_runner_uses_fresh_zero_controls_and_native_solver_settings(monkeypatch):
    problema = _RunnerProblem()
    fbsm_calls = []

    def fake_fbsm(problem, control, h, method, **kwargs):
        fbsm_calls.append((problem, control.copy(), h, method, kwargs))
        control[:] = 9.0
        return SimpleNamespace(
            control_optimo=np.full_like(control, 0.5),
            estado=np.zeros((len(control), 1)),
            adjunto=np.zeros((len(control), 1)),
            iteraciones=3,
            convergio=False,
        )

    monkeypatch.setattr(reporte, "fbsm", fake_fbsm)
    filas = reporte._ejecutar_metodos(
        problema, np.linspace(0.0, 8.0, 161), 0.05,
        "crank_nicolson", max_iter=200, tol=1e-6, omega=0.2,
        orden_caso=1.0,
    )

    assert np.all(fbsm_calls[0][1] == 0.0)
    assert fbsm_calls[0][2:] == (0.05, "crank_nicolson", {"max_iter": 200, "tol": 1e-6, "omega": 0.2})
    projected_control, projected_kwargs = problema.projected_calls[0]
    assert np.all(projected_control == 0.0)
    assert projected_kwargs == {
        "max_iter": 200, "tolerancia": 1e-6,
        "metodo_integracion": "crank_nicolson", "r": 10, "a": 1e-4,
        "b": 0.5, "t_min": 1e-12, "max_reducciones": 50,
    }
    assert [fila.metodo for fila in filas] == ["FBSM", "Projected gradient"]
    assert [fila.iteraciones_nativas for fila in filas] == [3, 4]


def test_runner_sir_quick_mode_changes_only_horizon_and_keeps_case_order(monkeypatch):
    factory_calls = []

    def fake_factory(*args):
        factory_calls.append(args)
        return SimpleNamespace(_m=1)

    monkeypatch.setattr(reporte, "crear_problema_sir", fake_factory)
    monkeypatch.setattr(reporte, "_ejecutar_metodos", lambda *args, **kwargs: [])

    quick_grid, quick_rows, _ = reporte._resolver_sir(True)
    official_grid, official_rows, _ = reporte._resolver_sir(False)

    assert reporte._configuracion_ejecucion(True) == (0.05, 1e-5, 8.0, 0.5, 1e-6)
    assert reporte._configuracion_ejecucion(False) == (0.01, 1e-6, 50.0, 0.5, 1e-6)
    assert (quick_grid[-1], len(quick_grid), quick_rows) == (8.0, 17, [])
    assert (official_grid[-1], len(official_grid), official_rows) == (50.0, 101, [])
    assert [(call[2], call[-1]) for call in factory_calls] == [
        (1.0, 8.0), (10.0, 8.0), (1.0, 50.0), (10.0, 50.0),
    ]


def _artifact_inputs():
    lqr = pd.DataFrame([dict(zip(reporte._LQR_COLUMNS, ["LQR", "FBSM"] + [0.0] * 8))])
    sir = pd.DataFrame([dict(zip(reporte._SIR_COLUMNS, [1.0, "FBSM"] + [0.0] * 6 + [reporte._SIR_DISCLAIMER]))])
    figures = tuple(plt.subplots()[0] for _ in range(3))
    return figures, lqr, sir


def test_artifact_publication_replaces_stale_exact_seven_files(tmp_path):
    destino = tmp_path / "report"
    destino.mkdir()
    for nombre in reporte._ARTIFACT_NAMES:
        (destino / nombre).write_bytes(b"stale")
    figures, lqr, sir = _artifact_inputs()

    reporte._publicar(destino, figures, lqr, sir)

    assert {path.name for path in destino.iterdir()} == set(reporte._ARTIFACT_NAMES)
    for nombre in reporte._FIGURE_NAMES:
        assert (destino / nombre).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert list(pd.read_csv(destino / "lqr_comparison.csv").columns) == reporte._LQR_COLUMNS
    assert list(pd.read_csv(destino / "sir_comparison.csv").columns) == reporte._SIR_COLUMNS
    assert "tabular" in (destino / "lqr_comparison.tex").read_text()
    plt.close("all")


def test_atomic_publication_restores_all_stale_files_on_promotion_failure(tmp_path, monkeypatch):
    destino = tmp_path / "report"
    destino.mkdir()
    for nombre in reporte._ARTIFACT_NAMES:
        (destino / nombre).write_bytes(f"old:{nombre}".encode())
    figures, lqr, sir = _artifact_inputs()
    real_replace = reporte.os.replace
    promoted = 0

    def failing_replace(source, target):
        nonlocal promoted
        source, target = Path(source), Path(target)
        if source.parent != destino and target.parent == destino:
            assert source.parent.parent == destino.parent
            promoted += 1
            if promoted == 2:
                raise OSError("promotion failed")
        return real_replace(source, target)

    monkeypatch.setattr(reporte.os, "replace", failing_replace)
    with pytest.raises(OSError, match="promotion failed"):
        reporte._publicar(destino, figures, lqr, sir)

    assert all((destino / name).read_bytes() == f"old:{name}".encode() for name in reporte._ARTIFACT_NAMES)
    plt.close("all")
