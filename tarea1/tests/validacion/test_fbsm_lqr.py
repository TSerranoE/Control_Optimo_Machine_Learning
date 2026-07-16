"""Validación de FBSM frente a la solución LQR vía Riccati."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from metodos_optimizacion import fbsm
from problemas_control import ProblemaLQR
from validacion_problema3 import crear_problema_lqr_fbsm


@pytest.fixture
def lqr_riccati_referencia():
    return ProblemaLQR(
        A=np.array([[-1.0]]), B=np.array([[1.0]]), Q=np.array([[1.0]]),
        R=np.array([[1.0]]), S=np.array([[1.0]]), t_span=(0.0, 2.0),
        x0=np.array([1.0]), h=1e-4,
    )


@pytest.fixture
def lqr_fbsm_problema():
    return crear_problema_lqr_fbsm(
        a=-1.0, b=1.0, q=1.0, r=1.0, s=1.0, T=2.0, x0=1.0,
    )


def _error_l2(problema_fbsm, referencia, h, metodo="rk4"):
    tiempos = np.linspace(0.0, 2.0, int(round(2.0 / h)) + 1)
    resultado = fbsm(
        problema_fbsm, np.zeros((tiempos.size, 1)), h, metodo,
        max_iter=100, tol=1e-10,
    )

    def lazo_cerrado(t, x):
        u = referencia.control_optimo_puntual(t, x, np.zeros(1))
        return referencia._f(t, x, u)

    estado_ref = solve_ivp(
        lazo_cerrado, referencia._t_span, referencia._x0, t_eval=tiempos,
        rtol=1e-11, atol=1e-13,
    ).y.T
    control_ref = np.array([
        referencia.control_optimo_puntual(t, x, np.zeros(1))
        for t, x in zip(tiempos, estado_ref)
    ])
    error = np.sqrt(np.trapezoid((resultado.control_optimo[:, 0] - control_ref[:, 0]) ** 2, tiempos))
    costo_ref = np.trapezoid(
        0.5 * (estado_ref[:, 0] ** 2 + control_ref[:, 0] ** 2), tiempos,
    ) + 0.5 * estado_ref[-1, 0] ** 2
    return float(error), resultado, control_ref, float(costo_ref)


def test_fbsm_vs_riccati_l2_error(lqr_fbsm_problema, lqr_riccati_referencia):
    error, resultado, control_ref, costo_ref = _error_l2(
        lqr_fbsm_problema, lqr_riccati_referencia, 0.005,
    )
    assert resultado.convergio
    assert control_ref.shape == resultado.control_optimo.shape
    assert error < 0.01
    assert resultado.historia_costo[-1] == pytest.approx(costo_ref, rel=1e-3)


def test_fbsm_l2_error_decreases_with_h(lqr_fbsm_problema, lqr_riccati_referencia):
    errores = [
        _error_l2(lqr_fbsm_problema, lqr_riccati_referencia, h)[0]
        for h in (0.1, 0.05, 0.025, 0.01)
    ]
    assert np.all(np.diff(errores) < 0.0)


def test_fbsm_convergence_rate_euler(lqr_fbsm_problema, lqr_riccati_referencia):
    hs = (0.05, 0.025, 0.0125)
    errores = [
        _error_l2(lqr_fbsm_problema, lqr_riccati_referencia, h, "euler_progresivo")[0]
        for h in hs
    ]
    tasas = np.log(np.array(errores[:-1]) / errores[1:]) / np.log(2.0)
    assert np.min(tasas) >= 0.8


def test_fbsm_convergence_rate_rk4(lqr_fbsm_problema, lqr_riccati_referencia):
    hs = (0.1, 0.05, 0.025)
    errores = [_error_l2(lqr_fbsm_problema, lqr_riccati_referencia, h)[0] for h in hs]
    tasas = np.log(np.array(errores[:-1]) / errores[1:]) / np.log(2.0)
    assert np.min(tasas) >= 1.5


def test_l2_norm_trapezoidal_consistency():
    t = np.array([0.0, 0.25, 1.0])
    u1 = np.array([0.0, 1.0, 2.0])
    u2 = np.array([0.5, 0.5, 0.5])
    calculado = np.sqrt(np.trapezoid((u1 - u2) ** 2, t))
    assert calculado == pytest.approx(1.0)


def test_lqr_fbsm_minimizador_depende_del_adjunto(lqr_fbsm_problema):
    u1 = lqr_fbsm_problema.control_optimo_puntual(0.0, np.array([1.0]), np.array([1.0]))
    u2 = lqr_fbsm_problema.control_optimo_puntual(0.0, np.array([1.0]), np.array([2.0]))
    np.testing.assert_allclose(u1, [-1.0])
    np.testing.assert_allclose(u2, [-2.0])
