"""Validación de FBSM frente a la solución LQR vía Riccati."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from metodos_optimizacion import fbsm
from problemas_control import ProblemaLQR


@pytest.fixture
def lqr_scalar_problema():
    return ProblemaLQR(
        A=np.array([[-1.0]]), B=np.array([[1.0]]), Q=np.array([[1.0]]),
        R=np.array([[1.0]]), S=np.array([[1.0]]), t_span=(0.0, 2.0),
        x0=np.array([1.0]), h=1e-4,
    )


def _error_l2(problema, h, metodo="rk4"):
    tiempos = np.linspace(0.0, 2.0, int(round(2.0 / h)) + 1)
    resultado = fbsm(
        problema, np.zeros((tiempos.size, 1)), h, metodo,
        max_iter=100, tol=1e-10,
    )

    def lazo_cerrado(t, x):
        u = problema.control_optimo_puntual(t, x, np.zeros(1))
        return problema._f(t, x, u)

    estado_ref = solve_ivp(
        lazo_cerrado, problema._t_span, problema._x0, t_eval=tiempos,
        rtol=1e-11, atol=1e-13,
    ).y.T
    control_ref = np.array([
        problema.control_optimo_puntual(t, x, np.zeros(1))
        for t, x in zip(tiempos, estado_ref)
    ])
    error = np.sqrt(np.trapezoid((resultado.control_optimo[:, 0] - control_ref[:, 0]) ** 2, tiempos))
    return float(error), resultado, control_ref


def test_fbsm_vs_riccati_l2_error(lqr_scalar_problema):
    error, resultado, control_ref = _error_l2(lqr_scalar_problema, 0.005)
    assert resultado.convergio
    assert control_ref.shape == resultado.control_optimo.shape
    assert error < 0.01


def test_fbsm_l2_error_decreases_with_h(lqr_scalar_problema):
    errores = [_error_l2(lqr_scalar_problema, h)[0] for h in (0.1, 0.05, 0.025, 0.01)]
    assert np.all(np.diff(errores) < 0.0)


def test_fbsm_convergence_rate_euler(lqr_scalar_problema):
    hs = (0.05, 0.025, 0.0125)
    errores = [_error_l2(lqr_scalar_problema, h, "euler_progresivo")[0] for h in hs]
    tasas = np.log(np.array(errores[:-1]) / errores[1:]) / np.log(2.0)
    assert np.min(tasas) >= 0.8


def test_fbsm_convergence_rate_rk4(lqr_scalar_problema):
    hs = (0.1, 0.05, 0.025)
    errores = [_error_l2(lqr_scalar_problema, h)[0] for h in hs]
    tasas = np.log(np.array(errores[:-1]) / errores[1:]) / np.log(2.0)
    assert np.min(tasas) >= 1.5


def test_l2_norm_trapezoidal_consistency():
    t = np.array([0.0, 0.25, 1.0])
    u1 = np.array([0.0, 1.0, 2.0])
    u2 = np.array([0.5, 0.5, 0.5])
    calculado = np.sqrt(np.trapezoid((u1 - u2) ** 2, t))
    assert calculado == pytest.approx(1.0)
