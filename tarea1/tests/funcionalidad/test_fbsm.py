"""Tests unitarios para el núcleo Forward-Backward Sweep Method (FBSM)."""

import numpy as np
import pytest

from integradores import EDOSolver
from metodos_optimizacion import ResultadoFBSM
from problemas_control import ProblemaLQR


@pytest.fixture
def lqr_scalar_problem():
    """Problema LQR escalar usado para validar FBSM."""
    return ProblemaLQR(
        A=np.array([[1.0]]),
        B=np.array([[1.0]]),
        Q=np.array([[1.0]]),
        R=np.array([[1.0]]),
        S=np.array([[1.0]]),
        t_span=(0.0, 1.0),
        x0=np.array([1.0]),
        h=1e-4,
    )


@pytest.fixture
def u_cero_lqr():
    """Control inicial nulo para el problema LQR escalar con h=0.01."""
    t0, tf = 0.0, 1.0
    h = 0.01
    N = int(np.round((tf - t0) / h))
    return np.zeros((N + 1, 1))


def test_resultado_fbsm_campos():
    """ResultadoFBSM debe exponer los campos esperados con tipos correctos."""
    u = np.array([[0.0], [1.0]])
    x = np.array([[1.0], [0.5]])
    p = np.array([[2.0], [1.5]])
    t = np.array([0.0, 1.0])
    historia_J = [10.0, 5.0]

    resultado = ResultadoFBSM(
        u=u, x=x, p=p, t=t, historia_J=historia_J, iteraciones=2, convergio=True
    )

    assert isinstance(resultado.u, np.ndarray)
    assert isinstance(resultado.x, np.ndarray)
    assert isinstance(resultado.p, np.ndarray)
    assert isinstance(resultado.t, np.ndarray)
    assert isinstance(resultado.historia_J, list)
    assert resultado.iteraciones == 2
    assert resultado.convergio is True


def test_resultado_fbsm_shapes():
    """ResultadoFBSM debe preservar las shapes (N+1, m), (N+1, n), (N+1, n)."""
    N = 10
    u = np.zeros((N + 1, 1))
    x = np.zeros((N + 1, 2))
    p = np.zeros((N + 1, 2))
    t = np.linspace(0.0, 1.0, N + 1)

    resultado = ResultadoFBSM(
        u=u, x=x, p=p, t=t, historia_J=[1.0], iteraciones=1, convergio=False
    )

    assert resultado.u.shape == (N + 1, 1)
    assert resultado.x.shape == (N + 1, 2)
    assert resultado.p.shape == (N + 1, 2)
    assert resultado.t.shape == (N + 1,)


def test_integrar_adjunto_atras_satisface_transversalidad(lqr_scalar_problem, u_cero_lqr):
    """El adjunto integrado hacia atrás debe terminar en la condición de transversalidad."""
    from metodos_optimizacion import _integrar_adjunto_atras

    problema = lqr_scalar_problem
    h = 0.01
    metodo = "rk4"
    N = u_cero_lqr.shape[0] - 1
    tiempos = np.linspace(0.0, 1.0, N + 1)

    u_callable = lambda t: np.zeros((1,))

    solver = EDOSolver()
    sol_x = solver.solve(problema._f, problema._x0, problema._t_span, h, method=metodo, u=u_callable)
    x_traj = sol_x.estados

    p_traj = _integrar_adjunto_atras(
        problema, x_traj, u_cero_lqr, tiempos, h, metodo
    )

    assert p_traj.shape == x_traj.shape
    np.testing.assert_allclose(
        p_traj[-1], problema.condicion_transversalidad(x_traj[-1]), atol=1e-8
    )


@pytest.mark.parametrize("metodo", EDOSolver.METODOS)
def test_integrar_adjunto_atras_todos_los_metodos(lqr_scalar_problem, metodo):
    """El helper adjunto debe funcionar con los 5 métodos de EDOSolver."""
    from metodos_optimizacion import _integrar_adjunto_atras

    problema = lqr_scalar_problem
    h = 0.05
    t0, tf = problema._t_span
    N = int(np.round((tf - t0) / h))
    tiempos = np.linspace(t0, tf, N + 1)
    u_traj = np.zeros((N + 1, 1))

    u_callable = lambda t: np.zeros((1,))
    solver = EDOSolver()
    sol_x = solver.solve(problema._f, problema._x0, problema._t_span, h, method=metodo, u=u_callable)

    p_traj = _integrar_adjunto_atras(
        problema, sol_x.estados, u_traj, tiempos, h, metodo
    )

    assert p_traj.shape == sol_x.estados.shape
    np.testing.assert_allclose(
        p_traj[-1], problema.condicion_transversalidad(sol_x.estados[-1]), atol=1e-6
    )
