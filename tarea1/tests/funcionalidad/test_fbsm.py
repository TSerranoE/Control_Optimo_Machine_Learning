"""Tests funcionales del núcleo Forward-Backward Sweep Method (FBSM)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from integradores import EDOSolver
from metodos_optimizacion import ResultadoFBSM, _integrar_adjunto_atras, fbsm
from problemas_control import ConjuntoAdmisible, ProblemaLQR


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
    t0, tf, h = 0.0, 1.0, 0.01
    return np.zeros((int(np.round((tf - t0) / h)) + 1, 1))


def test_fbsm_resultado_campos_shapes():
    """ResultadoFBSM expone seis campos inmutables con las shapes contratadas."""
    N = 10
    resultado = ResultadoFBSM(
        control_optimo=np.zeros((N + 1, 1)),
        estado=np.zeros((N + 1, 2)),
        adjunto=np.zeros((N + 1, 2)),
        historia_costo=(1.0,),
        iteraciones=1,
        convergio=False,
    )

    assert resultado.control_optimo.shape == (N + 1, 1)
    assert resultado.estado.shape == (N + 1, 2)
    assert resultado.adjunto.shape == (N + 1, 2)
    assert resultado.historia_costo == (1.0,)
    with pytest.raises(FrozenInstanceError):
        resultado.iteraciones = 2


def test_fbsm_api_publica(lqr_scalar_problem, u_cero_lqr):
    """El paquete src exporta el solver y su tipo de resultado."""
    from src import ResultadoFBSM as ResultadoPublico, fbsm as fbsm_publico

    assert tuple(ResultadoPublico.__dataclass_fields__) == tuple(ResultadoFBSM.__dataclass_fields__)
    with pytest.raises(ValueError, match="h"):
        fbsm_publico(lqr_scalar_problem, u_cero_lqr, h=0.0)


def test_integrar_adjunto_atras_satisface_transversalidad(lqr_scalar_problem, u_cero_lqr):
    """El adjunto integrado hacia atrás debe terminar en la condición de transversalidad."""
    problema = lqr_scalar_problem
    h = 0.01
    metodo = "rk4"
    N = u_cero_lqr.shape[0] - 1
    tiempos = np.linspace(0.0, 1.0, N + 1)

    solver = EDOSolver()
    sol_x = solver.solve(
        problema._f,
        problema._x0,
        problema._t_span,
        h,
        method=metodo,
        u=lambda _t: np.zeros(1),
    )
    x_traj = sol_x.estados

    p_traj = _integrar_adjunto_atras(
        problema, x_traj, u_cero_lqr, tiempos, h, metodo
    )

    assert p_traj.shape == x_traj.shape
    np.testing.assert_allclose(
        p_traj[-1], problema.condicion_transversalidad(x_traj[-1]), atol=1e-8
    )


def test_fbsm_converge_lqr_scalar(lqr_scalar_problem, u_cero_lqr):
    """FBSM debe converger para el problema LQR escalar con tolerancia razonable."""
    resultado = fbsm(
        lqr_scalar_problem,
        u_cero_lqr,
        h=0.01,
        metodo_integracion="rk4",
        max_iter=100,
        tol=1e-4,
        omega=0.99,
    )

    assert resultado.convergio
    assert resultado.iteraciones <= 100
    assert resultado.control_optimo.shape == u_cero_lqr.shape
    assert resultado.estado.shape == u_cero_lqr.shape
    assert resultado.adjunto.shape == u_cero_lqr.shape


def test_fbsm_max_iter_flag(lqr_scalar_problem, u_cero_lqr):
    """FBSM debe retornar convergio=False cuando se agota max_iter."""
    resultado = fbsm(
        lqr_scalar_problem,
        u_cero_lqr,
        h=0.01,
        metodo_integracion="rk4",
        max_iter=5,
        tol=1e-15,
        omega=0.99,
    )

    assert not resultado.convergio
    assert resultado.iteraciones == 5


def test_fbsm_converge_temprano(lqr_scalar_problem, u_cero_lqr):
    """El criterio relativo de costo detiene FBSM antes del máximo."""
    resultado = fbsm(
        lqr_scalar_problem,
        u_cero_lqr,
        h=0.01,
        metodo_integracion="rk4",
        max_iter=100,
        tol=1e-4,
        omega=0.99,
    )

    assert resultado.convergio
    assert resultado.iteraciones < 100
    assert len(resultado.historia_costo) == resultado.iteraciones


def test_fbsm_relajacion_omega_1(lqr_scalar_problem, u_cero_lqr):
    """omega=1 aplica exactamente el control puntual, sin promediar con u previo."""
    lqr_scalar_problem.control_optimo_puntual = lambda _t, _x, _p: np.array([-0.25])
    resultado = fbsm(
        lqr_scalar_problem,
        u_cero_lqr,
        h=0.01,
        metodo_integracion="rk4",
        max_iter=1,
        tol=1e-15,
        omega=1.0,
    )
    np.testing.assert_allclose(resultado.control_optimo, -0.25)


@pytest.mark.parametrize("metodo", EDOSolver.METODOS)
def test_fbsm_cinco_metodos(lqr_scalar_problem, metodo):
    """FBSM debe converger con los 5 métodos de EDOSolver."""
    h = 0.01
    t0, tf = lqr_scalar_problem._t_span
    N = int(np.round((tf - t0) / h))
    u0 = np.zeros((N + 1, 1))

    resultado = fbsm(
        lqr_scalar_problem,
        u0,
        h=h,
        metodo_integracion=metodo,
        max_iter=100,
        tol=1e-4,
        omega=0.99,
    )

    assert resultado.convergio
    assert resultado.iteraciones < 100


def test_fbsm_adjunto_condicion_terminal(lqr_scalar_problem, u_cero_lqr):
    """La trayectoria final consistente conserva la condición transversal."""
    resultado = fbsm(lqr_scalar_problem, u_cero_lqr, h=0.01, tol=1e-4)

    np.testing.assert_allclose(
        resultado.adjunto[-1],
        lqr_scalar_problem.condicion_transversalidad(resultado.estado[-1]),
    )


def test_fbsm_caja_respetada():
    """FBSM debe respetar el conjunto admisible tipo caja [0, 0.5]."""
    problema = ProblemaLQR(
        A=np.array([[1.0]]),
        B=np.array([[1.0]]),
        Q=np.array([[1.0]]),
        R=np.array([[1.0]]),
        S=np.array([[1.0]]),
        t_span=(0.0, 1.0),
        x0=np.array([1.0]),
        h=1e-4,
        conjunto_admisible=ConjuntoAdmisible(limites=[(0.0, 0.5)]),
    )
    h = 0.01
    N = int(np.round(1.0 / h))
    u0 = np.zeros((N + 1, 1))

    resultado = fbsm(problema, u0, h=h, metodo_integracion="rk4", max_iter=100, tol=1e-4)

    assert resultado.convergio
    assert np.all(resultado.control_optimo >= 0.0)
    assert np.all(resultado.control_optimo <= 0.5)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"h": -0.01}, "h"),
        ({"h": 0.0}, "h"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": -1e-6}, "tol"),
        ({"tol": 0.0}, "tol"),
        ({"omega": 0.0}, "omega"),
        ({"omega": 1.01}, "omega"),
    ],
)
def test_fbsm_input_validation(lqr_scalar_problem, u_cero_lqr, kwargs, match):
    """FBSM debe rechazar parámetros numéricos fuera de rango."""
    base = {
        "problema": lqr_scalar_problem,
        "u_inicial": u_cero_lqr,
        "h": 0.01,
        "metodo_integracion": "rk4",
        "max_iter": 100,
        "tol": 1e-6,
        "omega": 0.99,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        fbsm(**base)
