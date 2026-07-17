"""Validación del problema SIR con control de vacunación."""

import numpy as np
import pytest

from metodos_optimizacion import fbsm
from problemas_control import ControlProblem
from validacion_problema3 import crear_problema_sir

H_SIR = 0.5


@pytest.fixture(scope="module")
def sir_problema():
    return crear_problema_sir(
        beta=0.3, gamma=0.1, A=10.0, B=1.0, u_max=0.4,
        S0=0.99, I0=0.01, T=50.0,
    )


@pytest.fixture(scope="module")
def sir_resultado(sir_problema):
    h = H_SIR
    return fbsm(
        sir_problema, np.zeros((int(50.0 / h) + 1, 1)), h,
        metodo_integracion="crank_nicolson", max_iter=200, tol=1e-6,
        omega=0.2,
    )


def test_sir_dynamics_evaluation(sir_problema):
    derivada = sir_problema._f(0.0, np.array([0.99, 0.01]), np.array([0.0]))
    np.testing.assert_allclose(derivada, [-0.00297, 0.00197])


def test_sir_cost_functional(sir_problema):
    costo = sir_problema._l(0.0, np.array([0.5, 0.5]), np.array([0.3]))
    assert costo == pytest.approx(5.045)


def test_sir_box_constraint(sir_problema):
    np.testing.assert_allclose(sir_problema._conjunto.proyectar(np.array([-1.0])), [0.0])
    np.testing.assert_allclose(sir_problema._conjunto.proyectar(np.array([1.0])), [0.4])


def test_sir_usa_minimizador_numerico_general(sir_problema):
    assert "control_optimo_puntual" not in vars(sir_problema)
    assert type(sir_problema).control_optimo_puntual is ControlProblem.control_optimo_puntual


def test_sir_minimizador_numerico_es_acotado(sir_problema):
    control_interior = sir_problema.control_optimo_puntual(
        3.0, np.array([0.5, 0.2]), np.array([0.4, -0.1])
    )
    control_saturado = sir_problema.control_optimo_puntual(
        3.0, np.array([0.9, 0.1]), np.array([2.0, 0.0])
    )

    np.testing.assert_allclose(control_interior, [0.2], atol=1e-6)
    assert 0.0 <= control_interior[0] <= 0.4
    assert control_saturado[0] == pytest.approx(0.4, abs=1e-6)


@pytest.mark.parametrize("B", [0.0, -1.0])
def test_sir_rejects_nonpositive_vaccination_cost(B):
    with pytest.raises(ValueError, match="B.*positivo"):
        crear_problema_sir(
            beta=0.3, gamma=0.1, A=10.0, B=B, u_max=0.4,
            S0=0.99, I0=0.01, T=50.0,
        )


@pytest.mark.slow
def test_sir_nonnegative_states(sir_resultado):
    assert np.min(sir_resultado.estado[:, 0]) >= 0.0
    assert np.min(sir_resultado.estado[:, 1]) >= 0.0


@pytest.mark.slow
def test_sir_control_in_bounds(sir_resultado):
    assert np.min(sir_resultado.control_optimo) >= 0.0
    assert np.max(sir_resultado.control_optimo) <= 0.4


@pytest.mark.slow
def test_sir_ab_ratio_qualitative(sir_resultado):
    problema_bajo = crear_problema_sir(
        beta=0.3, gamma=0.1, A=1.0, B=1.0, u_max=0.4,
        S0=0.99, I0=0.01, T=50.0,
    )
    h = H_SIR
    resultado_bajo = fbsm(
        problema_bajo, np.zeros((int(50.0 / h) + 1, 1)), h,
        metodo_integracion="crank_nicolson", max_iter=200, tol=1e-6,
        omega=0.2,
    )
    assert resultado_bajo.convergio
    assert np.mean(sir_resultado.control_optimo) > np.mean(resultado_bajo.control_optimo)


@pytest.mark.slow
def test_sir_fbsm_convergence(sir_resultado):
    assert sir_resultado.convergio
    assert sir_resultado.iteraciones < 200
    assert np.all(np.diff(sir_resultado.historia_costo) <= 0.0)
    assert np.all(np.isfinite(sir_resultado.estado))
    assert np.all(np.isfinite(sir_resultado.control_optimo))


@pytest.mark.slow
def test_sir_fbsm_preserves_fixed_omega_after_cost_increase(sir_problema):
    h = H_SIR
    resultado = fbsm(
        sir_problema, np.zeros((int(50.0 / h) + 1, 1)), h,
        metodo_integracion="crank_nicolson", max_iter=4, tol=1e-15,
        omega=0.99,
    )
    assert resultado.historia_costo[2] > resultado.historia_costo[1]
    assert resultado.historia_costo[3] == pytest.approx(3.0462046413, rel=1e-6)
