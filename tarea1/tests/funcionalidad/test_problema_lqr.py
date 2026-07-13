"""Tests unitarios para la subclase ``ProblemaLQR`` del Problema 2."""

import numpy as np
import pytest

from problemas_control import ControlProblem, ProblemaLQR


def _p_escalar_analitico(t, T=1.0):
    """Solución analítica escalar de la DRE para A=B=Q=R=S=1.

    La ecuación de Riccati hacia atrás es ``-dP/dt = 2P - P^2 + 1``,
    cuya solución con ``P(T)=1`` es ``P(t)=1 + sqrt(2)*tanh(sqrt(2)*(T-t))``.
    """
    return 1.0 + np.sqrt(2.0) * np.tanh(np.sqrt(2.0) * (T - t))


@pytest.fixture
def scalar_lqr_problem():
    """Problema LQR escalar con solución analítica conocida."""
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


class TestProblemaLQR:
    """Pruebas de herencia, precomputo Riccati y control óptimo analítico."""

    def test_lqr_inherits_control_problem(self, scalar_lqr_problem):
        """``ProblemaLQR`` debe ser una instancia de ``ControlProblem``."""
        assert isinstance(scalar_lqr_problem, ControlProblem)

    def test_lqr_riccati_precomputed(self, scalar_lqr_problem):
        """La matriz Riccati ``P(t)`` debe estar precomputada e interpolada."""
        problema = scalar_lqr_problem

        assert callable(problema._P_interp)

        # Condición terminal P(T) = S
        P_T = problema._P_interp(1.0).reshape(1, 1)
        np.testing.assert_allclose(P_T, np.array([[1.0]]), atol=1e-8)

        # Comparación con la solución analítica en t = 0.5
        P_media = problema._P_interp(0.5).reshape(1, 1)
        esperado = _p_escalar_analitico(0.5)
        assert P_media[0, 0] == pytest.approx(esperado, abs=1e-4)

    def test_lqr_control_optimo_matches_analytical(self, scalar_lqr_problem):
        """El control óptimo puntual debe coincidir con ``-R^{-1} B^T P(t) x``."""
        problema = scalar_lqr_problem
        t = 0.5
        x = np.array([2.0])

        u_opt = problema.control_optimo_puntual(t, x, np.array([0.0]))

        P_t = _p_escalar_analitico(t)
        esperado = -P_t * x[0]

        assert u_opt.shape == (1,)
        assert u_opt[0] == pytest.approx(esperado, abs=1e-4)
