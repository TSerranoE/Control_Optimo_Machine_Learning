"""Validación de convergencia del LQR analítico del Problema 2."""

import numpy as np
import pytest
from scipy.linalg import solve_continuous_are

from problemas_control import ProblemaLQR


def _p_escalar_analitico(t, T=1.0):
    """Solución analítica escalar de la DRE para A=B=Q=R=S=1.

    La solución con ``P(T)=1`` es ``P(t)=1 + sqrt(2)*tanh(sqrt(2)*(T-t))``.
    """
    return 1.0 + np.sqrt(2.0) * np.tanh(np.sqrt(2.0) * (T - t))


class TestLQRConvergencia:
    """Convergencia de la matriz Riccati y costo óptimo frente a soluciones de referencia."""

    def test_lqr_scalar_convergence(self):
        """La matriz Riccati numérica debe coincidir con la solución analítica."""
        h = 1e-4
        problema = ProblemaLQR(
            A=np.array([[1.0]]),
            B=np.array([[1.0]]),
            Q=np.array([[1.0]]),
            R=np.array([[1.0]]),
            S=np.array([[1.0]]),
            t_span=(0.0, 1.0),
            x0=np.array([1.0]),
            h=h,
        )

        tiempos = np.linspace(0.0, 1.0, 11)
        errores = [
            abs(
                problema._P_interp(t).reshape(1, 1)[0, 0]
                - _p_escalar_analitico(t)
            )
            for t in tiempos
        ]
        assert max(errores) < 1e-6

    def test_lqr_vector_convergence(self):
        """Para una condición terminal estacionaria, ``P(t)`` es constante."""
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        Q = np.eye(2)
        R = np.eye(1)
        S = solve_continuous_are(A, B, Q, R)

        problema = ProblemaLQR(
            A=A,
            B=B,
            Q=Q,
            R=R,
            S=S,
            t_span=(0.0, 2.0),
            x0=np.array([1.0, 0.0]),
            h=1e-3,
        )

        tiempos = np.linspace(0.0, 2.0, 5)
        for t in tiempos:
            P_t = problema._P_interp(t).reshape(2, 2)
            np.testing.assert_allclose(P_t, S, atol=1e-6)

    def test_lqr_cost_matches_analytical(self):
        """El costo de la trayectoria óptima integrada coincide con ``x0^T P(0) x0``."""
        h_riccati = 1e-4
        problema = ProblemaLQR(
            A=np.array([[1.0]]),
            B=np.array([[1.0]]),
            Q=np.array([[1.0]]),
            R=np.array([[1.0]]),
            S=np.array([[1.0]]),
            t_span=(0.0, 1.0),
            x0=np.array([1.0]),
            h=h_riccati,
        )

        n = problema._n

        def dinamica_lazo_cerrado(t, x, u=None):
            """``dx/dt = (A - B R^{-1} B^T P(t)) x``."""
            P_t = problema._P_interp(t).reshape(n, n)
            return (problema._A - problema._B @ problema._R_inv @ problema._B.T @ P_t) @ x

        h = 1e-3
        sol = problema._solver.solve(
            dinamica_lazo_cerrado, problema._x0, problema._t_span, h
        )

        u_opt = np.array(
            [
                -problema._R_inv @ problema._B.T @ problema._P_interp(t).reshape(n, n) @ x
                for t, x in zip(sol.tiempos, sol.estados)
            ]
        )

        costo = problema.evaluar_costo(
            u_opt, h=h, metodo_integracion="euler_progresivo"
        )
        P_0 = problema._P_interp(0.0).reshape(n, n)
        esperado = 0.5 * float(problema._x0 @ P_0 @ problema._x0)

        assert costo == pytest.approx(esperado, rel=1e-3)
