"""Tests unitarios del esqueleto de ``ControlProblem`` del Problema 2."""

import numpy as np
import pytest

from integradores import EDOSolver
from problemas_control import ConjuntoAdmisible, ControlProblem


class TestControlProblemSkeleton:
    """Pruebas del constructor y métodos base del esqueleto."""

    def test_constructor_valid(self, simple_control_problem):
        """El constructor debe almacenar correctamente los parámetros básicos."""
        problema = simple_control_problem

        assert problema._T == 1.0
        np.testing.assert_array_equal(problema._x0, np.array([1.0]))
        assert problema._n == 1
        assert problema._m == 1
        assert isinstance(problema._solver, EDOSolver)

    def test_constructor_invalid_T(self):
        """``T`` debe ser estrictamente positivo."""
        f = lambda t, x, u: -np.asarray(x)
        l = lambda t, x, u: float(np.dot(x, x))
        phi = lambda x: float(np.dot(x, x))

        with pytest.raises(ValueError, match="T"):
            ControlProblem(
                f=f,
                l=l,
                phi=phi,
                T=0.0,
                x0=np.array([1.0]),
                m=1,
            )

    def test_constructor_invalid_m(self):
        """``m`` debe ser un entero positivo."""
        f = lambda t, x, u: -np.asarray(x)
        l = lambda t, x, u: float(np.dot(x, x))
        phi = lambda x: float(np.dot(x, x))

        with pytest.raises(ValueError, match="m"):
            ControlProblem(
                f=f,
                l=l,
                phi=phi,
                T=1.0,
                x0=np.array([1.0]),
                m=0,
            )

    def test_hamiltoniano_scalar(self, simple_control_problem):
        """El Hamiltoniano debe ser ``l + p @ f`` para un caso escalar."""
        problema = simple_control_problem
        t = 0.0
        x = np.array([2.0])
        p = np.array([3.0])
        u = np.array([1.0])

        resultado = problema.hamiltoniano(t, x, p, u)

        # f = -x + u = -1, l = x^2 + u^2 = 5, p @ f = -3
        esperado = 5.0 + 3.0 * (-1.0)
        assert resultado == pytest.approx(esperado)

    def test_sistema_adjunto_shape(self, simple_control_problem):
        """El sistema adjunto debe devolver un ndarray de shape ``(n,)``."""
        problema = simple_control_problem
        x = np.array([2.0])
        p = np.array([3.0])
        u = np.array([1.0])

        resultado = problema.sistema_adjunto(0.0, x, p, u)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (1,)

    def test_condicion_transversalidad(self):
        """La transversalidad debe devolver el gradiente del costo terminal."""
        S = np.diag([1.0, 2.0])
        phi = lambda x: float(x @ S @ x)
        f = lambda t, x, u: np.zeros_like(x)
        l = lambda t, x, u: 0.0
        x_T = np.array([1.0, 2.0])

        problema = ControlProblem(
            f=f,
            l=l,
            phi=phi,
            T=1.0,
            x0=np.zeros(2),
            m=1,
        )

        resultado = problema.condicion_transversalidad(x_T)
        esperado = 2.0 * S @ x_T

        np.testing.assert_allclose(resultado, esperado, rtol=1e-6)
