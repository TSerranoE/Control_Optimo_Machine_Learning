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

        assert problema._t_span == (0.0, 1.0)
        assert problema._t0 == 0.0
        assert problema._T == 1.0
        np.testing.assert_array_equal(problema._x0, np.array([1.0]))
        assert problema._n == 1
        assert problema._m == 1
        assert isinstance(problema._solver, EDOSolver)

    def test_constructor_invalid_t_span(self):
        """``t_span`` debe tener dos elementos con ``tf > t0``."""
        f = lambda t, x, u: -np.asarray(x)
        l = lambda t, x, u: float(np.dot(x, x))
        phi = lambda x: float(np.dot(x, x))
        derivadas = _derivadas_simples()

        with pytest.raises(ValueError, match="t_span"):
            ControlProblem(
                f=f,
                l=l,
                phi=phi,
                t_span=(1.0, 0.5),
                x0=np.array([1.0]),
                m=1,
                **derivadas,
            )

    def test_constructor_invalid_m(self):
        """``m`` debe ser un entero positivo."""
        f = lambda t, x, u: -np.asarray(x)
        l = lambda t, x, u: float(np.dot(x, x))
        phi = lambda x: float(np.dot(x, x))
        derivadas = _derivadas_simples()

        with pytest.raises(ValueError, match="m"):
            ControlProblem(
                f=f,
                l=l,
                phi=phi,
                t_span=(0.0, 1.0),
                x0=np.array([1.0]),
                m=0,
                **derivadas,
            )

    @pytest.mark.parametrize("derivada_faltante", [
        "df_dx",
        "df_du",
        "dl_dx",
        "dl_du",
        "dphi_dx",
    ])
    def test_constructor_missing_derivative_raises(self, derivada_faltante):
        """Cada una de las cinco derivadas es obligatoria."""
        f = lambda t, x, u: -np.asarray(x)
        l = lambda t, x, u: float(np.dot(x, x))
        phi = lambda x: float(np.dot(x, x))
        derivadas = _derivadas_simples()
        derivadas[derivada_faltante] = None

        with pytest.raises(TypeError, match=derivada_faltante):
            ControlProblem(
                f=f,
                l=l,
                phi=phi,
                t_span=(0.0, 1.0),
                x0=np.array([1.0]),
                m=1,
                **derivadas,
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
            df_dx=lambda t, x, u: np.zeros((2, 2)),
            df_du=lambda t, x, u: np.zeros((2, 1)),
            dl_dx=lambda t, x, u: np.zeros(2),
            dl_du=lambda t, x, u: np.zeros(1),
            dphi_dx=lambda x: 2.0 * S @ x,
            t_span=(0.0, 1.0),
            x0=np.zeros(2),
            m=1,
        )

        resultado = problema.condicion_transversalidad(x_T)
        esperado = 2.0 * S @ x_T

        np.testing.assert_allclose(resultado, esperado, rtol=1e-6)


def _derivadas_simples():
    """Devuelve derivadas analíticas para un problema escalar simple.

    Dinámica f = -x, costo l = x @ x, terminal phi = x @ x.
    """
    return {
        "df_dx": lambda t, x, u: np.array([[-1.0]]),
        "df_du": lambda t, x, u: np.array([[0.0]]),
        "dl_dx": lambda t, x, u: 2.0 * np.asarray(x),
        "dl_du": lambda t, x, u: np.array([0.0]),
        "dphi_dx": lambda x: 2.0 * np.asarray(x),
    }
