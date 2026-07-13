"""Fixtures compartidas para los tests de funcionalidad."""

import numpy as np
import pytest

from problemas_control import ConjuntoAdmisible, ControlProblem


@pytest.fixture
def campo_lineal():
    """Campo vectorial lineal decreciente dx/dt = -x.

    Returns
    -------
    callable
        Función f(t, x, u) que retorna -x como ndarray.
    """
    return lambda t, x, u: -np.asarray(x)


@pytest.fixture
def scalar_lqr_matrices():
    """Matrices escalares para un LQR con solución conocida.

    Returns
    -------
    tuple
        ``(A, B, Q, R, S)`` como escalares (todos iguales a 1).
    """
    return (1.0, 1.0, 1.0, 1.0, 1.0)


@pytest.fixture
def box_conjunto():
    """Conjunto admisible tipo caja con límites ``(-1, 1)``.

    Returns
    -------
    ConjuntoAdmisible
        Conjunto de dimensión 1 acotado.
    """
    return ConjuntoAdmisible(limites=((-1.0, 1.0),))


@pytest.fixture
def unrestricted_conjunto():
    """Conjunto admisible irrestricto.

    Returns
    -------
    ConjuntoAdmisible
        Conjunto sin límites.
    """
    return ConjuntoAdmisible(limites=None)


@pytest.fixture
def simple_control_problem(unrestricted_conjunto):
    """Problema de control escalar sencillo para los tests del esqueleto.

    La dinámica es ``f(t, x, u) = -x + u``, el costo de operación
    ``l(t, x, u) = x^2 + u^2`` y el costo terminal ``phi(x) = x^2``.

    Parameters
    ----------
    unrestricted_conjunto : ConjuntoAdmisible
        Conjunto irrestricto inyectado por fixture.

    Returns
    -------
    ControlProblem
        Instancia con ``T=1.0``, ``x0=[1.0]`` y ``m=1``.
    """
    f = lambda t, x, u: -np.asarray(x) + np.asarray(u)
    l = lambda t, x, u: float(np.dot(x, x) + np.dot(u, u))
    phi = lambda x: float(np.dot(x, x))

    df_dx = lambda t, x, u: np.array([[-1.0]])
    df_du = lambda t, x, u: np.array([[1.0]])
    dl_dx = lambda t, x, u: 2.0 * np.asarray(x)
    dl_du = lambda t, x, u: 2.0 * np.asarray(u)
    dphi_dx = lambda x: 2.0 * np.asarray(x)

    return ControlProblem(
        f=f,
        l=l,
        phi=phi,
        df_dx=df_dx,
        df_du=df_du,
        dl_dx=dl_dx,
        dl_du=dl_du,
        dphi_dx=dphi_dx,
        t_span=(0.0, 1.0),
        x0=np.array([1.0]),
        m=1,
        conjunto_admisible=unrestricted_conjunto,
    )
