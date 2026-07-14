"""Tests unitarios para el núcleo Forward-Backward Sweep Method (FBSM)."""

import numpy as np
import pytest

from metodos_optimizacion import ResultadoFBSM


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
