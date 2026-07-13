"""Fixtures compartidas para los tests de funcionalidad."""

import numpy as np
import pytest


@pytest.fixture
def campo_lineal():
    """Campo vectorial lineal decreciente dx/dt = -x.

    Returns
    -------
    callable
        Función f(t, x, u) que retorna -x como ndarray.
    """
    return lambda t, x, u: -np.asarray(x)
