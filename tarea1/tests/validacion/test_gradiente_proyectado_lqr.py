"""Validación integral del gradiente proyectado sobre un LQR escalar."""

import numpy as np
import pytest

from integradores import EDOSolver
from problemas_control import ConjuntoAdmisible, ProblemaLQR


@pytest.mark.parametrize("metodo", EDOSolver.METODOS)
def test_gradiente_proyectado_lqr_reduce_costo_y_respeta_caja(metodo):
    problema = ProblemaLQR(
        np.array([[-1.0]]), np.array([[1.0]]), np.array([[1.0]]),
        np.array([[0.1]]), np.array([[1.0]]), (0.0, 1.0), np.array([1.0]),
        0.05, ConjuntoAdmisible(((-2.0, 2.0),)),
    )
    inicial = np.zeros(21)
    costo_inicial = problema._evaluar_costo_nodal(inicial, metodo)

    resultado = problema.gradiente_proyectado(inicial, 10, 1e-7, metodo)

    assert resultado.historial_costos[-1] < costo_inicial
    assert np.all(resultado.control >= -2.0)
    assert np.all(resultado.control <= 2.0)
