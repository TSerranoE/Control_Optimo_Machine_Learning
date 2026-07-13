"""Tests de convergencia de órdenes para los integradores de EDO.

Se resuelve el problema modelo dx/dt = -x, cuya solución analítica es
exp(-t), con dos pasos temporales (h y h/2). El cociente de errores al
reducir a la mitad el paso debe reflejar el orden teórico del método.
"""

import numpy as np
import pytest

from integradores import EDOSolver


def error_en_t_final(campo, x0, intervalo, paso, metodo, solucion_analitica):
    """Calcula la norma del error absoluto en el tiempo final.

    Parameters
    ----------
    campo : callable
        Campo vectorial f(t, x, u).
    x0 : np.ndarray
        Estado inicial.
    intervalo : tuple[float, ...]
        Extremos temporales.
    paso : float
        Paso temporal (escalar).
    metodo : str
        Método numérico a evaluar.
    solucion_analitica : callable
        Función que recibe t y devuelve el estado exacto.

    Returns
    -------
    float
        Norma L2 del error absoluto en el último instante.
    """
    resolutor = EDOSolver()
    solucion = resolutor.solve(
        campo,
        x0,
        intervalo,
        paso,
        method=metodo,
    )
    tiempo_final = solucion.tiempos[-1]
    estado_numerico = solucion.estados[-1]
    estado_exacto = np.asarray(solucion_analitica(tiempo_final))
    return np.linalg.norm(estado_numerico - estado_exacto)


@pytest.fixture
def problema_lineal():
    """Problema dx/dt = -x con solución analítica exp(-t)."""
    x0 = np.array([1.0])
    campo = lambda t, x, u: -np.asarray(x)
    solucion = lambda t: np.exp(-t) * x0
    return campo, x0, solucion


@pytest.mark.parametrize(
    "metodo, paso_base, ratio_esperado, tolerancia",
    [
        ("euler_progresivo", 0.02, 2.0, 0.5),
        ("heun", 0.02, 4.0, 1.0),
        ("crank_nicolson", 0.02, 4.0, 1.0),
        ("rk4", 0.05, 16.0, 4.0),
    ],
)
def test_convergencia_orden(
    problema_lineal, metodo, paso_base, ratio_esperado, tolerancia
):
    """El cociente de errores al refinar el paso refleja el orden teórico."""
    campo, x0, solucion = problema_lineal
    intervalo = (0.0, 1.0)

    error_grueso = error_en_t_final(
        campo, x0, intervalo, paso_base, metodo, solucion
    )
    error_fino = error_en_t_final(
        campo, x0, intervalo, paso_base / 2.0, metodo, solucion
    )

    assert error_grueso / error_fino == pytest.approx(ratio_esperado, abs=tolerancia)
