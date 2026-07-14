"""Métodos de optimización para control óptimo.

Este módulo contiene implementaciones de métodos iterativos para resolver
problemas de control óptimo, incluyendo el Forward-Backward Sweep Method
(FBSM) y el método del gradiente proyectado.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ResultadoFBSM:
    """Resultado del Forward-Backward Sweep Method.

    Attributes
    ----------
    u : np.ndarray
        Control óptimo u*(t_k), shape (N+1, m).
    x : np.ndarray
        Trayectoria de estado x(t_k), shape (N+1, n).
    p : np.ndarray
        Trayectoria del costado p(t_k), shape (N+1, n).
    t : np.ndarray
        Grilla temporal t_k, shape (N+1,).
    historia_J : list[float]
        Valor del funcional de costo en cada iteración.
    iteraciones : int
        Número de iteraciones ejecutadas.
    convergio : bool
        True si el criterio de convergencia se satisfizo antes de max_iter.
    """

    u: np.ndarray
    x: np.ndarray
    p: np.ndarray
    t: np.ndarray
    historia_J: list[float]
    iteraciones: int
    convergio: bool
