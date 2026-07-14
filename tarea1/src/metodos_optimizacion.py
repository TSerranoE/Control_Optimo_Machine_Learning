"""Métodos de optimización para control óptimo.

Este módulo contiene implementaciones de métodos iterativos para resolver
problemas de control óptimo, incluyendo el Forward-Backward Sweep Method
(FBSM) y el método del gradiente proyectado.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from integradores import EDOSolver
from problemas_control import ControlProblem


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


def _integrar_adjunto_atras(
    problema: ControlProblem,
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    tiempos: np.ndarray,
    h: float,
    metodo: str,
) -> np.ndarray:
    """Integra el sistema adjunto hacia atrás mediante reversión temporal.

    Usa la transformación ``τ = T - t`` para integrar hacia adelante en ``τ``
    con ``EDOSolver``, evitando así modificar el integrador para pasos
    negativos. Las trayectorias de estado y control se interpolan
    linealmente sobre la grilla directa.

    Parameters
    ----------
    problema : ControlProblem
        Problema de control con ``sistema_adjunto`` y ``condicion_transversalidad``.
    x_traj : np.ndarray
        Trayectoria de estado en la grilla directa, shape (N+1, n).
    u_traj : np.ndarray
        Trayectoria de control en la grilla directa, shape (N+1, m).
    tiempos : np.ndarray
        Grilla temporal directa, shape (N+1,).
    h : float
        Paso de integración (positivo).
    metodo : str
        Método de ``EDOSolver``.

    Returns
    -------
    np.ndarray
        Trayectoria del costado ``p(t)`` en la grilla directa, shape (N+1, n).
    """
    t0 = tiempos[0]
    T = tiempos[-1] - t0

    x_interp = interp1d(
        tiempos, x_traj, axis=0, kind="linear", fill_value="extrapolate"
    )
    u_interp = interp1d(
        tiempos, u_traj, axis=0, kind="linear", fill_value="extrapolate"
    )

    def dp_dtau(tau: float, p: np.ndarray, _u=None) -> np.ndarray:
        t = T - tau + t0
        x_t = np.asarray(x_interp(t), dtype=float)
        u_t = np.asarray(u_interp(t), dtype=float)
        # sistema_adjunto retorna -∂H/∂x = dp/dt;
        # dp/dτ = -dp/dt = +∂H/∂x
        return -problema.sistema_adjunto(t, x_t, p, u_t)

    p_terminal = np.asarray(
        problema.condicion_transversalidad(x_traj[-1]), dtype=float
    )

    solver = EDOSolver()
    sol = solver.solve(dp_dtau, p_terminal, (0.0, T), h, method=metodo)

    return sol.estados[::-1]
