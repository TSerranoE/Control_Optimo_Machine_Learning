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


@dataclass(frozen=True)
class ResultadoFBSM:
    """Trayectorias finales e historial inmutable de una ejecución de FBSM."""

    control_optimo: np.ndarray
    estado: np.ndarray
    adjunto: np.ndarray
    historia_costo: tuple[float, ...]
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
    """Integra el estado adjunto hacia atrás usando ``τ = tf - t`` y ``EDOSolver``."""
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


def fbsm(
    problema: ControlProblem,
    u_inicial: np.ndarray,
    h: float,
    metodo_integracion: str = "rk4",
    max_iter: int = 100,
    tol: float = 1e-6,
    omega: float = 0.99,
) -> ResultadoFBSM:
    """Resuelve un problema de control óptimo mediante barridos sucesivos."""
    if h <= 0:
        raise ValueError("h debe ser positivo.")
    if max_iter < 1:
        raise ValueError("max_iter debe ser al menos 1.")
    if tol <= 0:
        raise ValueError("tol debe ser positiva.")
    if not 0 < omega <= 1:
        raise ValueError("omega debe pertenecer a (0, 1].")

    t0, tf = problema._t_span
    numero_pasos = int(np.round((tf - t0) / h))
    tiempos = np.linspace(t0, tf, numero_pasos + 1)
    u_actual = np.asarray(u_inicial, dtype=float).copy()
    shape_esperada = (numero_pasos + 1, problema._m)
    if u_actual.shape != shape_esperada:
        raise ValueError(f"u_inicial debe tener shape {shape_esperada}.")

    def control_interpolado(u_traj: np.ndarray):
        interpolador = interp1d(tiempos, u_traj, axis=0, kind="linear")
        return lambda t: np.asarray(interpolador(t), dtype=float)

    def control_para_solver(u_traj: np.ndarray):
        if metodo_integracion == "rk4":
            return control_interpolado(u_traj)
        return u_traj

    def integrar_estado(u_traj: np.ndarray) -> np.ndarray:
        solucion = problema._solver.solve(
            problema._f,
            problema._x0,
            problema._t_span,
            h,
            method=metodo_integracion,
            u=control_para_solver(u_traj),
        )
        return solucion.estados

    historia_costo: list[float] = []
    costo_anterior: float | None = None
    convergio = False

    for iteracion in range(1, max_iter + 1):
        estado = integrar_estado(u_actual)
        adjunto = _integrar_adjunto_atras(
            problema, estado, u_actual, tiempos, h, metodo_integracion
        )
        u_puntual = np.array(
            [
                problema.control_optimo_puntual(t, x, p)
                for t, x, p in zip(tiempos, estado, adjunto)
            ],
            dtype=float,
        )
        u_nuevo = (1.0 - omega) * u_actual + omega * u_puntual
        if problema._conjunto is not None:
            u_nuevo = np.array([problema._conjunto.proyectar(u) for u in u_nuevo])

        costo_nuevo = problema.evaluar_costo(
            control_para_solver(u_nuevo), h, metodo_integracion
        )
        historia_costo.append(costo_nuevo)
        if costo_anterior is not None:
            cambio_relativo = abs(costo_nuevo - costo_anterior) / max(
                abs(costo_nuevo), np.finfo(float).eps
            )
            if cambio_relativo < tol:
                convergio = True
        u_actual = u_nuevo
        costo_anterior = costo_nuevo
        if convergio:
            break

    # Las trayectorias retornadas corresponden al control final, no al barrido previo.
    estado = integrar_estado(u_actual)
    adjunto = _integrar_adjunto_atras(
        problema, estado, u_actual, tiempos, h, metodo_integracion
    )
    return ResultadoFBSM(
        control_optimo=u_actual,
        estado=estado,
        adjunto=adjunto,
        historia_costo=tuple(historia_costo),
        iteraciones=iteracion,
        convergio=convergio,
    )
