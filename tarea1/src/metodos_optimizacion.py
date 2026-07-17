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


def _normalizar_grilla_fbsm(
    t_span: tuple[float, float], h: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Construye la grilla concreta y sus pasos efectivos para FBSM."""
    t0, tf = t_span
    horizonte = tf - t0
    h_array = np.asarray(h, dtype=float)
    if h_array.ndim == 0:
        paso = float(h_array)
        if not np.isfinite(paso) or paso <= 0.0:
            raise ValueError("h debe ser positivo y finito.")
        numero_pasos = max(1, int(np.round(horizonte / paso)))
        tiempos = np.linspace(t0, tf, numero_pasos + 1)
        return tiempos, np.diff(tiempos)

    if h_array.ndim != 1 or h_array.size == 0:
        raise ValueError("h debe ser un vector unidimensional no vacío.")
    if not np.all(np.isfinite(h_array)):
        raise ValueError("Todos los pasos de h deben ser finitos.")
    if not np.all(h_array > 0.0):
        raise ValueError("Todos los pasos de h deben ser positivos.")

    tolerancia_absoluta = 1e-12 * max(1.0, abs(horizonte))
    if not np.isclose(
        np.sum(h_array), horizonte, rtol=1e-10, atol=tolerancia_absoluta
    ):
        raise ValueError("La suma de h debe coincidir con tf - t0.")

    tiempos = t0 + np.concatenate(([0.0], np.cumsum(h_array)))
    tiempos[-1] = tf
    return tiempos, np.diff(tiempos)


def _integrar_adjunto_atras(
    problema: ControlProblem,
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    tiempos: np.ndarray,
    h: float | np.ndarray,
    metodo: str,
) -> np.ndarray:
    """Integra el adjunto con los pasos efectivos invertidos en ``τ = tf - t``."""
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

    pasos = np.asarray(h, dtype=float)
    if pasos.ndim == 0:
        pasos = np.diff(tiempos)
    pasos_reversos = pasos[::-1]
    tiempos_tau = np.concatenate(([0.0], np.cumsum(pasos_reversos)))
    tiempos_tau[-1] = T
    solver = EDOSolver()
    sol = solver.solve(
        dp_dtau, p_terminal, tuple(tiempos_tau), pasos_reversos, method=metodo
    )

    return sol.estados[::-1]


def fbsm(
    problema: ControlProblem,
    u_inicial: np.ndarray,
    h: float | np.ndarray,
    metodo_integracion: str = "rk4",
    max_iter: int = 100,
    tol: float = 1e-6,
    omega: float = 0.99,
) -> ResultadoFBSM:
    """Resuelve FBSM con un paso escalar o una secuencia de pasos consecutivos."""
    if max_iter < 1:
        raise ValueError("max_iter debe ser al menos 1.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol debe ser positiva.")
    if not 0 < omega <= 1:
        raise ValueError("omega debe pertenecer a (0, 1].")

    tiempos, pasos = _normalizar_grilla_fbsm(problema._t_span, h)
    numero_pasos = len(pasos)
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
            tuple(tiempos),
            pasos,
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
            problema, estado, u_actual, tiempos, pasos, metodo_integracion
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

        estado_nuevo = integrar_estado(u_nuevo)
        costos_nodales = np.array(
            [
                problema._l(t, x, u)
                for t, x, u in zip(tiempos, estado_nuevo, u_nuevo)
            ],
            dtype=float,
        )
        costo_nuevo = float(
            np.trapezoid(costos_nodales, tiempos) + problema._phi(estado_nuevo[-1])
        )
        historia_costo.append(costo_nuevo)
        if costo_anterior is not None:
            cambio_relativo = abs(costo_nuevo - costo_anterior) / max(
                abs(costo_nuevo), np.finfo(float).eps
            )
            cambio_control = np.linalg.norm(u_nuevo - u_actual) / max(
                np.linalg.norm(u_nuevo), np.linalg.norm(u_actual), np.finfo(float).eps
            )
            if cambio_relativo < tol and cambio_control < tol:
                convergio = True
        u_actual = u_nuevo
        costo_anterior = costo_nuevo
        if convergio:
            break

    # Las trayectorias retornadas corresponden al control final, no al barrido previo.
    estado = integrar_estado(u_actual)
    adjunto = _integrar_adjunto_atras(
        problema, estado, u_actual, tiempos, pasos, metodo_integracion
    )
    return ResultadoFBSM(
        control_optimo=u_actual,
        estado=estado,
        adjunto=adjunto,
        historia_costo=tuple(historia_costo),
        iteraciones=iteracion,
        convergio=convergio,
    )
