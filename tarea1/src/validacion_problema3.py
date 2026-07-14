"""Problemas de validación para el Forward-Backward Sweep Method."""

import numpy as np

from problemas_control import ConjuntoAdmisible, ControlProblem


def crear_problema_sir(
    beta: float,
    gamma: float,
    A: float,
    B: float,
    u_max: float,
    S0: float,
    I0: float,
    T: float,
    h_riccati: float = 0.01,
) -> ControlProblem:
    """Crea un problema SIR con vacunación acotada y costo de infecciones."""
    del h_riccati  # Conserva la firma acordada; el modelo SIR no usa Riccati.

    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        del t
        S, I = x
        vacunacion = float(np.asarray(u).reshape(-1)[0])
        contagios = beta * S * I
        return np.array([-contagios - vacunacion * S, contagios - gamma * I])

    def l(t: float, x: np.ndarray, u: np.ndarray) -> float:
        del t
        vacunacion = float(np.asarray(u).reshape(-1)[0])
        return float(A * x[1] + 0.5 * B * vacunacion**2)

    def df_dx(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        del t
        S, I = x
        vacunacion = float(np.asarray(u).reshape(-1)[0])
        return np.array(
            [
                [-beta * I - vacunacion, -beta * S],
                [beta * I, beta * S - gamma],
            ]
        )

    def df_du(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        del t, u
        return np.array([[-x[0]], [0.0]])

    def dl_dx(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        del t, x, u
        return np.array([0.0, A])

    def dl_du(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        del t, x
        vacunacion = float(np.asarray(u).reshape(-1)[0])
        return np.array([B * vacunacion])

    problema = ControlProblem(
        f=f,
        l=l,
        phi=lambda x: 0.0,
        df_dx=df_dx,
        df_du=df_du,
        dl_dx=dl_dx,
        dl_du=dl_du,
        dphi_dx=lambda x: np.zeros(2),
        t_span=(0.0, T),
        x0=np.array([S0, I0], dtype=float),
        m=1,
        conjunto_admisible=ConjuntoAdmisible(((0.0, u_max),)),
    )

    def control_optimo_puntual(t: float, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        del t
        vacunacion = p[0] * x[0] / B
        return problema._conjunto.proyectar(np.array([vacunacion]))

    problema.control_optimo_puntual = control_optimo_puntual
    return problema
