"""Problemas de validación para el Forward-Backward Sweep Method."""

import numpy as np

from problemas_control import ConjuntoAdmisible, ControlProblem, ProblemaLQR


def crear_problema_lqr_fbsm(
    a: float,
    b: float,
    q: float,
    r: float,
    s: float,
    T: float,
    x0: float,
    h_riccati: float = 0.01,
) -> ProblemaLQR:
    """Crea un LQR escalar con minimizador adjunto y referencia Riccati."""
    return ProblemaLQR(
        A=np.array([[a]], dtype=float),
        B=np.array([[b]], dtype=float),
        Q=np.array([[q]], dtype=float),
        R=np.array([[r]], dtype=float),
        S=np.array([[s]], dtype=float),
        t_span=(0.0, T),
        x0=np.array([x0], dtype=float),
        h=h_riccati,
    )


def crear_problema_sir(
    beta: float,
    gamma: float,
    A: float,
    B: float,
    u_max: float,
    S0: float,
    I0: float,
    T: float,
) -> ControlProblem:
    """Crea un SIR acotado que usa el minimizador numérico general."""
    if B <= 0:
        raise ValueError("B debe ser positivo.")

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

    return ControlProblem(
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
