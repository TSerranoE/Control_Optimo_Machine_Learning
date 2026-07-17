"""Métodos de optimización para control óptimo.

Este módulo contiene implementaciones de métodos iterativos para resolver
problemas de control óptimo, incluyendo el Forward-Backward Sweep Method
(FBSM) y el método del gradiente proyectado.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from integradores import EDOSolver

if TYPE_CHECKING:
    from problemas_control import ControlProblem, ResultadoGradienteProyectado


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
    problema: "ControlProblem",
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    tiempos: np.ndarray,
    h: float | np.ndarray,
    metodo: str,
) -> np.ndarray:
    """Integra el adjunto con los pasos efectivos invertidos en ``τ = tf - t``."""
    pasos = np.asarray(h, dtype=float)
    if pasos.ndim == 0:
        pasos = np.diff(tiempos)
    return problema.integrar_adjunto(tiempos, x_traj, u_traj, pasos, metodo)


def _normalizar_control(
    problema: "ControlProblem", u: np.ndarray, metodo_integracion: str
) -> tuple[np.ndarray, float]:
    """Valida un control nodal e infiere su paso uniforme."""
    if metodo_integracion not in EDOSolver.METODOS:
        raise ValueError(
            f"Método '{metodo_integracion}' no válido. Disponibles: {EDOSolver.METODOS}"
        )
    control = np.asarray(u, dtype=float)
    if control.ndim == 1 and problema.dimension_control == 1:
        control = control.reshape(-1, 1)
    if control.ndim != 2 or control.shape[1] != problema.dimension_control:
        raise ValueError(
            f"El control debe tener shape (N, {problema.dimension_control}) con N >= 2."
        )
    if control.shape[0] < 2:
        raise ValueError("El control debe contener al menos dos nodos.")
    if not np.all(np.isfinite(control)):
        raise ValueError("El control debe contener solo valores finitos.")
    t0, tf = problema.t_span
    return control.copy(), (tf - t0) / (control.shape[0] - 1)


def _integrar_valores(
    valores: np.ndarray,
    h: float,
    metodo_integracion: str,
    valores_medios: np.ndarray | None = None,
) -> float:
    """Integra valores escalares con la cuadratura del método."""
    if metodo_integracion == "euler_progresivo":
        return float(h * np.sum(valores[:-1]))
    if metodo_integracion == "euler_implicito":
        return float(h * np.sum(valores[1:]))
    if metodo_integracion in ("heun", "crank_nicolson"):
        return float(h * np.sum((valores[:-1] + valores[1:]) / 2.0))
    return float(
        h * np.sum((valores[:-1] + 4.0 * valores_medios + valores[1:]) / 6.0)
    )


def grad(
    problema: "ControlProblem", u: np.ndarray, metodo_integracion: str
) -> np.ndarray:
    """Calcula el gradiente reducido mediante el adjunto continuo."""
    control, h = _normalizar_control(problema, u, metodo_integracion)
    tiempos, estados = problema.integrar_estado(control, h, metodo_integracion)
    adjuntos = problema.integrar_adjunto(
        tiempos, estados, control, h, metodo_integracion
    )
    return np.array(
        [
            problema.gradiente_hamiltoniano_control(t, x, p, c)
            for t, x, c, p in zip(tiempos, estados, control, adjuntos)
        ],
        dtype=float,
    )


def L2InnerProd(
    problema: "ControlProblem",
    u_1: np.ndarray,
    u_2: np.ndarray,
    metodo_integracion: str,
) -> float:
    """Calcula el producto interno L2 con cuadratura según el método."""
    primero, h = _normalizar_control(problema, u_1, metodo_integracion)
    segundo, _ = _normalizar_control(problema, u_2, metodo_integracion)
    if primero.shape != segundo.shape:
        raise ValueError("Los controles deben tener el mismo shape.")
    valores = np.einsum("ij,ij->i", primero, segundo)
    valores_medios = None
    if metodo_integracion == "rk4":
        primero_medio = (primero[:-1] + primero[1:]) / 2.0
        segundo_medio = (segundo[:-1] + segundo[1:]) / 2.0
        valores_medios = np.einsum("ij,ij->i", primero_medio, segundo_medio)
    return _integrar_valores(valores, h, metodo_integracion, valores_medios)


def L2Norm(
    problema: "ControlProblem", u: np.ndarray, metodo_integracion: str
) -> float:
    """Calcula la norma inducida por ``L2InnerProd``."""
    return float(
        np.sqrt(max(0.0, problema.L2InnerProd(u, u, metodo_integracion)))
    )


def proj(
    problema: "ControlProblem", u: np.ndarray, metodo_integracion: str
) -> np.ndarray:
    """Proyecta un control nodal punto a punto sobre el conjunto admisible."""
    control, _ = _normalizar_control(problema, u, metodo_integracion)
    return problema.proyectar_control(control)


def BBStep(
    problema: "ControlProblem",
    u_1: np.ndarray,
    u_2: np.ndarray,
    g_1: np.ndarray,
    g_2: np.ndarray,
    metodo_integracion: str,
    *,
    t_min: float = 1e-12,
) -> float:
    """Calcula el paso espectral BB con salvaguardas."""
    if not np.isfinite(t_min) or not 0.0 < t_min <= 1.0:
        raise ValueError("t_min debe pertenecer a (0, 1].")
    controles = [
        _normalizar_control(problema, valor, metodo_integracion)[0]
        for valor in (u_1, u_2, g_1, g_2)
    ]
    if len({valor.shape for valor in controles}) != 1:
        raise ValueError("Los controles y gradientes deben tener el mismo shape.")
    s = controles[1] - controles[0]
    y = controles[3] - controles[2]
    numerador = problema.L2InnerProd(s, s, metodo_integracion)
    denominador = problema.L2InnerProd(s, y, metodo_integracion)
    umbral = np.finfo(float).eps * abs(numerador)
    if not np.isfinite(denominador) or denominador <= umbral:
        return 1.0
    paso = numerador / denominador
    if not np.isfinite(paso) or paso <= 0.0:
        return 1.0
    return float(np.clip(paso, t_min, 1.0))


def backtracking(
    problema: "ControlProblem",
    u: np.ndarray,
    g: np.ndarray,
    v: np.ndarray,
    a: float,
    b: float,
    J_hat: float,
    metodo_integracion: str,
    t_inicial: float = 1,
    *,
    max_reducciones: int = 50,
) -> float:
    """Busca un paso que satisfaga Armijo no monótono."""
    if not 0.0 < a < 1.0 or not 0.0 < b < 1.0:
        raise ValueError("a y b deben pertenecer a (0, 1).")
    if not np.isfinite(J_hat):
        raise ValueError("J_hat debe ser finito.")
    if not np.isfinite(t_inicial) or t_inicial <= 0.0:
        raise ValueError("t_inicial debe ser positivo y finito.")
    if not isinstance(max_reducciones, int) or max_reducciones < 1:
        raise ValueError("max_reducciones debe ser un entero positivo.")
    control, _ = _normalizar_control(problema, u, metodo_integracion)
    gradiente, _ = _normalizar_control(problema, g, metodo_integracion)
    direccion, _ = _normalizar_control(problema, v, metodo_integracion)
    if control.shape != gradiente.shape or control.shape != direccion.shape:
        raise ValueError("u, g y v deben tener el mismo shape.")
    producto = problema.L2InnerProd(gradiente, direccion, metodo_integracion)
    paso = float(t_inicial)
    for reduccion in range(max_reducciones + 1):
        candidato = control + paso * direccion
        costo = problema.evaluar_costo_nodal(candidato, metodo_integracion)
        if costo <= J_hat + a * paso * producto:
            return paso
        if reduccion < max_reducciones:
            paso *= b
    raise RuntimeError("El backtracking agotó las reducciones permitidas.")


def gradiente_proyectado(
    problema: "ControlProblem",
    u_inicial,
    max_iter,
    tolerancia,
    metodo_integracion,
    *,
    r=10,
    a=1e-4,
    b=0.5,
    t_min=1e-12,
    max_reducciones=50,
) -> "ResultadoGradienteProyectado":
    """Minimiza el costo mediante gradiente proyectado y búsqueda Armijo."""
    control, h = _normalizar_control(problema, u_inicial, metodo_integracion)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter debe ser un entero positivo.")
    if not np.isfinite(tolerancia) or tolerancia < 0.0:
        raise ValueError("tolerancia debe ser finita y no negativa.")
    if isinstance(r, bool) or not isinstance(r, int) or r < 1:
        raise ValueError("r debe ser un entero positivo.")
    if not 0.0 < a < 1.0 or not 0.0 < b < 1.0:
        raise ValueError("a y b deben pertenecer a (0, 1).")
    if not np.isfinite(t_min) or not 0.0 < t_min <= 1.0:
        raise ValueError("t_min debe pertenecer a (0, 1].")
    if (
        isinstance(max_reducciones, bool)
        or not isinstance(max_reducciones, int)
        or max_reducciones < 1
    ):
        raise ValueError("max_reducciones debe ser un entero positivo.")

    costo_anterior = problema.evaluar_costo_nodal(control, metodo_integracion)
    historial = [costo_anterior]
    control_anterior = gradiente_anterior = None
    convergio = False
    for iteraciones in range(1, max_iter + 1):
        gradiente_actual = problema.grad(control, metodo_integracion)
        direccion = problema.proj(
            control - gradiente_actual, metodo_integracion
        ) - control
        semilla = 1
        if control_anterior is not None:
            semilla = min(
                1,
                problema.BBStep(
                    control_anterior,
                    control,
                    gradiente_anterior,
                    gradiente_actual,
                    metodo_integracion,
                    t_min=t_min,
                ),
            )
        paso = problema.backtracking(
            control,
            gradiente_actual,
            direccion,
            a,
            b,
            max(historial[-r:]),
            metodo_integracion,
            t_inicial=semilla,
            max_reducciones=max_reducciones,
        )
        nuevo_control = control + paso * direccion
        nuevo_costo = problema.evaluar_costo_nodal(
            nuevo_control, metodo_integracion
        )
        historial.append(nuevo_costo)
        control_anterior, gradiente_anterior = control, gradiente_actual
        control = nuevo_control
        cambio_relativo = abs(nuevo_costo - costo_anterior) / max(
            1.0, abs(costo_anterior)
        )
        costo_anterior = nuevo_costo
        if cambio_relativo <= tolerancia:
            convergio = True
            break

    tiempos, estados = problema.integrar_estado(control, h, metodo_integracion)
    adjuntos = problema.integrar_adjunto(
        tiempos, estados, control, h, metodo_integracion
    )
    historial[-1] = problema.evaluar_costo_nodal(control, metodo_integracion)
    try:
        from .problemas_control import ResultadoGradienteProyectado
    except ImportError:
        from problemas_control import ResultadoGradienteProyectado
    return ResultadoGradienteProyectado(
        control, estados, adjuntos, tuple(historial), iteraciones, convergio
    )


def fbsm(
    problema: "ControlProblem",
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

    tiempos, pasos = _normalizar_grilla_fbsm(problema.t_span, h)
    numero_pasos = len(pasos)
    u_actual = np.asarray(u_inicial, dtype=float).copy()
    shape_esperada = (numero_pasos + 1, problema.dimension_control)
    if u_actual.shape != shape_esperada:
        raise ValueError(f"u_inicial debe tener shape {shape_esperada}.")

    def integrar_estado(u_traj: np.ndarray) -> np.ndarray:
        _, estados = problema.integrar_estado(
            u_traj,
            pasos,
            metodo_integracion,
        )
        return estados

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
        u_nuevo = problema.proyectar_control(u_nuevo)

        estado_nuevo = integrar_estado(u_nuevo)
        costo_nuevo = problema.evaluar_costo_trayectoria(
            tiempos, estado_nuevo, u_nuevo
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
