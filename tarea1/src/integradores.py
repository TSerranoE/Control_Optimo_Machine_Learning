"""Integradores numéricos para problemas de valor inicial.

Este módulo implementa la clase ``EDOSolver`` para integrar sistemas de
ecuaciones diferenciales ordinarias y la clase ``EDOSolution`` para almacenar
los resultados de la integración.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EDOSolution:
    """Contenedor de resultados de integración de EDO.

    Attributes
    ----------
    tiempos : np.ndarray
        Valores t_k de la grilla temporal, shape (N+1,).
    estados : np.ndarray
        x(t_k), solución aproximada evaluada en cada tiempo de la grilla,
        shape (N+1, n).
    t_span : tuple[float, ...]
        Puntos de ruptura temporales originales.
    intermedios : list[dict] | None
        Valores intermedios del método (k_i de RK4, z de Heun, etc.).
        None si guardar_intermedios=False.
    """

    tiempos: np.ndarray
    estados: np.ndarray
    t_span: tuple[float, ...]
    intermedios: list[dict] | None = None


class EDOSolver:
    """Resuelve EDOs dx/dt = f(t, x, u) con 5 métodos numéricos."""

    METODOS = ("euler", "euler_implicito", "heun", "crank_nicolson", "rk4")

    def solve(
        self,
        f: Callable,
        x0: np.ndarray,
        t_span: tuple[float, ...],
        h: float | np.ndarray,
        method: str = "rk4",
        u: Callable | np.ndarray | None = None,
        guardar_intermedios: bool = False,
        argumentos_fsolve: dict | None = None,
    ) -> EDOSolution:
        """Resuelve el PVI dx/dt = f(t, x, u), x(t0) = x0.

        Parameters
        ----------
        f : callable
            Campo vectorial f(t, x, u) -> ndarray shape (n,).
        x0 : np.ndarray
            Estado inicial, shape (n,).
        t_span : tuple of float
            Puntos temporales ordenados, len >= 2.
        h : float or np.ndarray
            Paso temporal (escalar positivo o ndarray positivo).
        method : str
            Método numérico: 'euler', 'euler_implicito', 'heun',
            'crank_nicolson', 'rk4'.
        u : callable or np.ndarray, optional
            Control u(t). Default None.
        guardar_intermedios : bool
            Almacenar valores intermedios del método (k_i, z, etc.).
            Default False.
        argumentos_fsolve : dict, optional
            Argumentos adicionales para scipy.optimize.fsolve (métodos
            implícitos). Default None.

        Returns
        -------
        EDOSolution
            Resultado de la integración.
        """
        self._validar_entradas(f, x0, t_span, h, method, u, argumentos_fsolve)
        tiempos, pasos = self._construir_grilla(t_span, h)
        control = self._preprocesar_control(u, tiempos, method)
        estados, intermedios = self._resolver_integracion(
            f, x0, tiempos, pasos, control, method, guardar_intermedios
        )
        return EDOSolution(
            tiempos=tiempos,
            estados=estados,
            t_span=t_span,
            intermedios=intermedios,
        )

    def _validar_entradas(
        self,
        f: Callable,
        x0: np.ndarray,
        t_span: tuple[float, ...],
        h: float | np.ndarray,
        method: str,
        u: Callable | np.ndarray | None,
        argumentos_fsolve: dict | None,
    ) -> None:
        """Valida las entradas de ``solve`` según las reglas de R5."""
        if not isinstance(x0, np.ndarray) or not np.issubdtype(x0.dtype, np.number):
            raise ValueError("x0 debe ser un ndarray numérico 1D")
        if x0.ndim != 1 or x0.size == 0:
            raise ValueError("x0 debe ser un ndarray numérico 1D")

        if not isinstance(t_span, (tuple, list)) or len(t_span) < 2:
            raise ValueError("t_span debe tener >= 2 puntos ordenados")
        t_span_array = np.asarray(t_span, dtype=float)
        if not np.all(np.diff(t_span_array) > 0):
            raise ValueError("t_span debe tener >= 2 puntos ordenados")

        h_array = np.asarray(h, dtype=float)
        if h_array.ndim == 0:
            h_array = np.full(len(t_span) - 1, h_array.item())
        if h_array.ndim != 1 or h_array.size != len(t_span) - 1:
            raise ValueError("h debe ser positivo y consistente con t_span")
        if not np.all(h_array > 0):
            raise ValueError("h debe ser positivo y consistente con t_span")

        if method not in self.METODOS:
            raise ValueError(
                f"Método '{method}' no válido. Disponibles: {self.METODOS}"
            )

        if argumentos_fsolve is not None and not isinstance(argumentos_fsolve, dict):
            raise TypeError("argumentos_fsolve debe ser un diccionario")

    def _construir_grilla(
        self, t_span: tuple[float, ...], h: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construye la grilla temporal y el arreglo de pasos.

        Expande un paso escalar al número de subintervalos definido por
        ``t_span``. Cada entrada de ``h`` gobierna el paso dentro del
        subintervalo correspondiente de ``t_span``.
        """
        t_span_array = np.asarray(t_span, dtype=float)
        pasos = np.asarray(h, dtype=float)
        if pasos.ndim == 0:
            pasos = np.full(len(t_span_array) - 1, pasos.item())

        subgrillas = [np.array([t_span_array[0]], dtype=float)]
        pasos_expandidos = []
        for j in range(len(pasos)):
            longitud = t_span_array[j + 1] - t_span_array[j]
            numero_pasos = int(np.round(longitud / pasos[j]))
            if numero_pasos <= 0:
                numero_pasos = 1
            paso_efectivo = longitud / numero_pasos
            tiempos_sub = np.linspace(
                t_span_array[j], t_span_array[j + 1], numero_pasos + 1
            )
            subgrillas.append(tiempos_sub[1:])
            pasos_expandidos.extend([paso_efectivo] * numero_pasos)

        tiempos = np.concatenate(subgrillas)
        return tiempos, np.asarray(pasos_expandidos, dtype=float)

    def _preprocesar_control(
        self,
        u: Callable | np.ndarray | None,
        tiempos: np.ndarray,
        method: str,
    ) -> Callable | np.ndarray | None:
        """Normaliza el control para el método seleccionado."""
        if u is None:
            return None
        return u

    def _resolver_integracion(
        self,
        f: Callable,
        x0: np.ndarray,
        tiempos: np.ndarray,
        pasos: np.ndarray,
        control: Callable | np.ndarray | None,
        method: str,
        guardar_intermedios: bool,
    ) -> tuple[np.ndarray, list[dict] | None]:
        """Despacha al método numérico solicitado."""
        metodos_disponibles = {
            "euler": self._euler,
        }
        if method not in metodos_disponibles:
            raise ValueError(
                f"Método '{method}' aún no implementado. Disponibles: {self.METODOS}"
            )
        estados, intermedios = metodos_disponibles[method](
            f, x0, tiempos, pasos, control, guardar_intermedios
        )
        return estados, intermedios if guardar_intermedios else None

    def _euler(
        self,
        f: Callable,
        x0: np.ndarray,
        tiempos: np.ndarray,
        pasos: np.ndarray,
        control: Callable | np.ndarray | None,
        guardar_intermedios: bool,
    ) -> tuple[np.ndarray, list[dict]]:
        """Euler progresivo: x_{k+1} = x_k + h_k * f(t_k, x_k, u_k)."""
        num_pasos = len(pasos)
        estados = np.empty((num_pasos + 1, x0.size), dtype=float)
        estados[0] = x0
        intermedios = []

        for k in range(num_pasos):
            t_k = tiempos[k]
            x_k = estados[k]
            u_k = None
            if control is not None:
                u_k = (
                    control(t_k)
                    if callable(control)
                    else control[k]
                )
            pendiente = f(t_k, x_k, u_k)
            estados[k + 1] = x_k + pasos[k] * np.asarray(pendiente)
            intermedios.append({})

        return estados, intermedios
