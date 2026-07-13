"""Integradores numéricos para problemas de valor inicial.

Este módulo implementa la clase ``EDOSolver`` para integrar sistemas de
ecuaciones diferenciales ordinarias y la clase ``EDOSolution`` para almacenar
los resultados de la integración.
"""

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import fsolve


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

    METODOS = ("euler_progresivo", "euler_implicito", "heun", "crank_nicolson", "rk4")

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
            Método numérico: 'euler_progresivo', 'euler_implicito', 'heun',
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
        argumentos_fsolve = argumentos_fsolve or {"xtol": 1e-8}
        estados, intermedios = self._resolver_integracion(
            f,
            x0,
            tiempos,
            pasos,
            control,
            method,
            guardar_intermedios,
            argumentos_fsolve,
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
        """Valida las entradas de ``solve`` antes de construir la grilla.

        Lanza ``ValueError`` o ``TypeError`` con mensajes descriptivos cuando
        algún parámetro no cumple las restricciones del problema de valor
        inicial (dimensiones, orden, positividad, método soportado, etc.).
        """
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
        """Construye la grilla temporal y el arreglo de pasos efectivos.

        Expande un paso escalar al número de subintervalos definido por
        ``t_span``. Cuando la longitud de un subintervalo no es múltiplo
        exacto del paso solicitado, se redondea el número de pasos y se
        ajusta el paso efectivo para que la grilla termine exactamente en
        los puntos de ruptura de ``t_span``.
        """
        t_span_array = np.asarray(t_span, dtype=float)
        pasos_solicitados = np.asarray(h, dtype=float)
        if pasos_solicitados.ndim == 0:
            pasos_solicitados = np.full(
                len(t_span_array) - 1, pasos_solicitados.item()
            )

        subgrillas = [np.array([t_span_array[0]], dtype=float)]
        pasos_expandidos = []
        for j in range(len(pasos_solicitados)):
            longitud = t_span_array[j + 1] - t_span_array[j]
            numero_pasos = int(np.round(longitud / pasos_solicitados[j]))
            if numero_pasos <= 0:
                numero_pasos = 1

            paso_efectivo = longitud / numero_pasos
            if not np.isclose(paso_efectivo, pasos_solicitados[j]):
                warnings.warn(
                    f"El paso solicitado {pasos_solicitados[j]} no divide "
                    f"exactamente el subintervalo [{t_span_array[j]}, "
                    f"{t_span_array[j + 1]}]; se usará paso efectivo "
                    f"{paso_efectivo:.6g} ({numero_pasos} pasos).",
                    UserWarning,
                    stacklevel=3,
                )

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
        """Normaliza el control para el método seleccionado.

        RK4 requiere un control callable para evaluarlo en los tiempos
        intermedios de cada etapa. Los métodos restantes vectorizan el
        control evaluándolo una sola vez sobre la grilla temporal.
        """
        if u is None:
            return None

        if method == "rk4":
            if not callable(u):
                raise ValueError(
                    "El método RK4 requiere que el control u sea una función callable."
                )
            return u

        if callable(u):
            warnings.warn(
                "El control callable se evaluará sobre la grilla; "
                "para mayor eficiencia use un arreglo.",
                UserWarning,
                stacklevel=2,
            )
            return np.array([u(t) for t in tiempos])

        u_array = np.asarray(u)
        if u_array.shape[0] != len(tiempos):
            raise ValueError(
                "El arreglo de control debe tener la misma longitud que la grilla temporal."
            )
        return u_array

    def _resolver_integracion(
        self,
        f: Callable,
        x0: np.ndarray,
        tiempos: np.ndarray,
        pasos: np.ndarray,
        control: Callable | np.ndarray | None,
        method: str,
        guardar_intermedios: bool,
        argumentos_fsolve: dict,
    ) -> tuple[np.ndarray, list[dict] | None]:
        """Despacha al método numérico solicitado."""
        metodos_disponibles = {
            "euler_progresivo": self._euler_progresivo,
            "euler_implicito": self._euler_implicito,
            "heun": self._heun,
            "crank_nicolson": self._crank_nicolson,
            "rk4": self._rk4,
        }
        estados, intermedios = metodos_disponibles[method](
            f, x0, tiempos, pasos, control, guardar_intermedios, argumentos_fsolve
        )
        if guardar_intermedios:
            return estados, intermedios
        return estados, None

    def _euler_progresivo(
        self,
        f: Callable,
        x0: np.ndarray,
        tiempos: np.ndarray,
        pasos: np.ndarray,
        control: np.ndarray | None,
        guardar_intermedios: bool,
        argumentos_fsolve: dict,
    ) -> tuple[np.ndarray, list[dict]]:
        """Euler progresivo: x_{k+1} = x_k + h_k * f(t_k, x_k, u_k)."""
        num_pasos = len(pasos)
        estados = np.empty((num_pasos + 1, x0.size), dtype=float)
        estados[0] = x0
        intermedios = []

        for k in range(num_pasos):
            t_k = tiempos[k]
            x_k = estados[k]
            u_k = control[k] if control is not None else None
            pendiente = f(t_k, x_k, u_k)
            estados[k + 1] = x_k + pasos[k] * np.asarray(pendiente)
            intermedios.append({})

        return estados, intermedios

    def _resolver_implicito(
        self,
        residual: Callable[[np.ndarray], np.ndarray],
        guess_inicial: np.ndarray,
        argumentos_fsolve: dict,
    ) -> np.ndarray:
        """Resuelve un sistema no lineal con scipy.optimize.fsolve.

        Los métodos implícitos definen un residual g(z) que debe anularse en
        cada paso temporal; este helper centraliza la llamada a fsolve.
        """
        return fsolve(residual, x0=guess_inicial, **argumentos_fsolve)

    def _euler_implicito(
        self,
        f: Callable,
        x0: np.ndarray,
        tiempos: np.ndarray,
        pasos: np.ndarray,
        control: np.ndarray | None,
        guardar_intermedios: bool,
        argumentos_fsolve: dict,
    ) -> tuple[np.ndarray, list[dict]]:
        """Euler implícito: resuelve x_{k+1} con fsolve.

        El residual es g(z) = z - x_k - h_k * f(t_{k+1}, z, u_{k+1}).
        """
        num_pasos = len(pasos)
        estados = np.empty((num_pasos + 1, x0.size), dtype=float)
        estados[0] = x0
        intermedios = []

        for k in range(num_pasos):
            t_siguiente = tiempos[k + 1]
            x_k = estados[k]
            h_k = pasos[k]
            u_siguiente = control[k + 1] if control is not None else None

            # La closure captura las variables del paso actual; fsolve la
            # evalúa iterativamente para distintos valores de z.
            def residual(z: np.ndarray) -> np.ndarray:
                return z - x_k - h_k * np.asarray(f(t_siguiente, z, u_siguiente))

            estados[k + 1] = self._resolver_implicito(
                residual, x_k, argumentos_fsolve
            )
            intermedios.append({})

        return estados, intermedios

    def _crank_nicolson(
        self,
        f: Callable,
        x0: np.ndarray,
        tiempos: np.ndarray,
        pasos: np.ndarray,
        control: np.ndarray | None,
        guardar_intermedios: bool,
        argumentos_fsolve: dict,
    ) -> tuple[np.ndarray, list[dict]]:
        """Crank-Nicolson: promedio de pendientes explícita e implícita.

        El residual es g(z) = z - x_k - (h_k/2) * [f(t_k, x_k, u_k) +
                                                   f(t_{k+1}, z, u_{k+1})].
        """
        num_pasos = len(pasos)
        estados = np.empty((num_pasos + 1, x0.size), dtype=float)
        estados[0] = x0
        intermedios = []

        for k in range(num_pasos):
            t_k = tiempos[k]
            t_siguiente = tiempos[k + 1]
            x_k = estados[k]
            h_k = pasos[k]
            u_k = control[k] if control is not None else None
            u_siguiente = control[k + 1] if control is not None else None

            pendiente_inicial = np.asarray(f(t_k, x_k, u_k))

            # La closure captura las variables del paso actual; fsolve la
            # evalúa iterativamente para distintos valores de z.
            def residual(z: np.ndarray) -> np.ndarray:
                pendiente_final = np.asarray(f(t_siguiente, z, u_siguiente))
                return z - x_k - (h_k / 2.0) * (
                    pendiente_inicial + pendiente_final
                )

            estados[k + 1] = self._resolver_implicito(
                residual, x_k, argumentos_fsolve
            )
            intermedios.append({})

        return estados, intermedios

    def _heun(
        self,
        f: Callable,
        x0: np.ndarray,
        tiempos: np.ndarray,
        pasos: np.ndarray,
        control: np.ndarray | None,
        guardar_intermedios: bool,
        argumentos_fsolve: dict,
    ) -> tuple[np.ndarray, list[dict]]:
        """Heun: predictor-corrector explícito de orden 2.

        Predictor: z = x_k + h_k * f(t_k, x_k, u_k)
        Corrector: x_{k+1} = x_k + (h_k/2) * [f(t_k, x_k, u_k) +
                                               f(t_{k+1}, z, u_{k+1})]
        """
        num_pasos = len(pasos)
        estados = np.empty((num_pasos + 1, x0.size), dtype=float)
        estados[0] = x0
        intermedios = []

        for k in range(num_pasos):
            t_k = tiempos[k]
            t_siguiente = tiempos[k + 1]
            x_k = estados[k]
            h_k = pasos[k]
            u_k = control[k] if control is not None else None
            u_siguiente = control[k + 1] if control is not None else None

            pendiente_inicial = np.asarray(f(t_k, x_k, u_k))
            predictor = x_k + h_k * pendiente_inicial
            pendiente_predictor = np.asarray(f(t_siguiente, predictor, u_siguiente))

            estados[k + 1] = x_k + (h_k / 2.0) * (
                pendiente_inicial + pendiente_predictor
            )
            intermedios.append({"z": predictor} if guardar_intermedios else {})

        return estados, intermedios

    def _rk4(
        self,
        f: Callable,
        x0: np.ndarray,
        tiempos: np.ndarray,
        pasos: np.ndarray,
        control: Callable | None,
        guardar_intermedios: bool,
        argumentos_fsolve: dict,
    ) -> tuple[np.ndarray, list[dict]]:
        """Runge-Kutta explícito de orden 4.

        RK4 evalúa el control en t_k, t_k + h_k/2 y t_k + h_k; por eso
        requiere que u sea callable.
        """
        num_pasos = len(pasos)
        estados = np.empty((num_pasos + 1, x0.size), dtype=float)
        estados[0] = x0
        intermedios = []

        for k in range(num_pasos):
            t_k = tiempos[k]
            h_k = pasos[k]
            x_k = estados[k]

            u_k = control(t_k) if control is not None else None
            k1 = np.asarray(f(t_k, x_k, u_k))

            t_medio = t_k + h_k / 2.0
            u_medio = control(t_medio) if control is not None else None
            k2 = np.asarray(f(t_medio, x_k + (h_k / 2.0) * k1, u_medio))
            k3 = np.asarray(f(t_medio, x_k + (h_k / 2.0) * k2, u_medio))

            t_siguiente = tiempos[k + 1]
            u_siguiente = control(t_siguiente) if control is not None else None
            k4 = np.asarray(f(t_siguiente, x_k + h_k * k3, u_siguiente))

            estados[k + 1] = x_k + (h_k / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            intermedios.append(
                {"k1": k1, "k2": k2, "k3": k3, "k4": k4}
                if guardar_intermedios
                else {}
            )

        return estados, intermedios
