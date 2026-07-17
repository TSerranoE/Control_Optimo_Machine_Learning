"""Formulación de problemas de control óptimo tipo Bolza.

Este módulo implementa las clases del Problema 2 de la tarea:
``ConjuntoAdmisible`` para representar restricciones de control,
``ControlProblem`` como formulación general del problema de Bolza, y
``ProblemaLQR`` como subclase con solución analítica vía Riccati.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import inv
from scipy.optimize import minimize, minimize_scalar

from integradores import EDOSolver


class ConjuntoAdmisible:
    """Conjunto admisible de controles con proyección por componentes.

    Soporta conjuntos tipo caja (límites por dimensión de control) y conjuntos
    irrestrictos. La proyección sobre una caja se realiza con ``numpy.clip``.

    Parameters
    ----------
    limites : tuple[tuple[float, float], ...] | None, optional
        Tupla con un par ``(inferior, superior)`` por dimensión de control.
        ``None`` indica un conjunto irrestricto. Default es ``None``.

    Attributes
    ----------
    _limites : tuple[tuple[float, float], ...] | None
        Límites almacenados internamente.

    Raises
    ------
    ValueError
        Si algún límite no tiene la forma ``(low, high)`` con ``low < high``.
    """

    def __init__(self, limites: tuple[tuple[float, float], ...] | None = None):
        """Inicializa el conjunto admisible validando los límites."""
        if limites is not None:
            limites_validados = []
            for par in limites:
                if len(par) != 2:
                    raise ValueError("Cada límite debe ser una tupla (inferior, superior).")
                inferior, superior = float(par[0]), float(par[1])
                if inferior >= superior:
                    raise ValueError(
                        "El límite inferior debe ser estrictamente menor que el superior."
                    )
                limites_validados.append((inferior, superior))
            limites = tuple(limites_validados)
        self._limites = limites

    def proyectar(self, u: np.ndarray) -> np.ndarray:
        """Proyecta el vector de control sobre el conjunto admisible.

        Parameters
        ----------
        u : np.ndarray
            Vector de control de dimensión ``m``.

        Returns
        -------
        np.ndarray
            Vector proyectado. Para conjuntos irrestrictos devuelve una copia.
        """
        u = np.asarray(u)
        if self._limites is None:
            return u.copy()

        bajo = np.array([lim[0] for lim in self._limites])
        alto = np.array([lim[1] for lim in self._limites])
        return np.clip(u, bajo, alto)

    def es_caja(self) -> bool:
        """Indica si el conjunto está definido por una caja de límites.

        Returns
        -------
        bool
            ``True`` si existen límites; ``False`` si es irrestricto.
        """
        return self._limites is not None

    def limites(self) -> tuple[tuple[float, float], ...] | None:
        """Devuelve los límites del conjunto.

        Returns
        -------
        tuple[tuple[float, float], ...] | None
            Tupla de límites o ``None`` si es irrestricto.
        """
        return self._limites


@dataclass(frozen=True)
class ResultadoGradienteProyectado:
    """Resultado inmutable del método de gradiente proyectado."""

    control: np.ndarray
    estados: np.ndarray
    adjuntos: np.ndarray
    historial_costos: tuple[float, ...]
    iteraciones: int
    convergio: bool

    def __post_init__(self):
        for nombre in ("control", "estados", "adjuntos"):
            copia = np.array(getattr(self, nombre), dtype=float, copy=True)
            copia.setflags(write=False)
            object.__setattr__(self, nombre, copia)
        object.__setattr__(
            self, "historial_costos", tuple(float(c) for c in self.historial_costos)
        )


class ControlProblem:
    """Formulación general de un problema de control óptimo tipo Bolza.

    La dinámica está dada por ``dx/dt = f(t, x, u)``, el costo de operación por
    ``l(t, x, u)`` y el costo terminal por ``phi(x)``. Las cinco derivadas
    parciales requeridas se proveen de forma analítica en el constructor.

    Parameters
    ----------
    f : callable
        Dinámica ``f(t, x, u) -> ndarray`` de shape ``(n,)``.
    l : callable
        Costo de operación ``l(t, x, u) -> float``.
    phi : callable
        Costo terminal ``phi(x) -> float``.
    df_dx : callable
        Jacobiano ``∂f/∂x(t, x, u) -> ndarray`` de shape ``(n, n)``.
    df_du : callable
        Jacobiano ``∂f/∂u(t, x, u) -> ndarray`` de shape ``(n, m)``.
    dl_dx : callable
        Gradiente ``∂l/∂x(t, x, u) -> ndarray`` de shape ``(n,)``.
    dl_du : callable
        Gradiente ``∂l/∂u(t, x, u) -> ndarray`` de shape ``(m,)``.
    dphi_dx : callable
        Gradiente ``∂phi/∂x(x) -> ndarray`` de shape ``(n,)``.
    t_span : tuple[float, float]
        Intervalo temporal ``(t0, tf)`` con ``tf > t0``.
    x0 : np.ndarray
        Estado inicial, shape ``(n,)``.
    m : int
        Dimensión del control. Debe ser un entero positivo.
    conjunto_admisible : ConjuntoAdmisible | None, optional
        Restricciones de control. Default ``None`` (irrestricto).
    solver : EDOSolver | None, optional
        Integrador de EDOs inyectado. Default ``EDOSolver()``.

    Raises
    ------
    ValueError
        Si ``t_span`` no tiene dos elementos, ``t_span[1] <= t_span[0]``,
        ``m`` no es entero positivo, ``x0`` no es un ndarray 1D,
        o la dinámica no devuelve la dimensión esperada.
    TypeError
        Si alguna de las cinco derivadas no es callable.
    """

    def __init__(
        self,
        f: Callable,
        l: Callable,
        phi: Callable,
        df_dx: Callable,
        df_du: Callable,
        dl_dx: Callable,
        dl_du: Callable,
        dphi_dx: Callable,
        t_span: tuple[float, float],
        x0: np.ndarray,
        m: int,
        conjunto_admisible: ConjuntoAdmisible | None = None,
        solver: EDOSolver | None = None,
    ):
        """Inicializa el problema de control validando las entradas."""
        if not isinstance(x0, np.ndarray) or x0.ndim != 1 or x0.size == 0:
            raise ValueError("x0 debe ser un ndarray numérico 1D no vacío.")

        try:
            t0, tf = float(t_span[0]), float(t_span[1])
        except Exception as exc:
            raise ValueError(
                "t_span debe ser una tupla de dos valores numéricos (t0, tf)."
            ) from exc

        if len(t_span) != 2 or tf <= t0:
            raise ValueError(
                "t_span debe tener dos elementos con el tiempo final "
                "estrictamente mayor al inicial."
            )

        if not isinstance(m, int) or m <= 0:
            raise ValueError("La dimensión del control m debe ser un entero positivo.")

        derivadas = {
            "df_dx": df_dx,
            "df_du": df_du,
            "dl_dx": dl_dx,
            "dl_du": dl_du,
            "dphi_dx": dphi_dx,
        }
        for nombre, derivada in derivadas.items():
            if not callable(derivada):
                raise TypeError(
                    f"La derivada '{nombre}' es obligatoria y debe ser callable."
                )

        self._f = f
        self._l = l
        self._phi = phi
        self._t_span = (t0, tf)
        self._t0 = t0
        self._T = tf - t0
        self._x0 = np.asarray(x0, dtype=float)
        self._n = self._x0.shape[0]
        self._m = m
        self._conjunto = conjunto_admisible
        self._solver = solver if solver is not None else EDOSolver()
        self._df_dx = df_dx
        self._df_du = df_du
        self._dl_dx = dl_dx
        self._dl_du = dl_du
        self._dphi_dx = dphi_dx

        u_prueba = np.zeros(self._m)
        try:
            f_prueba = np.asarray(self._f(0.0, self._x0, u_prueba))
        except Exception as exc:
            raise ValueError(
                "No fue posible evaluar la dinámica f(0, x0, 0). "
                "Verifique las dimensiones de f."
            ) from exc

        if f_prueba.shape != (self._n,):
            raise ValueError(
                f"La dinámica f debe devolver un ndarray de shape ({self._n},), "
                f"pero devolvió {f_prueba.shape}."
            )

    @property
    def t_span(self) -> tuple[float, float]:
        """Devuelve el intervalo temporal inmutable del problema."""
        return self._t_span

    @property
    def estado_inicial(self) -> np.ndarray:
        """Devuelve una copia defensiva del estado inicial."""
        return self._x0.copy()

    @property
    def dimension_estado(self) -> int:
        """Devuelve la dimensión del estado."""
        return self._n

    @property
    def dimension_control(self) -> int:
        """Devuelve la dimensión del control."""
        return self._m

    def hamiltoniano(
        self, t: float, x: np.ndarray, p: np.ndarray, u: np.ndarray
    ) -> float:
        """Calcula el Hamiltoniano ``H = l(t, x, u) + p @ f(t, x, u)``.

        Parameters
        ----------
        t : float
            Instante de tiempo.
        x : np.ndarray
            Estado, shape ``(n,)``.
        p : np.ndarray
            Estado adjunto, shape ``(n,)``.
        u : np.ndarray
            Control, shape ``(m,)``.

        Returns
        -------
        float
            Valor escalar del Hamiltoniano.
        """
        return float(self._l(t, x, u) + np.dot(p, self._f(t, x, u)))

    def sistema_adjunto(
        self, t: float, x: np.ndarray, p: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        """Calcula ``-∂H/∂x`` para el sistema adjunto.

        Parameters
        ----------
        t : float
            Instante de tiempo.
        x : np.ndarray
            Estado, shape ``(n,)``.
        p : np.ndarray
            Estado adjunto, shape ``(n,)``.
        u : np.ndarray
            Control, shape ``(m,)``.

        Returns
        -------
        np.ndarray
            Derivada del estado adjunto, shape ``(n,)``.
        """
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float)

        grad_l = np.asarray(self._dl_dx(t, x, u))
        J_f = np.asarray(self._df_dx(t, x, u))

        return -(grad_l + J_f.T @ p)

    def condicion_transversalidad(self, x_T: np.ndarray) -> np.ndarray:
        """Evalúa la condición de transversalidad ``∂phi/∂x`` en ``x_T``.

        Parameters
        ----------
        x_T : np.ndarray
            Estado final, shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Gradiente del costo terminal, shape ``(n,)``.
        """
        x_T = np.asarray(x_T, dtype=float)
        return np.asarray(self._dphi_dx(x_T))

    def proyectar_control(self, u: np.ndarray) -> np.ndarray:
        """Proyecta uno o más controles sobre el conjunto admisible."""
        control = np.asarray(u, dtype=float)
        if self._conjunto is None:
            return control.copy()
        if control.ndim == 1:
            return self._conjunto.proyectar(control)
        return np.array([self._conjunto.proyectar(nodo) for nodo in control])

    def evaluar_costo_trayectoria(
        self, tiempos: np.ndarray, estados: np.ndarray, controles: np.ndarray
    ) -> float:
        """Evalúa el costo sobre trayectorias ya integradas."""
        costos = np.array(
            [self._l(t, x, u) for t, x, u in zip(tiempos, estados, controles)],
            dtype=float,
        )
        return float(np.trapezoid(costos, tiempos) + self._phi(estados[-1]))

    def evaluar_costo(
        self,
        u_traj: Callable | np.ndarray,
        h: float,
        metodo_integracion: str | None = None,
    ) -> float:
        """Evalúa el funcional de costo Bolza para una trayectoria de control.

        Integra la dinámica con el paso ``h`` y el método indicado, evalúa el
        costo de operación ``l`` en cada punto de la grilla mediante cuadratura
        trapezoidal y suma el costo terminal ``phi`` en el estado final.

        Parameters
        ----------
        u_traj : callable o np.ndarray
            Control definido sobre la grilla temporal. Si es callable, se
            evalúa en cada ``t_k``; si es ndarray, se usa directamente.
        h : float
            Paso temporal obligatorio para la integración.
        metodo_integracion : str | None, optional
            Método numérico a utilizar. ``None`` delega al integrador por
            defecto (RK4).

        Returns
        -------
        float
            Valor aproximado del costo ``J(u)``.

        Raises
        ------
        TypeError
            Si no se proporciona el paso de integración ``h``.
        """
        sol = self._solver.solve(
            self._f,
            self._x0,
            self._t_span,
            h,
            method=metodo_integracion,
            u=u_traj,
        )
        tiempos = sol.tiempos
        estados = sol.estados

        if callable(u_traj):
            controles = np.array([u_traj(t) for t in tiempos])
        else:
            controles = np.asarray(u_traj)

        if controles.ndim == 1 and self._m == 1:
            controles = controles.reshape(-1, 1)

        if controles.shape[0] != len(tiempos):
            raise ValueError(
                "La trayectoria de control debe tener la misma longitud que la "
                "grilla temporal obtenida de la integración."
            )

        costos = np.array(
            [
                float(self._l(t, x, u))
                for t, x, u in zip(tiempos, estados, controles)
            ]
        )
        integral = np.trapezoid(costos, tiempos)
        return float(integral + self._phi(estados[-1]))

    def _normalizar_control(
        self, u: np.ndarray, metodo_integracion: str
    ) -> tuple[np.ndarray, float]:
        """Valida un control nodal y devuelve su paso temporal inferido."""
        if metodo_integracion not in EDOSolver.METODOS:
            raise ValueError(
                f"Método '{metodo_integracion}' no válido. "
                f"Disponibles: {EDOSolver.METODOS}"
            )

        control = np.asarray(u, dtype=float)
        if control.ndim == 1 and self._m == 1:
            control = control.reshape(-1, 1)
        if control.ndim != 2 or control.shape[1] != self._m:
            raise ValueError(
                f"El control debe tener shape (N, {self._m}) con N >= 2."
            )
        if control.shape[0] < 2:
            raise ValueError("El control debe contener al menos dos nodos.")
        if not np.all(np.isfinite(control)):
            raise ValueError("El control debe contener solo valores finitos.")

        h = self._T / (control.shape[0] - 1)
        return control.copy(), h

    def _integrar_estado(
        self, control: np.ndarray, h: float | np.ndarray, metodo_integracion: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integra el estado usando el convenio de control de cada método."""
        pasos = np.asarray(h, dtype=float)
        grilla: tuple[float, ...] = self._t_span
        if pasos.ndim == 1:
            grilla_array = self._t0 + np.concatenate(([0.0], np.cumsum(pasos)))
            grilla_array[-1] = self._t_span[1]
            grilla = tuple(grilla_array)
        u_solver = control
        if metodo_integracion == "rk4":
            tiempos = np.linspace(self._t0, self._t_span[1], control.shape[0])
            if pasos.ndim == 1:
                tiempos = np.asarray(grilla)
            u_solver = interp1d(tiempos, control, axis=0, kind="linear")
        solucion = self._solver.solve(
            self._f,
            self._x0,
            grilla,
            h,
            method=metodo_integracion,
            u=u_solver,
        )
        return solucion.tiempos, solucion.estados

    def integrar_estado(
        self, control: np.ndarray, h: float | np.ndarray, metodo_integracion: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integra públicamente el estado con control nodal."""
        return self._integrar_estado(control, h, metodo_integracion)

    def _integrar_adjunto(
        self,
        tiempos: np.ndarray,
        estados: np.ndarray,
        control: np.ndarray,
        h: float | np.ndarray,
        metodo_integracion: str,
    ) -> np.ndarray:
        """Integra el adjunto continuo por reversión temporal."""
        pasos = np.asarray(h, dtype=float)
        if pasos.ndim == 1:
            return self._integrar_adjunto_en_grilla(
                tiempos, estados, control, pasos, metodo_integracion
            )

        tf = self._t_span[1]
        q_0 = self.condicion_transversalidad(estados[-1])

        def campo_reverso(tau, q, datos):
            x = np.asarray(datos[: self._n], dtype=float)
            u = np.asarray(datos[self._n :], dtype=float)
            return -self.sistema_adjunto(tf - tau, x, q, u)

        datos_nodales = np.hstack((estados, control))
        if metodo_integracion == "rk4":
            interpolador = interp1d(tiempos, datos_nodales, axis=0, kind="linear")
            datos_reversos = lambda tau: np.asarray(interpolador(tf - tau))
        else:
            datos_reversos = datos_nodales[::-1]

        solucion = self._solver.solve(
            campo_reverso,
            q_0,
            (0.0, self._T),
            h,
            method=metodo_integracion,
            u=datos_reversos,
        )
        return solucion.estados[::-1]

    def _integrar_adjunto_en_grilla(
        self,
        tiempos: np.ndarray,
        estados: np.ndarray,
        control: np.ndarray,
        pasos: np.ndarray,
        metodo_integracion: str,
    ) -> np.ndarray:
        """Integra el adjunto sobre una grilla posiblemente no uniforme."""
        t0, tf = tiempos[0], tiempos[-1]
        horizonte = tf - t0
        x_interp = interp1d(
            tiempos, estados, axis=0, kind="linear", fill_value="extrapolate"
        )
        u_interp = interp1d(
            tiempos, control, axis=0, kind="linear", fill_value="extrapolate"
        )

        def campo_reverso(tau, p, _u=None):
            t = tf - tau
            return -self.sistema_adjunto(t, x_interp(t), p, u_interp(t))

        pasos_reversos = pasos[::-1]
        tiempos_tau = np.concatenate(([0.0], np.cumsum(pasos_reversos)))
        tiempos_tau[-1] = horizonte
        solucion = EDOSolver().solve(
            campo_reverso,
            self.condicion_transversalidad(estados[-1]),
            tuple(tiempos_tau),
            pasos_reversos,
            method=metodo_integracion,
        )
        return solucion.estados[::-1]

    def integrar_adjunto(
        self,
        tiempos: np.ndarray,
        estados: np.ndarray,
        control: np.ndarray,
        h: float | np.ndarray,
        metodo_integracion: str,
    ) -> np.ndarray:
        """Integra públicamente el sistema adjunto continuo."""
        return self._integrar_adjunto(tiempos, estados, control, h, metodo_integracion)

    @staticmethod
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
            h
            * np.sum(
                (valores[:-1] + 4.0 * valores_medios + valores[1:]) / 6.0
            )
        )

    def _evaluar_costo_nodal(
        self, u: np.ndarray, metodo_integracion: str
    ) -> float:
        """Evalúa el costo con cuadratura consistente con el método."""
        control, h = self._normalizar_control(u, metodo_integracion)
        if metodo_integracion == "rk4":
            tiempos = np.linspace(self._t0, self._t_span[1], control.shape[0])
            control_interpolado = interp1d(tiempos, control, axis=0, kind="linear")

            def sistema_aumentado(t, estado, control_t):
                x = estado[: self._n]
                return np.concatenate(
                    (self._f(t, x, control_t), [self._l(t, x, control_t)])
                )

            estado_inicial = np.concatenate((self._x0, [0.0]))
            solucion = self._solver.solve(
                sistema_aumentado,
                estado_inicial,
                self._t_span,
                h,
                method="rk4",
                u=control_interpolado,
            )
            estado_final = solucion.estados[-1]
            return float(estado_final[-1] + self._phi(estado_final[: self._n]))

        tiempos, estados = self._integrar_estado(control, h, metodo_integracion)
        valores = np.array(
            [self._l(t, x, c) for t, x, c in zip(tiempos, estados, control)],
            dtype=float,
        )
        integral = self._integrar_valores(valores, h, metodo_integracion)
        return float(integral + self._phi(estados[-1]))

    def grad(self, u: np.ndarray, metodo_integracion: str) -> np.ndarray:
        """Calcula el gradiente reducido mediante el adjunto continuo."""
        control, h = self._normalizar_control(u, metodo_integracion)
        tiempos, estados = self._integrar_estado(control, h, metodo_integracion)
        adjuntos = self._integrar_adjunto(
            tiempos, estados, control, h, metodo_integracion
        )
        return np.array(
            [
                np.asarray(self._dl_du(t, x, c), dtype=float)
                + np.asarray(self._df_du(t, x, c), dtype=float).T @ p
                for t, x, c, p in zip(tiempos, estados, control, adjuntos)
            ],
            dtype=float,
        )

    def L2InnerProd(
        self, u_1: np.ndarray, u_2: np.ndarray, metodo_integracion: str
    ) -> float:
        """Calcula el producto interno L2 con cuadratura según el método."""
        primero, h = self._normalizar_control(u_1, metodo_integracion)
        segundo, _ = self._normalizar_control(u_2, metodo_integracion)
        if primero.shape != segundo.shape:
            raise ValueError("Los controles deben tener el mismo shape.")

        valores = np.einsum("ij,ij->i", primero, segundo)
        valores_medios = None
        if metodo_integracion == "rk4":
            primero_medio = (primero[:-1] + primero[1:]) / 2.0
            segundo_medio = (segundo[:-1] + segundo[1:]) / 2.0
            valores_medios = np.einsum("ij,ij->i", primero_medio, segundo_medio)
        return self._integrar_valores(
            valores, h, metodo_integracion, valores_medios
        )

    def L2Norm(self, u: np.ndarray, metodo_integracion: str) -> float:
        """Calcula la norma inducida por ``L2InnerProd``."""
        producto = self.L2InnerProd(u, u, metodo_integracion)
        return float(np.sqrt(max(0.0, producto)))

    def proj(self, u: np.ndarray, metodo_integracion: str) -> np.ndarray:
        """Proyecta un control nodal punto a punto sobre el conjunto admisible."""
        control, _ = self._normalizar_control(u, metodo_integracion)
        if self._conjunto is None:
            return control
        return np.array([self._conjunto.proyectar(nodo) for nodo in control])

    def BBStep(
        self,
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
            self._normalizar_control(valor, metodo_integracion)[0]
            for valor in (u_1, u_2, g_1, g_2)
        ]
        if len({valor.shape for valor in controles}) != 1:
            raise ValueError("Los controles y gradientes deben tener el mismo shape.")

        s = controles[1] - controles[0]
        y = controles[3] - controles[2]
        numerador = self.L2InnerProd(s, s, metodo_integracion)
        denominador = self.L2InnerProd(s, y, metodo_integracion)
        umbral = np.finfo(float).eps * abs(numerador)
        if not np.isfinite(denominador) or denominador <= umbral:
            return 1.0
        paso = numerador / denominador
        if not np.isfinite(paso) or paso <= 0.0:
            return 1.0
        return float(np.clip(paso, t_min, 1.0))

    def backtracking(
        self,
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

        control, _ = self._normalizar_control(u, metodo_integracion)
        gradiente, _ = self._normalizar_control(g, metodo_integracion)
        direccion, _ = self._normalizar_control(v, metodo_integracion)
        if control.shape != gradiente.shape or control.shape != direccion.shape:
            raise ValueError("u, g y v deben tener el mismo shape.")

        producto = self.L2InnerProd(gradiente, direccion, metodo_integracion)
        paso = float(t_inicial)
        for reduccion in range(max_reducciones + 1):
            candidato = control + paso * direccion
            costo = self._evaluar_costo_nodal(candidato, metodo_integracion)
            if costo <= J_hat + a * paso * producto:
                return paso
            if reduccion < max_reducciones:
                paso *= b
        raise RuntimeError("El backtracking agotó las reducciones permitidas.")

    def gradiente_proyectado(
        self,
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
    ) -> ResultadoGradienteProyectado:
        """Minimiza el costo mediante gradiente proyectado y búsqueda Armijo."""
        control, h = self._normalizar_control(u_inicial, metodo_integracion)
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

        costo_anterior = self._evaluar_costo_nodal(control, metodo_integracion)
        historial = [costo_anterior]
        control_anterior = gradiente_anterior = None
        convergio = False

        for iteraciones in range(1, max_iter + 1):
            gradiente = self.grad(control, metodo_integracion)
            direccion = self.proj(
                control - gradiente, metodo_integracion
            ) - control
            semilla = 1
            if control_anterior is not None:
                semilla = min(
                    1,
                    self.BBStep(
                        control_anterior,
                        control,
                        gradiente_anterior,
                        gradiente,
                        metodo_integracion,
                        t_min=t_min,
                    ),
                )
            paso = self.backtracking(
                control,
                gradiente,
                direccion,
                a,
                b,
                max(historial[-r:]),
                metodo_integracion,
                t_inicial=semilla,
                max_reducciones=max_reducciones,
            )
            nuevo_control = control + paso * direccion
            nuevo_costo = self._evaluar_costo_nodal(
                nuevo_control, metodo_integracion
            )
            historial.append(nuevo_costo)
            control_anterior, gradiente_anterior = control, gradiente
            control = nuevo_control
            cambio_relativo = abs(nuevo_costo - costo_anterior) / max(
                1.0, abs(costo_anterior)
            )
            costo_anterior = nuevo_costo
            if cambio_relativo <= tolerancia:
                convergio = True
                break

        tiempos, estados = self._integrar_estado(control, h, metodo_integracion)
        adjuntos = self._integrar_adjunto(
            tiempos, estados, control, h, metodo_integracion
        )
        historial[-1] = self._evaluar_costo_nodal(control, metodo_integracion)
        return ResultadoGradienteProyectado(
            control, estados, adjuntos, tuple(historial), iteraciones, convergio
        )

    def fbsm(
        self,
        u_inicial: np.ndarray,
        h: float | np.ndarray,
        metodo_integracion: str = "rk4",
        max_iter: int = 100,
        tol: float = 1e-6,
        omega: float = 0.99,
    ):
        """Resuelve FBSM delegando en la función externa."""
        if __package__:
            from .metodos_optimizacion import fbsm
        else:
            from metodos_optimizacion import fbsm

        return fbsm(self, u_inicial, h, metodo_integracion, max_iter, tol, omega)

    def control_optimo_puntual(
        self, t: float, x: np.ndarray, p: np.ndarray
    ) -> np.ndarray:
        """Devuelve el control que minimiza el Hamiltoniano puntualmente.

        Resuelve ``argmin_u H(t, x, p, u)`` respetando el conjunto admisible.
        Para controles escalares usa ``minimize_scalar``; para dimensión mayor
        usa ``minimize`` con ``L-BFGS-B``.

        Parameters
        ----------
        t : float
            Instante de tiempo.
        x : np.ndarray
            Estado, shape ``(n,)``.
        p : np.ndarray
            Estado adjunto, shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Control óptimo puntual, shape ``(m,)``.
        """
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float)

        def objetivo(u: np.ndarray) -> float:
            return self.hamiltoniano(t, x, p, u)

        if self._m == 1:
            if self._conjunto is not None and self._conjunto.es_caja():
                bounds = self._conjunto.limites()[0]
                resultado = minimize_scalar(
                    objetivo, bounds=bounds, method="bounded", options={"xatol": 1e-8}
                )
            else:
                resultado = minimize_scalar(objetivo, method="brent", tol=1e-8)
            if not resultado.success or not np.all(np.isfinite(resultado.x)):
                raise RuntimeError("La minimización puntual del Hamiltoniano falló.")
            return np.array([float(resultado.x)])

        u0 = np.zeros(self._m)
        bounds = None
        if self._conjunto is not None and self._conjunto.es_caja():
            bounds = self._conjunto.limites()

        resultado = minimize(
            objetivo, u0, method="L-BFGS-B", bounds=bounds, tol=1e-6
        )
        if not resultado.success or not np.all(np.isfinite(resultado.x)):
            raise RuntimeError("La minimización puntual del Hamiltoniano falló.")
        return np.asarray(resultado.x, dtype=float)


class ProblemaLQR(ControlProblem):
    """Problema de control lineal cuadrático regulador (LQR) finito horizonte.

    Construye automáticamente la dinámica lineal, el costo cuadrático y las
    cinco derivadas parciales analíticas requeridas por ``ControlProblem``.
    Precomputa la matriz Riccati ``P(t)`` resolviendo la ecuación diferencial
    de Riccati hacia atrás mediante reversión temporal con ``EDOSolver``.

    Parameters
    ----------
    A : np.ndarray
        Matriz de estado, shape ``(n, n)``.
    B : np.ndarray
        Matriz de control, shape ``(n, m)``.
    Q : np.ndarray
        Peso del estado en el costo de operación, shape ``(n, n)``.
    R : np.ndarray
        Peso del control en el costo de operación, shape ``(m, m)``.
    S : np.ndarray
        Peso terminal, shape ``(n, n)``.
    t_span : tuple[float, float]
        Intervalo temporal ``(t0, tf)`` con ``tf > t0``.
    x0 : np.ndarray
        Estado inicial, shape ``(n,)``.
    h : float
        Paso de integración para la DRE de Riccati.
    conjunto_admisible : ConjuntoAdmisible | None, optional
        Restricciones de control. Default ``None``.
    solver : EDOSolver | None, optional
        Integrador de EDOs. Default ``EDOSolver()``.

    Raises
    ------
    ValueError
        Si las dimensiones de las matrices no son consistentes.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        S: np.ndarray,
        t_span: tuple[float, float],
        x0: np.ndarray,
        h: float,
        conjunto_admisible: ConjuntoAdmisible | None = None,
        solver: EDOSolver | None = None,
    ):
        """Inicializa el problema LQR validando matrices y precomputando P(t)."""
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        S = np.asarray(S, dtype=float)

        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError("A debe tener shape (n, n).")
        if B.ndim != 2 or B.shape[0] != n:
            raise ValueError("B debe tener shape (n, m).")
        m = B.shape[1]
        if Q.shape != (n, n):
            raise ValueError("Q debe tener shape (n, n).")
        if R.shape != (m, m):
            raise ValueError("R debe tener shape (m, m).")
        if S.shape != (n, n):
            raise ValueError("S debe tener shape (n, n).")

        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (n,):
            raise ValueError("x0 debe tener shape (n,).")

        f = lambda t, x, u: A @ np.asarray(x, dtype=float) + B @ np.asarray(
            u, dtype=float
        )
        l = lambda t, x, u: 0.5 * float(
            np.asarray(x, dtype=float) @ Q @ np.asarray(x, dtype=float)
            + np.asarray(u, dtype=float) @ R @ np.asarray(u, dtype=float)
        )
        phi = lambda x: 0.5 * float(
            np.asarray(x, dtype=float) @ S @ np.asarray(x, dtype=float)
        )

        df_dx = lambda t, x, u: A
        df_du = lambda t, x, u: B
        dl_dx = lambda t, x, u: Q @ np.asarray(x, dtype=float)
        dl_du = lambda t, x, u: R @ np.asarray(u, dtype=float)
        dphi_dx = lambda x: S @ np.asarray(x, dtype=float)

        super().__init__(
            f=f,
            l=l,
            phi=phi,
            df_dx=df_dx,
            df_du=df_du,
            dl_dx=dl_dx,
            dl_du=dl_du,
            dphi_dx=dphi_dx,
            t_span=t_span,
            x0=x0,
            m=m,
            conjunto_admisible=conjunto_admisible,
            solver=solver,
        )

        self._A = A
        self._B = B
        self._Q = Q
        self._R = R
        self._S = S
        self._h = h
        self._R_inv = inv(R)
        self._precomputar_riccati(h)

    def _precomputar_riccati(self, h: float) -> None:
        """Resuelve la DRE hacia atrás mediante reversión temporal.

        Integra ``dP/dτ = A^T P + P A - P B R^{-1} B^T P + Q`` en
        ``τ ∈ [0, tf - t0]`` con ``P(τ=0) = S`` y construye un interpolador
        ``P(t)`` para ``t ∈ [t0, tf]`` mediante ``t = tf - τ``.
        """
        n = self._n

        def riccati_ode(tau: float, P_flat: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
            P = P_flat.reshape(n, n)
            dP = (
                self._A.T @ P
                + P @ self._A
                - P @ self._B @ self._R_inv @ self._B.T @ P
                + self._Q
            )
            return dP.flatten()

        P_T = self._S.flatten()
        sol = self._solver.solve(riccati_ode, P_T, (0.0, self._T), h)

        tf = self._t_span[1]
        # τ = tf - t. Invertimos para obtener P(t) en tiempo directo.
        self._P_tiempos = tf - sol.tiempos[::-1]
        self._P_estados = sol.estados[::-1]
        self._P_interp = interp1d(
            self._P_tiempos, self._P_estados, axis=0, kind="linear"
        )

    def control_optimo_puntual(
        self, t: float, x: np.ndarray, p: np.ndarray
    ) -> np.ndarray:
        """Devuelve el minimizador puntual del Hamiltoniano LQR.

        Parameters
        ----------
        t : float
            Instante de tiempo.
        x : np.ndarray
            Estado, shape ``(n,)`` (no utilizado por la fórmula).
        p : np.ndarray
            Estado adjunto, shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Control ``u*(t) = -R^{-1} B^T p`` proyectado sobre el conjunto
            admisible cuando corresponda, shape ``(m,)``.
        """
        p = np.asarray(p, dtype=float)
        u_libre = -self._R_inv @ self._B.T @ p

        if self._conjunto is not None:
            return self._conjunto.proyectar(u_libre)
        return u_libre

    def control_riccati(self, t: float, x: np.ndarray) -> np.ndarray:
        """Devuelve el control de realimentación obtenido mediante Riccati.

        Parameters
        ----------
        t : float
            Instante de tiempo.
        x : np.ndarray
            Estado, shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Control ``u*(t) = -R^{-1} B^T P(t) x`` proyectado sobre el
            conjunto admisible cuando corresponda, shape ``(m,)``.
        """
        x = np.asarray(x, dtype=float)
        P_t = self._P_interp(t).reshape(self._n, self._n)
        u_libre = -self._R_inv @ self._B.T @ P_t @ x

        if self._conjunto is not None:
            return self._conjunto.proyectar(u_libre)
        return u_libre
