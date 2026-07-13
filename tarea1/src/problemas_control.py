"""Formulación de problemas de control óptimo tipo Bolza.

Este módulo implementa las clases del Problema 2 de la tarea:
``ConjuntoAdmisible`` para representar restricciones de control,
``ControlProblem`` como formulación general del problema de Bolza, y
``ProblemaLQR`` como subclase con solución analítica vía Riccati.
"""

from collections.abc import Callable

import numpy as np

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


class ControlProblem:
    """Formulación general de un problema de control óptimo tipo Bolza.

    La dinámica está dada por ``dx/dt = f(t, x, u)``, el costo de operación por
    ``l(t, x, u)`` y el costo terminal por ``phi(x)``. Las cinco derivadas
    parciales requeridas se proveen de forma analítica en el constructor; no
    existe fallback por diferencias finitas.

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
            Costado, shape ``(n,)``.
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
            Costado, shape ``(n,)``.
        u : np.ndarray
            Control, shape ``(m,)``.

        Returns
        -------
        np.ndarray
            Derivada del costado, shape ``(n,)``.
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
