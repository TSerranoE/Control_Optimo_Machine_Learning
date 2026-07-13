"""Formulación de problemas de control óptimo tipo Bolza.

Este módulo implementa las clases del Problema 2 de la tarea:
``ConjuntoAdmisible`` para representar restricciones de control,
``ControlProblem`` como formulación general del problema de Bolza, y
``ProblemaLQR`` como subclase con solución analítica vía Riccati.
"""

from typing import Callable

import numpy as np


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
