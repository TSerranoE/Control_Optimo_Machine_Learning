"""Validated contracts and metrics for the Problem 4c method comparison."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["ResultadoComparacion", "generar_reporte_problema4"]


@dataclass(frozen=True)
class ResultadoComparacion:
    """Immutable boundary returned by a successfully published comparison."""

    figuras: tuple[Any, ...]
    tabla_lqr: pd.DataFrame
    tabla_sir: pd.DataFrame
    ruta_salida: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "figuras", tuple(self.figuras))
        object.__setattr__(self, "tabla_lqr", self.tabla_lqr.copy(deep=True))
        object.__setattr__(self, "tabla_sir", self.tabla_sir.copy(deep=True))
        object.__setattr__(self, "ruta_salida", Path(self.ruta_salida))


@dataclass(frozen=True)
class _FilaComparacion:
    """Normalized solver result before conversion to report tables."""

    metodo: str
    control: np.ndarray
    estados: np.ndarray
    adjuntos: np.ndarray
    costo_comun: float
    convergio: bool
    iteraciones_nativas: int
    etiqueta_historia: str
    tiempo_exploratorio_segundos: float | None = None
    orden_caso: float = 0.0

    def __post_init__(self) -> None:
        for nombre in ("control", "estados", "adjuntos"):
            copia = np.array(getattr(self, nombre), dtype=float, copy=True)
            copia.setflags(write=False)
            object.__setattr__(self, nombre, copia)
        if not np.isfinite(self.costo_comun):
            raise ValueError("common cost must be finite")
        if self.tiempo_exploratorio_segundos is not None and (
            not np.isfinite(self.tiempo_exploratorio_segundos)
            or self.tiempo_exploratorio_segundos < 0.0
        ):
            raise ValueError("exploratory solver time must be finite and nonnegative")


def _validar_ruta_salida(ruta_salida: Path) -> Path:
    ruta = Path(ruta_salida)
    if not ruta.parent.exists() or not ruta.parent.is_dir():
        raise ValueError("output parent must be an existing directory")
    if ruta.exists() and not ruta.is_dir():
        raise ValueError("output path must be a directory")
    return ruta


def _validar_grilla(tiempos: np.ndarray, h: float | None = None) -> np.ndarray:
    grilla = np.asarray(tiempos, dtype=float)
    if grilla.ndim != 1 or grilla.size < 2:
        raise ValueError("grid must contain at least two nodes")
    if not np.all(np.isfinite(grilla)):
        raise ValueError("grid values must be finite")
    pasos = np.diff(grilla)
    if np.any(pasos <= 0.0):
        raise ValueError("grid nodes must be strictly increasing")
    if h is not None and (not np.isfinite(h) or h <= 0.0):
        raise ValueError("grid step must be positive and finite")
    if h is not None and not np.allclose(pasos, h):
        raise ValueError("grid nodes must be aligned with h")
    return grilla.copy()


def _validar_trayectorias(
    tiempos: np.ndarray,
    control: np.ndarray,
    estados: np.ndarray,
    adjuntos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grilla = _validar_grilla(tiempos)
    u = np.asarray(control, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if u.ndim != 2:
        raise ValueError("control must have shape (N, m)")
    x, p = np.asarray(estados, dtype=float), np.asarray(adjuntos, dtype=float)
    if x.ndim != 2:
        raise ValueError("states must have shape (N, n)")
    if p.ndim != 2:
        raise ValueError("adjoints must have shape (N, n)")
    if u.shape[0] != grilla.size or x.shape[0] != grilla.size or p.shape[0] != grilla.size:
        raise ValueError("all trajectories must have the same number of nodes")
    if x.shape != p.shape:
        raise ValueError("states and adjoints must have matching shapes")
    if not all(np.all(np.isfinite(valor)) for valor in (u, x, p)):
        raise ValueError("trajectory values must be finite")
    return u.copy(), x.copy(), p.copy()


def _costo_trapezoidal(
    tiempos: np.ndarray, costos_operacion: np.ndarray, costo_terminal: float
) -> float:
    grilla = _validar_grilla(tiempos)
    costos = np.asarray(costos_operacion, dtype=float)
    if costos.shape != grilla.shape:
        raise ValueError("running costs must match a one-dimensional grid")
    if not np.all(np.isfinite(costos)) or not np.isfinite(costo_terminal):
        raise ValueError("cost inputs must be finite")
    return float(np.trapezoid(costos, grilla) + costo_terminal)


def _residuo_estacionariedad(
    problema: Any,
    tiempos: np.ndarray,
    estados: np.ndarray,
    control: np.ndarray,
    adjuntos: np.ndarray,
    metodo_integracion: str,
) -> float:
    u, x, p = _validar_trayectorias(tiempos, control, estados, adjuntos)
    if u.shape[1] != problema._m or x.shape[1] != problema._n:
        raise ValueError("trajectory shapes are incompatible with the problem")
    gradiente = np.array([
        np.asarray(problema._dl_du(t, xk, uk), dtype=float)
        + np.asarray(problema._df_du(t, xk, uk), dtype=float).T @ pk
        for t, xk, uk, pk in zip(tiempos, x, u, p)
    ])
    direccion = problema.proj(u - gradiente, metodo_integracion) - u
    integrando = np.sum(direccion**2, axis=1)
    return float(np.sqrt(max(0.0, np.trapezoid(integrando, tiempos))))


def _metricas_riccati(
    tiempos: np.ndarray,
    control: np.ndarray,
    control_referencia: np.ndarray,
    costo: float,
    costo_referencia: float,
) -> tuple[float, float]:
    tiempos = _validar_grilla(tiempos)
    u = np.asarray(control, dtype=float)
    referencia = np.asarray(control_referencia, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if referencia.ndim == 1:
        referencia = referencia.reshape(-1, 1)
    if u.shape != referencia.shape or u.shape[0] != len(tiempos):
        raise ValueError("control and Riccati reference must share the grid shape")
    if not all(np.all(np.isfinite(valor)) for valor in (u, referencia, costo, costo_referencia)):
        raise ValueError("Riccati metric inputs must be finite")
    error = np.sqrt(np.trapezoid(np.sum((u - referencia) ** 2, axis=1), tiempos))
    brecha = (float(costo) - float(costo_referencia)) / max(
        abs(float(costo_referencia)), np.finfo(float).eps
    )
    return float(error), float(brecha)


def _control_riccati(referencia: Any, tiempos: np.ndarray, estados: np.ndarray) -> np.ndarray:
    return np.array([
        referencia.control_optimo_puntual(t, x, np.zeros(referencia._n))
        for t, x in zip(tiempos, estados)
    ])


def _etiqueta_historia(metodo: str, iteraciones: int) -> str:
    if metodo == "FBSM":
        return f"FBSM accepted control k=1..{iteraciones}"
    if metodo == "Projected gradient":
        return f"Projected gradient initial control k=0; accepted controls k=1..{iteraciones}"
    raise ValueError("unknown comparison method")


def _ordenar_filas(filas: Iterable[_FilaComparacion]) -> list[_FilaComparacion]:
    orden_metodo = {"FBSM": 0, "Projected gradient": 1}
    return sorted(filas, key=lambda fila: (fila.orden_caso, orden_metodo[fila.metodo]))


def _cronometrar_solver(solver: Callable[[], Any]) -> tuple[Any, float]:
    inicio = perf_counter()
    resultado = solver()
    return resultado, perf_counter() - inicio


def generar_reporte_problema4(
    ruta_salida: Path, modo_rapido: bool = False
) -> ResultadoComparacion:
    """Validate the report boundary; execution/publication is delivered in PR2."""
    _validar_ruta_salida(ruta_salida)
    if not isinstance(modo_rapido, bool):
        raise ValueError("modo_rapido must be boolean")
    raise NotImplementedError("Problem 4 report runner and publication are assigned to PR2")
