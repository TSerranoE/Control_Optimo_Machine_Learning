"""Validated contracts and metrics for the Problem 4c method comparison."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metodos_optimizacion import fbsm
from validacion_problema3 import crear_problema_lqr_fbsm, crear_problema_sir

__all__ = ["ResultadoComparacion", "generar_reporte_problema4"]

_FIGURE_NAMES = ("lqr_comparison.png", "sir_a_over_b_1.png", "sir_a_over_b_10.png")
_ARTIFACT_NAMES = _FIGURE_NAMES + (
    "lqr_comparison.csv", "lqr_comparison.tex",
    "sir_comparison.csv", "sir_comparison.tex",
)
_LQR_COLUMNS = [
    "case", "method", "common_final_cost", "riccati_l2_error",
    "relative_cost_gap", "stationarity_residual", "converged",
    "native_iterations", "native_history_label", "runtime_seconds",
]
_SIR_COLUMNS = [
    "a_over_b", "method", "common_final_cost", "stationarity_residual",
    "converged", "native_iterations", "native_history_label",
    "runtime_seconds", "optimality_disclaimer",
]
_SIR_DISCLAIMER = "First-order proxy only; not evidence of global optimality."


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
        referencia.control_riccati(t, x)
        for t, x in zip(tiempos, estados)
    ])


def _dinamica_riccati(referencia: Any, t: float, x: np.ndarray) -> np.ndarray:
    return referencia._f(t, x, referencia.control_riccati(t, x))


def _configuracion_ejecucion(modo_rapido: bool) -> tuple[float, float, float, float, float]:
    return (0.05, 1e-5, 8.0, 0.5, 1e-6) if modo_rapido else (0.01, 1e-6, 50.0, 0.5, 1e-6)


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


def _costo_comun(problema: Any, tiempos: np.ndarray, control: np.ndarray,
                 h: float, metodo: str) -> float:
    entrada: Any = control
    if metodo == "rk4":
        interpolador = interp1d(tiempos, control, axis=0, kind="linear")
        entrada = lambda t: np.asarray(interpolador(t), dtype=float)
    return float(problema.evaluar_costo(entrada, h, metodo))


def _ejecutar_metodos(
    problema: Any, tiempos: np.ndarray, h: float, metodo: str, *,
    max_iter: int, tol: float, omega: float, orden_caso: float,
) -> list[_FilaComparacion]:
    """Run both unchanged solvers from independent zero controls."""
    ceros = np.zeros((len(tiempos), problema._m))
    resultado_fbsm, tiempo_fbsm = _cronometrar_solver(lambda: fbsm(
        problema, ceros.copy(), h, metodo, max_iter=max_iter, tol=tol, omega=omega,
    ))
    resultado_pg, tiempo_pg = _cronometrar_solver(
        lambda: problema.gradiente_proyectado(
            ceros.copy(), max_iter=max_iter, tolerancia=tol,
            metodo_integracion=metodo, r=10, a=1e-4, b=0.5,
            t_min=1e-12, max_reducciones=50,
        )
    )
    datos = (
        ("FBSM", resultado_fbsm.control_optimo, resultado_fbsm.estado,
         resultado_fbsm.adjunto, resultado_fbsm, tiempo_fbsm),
        ("Projected gradient", resultado_pg.control, resultado_pg.estados,
         resultado_pg.adjuntos, resultado_pg, tiempo_pg),
    )
    return [
        _FilaComparacion(
            nombre, control, estados, adjuntos,
            _costo_comun(problema, tiempos, control, h, metodo),
            bool(resultado.convergio), int(resultado.iteraciones),
            _etiqueta_historia(nombre, int(resultado.iteraciones)), tiempo, orden_caso,
        )
        for nombre, control, estados, adjuntos, resultado, tiempo in datos
    ]


def _resolver_lqr(modo_rapido: bool) -> tuple[np.ndarray, list[_FilaComparacion], np.ndarray, float]:
    h, tol, _, _, _ = _configuracion_ejecucion(modo_rapido)
    final = 2.0
    tiempos = np.linspace(0.0, final, int(round(final / h)) + 1)
    problema = crear_problema_lqr_fbsm(-1.0, 1.0, 1.0, 1.0, 1.0, final, 1.0, h)
    filas = _ejecutar_metodos(
        problema, tiempos, h, "rk4", max_iter=100, tol=tol,
        omega=0.99, orden_caso=0.0,
    )
    referencia = problema

    def cerrado(t, x, _u=None):
        return _dinamica_riccati(referencia, t, x)

    estado_ref = referencia._solver.solve(cerrado, referencia._x0, (0.0, final), h).estados
    control_ref = _control_riccati(referencia, tiempos, estado_ref)
    costo_ref = _costo_comun(problema, tiempos, control_ref, h, "rk4")
    return tiempos, filas, control_ref, costo_ref


def _tablas_lqr(tiempos, filas, control_ref, costo_ref) -> pd.DataFrame:
    registros = []
    for fila in _ordenar_filas(filas):
        error, brecha = _metricas_riccati(
            tiempos, fila.control, control_ref, fila.costo_comun, costo_ref,
        )
        registros.append({
            "case": "LQR", "method": fila.metodo,
            "common_final_cost": fila.costo_comun, "riccati_l2_error": error,
            "relative_cost_gap": brecha,
            "stationarity_residual": _residuo_estacionariedad(
                crear_problema_lqr_fbsm(-1, 1, 1, 1, 1, 2, 1), tiempos,
                fila.estados, fila.control, fila.adjuntos, "rk4",
            ),
            "converged": fila.convergio, "native_iterations": fila.iteraciones_nativas,
            "native_history_label": fila.etiqueta_historia,
            "runtime_seconds": fila.tiempo_exploratorio_segundos,
        })
    return pd.DataFrame(registros, columns=_LQR_COLUMNS)


def _resolver_sir(modo_rapido: bool):
    _, _, final, h, tol = _configuracion_ejecucion(modo_rapido)
    tiempos = np.linspace(0.0, final, int(round(final / h)) + 1)
    filas, problemas = [], {}
    for razon in (1.0, 10.0):
        problema = crear_problema_sir(0.3, 0.1, razon, 1.0, 0.4, 0.99, 0.01, final)
        problemas[razon] = problema
        filas.extend(_ejecutar_metodos(
            problema, tiempos, h, "crank_nicolson", max_iter=200,
            tol=tol, omega=0.2, orden_caso=razon,
        ))
    return tiempos, _ordenar_filas(filas), problemas


def _tabla_sir(tiempos, filas, problemas) -> pd.DataFrame:
    return pd.DataFrame([{
        "a_over_b": fila.orden_caso, "method": fila.metodo,
        "common_final_cost": fila.costo_comun,
        "stationarity_residual": _residuo_estacionariedad(
            problemas[fila.orden_caso], tiempos, fila.estados, fila.control,
            fila.adjuntos, "crank_nicolson",
        ),
        "converged": fila.convergio, "native_iterations": fila.iteraciones_nativas,
        "native_history_label": fila.etiqueta_historia,
        "runtime_seconds": fila.tiempo_exploratorio_segundos,
        "optimality_disclaimer": _SIR_DISCLAIMER,
    } for fila in filas], columns=_SIR_COLUMNS)


def _crear_figuras(t_lqr, filas_lqr, control_ref, t_sir, filas_sir):
    figura_lqr, eje = plt.subplots(figsize=(8, 4.5))
    for fila in filas_lqr:
        eje.plot(t_lqr, fila.control[:, 0], label=fila.metodo)
    eje.plot(t_lqr, control_ref[:, 0], "--", label="Riccati")
    eje.set(xlabel="t", ylabel="u(t)", title="LQR method comparison")
    eje.legend(); eje.grid(alpha=0.3)
    figuras = [figura_lqr]
    for razon in (1.0, 10.0):
        figura, ejes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        for fila in filas_sir:
            if fila.orden_caso == razon:
                ejes[0].plot(t_sir, fila.estados[:, 0], label=fila.metodo)
                ejes[1].plot(t_sir, fila.estados[:, 1], label=fila.metodo)
                ejes[2].plot(t_sir, fila.control[:, 0], label=fila.metodo)
        for eje, etiqueta in zip(ejes, ("S(t)", "I(t)", "u(t)")):
            eje.set_ylabel(etiqueta); eje.grid(alpha=0.3); eje.legend()
        ejes[0].set_title(f"SIR method comparison, A/B={razon:g}")
        ejes[-1].set_xlabel("t")
        figuras.append(figura)
    return tuple(figuras)


def _validar_staging(staging: Path) -> None:
    if {path.name for path in staging.iterdir()} != set(_ARTIFACT_NAMES):
        raise RuntimeError("staging must contain exactly the seven approved artifacts")
    for nombre in _FIGURE_NAMES:
        if not (staging / nombre).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("invalid PNG artifact")
    for nombre, columnas in (("lqr_comparison", _LQR_COLUMNS), ("sir_comparison", _SIR_COLUMNS)):
        if list(pd.read_csv(staging / f"{nombre}.csv").columns) != columnas:
            raise RuntimeError("invalid CSV schema")
        tex = (staging / f"{nombre}.tex").read_text()
        if "tabular" not in tex or any(col.replace("_", r"\_") not in tex for col in columnas):
            raise RuntimeError("invalid LaTeX schema")


def _guardar_latex(tabla: pd.DataFrame, ruta: Path) -> None:
    escapar = lambda valor: str(valor).replace("_", r"\_")
    filas = [" & ".join(map(escapar, tabla.columns)) + r" \\"]
    filas.extend(" & ".join(map(escapar, fila)) + r" \\" for fila in tabla.itertuples(index=False, name=None))
    formato = "l" * len(tabla.columns)
    ruta.write_text("\\begin{tabular}{" + formato + "}\n" + "\n".join(filas) + "\n\\end{tabular}\n")


def _publicar(ruta: Path, figuras, tabla_lqr, tabla_sir) -> None:
    creada = not ruta.exists()
    with TemporaryDirectory(dir=ruta.parent) as staging_raw, TemporaryDirectory(dir=ruta.parent) as backup_raw:
        staging, backup = Path(staging_raw), Path(backup_raw)
        for figura, nombre in zip(figuras, _FIGURE_NAMES):
            figura.tight_layout(); figura.savefig(staging / nombre, dpi=150, bbox_inches="tight")
        for base, tabla in (("lqr_comparison", tabla_lqr), ("sir_comparison", tabla_sir)):
            tabla.to_csv(staging / f"{base}.csv", index=False)
            _guardar_latex(tabla, staging / f"{base}.tex")
        _validar_staging(staging)
        ruta.mkdir(exist_ok=True)
        respaldados, promovidos = [], []
        try:
            for nombre in _ARTIFACT_NAMES:
                destino = ruta / nombre
                if destino.exists():
                    os.replace(destino, backup / nombre); respaldados.append(nombre)
            for nombre in _ARTIFACT_NAMES:
                os.replace(staging / nombre, ruta / nombre); promovidos.append(nombre)
        except Exception:
            for nombre in promovidos:
                (ruta / nombre).unlink(missing_ok=True)
            for nombre in respaldados:
                os.replace(backup / nombre, ruta / nombre)
            if creada:
                ruta.rmdir()
            raise


def generar_reporte_problema4(
    ruta_salida: Path, modo_rapido: bool = False
) -> ResultadoComparacion:
    """Run the comparison and atomically publish its seven approved artifacts."""
    ruta = _validar_ruta_salida(ruta_salida)
    if not isinstance(modo_rapido, bool):
        raise ValueError("modo_rapido must be boolean")
    t_lqr, filas_lqr, control_ref, costo_ref = _resolver_lqr(modo_rapido)
    t_sir, filas_sir, problemas_sir = _resolver_sir(modo_rapido)
    tabla_lqr = _tablas_lqr(t_lqr, filas_lqr, control_ref, costo_ref)
    tabla_sir = _tabla_sir(t_sir, filas_sir, problemas_sir)
    figuras = _crear_figuras(t_lqr, filas_lqr, control_ref, t_sir, filas_sir)
    _publicar(ruta, figuras, tabla_lqr, tabla_sir)
    return ResultadoComparacion(figuras, tabla_lqr, tabla_sir, ruta)
