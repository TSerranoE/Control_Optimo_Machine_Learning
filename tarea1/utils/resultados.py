"""Construcción y serialización pura de tablas de resultados numéricos."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable, Mapping
from numbers import Real

import pandas as pd

COLUMNAS_RESULTADOS = ("metodo", "h", "estado", "error_inf", "tiempo_s")
_CLAVES = {"metodo", "h", "error_inf", "tiempo_s"}
_NO_FINITOS = {"NaN", "+Inf", "-Inf"}
_NO_DISPONIBLE = "No disponible"


class ResultadoInvalidoError(ValueError):
    """Indica que un resultado no satisface el contrato reportable."""


def _real(valor: object, campo: str) -> float:
    if isinstance(valor, bool) or not isinstance(valor, Real):
        raise ResultadoInvalidoError(f"{campo} debe ser un número real")
    return float(valor)


def _formatear(valor: float) -> str:
    return f"{valor:.8e}"


def _fila(resultado: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(resultado, Mapping) or "estado" in resultado:
        raise ResultadoInvalidoError("estado se deriva de error_inf")
    if set(resultado) != _CLAVES:
        raise ResultadoInvalidoError("se requieren metodo, h, error_inf y tiempo_s")
    metodo = resultado["metodo"]
    if not isinstance(metodo, str) or not metodo.strip():
        raise ResultadoInvalidoError("metodo debe ser texto no vacío")
    h = _real(resultado["h"], "h")
    if not math.isfinite(h):
        raise ResultadoInvalidoError("h debe ser finito")
    error = _real(resultado["error_inf"], "error_inf")
    if math.isfinite(error):
        estado, error_texto = "finito", _formatear(error)
    elif math.isnan(error):
        estado, error_texto = "no_finito", "NaN"
    else:
        estado, error_texto = "no_finito", "+Inf" if error > 0 else "-Inf"
    tiempo = resultado["tiempo_s"]
    if tiempo is None:
        tiempo_texto = _NO_DISPONIBLE
    else:
        tiempo = _real(tiempo, "tiempo_s")
        if not math.isfinite(tiempo) or tiempo < 0:
            raise ResultadoInvalidoError("tiempo_s debe ser finito y no negativo")
        tiempo_texto = _formatear(tiempo)
    return {"metodo": metodo, "h": _formatear(h), "estado": estado,
            "error_inf": error_texto, "tiempo_s": tiempo_texto}


def tabla_resultados(resultados: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Normaliza resultados en el orden canónico, sin realizar E/S."""
    return pd.DataFrame([_fila(r) for r in resultados],
                        columns=COLUMNAS_RESULTADOS, dtype=str)


def _canonico(texto: str, campo: str, *, no_negativo: bool = False) -> float:
    try:
        valor = float(texto)
    except ValueError as exc:
        raise ResultadoInvalidoError(f"{campo} canónico inválido") from exc
    if (not math.isfinite(valor) or (no_negativo and valor < 0)
            or texto != _formatear(valor)):
        raise ResultadoInvalidoError(f"{campo} canónico inválido")
    return valor


def _validar(tabla: pd.DataFrame) -> list[dict[str, str]]:
    if not isinstance(tabla, pd.DataFrame) or tuple(tabla.columns) != COLUMNAS_RESULTADOS:
        raise ResultadoInvalidoError("la tabla no tiene las columnas canónicas")
    filas = tabla.to_dict("records")
    for fila in filas:
        if any(not isinstance(fila[c], str) for c in COLUMNAS_RESULTADOS):
            raise ResultadoInvalidoError("las celdas canónicas deben ser texto")
        if not fila["metodo"].strip():
            raise ResultadoInvalidoError("metodo debe ser texto no vacío")
        _canonico(fila["h"], "h")
        error = fila["error_inf"]
        estado = "no_finito" if error in _NO_FINITOS else "finito"
        if error not in _NO_FINITOS:
            _canonico(error, "error_inf")
        if fila["estado"] != estado:
            raise ResultadoInvalidoError("estado contradice error_inf")
        if fila["tiempo_s"] != _NO_DISPONIBLE:
            _canonico(fila["tiempo_s"], "tiempo_s", no_negativo=True)
    return filas


def serializar_csv(tabla: pd.DataFrame) -> str:
    """Serializa una tabla canónica como CSV UTF-8 listo, con finales LF."""
    salida = io.StringIO(newline="")
    escritor = csv.DictWriter(salida, fieldnames=COLUMNAS_RESULTADOS,
                              lineterminator="\n")
    escritor.writeheader()
    escritor.writerows(_validar(tabla))
    return salida.getvalue()


_ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
            "$": r"\$", "&": r"\&", "#": r"\#", "_": r"\_", "%": r"\%",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _latex(texto: str) -> str:
    return "".join(_ESCAPES.get(c, c) for c in texto)


def renderizar_latex(tabla: pd.DataFrame) -> str:
    """Renderiza la comparación mu=1000 desde una tabla canónica."""
    lineas = [r"\begin{table}[htbp]", r"\centering",
              r"\caption{Van der Pol mu=1000: time-versus-precision comparison}",
              r"\label{tab:van-der-pol-mu-1000}", r"\begin{tabular}{lllll}",
              r"\hline", "metodo & h & estado & error\\_inf & tiempo\\_s \\\\", r"\hline"]
    for fila in _validar(tabla):
        error = fila["error_inf"]
        if error in _NO_FINITOS:
            error = f"No finito ({error})"
        celdas = (_latex(fila["metodo"]), fila["h"], _latex(fila["estado"]),
                  error, _latex(fila["tiempo_s"]))
        lineas.append(" & ".join(celdas) + r" \\")
    lineas += [r"\hline", r"\end{tabular}", r"\par\smallskip",
               r"\footnotesize\detokenize{tiempo_s is one observed execution, not a stable benchmark.}",
               r"\end{table}"]
    return "\n".join(lineas) + "\n"
