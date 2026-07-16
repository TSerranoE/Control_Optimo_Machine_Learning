"""Paquete principal para la resolución de la Tarea 1.

Este paquete contiene las herramientas computacionales desarrolladas para el
curso MA6914: integradores de EDO, formulación de problemas de control óptimo
y métodos de optimización asociados.
"""

from .metodos_optimizacion import ResultadoFBSM, fbsm
from .problemas_control import (
    ConjuntoAdmisible,
    ControlProblem,
    ProblemaLQR,
    ResultadoGradienteProyectado,
)

__all__ = [
    "ConjuntoAdmisible",
    "ControlProblem",
    "ProblemaLQR",
    "ResultadoFBSM",
    "ResultadoGradienteProyectado",
    "fbsm",
]
