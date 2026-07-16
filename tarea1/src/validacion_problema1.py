"""Validación del Problema 1b de la Tarea 1.

Este módulo define los campos vectoriales de Lotka-Volterra y Van der Pol,
la generación de trayectorias de referencia, el cálculo de errores, la
medición de tiempos de cómputo y la ejecución de experimentos comparativos.
"""

import time

import numpy as np
from scipy.interpolate import interp1d

from src.integradores import EDOSolution, EDOSolver


def _campo_con_parametros(campo, parametros):
    """Fija los parámetros de un campo vectorial f(t, x, u, parametros).

    Parameters
    ----------
    campo : callable
        Campo vectorial con firma ``f(t, x, u, parametros)``.
    parametros : dict
        Diccionario de parámetros del campo.

    Returns
    -------
    callable
        Campo con firma ``f(t, x, u)`` lista para ``EDOSolver``.
    """

    def campo_parametrizado(t, x, u):
        return campo(t, x, u, parametros)

    return campo_parametrizado


def _control_a_array(u, t):
    """Normaliza el control a un arreglo numérico.

    Parameters
    ----------
    u : callable, np.ndarray or None
        Control a evaluar.
    t : float
        Instante de evaluación cuando ``u`` es callable.

    Returns
    -------
    np.ndarray or None
        Control como arreglo numérico o ``None`` si no hay control.
    """
    if u is None:
        return None
    if callable(u):
        return np.asarray(u(t))
    return np.asarray(u)


def campo_lotka_volterra(t, x, u, parametros):
    """Campo vectorial generalizado de Lotka-Volterra.

    El sistema tiene la forma

    .. math::
        \\dot{x}_0 = \\alpha x_0 - \\beta x_0 x_1 + u_0,
        \\dot{x}_1 = \\delta x_0 x_1 - \\gamma x_1 + u_1.

    Parameters
    ----------
    t : float
        Tiempo actual (no aparece explícitamente en el campo autónomo).
    x : array-like
        Estado del sistema, shape (2,).
    u : callable, np.ndarray or None
        Control. Si es callable se evalúa en ``t``.
    parametros : dict
        Diccionario con claves ``alpha``, ``beta``, ``delta`` y ``gamma``.

    Returns
    -------
    np.ndarray
        Derivada del estado, shape (2,).
    """
    x = np.asarray(x)
    alpha = parametros["alpha"]
    beta = parametros["beta"]
    delta = parametros["delta"]
    gamma = parametros["gamma"]
    control = _control_a_array(u, t)

    dx0 = alpha * x[0] - beta * x[0] * x[1]
    dx1 = delta * x[0] * x[1] - gamma * x[1]

    if control is not None:
        control = np.asarray(control)
        dx0 += control[0]
        dx1 += control[1]

    return np.array([dx0, dx1])


def campo_van_der_pol(t, x, u, parametros):
    """Campo vectorial de la ecuación de Van der Pol.

    La ecuación de segundo orden

    .. math::
        \\ddot{x} - \\mu(1 - x^2)\\dot{x} + x = u

    se reduce al sistema

    .. math::
        \\dot{x}_0 = x_1,
        \\dot{x}_1 = \\mu(1 - x_0^2)x_1 - x_0 + u.

    Parameters
    ----------
    t : float
        Tiempo actual.
    x : array-like
        Estado ``[x, \\dot{x}]``.
    u : callable, np.ndarray or None
        Control. Si es callable se evalúa en ``t``.
    parametros : dict
        Diccionario con clave ``mu``.

    Returns
    -------
    np.ndarray
        Derivada del estado, shape (2,).
    """
    x = np.asarray(x)
    mu = parametros["mu"]
    control = _control_a_array(u, t)

    dx0 = x[1]
    dx1 = mu * (1.0 - x[0] ** 2) * x[1] - x[0]

    if control is not None:
        control = np.asarray(control)
        dx0 += control[0]
        dx1 += control[1]

    return np.array([dx0, dx1])


def crear_referencia(
    campo,
    x0,
    t0,
    tf,
    parametros,
    h=1e-4,
    metodo="crank_nicolson",
    argumentos_fsolve=None,
):
    """Genera una trayectoria de referencia con paso muy pequeño.

    Parameters
    ----------
    campo : callable
        Campo vectorial ``f(t, x, u, parametros)``.
    x0 : array-like
        Condición inicial.
    t0 : float
        Tiempo inicial.
    tf : float
        Tiempo final.
    parametros : dict
        Parámetros del campo.
    h : float, optional
        Paso temporal de referencia. Default 1e-4.
    metodo : str, optional
        Método numérico para la referencia. Default ``'crank_nicolson'``.
    argumentos_fsolve : dict, optional
        Argumentos para ``scipy.optimize.fsolve``. Si es ``None`` se usan
        valores robustos por defecto (``xtol=1e-12``, ``maxfev=300``).

    Returns
    -------
    t_ref : np.ndarray
        Tiempos de la referencia.
    x_ref : EDOSolution
        Solución de referencia con tiempos y estados.
    """
    if argumentos_fsolve is None:
        argumentos_fsolve = {"xtol": 1e-12, "maxfev": 300}

    resolutor = EDOSolver()
    campo_parametrizado = _campo_con_parametros(campo, parametros)
    solucion = resolutor.solve(
        campo_parametrizado,
        np.asarray(x0),
        (t0, tf),
        h,
        method=metodo,
        argumentos_fsolve=argumentos_fsolve,
    )
    return solucion.tiempos, solucion


def calcular_error_inf(solucion_aprox, solucion_ref):
    """Calcula el error en norma infinito entre dos soluciones.

    Interpola la referencia a los tiempos de la aproximación y computa el
    máximo error absoluto por componente y de forma global.

    Parameters
    ----------
    solucion_aprox : EDOSolution
        Solución aproximada.
    solucion_ref : EDOSolution
        Solución de referencia.

    Returns
    -------
    dict
        Diccionario con ``global`` (float) y ``componentes`` (np.ndarray).
    """
    interpolador = interp1d(
        solucion_ref.tiempos,
        solucion_ref.estados,
        axis=0,
        kind="cubic",
        fill_value="extrapolate",
    )
    ref_en_tiempos_aprox = interpolador(solucion_aprox.tiempos)
    diferencia = solucion_aprox.estados - ref_en_tiempos_aprox
    error_componentes = np.max(np.abs(diferencia), axis=0)
    error_global = float(np.max(error_componentes))

    return {"global": error_global, "componentes": error_componentes}


def medir_tiempo(solver, campo, x0, t_span, parametros, metodo, h):
    """Mide el tiempo de una integración con ``time.perf_counter``.

    Parameters
    ----------
    solver : EDOSolver
        Instancia del integrador.
    campo : callable
        Campo vectorial ``f(t, x, u, parametros)``.
    x0 : array-like
        Condición inicial.
    t_span : tuple[float, float]
        Intervalo temporal.
    parametros : dict
        Parámetros del campo.
    metodo : str
        Método numérico.
    h : float
        Paso temporal.

    Returns
    -------
    float
        Tiempo transcurrido en segundos.
    """
    campo_parametrizado = _campo_con_parametros(campo, parametros)
    inicio = time.perf_counter()
    solver.solve(
        campo_parametrizado,
        np.asarray(x0),
        t_span,
        h,
        method=metodo,
    )
    fin = time.perf_counter()
    return fin - inicio


def ejecutar_experimento(
    campo,
    x0,
    t0,
    tf,
    parametros,
    hs,
    metodos,
    h_referencia=1e-4,
    argumentos_fsolve=None,
    solucion_ref=None,
):
    """Ejecuta un experimento comparativo para varios métodos y pasos.

    Parameters
    ----------
    campo : callable
        Campo vectorial ``f(t, x, u, parametros)``.
    x0 : array-like
        Condición inicial.
    t0 : float
        Tiempo inicial.
    tf : float
        Tiempo final.
    parametros : dict
        Parámetros del campo.
    hs : list[float]
        Pasos temporales a evaluar.
    metodos : list[str]
        Métodos numéricos a evaluar.
    h_referencia : float, optional
        Paso temporal para la trayectoria de referencia. Default 1e-4.
    argumentos_fsolve : dict, optional
        Argumentos para ``fsolve`` al generar la referencia. Se reenvían a
        ``crear_referencia``.
    solucion_ref : EDOSolution, optional
        Referencia precalculada. Si se proporciona, no se llama a
        ``crear_referencia``.

    Returns
    -------
    list[dict]
        Lista de resultados con ``metodo``, ``h``, ``error_inf`` y ``tiempo_s``.
    """
    if solucion_ref is None:
        _, solucion_ref = crear_referencia(
            campo,
            x0,
            t0,
            tf,
            parametros,
            h=h_referencia,
            metodo="crank_nicolson",
            argumentos_fsolve=argumentos_fsolve,
        )
    resultados = []

    for metodo in metodos:
        for h in hs:
            resolutor = EDOSolver()
            campo_parametrizado = _campo_con_parametros(campo, parametros)

            inicio = time.perf_counter()
            solucion = resolutor.solve(
                campo_parametrizado,
                np.asarray(x0),
                (t0, tf),
                h,
                method=metodo,
            )
            fin = time.perf_counter()

            errores = calcular_error_inf(solucion, solucion_ref)
            resultados.append(
                {
                    "metodo": metodo,
                    "h": h,
                    "error_inf": errores["global"],
                    "tiempo_s": fin - inicio,
                }
            )

    return resultados
