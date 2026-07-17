"""Generación reproducible de resultados y figuras del Problema 3."""

from pathlib import Path
import sys

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metodos_optimizacion import fbsm
from problemas_control import ProblemaLQR
from validacion_problema3 import crear_problema_lqr_fbsm, crear_problema_sir


def _comparar_lqr(h: float):
    """Resuelve el LQR con FBSM genérico y una referencia Riccati independiente."""
    problema = crear_problema_lqr_fbsm(-1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0)
    referencia = ProblemaLQR(
        A=np.array([[-1.0]]), B=np.array([[1.0]]), Q=np.array([[1.0]]),
        R=np.array([[1.0]]), S=np.array([[1.0]]), t_span=(0.0, 2.0),
        x0=np.array([1.0]), h=1e-4,
    )
    tiempos = np.linspace(0.0, 2.0, int(round(2.0 / h)) + 1)
    resultado = fbsm(
        problema, np.zeros((tiempos.size, 1)), h, "rk4",
        max_iter=100, tol=1e-10,
    )

    def lazo_cerrado(t, x):
        u = referencia.control_riccati(t, x)
        return referencia._f(t, x, u)

    estado_ref = solve_ivp(
        lazo_cerrado, referencia._t_span, referencia._x0, t_eval=tiempos,
        rtol=1e-11, atol=1e-13,
    ).y.T
    control_ref = np.array([
        referencia.control_riccati(t, x)
        for t, x in zip(tiempos, estado_ref)
    ])
    error = np.sqrt(np.trapezoid(
        (resultado.control_optimo[:, 0] - control_ref[:, 0]) ** 2,
        tiempos,
    ))
    return tiempos, resultado, control_ref, float(error)


def _guardar(figura, ruta: Path) -> None:
    figura.tight_layout()
    figura.savefig(ruta, dpi=150, bbox_inches="tight")


def _resolver_sir(A: float, T: float, h: float, tol: float):
    problema = crear_problema_sir(
        beta=0.3, gamma=0.1, A=A, B=1.0, u_max=0.4,
        S0=0.99, I0=0.01, T=T,
    )
    return fbsm(
        problema, np.zeros((int(round(T / h)) + 1, 1)), h,
        metodo_integracion="crank_nicolson", max_iter=200, tol=tol,
        omega=0.2,
    )


def generar_reporte_problema3(ruta_salida: Path, modo_rapido: bool = False) -> dict:
    """Ejecuta los experimentos 3a--3c y guarda sus cinco figuras.

    ``modo_rapido`` conserva las ecuaciones y parámetros físicos, pero acorta el
    horizonte SIR para que las pruebas de humo no repitan el barrido costoso. El
    resultado retiene las figuras para que el llamador pueda mostrarlas y cerrarlas.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.mkdir(parents=True, exist_ok=True)
    figuras = []

    hs = (0.1, 0.05, 0.025, 0.01)
    comparaciones = [_comparar_lqr(h) for h in hs]
    tiempos, resultado_lqr, control_riccati, error_h_001 = comparaciones[-1]
    errores = [comparacion[3] for comparacion in comparaciones]

    figura, ejes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    ejes[0].plot(tiempos, resultado_lqr.estado[:, 0], color="tab:blue")
    ejes[0].set(ylabel="x(t)", title="Problema 3a: trayectoria de estado")
    ejes[1].plot(tiempos, resultado_lqr.control_optimo[:, 0], color="tab:orange")
    ejes[1].set(xlabel="t", ylabel="u*(t)", title="Control óptimo por FBSM")
    ejes[2].plot(range(1, resultado_lqr.iteraciones + 1), resultado_lqr.historia_costo)
    ejes[2].set(xlabel="Iteración", ylabel="J", title="Convergencia del costo")
    for eje in ejes:
        eje.grid(alpha=0.3)
    _guardar(figura, ruta_salida / "3a_fbsm_trayectorias.png")
    figuras.append(figura)

    figura, eje = plt.subplots(figsize=(8, 4.5))
    eje.plot(tiempos, resultado_lqr.control_optimo[:, 0], label="FBSM")
    eje.plot(tiempos, control_riccati[:, 0], "--", label="Riccati")
    eje.set(xlabel="t", ylabel="u(t)", title="Problema 3b: FBSM vs. Riccati")
    eje.grid(alpha=0.3)
    eje.legend()
    _guardar(figura, ruta_salida / "3b_fbsm_vs_riccati.png")
    figuras.append(figura)

    figura, eje = plt.subplots(figsize=(7, 4.5))
    eje.loglog(hs, errores, "o-")
    eje.set(xlabel="Paso h", ylabel="Error L²", title="Error FBSM vs. Riccati")
    eje.grid(which="both", alpha=0.3)
    _guardar(figura, ruta_salida / "3b_error_l2_vs_h.png")
    figuras.append(figura)
    print("Problema 3b — error en norma L²")
    for h, error in zip(hs, errores):
        print(f"h={h:0.3f}: {error:.8e}")

    T_sir, h_sir, tol_sir = (8.0, 0.1, 1e-5) if modo_rapido else (50.0, 0.5, 1e-6)
    sir_alto = _resolver_sir(10.0, T_sir, h_sir, tol_sir)
    sir_bajo = _resolver_sir(1.0, T_sir, h_sir, tol_sir)
    tiempos_sir = np.linspace(0.0, T_sir, sir_alto.estado.shape[0])

    figura, ejes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ejes[0].plot(tiempos_sir, sir_alto.estado[:, 0], label="S(t)")
    ejes[0].plot(tiempos_sir, sir_alto.estado[:, 1], label="I(t)")
    ejes[0].set(ylabel="Proporción", title="Problema 3c: dinámica SIR controlada")
    ejes[0].legend()
    ejes[1].plot(tiempos_sir, sir_alto.control_optimo[:, 0], color="tab:green")
    ejes[1].set(xlabel="t", ylabel="u*(t)", title="Tasa óptima de vacunación")
    for eje in ejes:
        eje.grid(alpha=0.3)
    _guardar(figura, ruta_salida / "3c_sir_trayectorias.png")
    figuras.append(figura)

    figura, eje = plt.subplots(figsize=(8, 4.5))
    eje.plot(tiempos_sir, sir_alto.control_optimo[:, 0], label="A/B = 10")
    eje.plot(tiempos_sir, sir_bajo.control_optimo[:, 0], "--", label="A/B = 1")
    eje.set(xlabel="t", ylabel="u*(t)", title="Dependencia del control respecto de A/B")
    eje.grid(alpha=0.3)
    eje.legend()
    _guardar(figura, ruta_salida / "3c_sir_comparacion_ab.png")
    figuras.append(figura)

    return {
        "figuras": figuras,
        "3a": {
            "convergio": resultado_lqr.convergio,
            "iteraciones": resultado_lqr.iteraciones,
            "costo_inicial": float(resultado_lqr.historia_costo[0]),
            "costo_final": float(resultado_lqr.historia_costo[-1]),
        },
        "3b": {"error_h_001": error_h_001, "errores_l2": errores},
        "3c": {
            "omega": 0.2,
            "min_estado": float(np.min(sir_alto.estado)),
            "control_max": float(np.max(sir_alto.control_optimo)),
            "control_medio_alto": float(np.mean(sir_alto.control_optimo)),
            "control_medio_bajo": float(np.mean(sir_bajo.control_optimo)),
            "convergio": sir_alto.convergio and sir_bajo.convergio,
        },
    }
