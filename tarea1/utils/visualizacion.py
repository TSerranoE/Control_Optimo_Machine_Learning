"""Helpers de visualización para resultados de integración de EDOs.

Este módulo contiene funciones puras para generar gráficos comparativos,
retratos de fase y tablas de resultados a partir de las salidas de los
experimentos numéricos.
"""

import matplotlib.pyplot as plt
import numpy as np

from utils.resultados import tabla_resultados


def _formatear_h(h):
    """Devuelve una representación compacta de un paso temporal.

    Parameters
    ----------
    h : float
        Paso temporal a formatear.

    Returns
    -------
    str
        Representación en notación científica con exponente simplificado
        (por ejemplo ``'1e-4'`` en vez de ``'1e-04'``).
    """
    texto = f"{h:.0e}"
    base, exp = texto.split("e")
    return f"{base}e{int(exp)}"


def graficar_error_vs_h(
    resultados,
    ruta_salida=None,
    titulo=None,
    metodo_referencia="Crank-Nicolson",
    h_referencia=None,
    caption=None,
):
    """Grafica el error en norma infinito versus el paso temporal.

    Parameters
    ----------
    resultados : list[dict]
        Lista de diccionarios con claves ``metodo``, ``h``, ``error_inf`` y
        ``tiempo_s``.
    ruta_salida : str or pathlib.Path, optional
        Ruta donde guardar la figura. Si es ``None`` no se guarda.
    titulo : str, optional
        Título del gráfico. Si es ``None`` se construye un título por defecto.
    metodo_referencia : str, optional
        Nombre del método usado para la solución de referencia. Default
        ``'Crank-Nicolson'``.
    h_referencia : float, optional
        Paso temporal de la referencia. Si se proporciona, se incluye en el
        ``caption`` por defecto.
    caption : str, optional
        Texto descriptivo que se muestra debajo del gráfico (referencia,
        parámetros del modelo, etc.). Si es ``None`` se construye uno por
        defecto a partir de ``metodo_referencia`` y ``h_referencia``.

    Returns
    -------
    matplotlib.figure.Figure
        Figura con el gráfico log-log de error versus paso.
    """
    fig, ejes = plt.subplots()
    fig.subplots_adjust(bottom=0.22)
    metodos = sorted({resultado["metodo"] for resultado in resultados})

    errores_finitos = [
        r["error_inf"]
        for r in resultados
        if np.isfinite(r["error_inf"]) and r["error_inf"] > 0
    ]
    if errores_finitos:
        max_error = max(errores_finitos)
        min_error = min(errores_finitos)
        ylim_max = max(10 * max_error, 1e2)
        ylim_min = min(min_error / 10, 1e-8)
    else:
        ylim_max = 1e2
        ylim_min = 1e-8
    ejes.set_ylim(bottom=ylim_min, top=ylim_max)

    for metodo in metodos:
        datos = [r for r in resultados if r["metodo"] == metodo]
        pasos = np.array([r["h"] for r in datos])
        errores = np.array([r["error_inf"] for r in datos])

        finitos = np.isfinite(errores) & (errores > 0)
        divergentes = ~np.isfinite(errores)

        if np.any(finitos):
            ejes.loglog(
                pasos[finitos], errores[finitos], marker="o", label=metodo
            )
        if np.any(divergentes):
            ejes.plot(
                pasos[divergentes],
                np.full(np.count_nonzero(divergentes), ylim_max),
                marker="^",
                linestyle="None",
                color="red",
                label=f"{metodo} (divergencia)",
            )

    if titulo is None:
        titulo = "Error vs paso temporal"

    if caption is None and h_referencia is not None:
        caption = (
            f"Referencia: {metodo_referencia} con "
            f"h={_formatear_h(h_referencia)}"
        )

    ejes.set_xlabel("Paso temporal $h$ (escala logarítmica)")
    ejes.set_ylabel(r"Error $\|\cdot\|_\infty$ (escala logarítmica)")
    ejes.set_title(titulo)
    ejes.legend()
    ejes.grid(True, which="both", linestyle="--")

    if caption is not None:
        fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9)

    if ruta_salida is not None:
        fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")

    return fig


def graficar_referencia_vs_aproximada(
    t_ref,
    x_ref,
    t_aprox,
    x_aprox,
    titulo,
    ruta_salida=None,
    nombres_componentes=None,
    descripcion_referencia="Referencia",
    descripcion_aproximacion="Aproximación",
    caption=None,
):
    """Superpone trayectorias de referencia y aproximadas.

    Parameters
    ----------
    t_ref : np.ndarray
        Tiempos de la referencia.
    x_ref : np.ndarray
        Estados de la referencia, shape (N, n).
    t_aprox : np.ndarray
        Tiempos de la aproximación.
    x_aprox : np.ndarray
        Estados de la aproximación, shape (M, n).
    titulo : str
        Título del gráfico.
    ruta_salida : str or pathlib.Path, optional
        Ruta donde guardar la figura. Si es ``None`` no se guarda.
    nombres_componentes : list[str], optional
        Nombre de cada componente del estado. Si es ``None`` se usan
        ``x_0``, ``x_1``, etc.
    descripcion_referencia : str, optional
        Prefijo para las curvas de referencia en la leyenda. Default
        ``'Referencia'``.
    descripcion_aproximacion : str, optional
        Prefijo para las curvas aproximadas en la leyenda. Default
        ``'Aproximación'``.
    caption : str, optional
        Texto descriptivo que se muestra debajo del gráfico (parámetros del
        modelo, métodos usados, etc.).

    Returns
    -------
    matplotlib.figure.Figure
        Figura con la comparación de trayectorias.
    """
    fig, ejes = plt.subplots()
    fig.subplots_adjust(bottom=0.18)
    num_componentes = x_ref.shape[1]
    nombres = nombres_componentes or [f"$x_{i}$" for i in range(num_componentes)]

    for i in range(num_componentes):
        nombre = nombres[i]
        ejes.plot(
            t_ref, x_ref[:, i], label=f"{descripcion_referencia} {nombre}"
        )
        ejes.plot(
            t_aprox,
            x_aprox[:, i],
            "--",
            label=f"{descripcion_aproximacion} {nombre}",
        )

    ejes.set_xlabel("Tiempo $t$")
    ejes.set_ylabel("Estado $x(t)$")
    ejes.set_title(titulo)
    ejes.legend()
    ejes.grid(True)

    if caption is not None:
        fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9)

    if ruta_salida is not None:
        fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")

    return fig


def graficar_diagrama_fase(
    x, titulo, ruta_salida=None, caption=None
):
    """Dibuja el diagrama de fase de una trayectoria bidimensional.

    Parameters
    ----------
    x : np.ndarray
        Estados de la trayectoria, shape (N, 2).
    titulo : str
        Título del gráfico.
    ruta_salida : str or pathlib.Path, optional
        Ruta donde guardar la figura. Si es ``None`` no se guarda.
    caption : str, optional
        Texto descriptivo que se muestra debajo del gráfico (parámetros del
        modelo, método y paso usados, etc.).

    Returns
    -------
    matplotlib.figure.Figure
        Figura con el diagrama de fase.
    """
    fig, ejes = plt.subplots()
    fig.subplots_adjust(bottom=0.18)
    ejes.plot(x[:, 0], x[:, 1])
    ejes.set_xlabel("$x_0$")
    ejes.set_ylabel("$x_1$")
    ejes.set_title(titulo)
    ejes.grid(True)

    if caption is not None:
        fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9)

    if ruta_salida is not None:
        fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")

    return fig
