"""Tests de validación para el Problema 1b de la Tarea 1.

Verifican los campos vectoriales de Lotka-Volterra y Van der Pol,
la generación de referencias, el cálculo de errores, la medición de
tiempo y los helpers de visualización.
"""

import time
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.integradores import EDOSolution, EDOSolver
from src.validacion_problema1 import (
    calcular_error_inf,
    campo_lotka_volterra,
    campo_van_der_pol,
    crear_referencia,
    ejecutar_experimento,
    medir_tiempo,
)
from utils.visualizacion import (
    _formatear_h,
    graficar_diagrama_fase,
    graficar_error_vs_h,
    graficar_referencia_vs_aproximada,
    tabla_resultados,
)


class TestCampoLotkaVolterra:
    """Suite para el campo vectorial de Lotka-Volterra."""

    def test_derivada_conocida(self):
        """El campo con parámetros conocidos produce la derivada esperada."""
        parametros = {
            "alpha": 1.0,
            "beta": 0.1,
            "delta": 0.075,
            "gamma": 1.5,
        }
        x = np.array([10.0, 5.0])
        derivada = campo_lotka_volterra(0.0, x, None, parametros)

        esperado = np.array(
            [
                parametros["alpha"] * x[0]
                - parametros["beta"] * x[0] * x[1],
                parametros["delta"] * x[0] * x[1]
                - parametros["gamma"] * x[1],
            ]
        )
        assert np.allclose(derivada, esperado)

    def test_control_callable_se_propaga(self):
        """Si el control es callable, se evalúa en t y se aplica al campo."""
        parametros = {"alpha": 1.0, "beta": 0.0, "delta": 0.0, "gamma": 1.0}
        x = np.array([1.0, 1.0])
        control = lambda t: np.array([0.5, -0.5])

        derivada = campo_lotka_volterra(2.0, x, control, parametros)

        esperado = np.array([1.5, -1.5])
        assert np.allclose(derivada, esperado)

    def test_control_array_se_propaga(self):
        """Si el control es un arreglo, se suma directamente al campo."""
        parametros = {"alpha": 1.0, "beta": 0.0, "delta": 0.0, "gamma": 1.0}
        x = np.array([1.0, 1.0])
        control = np.array([0.5, -0.5])

        derivada = campo_lotka_volterra(0.0, x, control, parametros)

        esperado = np.array([1.5, -1.5])
        assert np.allclose(derivada, esperado)


class TestCampoVanDerPol:
    """Suite para el campo vectorial de Van der Pol."""

    def test_mu_cero_es_oscilador_armonico(self):
        """Con mu=0 el campo se reduce al oscilador armónico."""
        parametros = {"mu": 0.0}
        x = np.array([1.0, 0.0])
        derivada = campo_van_der_pol(0.0, x, None, parametros)

        esperado = np.array([0.0, -x[0]])
        assert np.allclose(derivada, esperado)

    def test_termino_no_lineal_con_mu_positivo(self):
        """Con mu>0 aparece el término de amortiguamiento de Van der Pol."""
        parametros = {"mu": 2.0}
        x = np.array([1.0, 1.0])
        derivada = campo_van_der_pol(0.0, x, None, parametros)

        esperado_x1 = x[1]
        esperado_x2 = parametros["mu"] * (1.0 - x[0] ** 2) * x[1] - x[0]
        assert np.allclose(derivada, np.array([esperado_x1, esperado_x2]))

    def test_control_array_se_aplica(self):
        """Si el control es un arreglo, se suma a la derivada de Van der Pol."""
        parametros = {"mu": 0.0}
        x = np.array([1.0, 0.0])
        control = np.array([0.0, 1.0])

        derivada = campo_van_der_pol(0.0, x, control, parametros)

        esperado = np.array([0.0, 0.0])
        assert np.allclose(derivada, esperado)


class TestCrearReferencia:
    """Suite para la generación de trayectorias de referencia."""

    def test_tiempos_crecientes_y_estados_consistentes(self):
        """La referencia devuelve tiempos crecientes y estados coherentes."""
        campo = lambda t, x, u, p: -np.asarray(x)
        x0 = np.array([1.0])
        t0, tf = 0.0, 1.0
        parametros = {}

        t_ref, x_ref = crear_referencia(
            campo, x0, t0, tf, parametros, h=0.1, metodo="rk4"
        )

        assert isinstance(t_ref, np.ndarray)
        assert isinstance(x_ref, EDOSolution)
        assert len(t_ref) == len(x_ref.estados)
        assert np.all(np.diff(t_ref) > 0)
        assert t_ref[0] == pytest.approx(t0)
        assert t_ref[-1] == pytest.approx(tf)
        assert x_ref.estados[0, 0] == pytest.approx(x0[0])

    @pytest.mark.filterwarnings(
        "ignore::RuntimeWarning"
    )
    def test_metodo_por_defecto_crank_nicolson(self):
        """Sin especificar método, se usa crank_nicolson."""
        campo = lambda t, x, u, p: -np.asarray(x)
        x0 = np.array([1.0])
        parametros = {}

        t_ref, x_ref = crear_referencia(
            campo, x0, 0.0, 0.5, parametros, h=0.1
        )

        assert t_ref[-1] == pytest.approx(0.5)
        assert len(t_ref) == len(x_ref.estados)

    def test_referencia_usa_argumentos_fsolve_robustos(self):
        """La referencia pasa argumentos fsolve con tolerancia y maxfev altos."""
        campo = lambda t, x, u, p: -np.asarray(x)
        x0 = np.array([1.0])
        parametros = {}

        with patch.object(EDOSolver, "solve") as mock_solve:
            mock_solve.return_value = EDOSolution(
                tiempos=np.array([0.0, 0.5]),
                estados=np.array([[1.0], [0.5]]),
                t_span=(0.0, 0.5),
            )
            crear_referencia(campo, x0, 0.0, 0.5, parametros)

        _, kwargs = mock_solve.call_args
        assert kwargs["argumentos_fsolve"] == {
            "xtol": 1e-12,
            "maxfev": 300,
        }

    def test_argumentos_fsolve_personalizados(self):
        """Se pueden sobrescribir los argumentos de fsolve de la referencia."""
        campo = lambda t, x, u, p: -np.asarray(x)
        x0 = np.array([1.0])
        parametros = {}
        argumentos = {"xtol": 1e-10, "maxfev": 200}

        with patch.object(EDOSolver, "solve") as mock_solve:
            mock_solve.return_value = EDOSolution(
                tiempos=np.array([0.0, 0.5]),
                estados=np.array([[1.0], [0.5]]),
                t_span=(0.0, 0.5),
            )
            crear_referencia(
                campo, x0, 0.0, 0.5, parametros, argumentos_fsolve=argumentos
            )

        _, kwargs = mock_solve.call_args
        assert kwargs["argumentos_fsolve"] == argumentos


class TestCalcularErrorInf:
    """Suite para el cálculo del error en norma infinito."""

    def test_error_cero_cuando_coinciden(self):
        """El error es cero cuando la aproximación coincide con la referencia."""
        tiempos = np.linspace(0.0, 1.0, 11)
        estados = np.column_stack([np.exp(-tiempos)])
        solucion = EDOSolution(
            tiempos=tiempos, estados=estados, t_span=(0.0, 1.0)
        )

        errores = calcular_error_inf(solucion, solucion)

        assert errores["global"] == pytest.approx(0.0, abs=1e-12)
        assert errores["componentes"][0] == pytest.approx(0.0, abs=1e-12)

    def test_error_positivo_cuando_difieren(self):
        """El error es positivo cuando la aproximación difiere de la referencia."""
        tiempos = np.linspace(0.0, 1.0, 11)
        estados_ref = np.column_stack([np.exp(-tiempos)])
        estados_aprox = estados_ref + 0.1
        solucion_ref = EDOSolution(
            tiempos=tiempos, estados=estados_ref, t_span=(0.0, 1.0)
        )
        solucion_aprox = EDOSolution(
            tiempos=tiempos, estados=estados_aprox, t_span=(0.0, 1.0)
        )

        errores = calcular_error_inf(solucion_aprox, solucion_ref)

        assert errores["global"] == pytest.approx(0.1)
        assert errores["componentes"][0] == pytest.approx(0.1)

    def test_interpolacion_a_tiempos_aproximados(self):
        """La referencia se interpola a los tiempos de la aproximación."""
        tiempos_ref = np.linspace(0.0, 1.0, 101)
        estados_ref = np.column_stack([np.sin(tiempos_ref)])
        solucion_ref = EDOSolution(
            tiempos=tiempos_ref, estados=estados_ref, t_span=(0.0, 1.0)
        )

        tiempos_aprox = np.array([0.25, 0.75])
        estados_aprox = np.column_stack([np.sin(tiempos_aprox)])
        solucion_aprox = EDOSolution(
            tiempos=tiempos_aprox, estados=estados_aprox, t_span=(0.0, 1.0)
        )

        errores = calcular_error_inf(solucion_aprox, solucion_ref)

        assert errores["global"] == pytest.approx(0.0, abs=1e-12)


class TestMedirTiempo:
    """Suite para la medición de tiempos de integración."""

    def test_devuelve_float_positivo(self):
        """La medición devuelve un número positivo."""
        solver = EDOSolver()
        campo = lambda t, x, u, p: -np.asarray(x)
        x0 = np.array([1.0])
        t_span = (0.0, 0.1)
        parametros = {}

        tiempo = medir_tiempo(
            solver, campo, x0, t_span, parametros, "rk4", 0.01
        )

        assert isinstance(tiempo, float)
        assert tiempo > 0.0

    def test_perf_counter_es_invocado(self):
        """La función usa time.perf_counter para medir."""
        solver = EDOSolver()
        campo = lambda t, x, u, p: -np.asarray(x)
        x0 = np.array([1.0])
        t_span = (0.0, 0.1)
        parametros = {}

        with patch("time.perf_counter", side_effect=[1.0, 2.0]):
            tiempo = medir_tiempo(
                solver, campo, x0, t_span, parametros, "rk4", 0.01
            )

        assert tiempo == pytest.approx(1.0)


class TestEjecutarExperimento:
    """Suite para la ejecución completa de un experimento."""

    @pytest.mark.filterwarnings(
        "ignore::RuntimeWarning"
    )
    def test_devuelve_resultados_para_cada_metodo_y_paso(self):
        """El experimento produce una entrada por cada método y paso."""
        campo = lambda t, x, u, p: -np.asarray(x)
        x0 = np.array([1.0])
        parametros = {}
        hs = [0.1, 0.05]
        metodos = ["euler_progresivo", "rk4"]

        resultados = ejecutar_experimento(
            campo, x0, 0.0, 0.5, parametros, hs, metodos
        )

        assert len(resultados) == len(hs) * len(metodos)
        for resultado in resultados:
            assert "metodo" in resultado
            assert "h" in resultado
            assert "error_inf" in resultado
            assert "tiempo_s" in resultado
            assert resultado["metodo"] in metodos
            assert resultado["h"] in hs

    def test_h_referencia_personalizado(self):
        """Se puede usar un paso de referencia diferente al default."""
        campo = lambda t, x, u, p: -np.asarray(x)
        x0 = np.array([1.0])
        parametros = {}
        hs = [0.1]
        metodos = ["rk4"]

        resultados = ejecutar_experimento(
            campo, x0, 0.0, 0.5, parametros, hs, metodos, h_referencia=0.01
        )

        assert len(resultados) == 1
        assert resultados[0]["error_inf"] >= 0.0

    def test_argumentos_fsolve_se_propagan_a_referencia(self):
        """``argumentos_fsolve`` se reenvía a ``crear_referencia``."""
        campo = lambda t, x, u, p: -np.asarray(x)
        x0 = np.array([1.0])
        parametros = {}
        hs = [0.1]
        metodos = ["rk4"]
        argumentos = {"xtol": 1e-10, "maxfev": 200}

        with patch(
            "src.validacion_problema1.crear_referencia"
        ) as mock_referencia:
            tiempos_ref = np.linspace(0.0, 0.5, 11)
            mock_referencia.return_value = (
                tiempos_ref,
                EDOSolution(
                    tiempos=tiempos_ref,
                    estados=np.column_stack([np.exp(-tiempos_ref)]),
                    t_span=(0.0, 0.5),
                ),
            )
            ejecutar_experimento(
                campo,
                x0,
                0.0,
                0.5,
                parametros,
                hs,
                metodos,
                argumentos_fsolve=argumentos,
            )

        _, kwargs = mock_referencia.call_args
        assert kwargs["argumentos_fsolve"] == argumentos

    def test_reutiliza_solucion_referencia_proporcionada(self):
        """Si se pasa una referencia, no se vuelve a calcular."""
        campo = lambda t, x, u, p: -np.asarray(x)
        x0 = np.array([1.0])
        parametros = {}
        hs = [0.1]
        metodos = ["rk4"]
        tiempos_ref = np.linspace(0.0, 0.5, 11)
        solucion_ref = EDOSolution(
            tiempos=tiempos_ref,
            estados=np.column_stack([np.exp(-tiempos_ref)]),
            t_span=(0.0, 0.5),
        )

        with patch(
            "src.validacion_problema1.crear_referencia"
        ) as mock_referencia:
            ejecutar_experimento(
                campo,
                x0,
                0.0,
                0.5,
                parametros,
                hs,
                metodos,
                solucion_ref=solucion_ref,
            )

        mock_referencia.assert_not_called()


class TestHelpersVisualizacion:
    """Suite para los helpers de visualización."""

    def test_graficar_error_vs_h_devuelve_figura(self):
        """La función devuelve una figura de matplotlib sin lanzar."""
        resultados = [
            {"metodo": "rk4", "h": 0.1, "error_inf": 1e-2, "tiempo_s": 0.01},
            {"metodo": "rk4", "h": 0.05, "error_inf": 1e-3, "tiempo_s": 0.02},
        ]

        fig = graficar_error_vs_h(resultados)

        assert fig is not None
        fig.clf()

    def test_graficar_error_vs_h_guarda_archivo(self, tmp_path):
        """Si se proporciona ruta, la figura se guarda en disco."""
        resultados = [
            {"metodo": "rk4", "h": 0.1, "error_inf": 1e-2, "tiempo_s": 0.01},
        ]
        ruta = tmp_path / "error.png"

        fig = graficar_error_vs_h(resultados, ruta_salida=str(ruta))

        assert ruta.exists()
        fig.clf()

    def test_ejes_indican_escala_logaritmica(self):
        """Los ejes x e y indican explícitamente la escala logarítmica."""
        resultados = [
            {"metodo": "rk4", "h": 0.1, "error_inf": 1e-2, "tiempo_s": 0.01},
        ]

        fig = graficar_error_vs_h(resultados)
        ejes = fig.axes[0]

        assert "logarítmica" in ejes.get_xlabel()
        assert "logarítmica" in ejes.get_ylabel()
        fig.clf()

    def test_caption_incluye_referencia_cuando_se_proporciona(self):
        """El caption incluye método y paso de referencia cuando se indican."""
        resultados = [
            {"metodo": "rk4", "h": 0.1, "error_inf": 1e-2, "tiempo_s": 0.01},
        ]

        fig = graficar_error_vs_h(
            resultados,
            metodo_referencia="Crank-Nicolson",
            h_referencia=1e-4,
        )
        texto_caption = fig.texts[0].get_text()

        assert "Crank-Nicolson" in texto_caption
        assert "1e-4" in texto_caption
        fig.clf()

    def test_titulo_personalizado_anula_default(self):
        """Un título explícito anula el título por defecto."""
        resultados = [
            {"metodo": "rk4", "h": 0.1, "error_inf": 1e-2, "tiempo_s": 0.01},
        ]
        titulo_esperado = "Error personalizado"

        fig = graficar_error_vs_h(
            resultados,
            titulo=titulo_esperado,
            metodo_referencia="Crank-Nicolson",
            h_referencia=1e-4,
        )
        titulo = fig.axes[0].get_title()

        assert titulo == titulo_esperado
        fig.clf()

    def test_nan_inf_se_marcan_como_divergencia(self):
        """Los errores NaN e inf se grafican con marcador de divergencia."""
        resultados = [
            {"metodo": "euler", "h": 0.1, "error_inf": np.nan, "tiempo_s": 0.01},
            {"metodo": "euler", "h": 0.05, "error_inf": np.inf, "tiempo_s": 0.01},
            {"metodo": "rk4", "h": 0.1, "error_inf": np.nan, "tiempo_s": 0.01},
        ]

        fig = graficar_error_vs_h(resultados)
        ejes = fig.axes[0]

        # Buscar líneas cuyo marcador sea triángulo hacia arriba (^).
        marcadores_divergencia = [
            line.get_marker() for line in ejes.lines
        ]
        assert "^" in marcadores_divergencia
        fig.clf()

    def test_ylim_superior_se_amplia_para_divergencias(self):
        """El límite superior del eje y se expande para mostrar divergencias."""
        resultados = [
            {"metodo": "rk4", "h": 0.1, "error_inf": 1e-2, "tiempo_s": 0.01},
            {"metodo": "rk4", "h": 0.05, "error_inf": np.nan, "tiempo_s": 0.01},
        ]

        fig = graficar_error_vs_h(resultados)
        ejes = fig.axes[0]
        _, ylim_max = ejes.get_ylim()

        assert ylim_max > 1e-2
        fig.clf()

    def test_puntos_finitos_siguen_ploteados_con_nan(self):
        """Los puntos finitos coexisten con los marcadores de divergencia."""
        resultados = [
            {"metodo": "rk4", "h": 0.1, "error_inf": 1e-2, "tiempo_s": 0.01},
            {"metodo": "rk4", "h": 0.05, "error_inf": 1e-3, "tiempo_s": 0.02},
            {"metodo": "rk4", "h": 0.025, "error_inf": np.nan, "tiempo_s": 0.04},
        ]

        fig = graficar_error_vs_h(resultados)
        ejes = fig.axes[0]
        marcadores = [line.get_marker() for line in ejes.lines]

        assert "o" in marcadores
        assert "^" in marcadores
        fig.clf()

    def test_leyenda_indica_divergencia(self):
        """La leyenda incluye una entrada que indica divergencia."""
        resultados = [
            {"metodo": "rk4", "h": 0.1, "error_inf": np.nan, "tiempo_s": 0.01},
        ]

        fig = graficar_error_vs_h(resultados)
        ejes = fig.axes[0]
        _, etiquetas = ejes.get_legend_handles_labels()

        assert any("divergencia" in etiqueta for etiqueta in etiquetas)
        fig.clf()

    def test_formatear_h_notacion_cientifica_compacta(self):
        """El helper formatea pasos pequeños en notación científica legible."""
        assert _formatear_h(1e-4) == "1e-4"
        assert _formatear_h(1e-5) == "1e-5"
        assert _formatear_h(2.5e-6) == "3e-6"

    def test_graficar_referencia_vs_aproximada_devuelve_figura(self):
        """La función superpone referencia y aproximación."""
        tiempos = np.linspace(0.0, 1.0, 11)
        estados = np.column_stack([np.exp(-tiempos)])

        fig = graficar_referencia_vs_aproximada(
            tiempos, estados, tiempos, estados, "Título"
        )

        assert fig is not None
        fig.clf()

    def test_graficar_referencia_vs_aproximada_guarda_archivo(self, tmp_path):
        """Si se proporciona ruta, la figura comparativa se guarda en disco."""
        tiempos = np.linspace(0.0, 1.0, 11)
        estados = np.column_stack([np.exp(-tiempos)])
        ruta = tmp_path / "comparacion.png"

        fig = graficar_referencia_vs_aproximada(
            tiempos, estados, tiempos, estados, "Título", ruta_salida=str(ruta)
        )

        assert ruta.exists()
        fig.clf()

    def test_leyendas_personalizadas_aparecen_en_figura(self):
        """Las descripciones personalizadas se reflejan en las leyendas."""
        tiempos = np.linspace(0.0, 1.0, 11)
        estados = np.column_stack([np.exp(-tiempos), np.sin(tiempos)])

        fig = graficar_referencia_vs_aproximada(
            tiempos,
            estados,
            tiempos,
            estados,
            "Título",
            nombres_componentes=["presas", "depredadores"],
            descripcion_referencia="Referencia (CN h=1e-4)",
            descripcion_aproximacion="Aproximación (RK4 h=0.1)",
        )
        _, etiquetas = fig.axes[0].get_legend_handles_labels()

        assert any("presas" in etiqueta for etiqueta in etiquetas)
        assert any("depredadores" in etiqueta for etiqueta in etiquetas)
        assert any("CN h=1e-4" in etiqueta for etiqueta in etiquetas)
        assert any("RK4 h=0.1" in etiqueta for etiqueta in etiquetas)
        fig.clf()

    def test_leyendas_default_usan_nombres_x_i(self):
        """Sin nombres personalizados se usan x_0, x_1, etc. como etiquetas."""
        tiempos = np.linspace(0.0, 1.0, 11)
        estados = np.column_stack([np.exp(-tiempos), np.sin(tiempos)])

        fig = graficar_referencia_vs_aproximada(
            tiempos, estados, tiempos, estados, "Título"
        )
        _, etiquetas = fig.axes[0].get_legend_handles_labels()

        assert any(r"$x_0$" in etiqueta for etiqueta in etiquetas)
        assert any(r"$x_1$" in etiqueta for etiqueta in etiquetas)
        fig.clf()

    def test_graficar_diagrama_fase_devuelve_figura(self):
        """La función dibuja el diagrama de fase."""
        estados = np.column_stack([np.sin(np.linspace(0.0, 2 * np.pi, 50))])
        estados = np.hstack([estados, np.cos(np.linspace(0.0, 2 * np.pi, 50)).reshape(-1, 1)])

        fig = graficar_diagrama_fase(estados, "Diagrama de fase")

        assert fig is not None
        fig.clf()

    def test_graficar_diagrama_fase_guarda_archivo(self, tmp_path):
        """Si se proporciona ruta, el diagrama de fase se guarda en disco."""
        estados = np.column_stack([np.sin(np.linspace(0.0, 2 * np.pi, 50))])
        estados = np.hstack([estados, np.cos(np.linspace(0.0, 2 * np.pi, 50)).reshape(-1, 1)])
        ruta = tmp_path / "fase.png"

        fig = graficar_diagrama_fase(estados, "Diagrama de fase", ruta_salida=str(ruta))

        assert ruta.exists()
        fig.clf()

    def test_caption_personalizado_aparece_en_error_vs_h(self):
        """Un caption explícito se muestra debajo del gráfico de error."""
        resultados = [
            {"metodo": "rk4", "h": 0.1, "error_inf": 1e-2, "tiempo_s": 0.01},
        ]
        caption_esperado = "Parámetros: μ=0.1, x₀=[2, 0]"

        fig = graficar_error_vs_h(
            resultados, caption=caption_esperado, h_referencia=1e-4
        )
        texto_caption = fig.texts[0].get_text()

        assert caption_esperado in texto_caption
        fig.clf()

    def test_caption_aparece_en_diagrama_fase(self):
        """Un caption explícito se muestra debajo del diagrama de fase."""
        estados = np.column_stack(
            [np.sin(np.linspace(0.0, 2 * np.pi, 50))]
        )
        estados = np.hstack(
            [estados, np.cos(np.linspace(0.0, 2 * np.pi, 50)).reshape(-1, 1)]
        )
        caption_esperado = "Calculado con Crank-Nicolson h=1e-4"

        fig = graficar_diagrama_fase(
            estados, "Diagrama de fase", caption=caption_esperado
        )
        texto_caption = fig.texts[0].get_text()

        assert caption_esperado in texto_caption
        fig.clf()

    def test_caption_aparece_en_referencia_vs_aproximada(self):
        """Un caption explícito se muestra debajo de la comparación."""
        tiempos = np.linspace(0.0, 1.0, 11)
        estados = np.column_stack([np.exp(-tiempos)])
        caption_esperado = "Referencia: CN h=1e-4. Aproximación: RK4 h=0.1"

        fig = graficar_referencia_vs_aproximada(
            tiempos,
            estados,
            tiempos,
            estados,
            "Título",
            caption=caption_esperado,
        )
        texto_caption = fig.texts[0].get_text()

        assert caption_esperado in texto_caption
        fig.clf()

    def test_tabla_resultados_devuelve_dataframe(self):
        """La función devuelve un DataFrame con las columnas esperadas."""
        resultados = [
            {"metodo": "rk4", "h": 0.1, "error_inf": 1e-2, "tiempo_s": 0.01},
            {"metodo": "heun", "h": 0.1, "error_inf": 1e-1, "tiempo_s": 0.005},
        ]

        tabla = tabla_resultados(resultados)

        assert isinstance(tabla, pd.DataFrame)
        assert list(tabla.columns) == [
            "metodo", "h", "estado", "error_inf", "tiempo_s"
        ]
        assert len(tabla) == 2
