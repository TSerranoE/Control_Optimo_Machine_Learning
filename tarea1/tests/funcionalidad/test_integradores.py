"""Tests de funcionalidad para el integrador de EDOs.

Verifican la API pública de ``EDOSolver`` y el contenedor ``EDOSolution``,
junto con las reglas de validación y el manejo de pasos temporales.
"""

import warnings

import numpy as np
import pytest

from integradores import EDOSolver, EDOSolution


class TestValidacion:
    """Suite para las reglas de validación de entradas."""

    @pytest.mark.parametrize(
        "condicion_inicial, intervalo, paso, metodo, mensaje_esperado, error_esperado",
        [
            (
                "foo",
                (0.0, 1.0),
                0.1,
                "euler_progresivo",
                "x0 debe ser un ndarray numérico 1D",
                ValueError,
            ),
            (
                np.array([]),
                (0.0, 1.0),
                0.1,
                "euler_progresivo",
                "x0 debe ser un ndarray numérico 1D",
                ValueError,
            ),
            (
                np.array([1.0]),
                (1.0, 0.0),
                0.1,
                "euler_progresivo",
                "t_span debe tener >= 2 puntos ordenados",
                ValueError,
            ),
            (
                np.array([1.0]),
                (0.0,),
                0.1,
                "euler_progresivo",
                "t_span debe tener >= 2 puntos ordenados",
                ValueError,
            ),
            (
                np.array([1.0]),
                (0.0, 1.0),
                0.0,
                "euler_progresivo",
                "h debe ser positivo y consistente con t_span",
                ValueError,
            ),
            (
                np.array([1.0]),
                (0.0, 1.0),
                -0.1,
                "euler_progresivo",
                "h debe ser positivo y consistente con t_span",
                ValueError,
            ),
            (
                np.array([1.0]),
                (0.0, 0.3, 0.7, 1.0),
                np.array([0.1]),
                "euler_progresivo",
                "h debe ser positivo y consistente con t_span",
                ValueError,
            ),
        ],
    )
    def test_validacion_parametrizada(
        self,
        campo_lineal,
        condicion_inicial,
        intervalo,
        paso,
        metodo,
        mensaje_esperado,
        error_esperado,
    ):
        """Las entradas inválidas producen errores descriptivos."""
        resolutor = EDOSolver()
        with pytest.raises(error_esperado, match=mensaje_esperado):
            resolutor.solve(
                campo_lineal,
                condicion_inicial,
                intervalo,
                paso,
                method=metodo,
            )

    def test_argumentos_fsolve_no_dict(self, campo_lineal):
        """argumentos_fsolve debe ser un diccionario o None."""
        resolutor = EDOSolver()
        with pytest.raises(TypeError, match="debe ser un diccionario"):
            resolutor.solve(
                campo_lineal,
                np.array([1.0]),
                (0.0, 1.0),
                0.1,
                method="euler_progresivo",
                argumentos_fsolve="no_es_dict",
            )


class TestGrillaTemporal:
    """Suite para el manejo del paso temporal escalar y arreglo."""

    def test_h_escalar(self, campo_lineal):
        """Un paso escalar genera pasos uniformes a lo largo del intervalo."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.1

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal, condicion_inicial, intervalo, paso, method="euler_progresivo"
        )

        diferencias = np.diff(solucion.tiempos)
        assert np.allclose(diferencias, paso)
        assert solucion.tiempos[-1] == pytest.approx(intervalo[-1])

    def test_h_arreglo(self, campo_lineal):
        """Un arreglo de pasos respeta los subintervalos definidos por t_span."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 0.3, 0.7, 1.0)
        pasos = np.array([0.1, 0.1, 0.1])

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal, condicion_inicial, intervalo, pasos, method="euler_progresivo"
        )

        assert solucion.tiempos[0] == pytest.approx(intervalo[0])
        assert solucion.tiempos[-1] == pytest.approx(intervalo[-1])
        assert np.isclose(solucion.tiempos[3], intervalo[1])
        assert np.isclose(solucion.tiempos[7], intervalo[2])

    def test_h_negativo_error(self, campo_lineal):
        """Un paso negativo debe ser rechazado antes de integrar."""
        condicion_inicial = np.array([1.0])

        resolutor = EDOSolver()
        with pytest.raises(ValueError, match="h debe ser positivo"):
            resolutor.solve(
                campo_lineal,
                condicion_inicial,
                (0.0, 0.3, 0.7, 1.0),
                np.array([0.1, -0.05, 0.1]),
                method="euler_progresivo",
            )

    def test_advertencia_redondeo_paso(self, campo_lineal):
        """Si el paso no divide exactamente el intervalo se advierte al usuario."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.3

        resolutor = EDOSolver()
        with pytest.warns(UserWarning, match="paso efectivo"):
            solucion = resolutor.solve(
                campo_lineal,
                condicion_inicial,
                intervalo,
                paso,
                method="euler_progresivo",
            )

        assert solucion.tiempos[-1] == pytest.approx(intervalo[-1])

    def test_paso_mayor_que_intervalo(self, campo_lineal):
        """Un paso mayor que el intervalo fuerza al menos un paso efectivo."""
        import warnings

        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 3.0

        resolutor = EDOSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solucion = resolutor.solve(
                campo_lineal,
                condicion_inicial,
                intervalo,
                paso,
                method="euler_progresivo",
            )

        assert len(solucion.tiempos) == 2
        assert solucion.tiempos[-1] == pytest.approx(intervalo[-1])


class TestEulerProgresivo:
    """Suite para el método de Euler progresivo."""

    def test_euler_progresivo_solucion_analitica(self, campo_lineal):
        """Euler aproxima correctamente la exponencial decreciente."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.01

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal, condicion_inicial, intervalo, paso, method="euler_progresivo"
        )

        valor_final_esperado = np.exp(-1.0)
        assert abs(solucion.estados[-1, 0] - valor_final_esperado) < 0.05

    def test_metodo_invalido_valueerror(self, campo_lineal):
        """Un método desconocido lista los métodos disponibles."""
        condicion_inicial = np.array([1.0])

        resolutor = EDOSolver()
        with pytest.raises(ValueError, match="foo.*Disponibles"):
            resolutor.solve(
                campo_lineal, condicion_inicial, (0.0, 1.0), 0.1, method="foo"
            )


class TestControlDual:
    """Suite para el manejo dual del control u(t)."""

    @pytest.mark.parametrize(
        "metodo, tipo_control, control, comportamiento_esperado, mensaje",
        [
            (
                "rk4",
                "callable",
                lambda t: 0.0,
                "ok",
                None,
            ),
            (
                "rk4",
                "arreglo",
                np.zeros(11),
                "error",
                "El método RK4 requiere que el control u sea una función callable.",
            ),
            (
                "euler_progresivo",
                "callable",
                lambda t: 0.0,
                "warning",
                "El control callable se evaluará sobre la grilla",
            ),
            (
                "euler_progresivo",
                "arreglo",
                np.zeros(11),
                "ok",
                None,
            ),
        ],
    )
    def test_control_dual_parametrizado(
        self,
        campo_lineal,
        metodo,
        tipo_control,
        control,
        comportamiento_esperado,
        mensaje,
    ):
        """RK4 exige callable; otros métodos aceptan callable o arreglo."""
        resolutor = EDOSolver()
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.1

        if comportamiento_esperado == "error":
            with pytest.raises(ValueError, match=mensaje):
                resolutor.solve(
                    campo_lineal,
                    condicion_inicial,
                    intervalo,
                    paso,
                    method=metodo,
                    u=control,
                )
        elif comportamiento_esperado == "warning":
            with pytest.warns(UserWarning, match=mensaje):
                solucion = resolutor.solve(
                    campo_lineal,
                    condicion_inicial,
                    intervalo,
                    paso,
                    method=metodo,
                    u=control,
                )
            assert solucion.estados.shape[0] == 11
        else:
            solucion = resolutor.solve(
                campo_lineal,
                condicion_inicial,
                intervalo,
                paso,
                method=metodo,
                u=control,
            )
            assert solucion.estados.shape[0] == 11

    def test_control_arreglo_longitud_incorrecta(self, campo_lineal):
        """Un arreglo de control debe coincidir con la grilla temporal."""
        resolutor = EDOSolver()
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.1

        with pytest.raises(ValueError, match="misma longitud"):
            resolutor.solve(
                campo_lineal,
                condicion_inicial,
                intervalo,
                paso,
                method="euler_progresivo",
                u=np.zeros(5),
            )


class TestHeun:
    """Suite para el método de Heun (predictor-corrector explícito)."""

    def test_heun_solucion_analitica(self, campo_lineal):
        """Heun aproxima la exponencial decreciente con orden 2."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.01

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal,
            condicion_inicial,
            intervalo,
            paso,
            method="heun",
        )

        valor_final_esperado = np.exp(-1.0)
        assert abs(solucion.estados[-1, 0] - valor_final_esperado) < 1e-4

    def test_heun_intermedios(self, campo_lineal):
        """Heun almacena el predictor z de cada paso cuando se solicita."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 0.5)
        paso = 0.1

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal,
            condicion_inicial,
            intervalo,
            paso,
            method="heun",
            guardar_intermedios=True,
        )

        assert solucion.intermedios is not None
        assert len(solucion.intermedios) == len(solucion.tiempos) - 1
        assert "z" in solucion.intermedios[0]
        assert solucion.intermedios[0]["z"].shape == condicion_inicial.shape


class TestEulerImplicito:
    """Suite para el método de Euler implícito."""

    def test_euler_implicito_convergencia(self, campo_lineal):
        """Euler implícito converge en el problema lineal decreciente."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.01

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal,
            condicion_inicial,
            intervalo,
            paso,
            method="euler_implicito",
        )

        valor_final_esperado = np.exp(-1.0)
        assert abs(solucion.estados[-1, 0] - valor_final_esperado) < 5e-3

    def test_argumentos_fsolve_custom(self, campo_lineal):
        """Los argumentos custom de fsolve se propagan sin errores."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.1

        resolutor = EDOSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            solucion = resolutor.solve(
                campo_lineal,
                condicion_inicial,
                intervalo,
                paso,
                method="euler_implicito",
                argumentos_fsolve={"xtol": 1e-12},
            )

        valor_final_esperado = np.exp(-1.0)
        assert abs(solucion.estados[-1, 0] - valor_final_esperado) < 5e-2


class TestCrankNicolson:
    """Suite para el método de Crank-Nicolson."""

    def test_crank_nicolson_convergencia(self, campo_lineal):
        """Crank-Nicolson converge con orden 2 en el problema lineal."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.01

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal,
            condicion_inicial,
            intervalo,
            paso,
            method="crank_nicolson",
        )

        valor_final_esperado = np.exp(-1.0)
        assert abs(solucion.estados[-1, 0] - valor_final_esperado) < 1e-5


class TestRK4:
    """Suite para el método de Runge-Kutta de orden 4."""

    def test_rk4_alta_precision(self, campo_lineal):
        """RK4 alcanza tolerancia menor a 1e-10 para la exponencial decreciente."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.01

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal,
            condicion_inicial,
            intervalo,
            paso,
            method="rk4",
        )

        valor_final_esperado = np.exp(-1.0)
        assert abs(solucion.estados[-1, 0] - valor_final_esperado) < 1e-10

    def test_rk4_intermedios(self, campo_lineal):
        """RK4 almacena las cuatro etapas k_i de cada paso."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 0.5)
        paso = 0.1

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal,
            condicion_inicial,
            intervalo,
            paso,
            method="rk4",
            guardar_intermedios=True,
        )

        assert solucion.intermedios is not None
        assert len(solucion.intermedios) == len(solucion.tiempos) - 1
        for clave in ("k1", "k2", "k3", "k4"):
            assert clave in solucion.intermedios[0]
            assert solucion.intermedios[0][clave].shape == condicion_inicial.shape


class TestEDOSolution:
    """Suite para el contenedor de resultados de integración."""

    def test_edosolution_atributos(self, campo_lineal):
        """La solución expone tiempos, estados y t_span con shapes coherentes."""
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.1

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal, condicion_inicial, intervalo, paso, method="euler_progresivo"
        )

        assert isinstance(solucion, EDOSolution)
        assert solucion.tiempos[0] == pytest.approx(intervalo[0])
        assert solucion.tiempos[-1] == pytest.approx(intervalo[1])
        assert len(solucion.tiempos) == len(solucion.estados)
        assert solucion.estados.shape == (len(solucion.tiempos), len(condicion_inicial))
        assert solucion.t_span == intervalo

    def test_intermedios_desactivados(self, campo_lineal):
        """Por defecto no se almacenan valores intermedios del método."""
        condicion_inicial = np.array([1.0])

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal, condicion_inicial, (0.0, 0.5), 0.1, method="euler_progresivo"
        )

        assert solucion.intermedios is None

    def test_intermedios_activados(self, campo_lineal):
        """Al pedir intermedios, se retorna una lista con un dict por paso."""
        condicion_inicial = np.array([1.0])

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo_lineal,
            condicion_inicial,
            (0.0, 0.5),
            0.1,
            method="euler_progresivo",
            guardar_intermedios=True,
        )

        assert solucion.intermedios is not None
        assert isinstance(solucion.intermedios, list)
        assert len(solucion.intermedios) == len(solucion.tiempos) - 1
