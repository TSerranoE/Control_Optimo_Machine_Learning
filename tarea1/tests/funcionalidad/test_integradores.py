"""Tests de funcionalidad para el integrador de EDOs.

Verifican la API pública de ``EDOSolver`` y el contenedor ``EDOSolution``,
junto con las reglas de validación y el manejo de pasos temporales.
"""

import numpy as np
import pytest

from integradores import EDOSolver, EDOSolution


class TestValidacion:
    """Suite para las reglas de validación de entradas (R5)."""

    @pytest.mark.parametrize(
        "campo, condicion_inicial, intervalo, paso, metodo, mensaje_esperado",
        [
            (
                lambda t, x, u: -np.asarray(x),
                "foo",
                (0.0, 1.0),
                0.1,
                "euler",
                "x0 debe ser un ndarray numérico 1D",
            ),
            (
                lambda t, x, u: -np.asarray(x),
                np.array([1.0]),
                (1.0, 0.0),
                0.1,
                "euler",
                "t_span debe tener >= 2 puntos ordenados",
            ),
            (
                lambda t, x, u: -np.asarray(x),
                np.array([1.0]),
                (0.0,),
                0.1,
                "euler",
                "t_span debe tener >= 2 puntos ordenados",
            ),
            (
                lambda t, x, u: -np.asarray(x),
                np.array([1.0]),
                (0.0, 1.0),
                0.0,
                "euler",
                "h debe ser positivo y consistente con t_span",
            ),
            (
                lambda t, x, u: -np.asarray(x),
                np.array([1.0]),
                (0.0, 1.0),
                -0.1,
                "euler",
                "h debe ser positivo y consistente con t_span",
            ),
            (
                lambda t, x, u: -np.asarray(x),
                np.array([1.0]),
                (0.0, 0.3, 0.7, 1.0),
                np.array([0.1]),
                "euler",
                "h debe ser positivo y consistente con t_span",
            ),
        ],
    )
    def test_validacion_parametrizada(
        self, campo, condicion_inicial, intervalo, paso, metodo, mensaje_esperado
    ):
        """Las entradas inválidas producen ValueError con mensaje descriptivo."""
        resolutor = EDOSolver()
        with pytest.raises(ValueError, match=mensaje_esperado):
            resolutor.solve(
                campo, condicion_inicial, intervalo, paso, method=metodo
            )


class TestGrillaTemporal:
    """Suite para el manejo del paso temporal escalar y arreglo."""

    def test_h_escalar(self):
        """Un paso escalar genera pasos uniformes a lo largo del intervalo."""
        campo = lambda t, x, u: -np.asarray(x)
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.1

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo, condicion_inicial, intervalo, paso, method="euler"
        )

        diferencias = np.diff(solucion.tiempos)
        assert np.allclose(diferencias, paso)
        assert solucion.tiempos[-1] == pytest.approx(intervalo[-1])

    def test_h_arreglo(self):
        """Un arreglo de pasos respeta los subintervalos definidos por t_span."""
        campo = lambda t, x, u: -np.asarray(x)
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 0.3, 0.7, 1.0)
        pasos = np.array([0.1, 0.1, 0.1])

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo, condicion_inicial, intervalo, pasos, method="euler"
        )

        assert solucion.tiempos[0] == pytest.approx(intervalo[0])
        assert solucion.tiempos[-1] == pytest.approx(intervalo[-1])
        assert np.isclose(solucion.tiempos[3], intervalo[1])
        assert np.isclose(solucion.tiempos[7], intervalo[2])

    def test_h_negativo_error(self):
        """Un paso negativo debe ser rechazado antes de integrar."""
        campo = lambda t, x, u: -np.asarray(x)
        condicion_inicial = np.array([1.0])

        resolutor = EDOSolver()
        with pytest.raises(ValueError, match="h debe ser positivo"):
            resolutor.solve(
                campo,
                condicion_inicial,
                (0.0, 0.3, 0.7, 1.0),
                np.array([0.1, -0.05, 0.1]),
                method="euler",
            )


class TestEuler:
    """Suite para el método de Euler progresivo."""

    def test_euler_solucion_analitica(self):
        """Euler aproxima correctamente la exponencial decreciente."""
        campo = lambda t, x, u: -np.asarray(x)
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.01

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo, condicion_inicial, intervalo, paso, method="euler"
        )

        valor_final_esperado = np.exp(-1.0)
        assert abs(solucion.estados[-1, 0] - valor_final_esperado) < 0.05

    def test_metodo_invalido_valueerror(self):
        """Un método desconocido lista los métodos disponibles."""
        campo = lambda t, x, u: -np.asarray(x)
        condicion_inicial = np.array([1.0])

        resolutor = EDOSolver()
        with pytest.raises(ValueError, match="foo.*Disponibles"):
            resolutor.solve(
                campo, condicion_inicial, (0.0, 1.0), 0.1, method="foo"
            )


class TestEDOSolution:
    """Suite para el contenedor de resultados de integración."""

    def test_edosolution_atributos(self):
        """La solución expone tiempos, estados y t_span con shapes coherentes."""
        campo = lambda t, x, u: -np.asarray(x)
        condicion_inicial = np.array([1.0])
        intervalo = (0.0, 1.0)
        paso = 0.1

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo, condicion_inicial, intervalo, paso, method="euler"
        )

        assert isinstance(solucion, EDOSolution)
        assert solucion.tiempos[0] == pytest.approx(intervalo[0])
        assert solucion.tiempos[-1] == pytest.approx(intervalo[1])
        assert len(solucion.tiempos) == len(solucion.estados)
        assert solucion.estados.shape == (len(solucion.tiempos), len(condicion_inicial))
        assert solucion.t_span == intervalo

    def test_intermedios_desactivados(self):
        """Por defecto no se almacenan valores intermedios del método."""
        campo = lambda t, x, u: -np.asarray(x)
        condicion_inicial = np.array([1.0])

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo, condicion_inicial, (0.0, 0.5), 0.1, method="euler"
        )

        assert solucion.intermedios is None

    def test_intermedios_activados(self):
        """Al pedir intermedios, se retorna una lista con un dict por paso."""
        campo = lambda t, x, u: -np.asarray(x)
        condicion_inicial = np.array([1.0])

        resolutor = EDOSolver()
        solucion = resolutor.solve(
            campo,
            condicion_inicial,
            (0.0, 0.5),
            0.1,
            method="euler",
            guardar_intermedios=True,
        )

        assert solucion.intermedios is not None
        assert isinstance(solucion.intermedios, list)
        assert len(solucion.intermedios) == len(solucion.tiempos) - 1
