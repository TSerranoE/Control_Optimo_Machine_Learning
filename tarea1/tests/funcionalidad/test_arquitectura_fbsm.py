"""Contratos de arquitectura para la interfaz publica usada por FBSM."""

import inspect

import numpy as np

import metodos_optimizacion
from problemas_control import ControlProblem


def test_interfaz_publica_protege_estado_inicial(simple_control_problem):
    estado = simple_control_problem.estado_inicial
    estado[0] = 99.0

    np.testing.assert_array_equal(simple_control_problem.estado_inicial, [1.0])
    assert simple_control_problem.t_span == (0.0, 1.0)
    assert simple_control_problem.dimension_estado == 1
    assert simple_control_problem.dimension_control == 1


def test_fbsm_y_helpers_no_acceden_atributos_privados_del_problema():
    for funcion in (
        metodos_optimizacion.fbsm,
        metodos_optimizacion._integrar_adjunto_atras,
    ):
        assert "problema._" not in inspect.getsource(funcion)


def test_wrapper_preserva_firma_y_delega(simple_control_problem, monkeypatch):
    firma_externa = list(
        inspect.signature(metodos_optimizacion.fbsm).parameters.values()
    )
    firma_metodo = list(inspect.signature(ControlProblem.fbsm).parameters.values())
    assert firma_metodo[1:] == firma_externa[1:]

    centinela = object()
    llamada = {}

    def registrar(problema, *args, **kwargs):
        llamada["argumentos"] = (problema, args, kwargs)
        return centinela

    monkeypatch.setattr(metodos_optimizacion, "fbsm", registrar)
    resultado = simple_control_problem.fbsm(
        np.zeros((3, 1)), 0.5, "heun", 7, 1e-4, 0.8
    )

    assert resultado is centinela
    problema, argumentos, keywords = llamada["argumentos"]
    assert problema is simple_control_problem
    np.testing.assert_array_equal(argumentos[0], np.zeros((3, 1)))
    assert argumentos[1:] == (0.5, "heun", 7, 1e-4, 0.8)
    assert keywords == {}


def test_funcion_externa_y_wrapper_son_numericamente_equivalentes(
    simple_control_problem,
):
    control = np.zeros((3, 1))

    externo = metodos_optimizacion.fbsm(
        simple_control_problem, control, 0.5, max_iter=2
    )
    metodo = simple_control_problem.fbsm(control, 0.5, max_iter=2)

    np.testing.assert_allclose(metodo.control_optimo, externo.control_optimo)
    np.testing.assert_allclose(metodo.estado, externo.estado)
    np.testing.assert_allclose(metodo.adjunto, externo.adjunto)
    assert metodo.historia_costo == externo.historia_costo
    assert metodo.iteraciones == externo.iteraciones
    assert metodo.convergio == externo.convergio


def test_integracion_publica_respeta_grilla_variable(simple_control_problem):
    pasos = np.array([0.2, 0.3, 0.5])
    control = np.zeros((4, 1))

    tiempos, estados = simple_control_problem.integrar_estado(
        control, pasos, "euler_progresivo"
    )
    adjuntos = simple_control_problem.integrar_adjunto(
        tiempos, estados, control, pasos, "euler_progresivo"
    )

    np.testing.assert_allclose(tiempos, [0.0, 0.2, 0.5, 1.0])
    assert estados.shape == adjuntos.shape == (4, 1)
    np.testing.assert_allclose(
        adjuntos[-1], simple_control_problem.condicion_transversalidad(estados[-1])
    )
