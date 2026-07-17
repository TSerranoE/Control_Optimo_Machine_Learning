"""Contratos de arquitectura para los métodos del Problema 4a."""

import inspect

import numpy as np

import metodos_optimizacion
from problemas_control import ControlProblem

NOMBRES_PROBLEMA_4A = (
    "grad",
    "proj",
    "BBStep",
    "backtracking",
    "L2InnerProd",
    "L2Norm",
    "gradiente_proyectado",
)


def test_implementaciones_problema_4a_residen_en_modulo_optimizacion():
    for nombre in NOMBRES_PROBLEMA_4A:
        funcion = getattr(metodos_optimizacion, nombre)
        assert funcion.__module__ == "metodos_optimizacion"


def test_modulo_optimizacion_no_accede_atributos_privados_del_problema():
    assert "problema._" not in inspect.getsource(metodos_optimizacion)


def test_wrappers_preservan_firmas_academicas():
    for nombre in NOMBRES_PROBLEMA_4A:
        firma_externa = list(
            inspect.signature(getattr(metodos_optimizacion, nombre)).parameters
        )
        firma_metodo = list(
            inspect.signature(getattr(ControlProblem, nombre)).parameters
        )
        assert firma_metodo[0] == "self"
        assert firma_externa[0] == "problema"
        assert firma_metodo[1:] == firma_externa[1:]


def test_wrappers_delegan_en_implementaciones_externas(
    simple_control_problem, monkeypatch
):
    centinelas = {
        "grad": np.full((3, 1), 1.0),
        "proj": np.full((3, 1), 2.0),
        "L2InnerProd": 3.0,
        "L2Norm": 4.0,
        "BBStep": 0.25,
        "backtracking": 0.5,
        "gradiente_proyectado": object(),
    }
    llamadas = []

    def fabricar(esperado):
        def falso(problema_recibido, *args, **kwargs):
            assert problema_recibido is simple_control_problem
            llamadas.append(args)
            return esperado

        return falso

    for nombre, esperado in centinelas.items():
        monkeypatch.setattr(metodos_optimizacion, nombre, fabricar(esperado))

    ceros = np.zeros(3)
    np.testing.assert_array_equal(
        simple_control_problem.grad(ceros, "heun"), centinelas["grad"]
    )
    np.testing.assert_array_equal(
        simple_control_problem.proj(ceros, "heun"), centinelas["proj"]
    )
    assert simple_control_problem.L2InnerProd(ceros, ceros, "heun") == 3.0
    assert simple_control_problem.L2Norm(ceros, "heun") == 4.0
    assert simple_control_problem.BBStep(
        ceros, ceros, ceros, ceros, "heun"
    ) == 0.25
    assert simple_control_problem.backtracking(
        ceros, ceros, ceros, 0.1, 0.5, 0.0, "heun"
    ) == 0.5
    assert (
        simple_control_problem.gradiente_proyectado(ceros, 1, 0.0, "heun")
        is centinelas["gradiente_proyectado"]
    )
    assert len(llamadas) == len(centinelas)


def test_gradiente_proyectado_externo_despacha_por_wrappers(
    simple_control_problem, monkeypatch
):
    llamadas = {"grad": 0, "proj": 0, "BBStep": 0, "backtracking": 0}

    grad_original = ControlProblem.grad
    proj_original = ControlProblem.proj
    backtracking_original = ControlProblem.backtracking

    def grad_espiado(self, u, metodo):
        llamadas["grad"] += 1
        return grad_original(self, u, metodo)

    def proj_espiado(self, u, metodo):
        llamadas["proj"] += 1
        return proj_original(self, u, metodo)

    def bbstep_espiado(self, *args, **kwargs):
        llamadas["BBStep"] += 1
        return 1.0

    def backtracking_espiado(self, *args, **kwargs):
        llamadas["backtracking"] += 1
        return backtracking_original(self, *args, **kwargs)

    monkeypatch.setattr(ControlProblem, "grad", grad_espiado)
    monkeypatch.setattr(ControlProblem, "proj", proj_espiado)
    monkeypatch.setattr(ControlProblem, "BBStep", bbstep_espiado)
    monkeypatch.setattr(ControlProblem, "backtracking", backtracking_espiado)

    metodos_optimizacion.gradiente_proyectado(
        simple_control_problem, np.zeros((3, 1)), 2, 0.0, "heun"
    )

    assert llamadas["grad"] >= 1
    assert llamadas["proj"] >= 1
    assert llamadas["BBStep"] >= 1
    assert llamadas["backtracking"] >= 1


def test_ambos_estilos_de_llamada_conservan_resultados_numericos(
    simple_control_problem,
):
    control = np.zeros((3, 1))

    grad_externo = metodos_optimizacion.grad(simple_control_problem, control, "heun")
    np.testing.assert_allclose(
        simple_control_problem.grad(control, "heun"), grad_externo
    )
    np.testing.assert_allclose(
        metodos_optimizacion.proj(simple_control_problem, control, "heun"),
        simple_control_problem.proj(control, "heun"),
    )
    assert metodos_optimizacion.L2InnerProd(
        simple_control_problem, control, control, "heun"
    ) == simple_control_problem.L2InnerProd(control, control, "heun")
    assert metodos_optimizacion.L2Norm(
        simple_control_problem, control, "heun"
    ) == simple_control_problem.L2Norm(control, "heun")

    gp_externo = metodos_optimizacion.gradiente_proyectado(
        simple_control_problem, control, 1, 0.0, "heun"
    )
    gp_metodo = simple_control_problem.gradiente_proyectado(control, 1, 0.0, "heun")
    np.testing.assert_allclose(gp_metodo.control, gp_externo.control)
    np.testing.assert_allclose(gp_metodo.estados, gp_externo.estados)
    assert gp_metodo.historial_costos == gp_externo.historial_costos
