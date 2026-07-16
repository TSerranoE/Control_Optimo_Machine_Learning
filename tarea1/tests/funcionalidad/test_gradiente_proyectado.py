"""Pruebas funcionales de los auxiliares de gradiente proyectado (Problema 4a)."""

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from integradores import EDOSolver
from problemas_control import ConjuntoAdmisible, ControlProblem


METODOS = EDOSolver.METODOS

def _problema_lineal(conjunto=None):
    """Construye un problema escalar con adjunto continuo de solución afín."""
    return ControlProblem(
        f=lambda t, x, u: np.zeros(1),
        l=lambda t, x, u: float(x[0]),
        phi=lambda x: 2.0 * float(x[0]),
        df_dx=lambda t, x, u: np.zeros((1, 1)),
        df_du=lambda t, x, u: np.ones((1, 1)),
        dl_dx=lambda t, x, u: np.ones(1),
        dl_du=lambda t, x, u: np.zeros(1),
        dphi_dx=lambda x: np.full(1, 2.0),
        t_span=(0.0, 1.0), x0=np.array([1.0]), m=1,
        conjunto_admisible=conjunto,
    )


@pytest.mark.parametrize("metodo", METODOS)
def test_helper_gradiente_infiere_h_y_revierte_adjunto(metodo):
    gradiente = _problema_lineal().grad(np.zeros(3), metodo)
    assert gradiente.shape == (3, 1)
    np.testing.assert_allclose(gradiente[:, 0], [3.0, 2.5, 2.0], atol=1e-10)

@pytest.mark.parametrize("control,metodo", [
    (np.zeros(1), "rk4"), (np.zeros((3, 2)), "rk4"),
    (np.array([0.0, np.nan]), "rk4"), (np.zeros(3), "desconocido"),
])
def test_helper_gradiente_rechaza_control_o_metodo_invalido(control, metodo):
    with pytest.raises(ValueError):
        _problema_lineal().grad(control, metodo)


def test_helper_costo_rk4_usa_etapas_del_estado_aumentado():
    problema = ControlProblem(
        f=lambda t, x, u: np.asarray(x),
        l=lambda t, x, u: float(x[0]),
        phi=lambda x: 0.0,
        df_dx=lambda t, x, u: np.ones((1, 1)),
        df_du=lambda t, x, u: np.zeros((1, 1)),
        dl_dx=lambda t, x, u: np.ones(1),
        dl_du=lambda t, x, u: np.zeros(1),
        dphi_dx=lambda x: np.zeros(1),
        t_span=(0.0, 1.0), x0=np.ones(1), m=1,
    )

    costo = problema._evaluar_costo_nodal(np.zeros(2), "rk4")

    assert costo == pytest.approx(1.7083333333333333)


def test_helper_costo_rk4_interpola_control_en_etapas():
    problema = ControlProblem(
        f=lambda t, x, u: np.asarray(u),
        l=lambda t, x, u: float(x[0]),
        phi=lambda x: 0.0,
        df_dx=lambda t, x, u: np.zeros((1, 1)),
        df_du=lambda t, x, u: np.ones((1, 1)),
        dl_dx=lambda t, x, u: np.ones(1),
        dl_du=lambda t, x, u: np.zeros(1),
        dphi_dx=lambda x: np.zeros(1),
        t_span=(0.0, 1.0), x0=np.zeros(1), m=1,
    )

    costo = problema._evaluar_costo_nodal(np.array([0.0, 2.0]), "rk4")

    assert costo == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("metodo,esperado", [
    ("euler_progresivo", 4.0), ("euler_implicito", 13.0),
    ("heun", 8.5), ("crank_nicolson", 8.5), ("rk4", 97.0 / 12.0),
])
def test_metric_producto_l2_usa_cuadratura_del_metodo(metodo, esperado):
    resultado = _problema_lineal().L2InnerProd(
        np.array([1.0, 2.0, 4.0]), np.array([2.0, 3.0, 5.0]), metodo
    )
    assert resultado == pytest.approx(esperado)

@pytest.mark.parametrize("metodo", METODOS)
def test_metric_norma_l2_es_raiz_del_producto_propio(metodo):
    problema, u = _problema_lineal(), np.array([1.0, 2.0, 4.0])
    assert problema.L2Norm(u, metodo) == pytest.approx(
        np.sqrt(problema.L2InnerProd(u, u, metodo))
    )


def test_metric_proyeccion_irrestricta_es_identidad_independiente():
    u = np.array([-2.0, 0.5, 3.0])
    proyectado = _problema_lineal().proj(u, "heun")
    np.testing.assert_array_equal(proyectado[:, 0], u)
    assert not np.shares_memory(proyectado, u)

def test_metric_proyeccion_caja_recorta_y_es_idempotente():
    problema = _problema_lineal(ConjuntoAdmisible(((-1.0, 1.0),)))
    proyectado = problema.proj(np.array([-2.0, 0.5, 3.0]), "rk4")
    np.testing.assert_array_equal(proyectado[:, 0], [-1.0, 0.5, 1.0])
    np.testing.assert_array_equal(problema.proj(proyectado, "rk4"), proyectado)


@pytest.mark.parametrize("g_2,t_min,esperado", [
    (np.full(3, 0.1), 1e-12, 1.0), (np.full(3, 2.0), 1e-12, 0.5),
    (np.full(3, 100.0), 0.1, 0.1), (np.full(3, -1.0), 1e-12, 1.0),
    (np.zeros(3), 1e-12, 1.0),
])
def test_bb_aplica_cociente_recorte_y_fallback(g_2, t_min, esperado):
    paso = _problema_lineal().BBStep(
        np.zeros(3), np.ones(3), np.zeros(3), g_2, "heun", t_min=t_min
    )
    assert paso == pytest.approx(esperado)


def test_bb_preserva_cociente_valido_a_escala_pequena():
    paso = _problema_lineal().BBStep(
        np.zeros(3), np.full(3, 1e-9), np.zeros(3), np.full(3, 2e-9), "heun"
    )
    assert paso == pytest.approx(0.5)


def test_backtracking_reduce_hasta_satisfacer_armijo(monkeypatch):
    problema, evaluados = _problema_lineal(), []

    def costo(control, metodo):
        paso = float(control[0, 0])
        evaluados.append((paso, metodo))
        return 0.0 if paso <= 0.25 else 1.0

    monkeypatch.setattr(problema, "_evaluar_costo_nodal", costo)
    paso = problema.backtracking(
        np.zeros(3), -np.ones(3), np.ones(3), 0.1, 0.5, 0.1, "rk4",
        t_inicial=1.0, max_reducciones=4,
    )
    assert paso == pytest.approx(0.25)
    assert evaluados == [(1.0, "rk4"), (0.5, "rk4"), (0.25, "rk4")]


def test_backtracking_respeta_controles_keyword(monkeypatch):
    problema = _problema_lineal()
    monkeypatch.setattr(problema, "_evaluar_costo_nodal", lambda u, m: 0.0 if u[0, 0] <= 0.2 else 1.0)
    paso = problema.backtracking(
        np.zeros(3), -np.ones(3), np.ones(3), 0.2, 0.25, 0.1,
        "euler_progresivo", t_inicial=0.8, max_reducciones=1,
    )
    assert paso == pytest.approx(0.2)


def test_backtracking_falla_al_agotar_reducciones(monkeypatch):
    problema, llamadas = _problema_lineal(), []
    monkeypatch.setattr(
        problema, "_evaluar_costo_nodal", lambda u, m: llamadas.append(1) or 1.0
    )
    with pytest.raises(RuntimeError, match="backtracking"):
        problema.backtracking(
            np.zeros(3), -np.ones(3), np.ones(3), 0.1, 0.5, 0.0, "heun",
            max_reducciones=2,
        )
    assert len(llamadas) == 3


@pytest.mark.parametrize("clave,valor", [
    ("a", 0.0), ("a", 1.0), ("b", 0.0), ("b", 1.0),
    ("J_hat", np.inf), ("t_inicial", 0.0), ("max_reducciones", 0),
])
def test_backtracking_rechaza_parametros_invalidos(clave, valor):
    parametros = dict(
        u=np.zeros(3), g=-np.ones(3), v=np.ones(3), a=0.1, b=0.5,
        J_hat=0.0, metodo_integracion="heun", t_inicial=1.0,
        max_reducciones=2,
    )
    parametros[clave] = valor
    with pytest.raises(ValueError):
        _problema_lineal().backtracking(**parametros)


def test_optimizer_firma_tiene_cuatro_requeridos_y_controles_keyword():
    firma = inspect.signature(ControlProblem.gradiente_proyectado)
    parametros = list(firma.parameters.values())[1:]
    assert [p.name for p in parametros[:4]] == [
        "u_inicial", "max_iter", "tolerancia", "metodo_integracion"
    ]
    assert all(p.default is inspect.Parameter.empty for p in parametros[:4])
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parametros[4:])
    assert [p.default for p in parametros[4:]] == [10, 1e-4, 0.5, 1e-12, 50]


@pytest.mark.parametrize("cambio", [
    {"u_inicial": np.zeros(1)}, {"max_iter": 0}, {"max_iter": 1.5},
    {"tolerancia": -1.0}, {"tolerancia": np.inf}, {"r": 0},
    {"a": 0.0}, {"a": 1.0}, {"b": 0.0}, {"b": 1.0},
    {"t_min": 0.0}, {"t_min": 1.1}, {"max_reducciones": 0},
    {"metodo_integracion": "desconocido"},
])
def test_optimizer_rechaza_parametros_invalidos(cambio):
    parametros = dict(
        u_inicial=np.zeros(3), max_iter=1, tolerancia=0.0,
        metodo_integracion="heun", r=2, a=0.1, b=0.5,
        t_min=1e-6, max_reducciones=2,
    )
    parametros.update(cambio)
    with pytest.raises(ValueError):
        _problema_lineal().gradiente_proyectado(**parametros)


def test_optimizer_usa_direccion_literal_y_semilla_inicial(monkeypatch):
    problema, llamadas = _problema_lineal(), {}
    monkeypatch.setattr(problema, "grad", lambda u, m: np.ones_like(u))

    def proyectar(candidato, metodo):
        llamadas["proyeccion"] = (candidato.copy(), metodo)
        return np.full_like(candidato, -0.5)

    def buscar(u, g, v, a, b, J_hat, metodo, t_inicial=1, *, max_reducciones=50):
        llamadas["backtracking"] = (v.copy(), a, b, J_hat, metodo, t_inicial, max_reducciones)
        return 0.25

    monkeypatch.setattr(problema, "proj", proyectar)
    monkeypatch.setattr(problema, "backtracking", buscar)
    monkeypatch.setattr(
        problema, "_evaluar_costo_nodal", lambda u, m: 10.0 + float(u[0, 0])
    )

    resultado = problema.gradiente_proyectado(np.zeros(3), 1, 0.0, "heun")

    np.testing.assert_array_equal(llamadas["proyeccion"][0], -np.ones((3, 1)))
    assert llamadas["proyeccion"][1] == "heun"
    np.testing.assert_array_equal(llamadas["backtracking"][0], -0.5 * np.ones((3, 1)))
    assert llamadas["backtracking"][1:] == (1e-4, 0.5, 10.0, "heun", 1, 50)
    np.testing.assert_allclose(resultado.control, -0.125)


def test_optimizer_usa_bb_ventana_y_controles_avanzados(monkeypatch):
    problema, semillas, referencias, minimos = _problema_lineal(), [], [], []
    monkeypatch.setattr(problema, "grad", lambda u, m: np.zeros_like(u))
    monkeypatch.setattr(problema, "proj", lambda u, m: u + np.ones_like(u))
    monkeypatch.setattr(
        problema, "_evaluar_costo_nodal", lambda u, m: 10.0 - float(u[0, 0])
    )

    def bb(*args, t_min):
        minimos.append(t_min)
        return 0.4

    def buscar(u, g, v, a, b, J_hat, metodo, t_inicial=1, *, max_reducciones=50):
        semillas.append(t_inicial)
        referencias.append(J_hat)
        assert (a, b, metodo, max_reducciones) == (0.2, 0.3, "heun", 7)
        return 0.1

    monkeypatch.setattr(problema, "BBStep", bb)
    monkeypatch.setattr(problema, "backtracking", buscar)

    resultado = problema.gradiente_proyectado(
        np.zeros(3), 3, 0.0, "heun", r=2, a=0.2, b=0.3,
        t_min=0.02, max_reducciones=7,
    )

    np.testing.assert_allclose(resultado.control, 0.3)
    assert semillas == [1, 0.4, 0.4]
    assert referencias == pytest.approx([10.0, 10.0, 9.9])
    assert minimos == [0.02, 0.02]


def test_resultado_es_inmutable_y_consistente_al_converger(monkeypatch):
    problema, costos = _problema_lineal(), []
    monkeypatch.setattr(problema, "grad", lambda u, m: np.ones_like(u))
    monkeypatch.setattr(problema, "proj", lambda u, m: u)
    monkeypatch.setattr(problema, "backtracking", lambda *args, **kwargs: 0.5)

    def costo(u, metodo):
        costos.append(u.copy())
        return 2.0 if np.allclose(u, 0.0) else 1.9995

    estados = np.full((3, 1), 9.5)
    adjuntos = np.full((3, 1), 29.5)
    monkeypatch.setattr(problema, "_evaluar_costo_nodal", costo)
    monkeypatch.setattr(
        problema, "_integrar_estado", lambda u, h, m: (np.linspace(0, 1, 3), estados)
    )
    monkeypatch.setattr(
        problema, "_integrar_adjunto", lambda t, x, u, h, m: adjuntos
    )

    resultado = problema.gradiente_proyectado(np.zeros(3), 5, 1e-3, "heun")
    estados[:] = -1
    adjuntos[:] = -1

    assert resultado.convergio and resultado.iteraciones == 1
    assert resultado.historial_costos == (2.0, 1.9995)
    np.testing.assert_allclose(resultado.control, -0.5)
    np.testing.assert_allclose(resultado.estados, 9.5)
    np.testing.assert_allclose(resultado.adjuntos, 29.5)
    assert len(costos) == 3 and np.array_equal(costos[-1], resultado.control)
    assert not resultado.control.flags.writeable
    assert not resultado.estados.flags.writeable
    assert not resultado.adjuntos.flags.writeable
    with pytest.raises(ValueError):
        resultado.control[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        resultado.convergio = False


def test_resultado_reporta_limite_y_se_exporta_publicamente(monkeypatch):
    from problemas_control import ResultadoGradienteProyectado
    from src import __all__

    problema = _problema_lineal()
    monkeypatch.setattr(problema, "grad", lambda u, m: np.zeros_like(u))
    monkeypatch.setattr(problema, "proj", lambda u, m: u + 1.0)
    monkeypatch.setattr(problema, "backtracking", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr(
        problema, "_evaluar_costo_nodal", lambda u, m: 10.0 - float(u[0, 0])
    )

    resultado = problema.gradiente_proyectado(np.zeros(3), 2, 0.0, "heun")

    assert isinstance(resultado, ResultadoGradienteProyectado)
    assert "ResultadoGradienteProyectado" in __all__
    assert not resultado.convergio and resultado.iteraciones == 2
