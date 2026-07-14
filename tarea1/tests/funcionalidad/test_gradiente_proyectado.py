"""Pruebas funcionales de los auxiliares de gradiente proyectado (Problema 4a)."""

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
