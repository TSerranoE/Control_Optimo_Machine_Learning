"""Tests unitarios para la clase ``ConjuntoAdmisible`` del Problema 2."""

import numpy as np
import pytest

from problemas_control import ConjuntoAdmisible


class TestConjuntoAdmisible:
    """Agrupa las pruebas de proyección y consulta del conjunto admisible."""

    def test_proyectar_box_clips(self):
        """Un punto fuera de la caja debe proyectarse al borde más cercano."""
        limites = ((-1.0, 1.0),)
        conjunto = ConjuntoAdmisible(limites=limites)
        u = np.array([2.0])

        resultado = conjunto.proyectar(u)

        esperado = np.array([1.0])
        np.testing.assert_array_equal(resultado, esperado)

    def test_proyectar_unrestricted_identity(self):
        """Un conjunto irrestricto debe devolver el vector sin modificar."""
        conjunto = ConjuntoAdmisible(limites=None)
        u = np.array([5.0, -3.0])

        resultado = conjunto.proyectar(u)

        np.testing.assert_array_equal(resultado, u)
        # Además, la proyección debe devolver una copia, no la referencia original.
        assert resultado is not u

    def test_es_caja_true(self):
        """``es_caja`` debe ser ``True`` cuando se definen límites."""
        conjunto = ConjuntoAdmisible(limites=((-1.0, 1.0),))

        assert conjunto.es_caja() is True

    def test_es_caja_false(self):
        """``es_caja`` debe ser ``False`` cuando el conjunto es irrestricto."""
        conjunto = ConjuntoAdmisible(limites=None)

        assert conjunto.es_caja() is False

    def test_limites_returns_tuple(self):
        """``limites`` debe devolver la tupla original de límites."""
        limites = ((-2.0, 2.0), (-1.0, 1.0))
        conjunto = ConjuntoAdmisible(limites=limites)

        assert conjunto.limites() == limites
