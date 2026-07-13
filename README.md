# MA6914 - Tarea 1: Resolución de EDO's y Control Óptimo

Repositorio con las herramientas computacionales desarrolladas para la Tarea 1
del curso MA6914 Seminario Avanzado I.

## Estructura del proyecto

```text
ma6914_tarea1/
├── src/
│   ├── integradores.py         # Clases EDOSolver y EDOSolution
│   ├── problemas_control.py    # Clase ControlProblem
│   └── metodos_optimizacion.py # Funciones FBSM y Gradiente Proyectado
├── tests/
│   ├── funcionalidad/          # Tests de métodos individuales
│   └── validacion/             # Tests de casos reales de la tarea
├── notebooks/
│   └── ejecucion_tarea1.py     # Script formato Jupytext (py:percent)
└── resultados_graficos/
    ├── 1_lotka_volterra/
    ├── 1_van_der_pol/
    ├── 3_lqr/
    ├── 3_modelo_sir/
    └── 4_gradiente_proyectado/
```

## Requisitos

- Python >= 3.12
- `uv` como gestor de paquetes

## Instalación

```bash
uv sync
```

## Ejecución de pruebas

```bash
uv run pytest
```

## Conversión de notebooks con Jupytext

```bash
uv run jupytext --to notebook ma6914_tarea1/notebooks/ejecucion_tarea1.py
```
