# MA6914 - Tarea 1: Resolución de EDO's y Control Óptimo

Repositorio con las herramientas computacionales desarrolladas para la Tarea 1
del curso MA6914 Seminario Avanzado I.

## Estructura del proyecto

```text
tarea1/
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

## Notebooks emparejados con Jupytext

Los notebooks se mantienen en formato dual:
- `.py` con formato `py:percent` (texto plano, ideal para diff y revisiones).
- `.ipynb` (formato nativo de Jupyter, listo para ejecutar en un entorno interactivo).

Ambos archivos se versionan. La configuración en `pyproject.toml` define el emparejamiento para las carpetas `tarea1/notebooks/` y `tarea2/notebooks/`.

### Sincronización manual

```bash
./scripts/sync_notebooks.sh
```

### Instalación del hook de pre-commit

Para que los notebooks se sincronicen automáticamente antes de cada commit:

```bash
./scripts/install_precommit_hook.sh
```

### Verificación por el orquestador (gga / sdd-verify)

El flujo de verificación debe ejecutar el script de sincronización y comprobar que no haya diferencias pendientes entre `.py` y `.ipynb`:

```bash
./scripts/sync_notebooks.sh
```

Si el script falla, significa que las dos representaciones de algún notebook no están alineadas.
