# Control óptimo y métodos numéricos

Repositorio de trabajo para el curso **MA6914 Seminario Avanzado I**.  
El contenido actualmente desarrollado corresponde principalmente a la **Tarea 1: resolución de EDO y problemas de control óptimo**.

## Tarea 1

La entrega completa se encuentra en la carpeta [`tarea1/`](./tarea1/).

### Acceso rápido

| Recurso | Ruta | Descripción |
|---|---|---|
| Informe | [`tarea1/informe/informe.tex`](./tarea1/informe/informe.tex) | Desarrollo matemático, resultados numéricos y conclusiones de los cuatro problemas. |
| Notebook de ejecución | [`tarea1/notebooks/ejecucion_tarea1.ipynb`](./tarea1/notebooks/ejecucion_tarea1.ipynb) | Ejecución ordenada de los experimentos, tablas y gráficos utilizados en el informe. |
| Implementación de integradores | [`tarea1/src/integradores.py`](./tarea1/src/integradores.py) | Clases `EDOSolver` y `EDOSolution`, junto con los cinco métodos de integración solicitados. |
| Problemas de control | [`tarea1/src/problemas_control.py`](./tarea1/src/problemas_control.py) | Clase `ControlProblem`, FBSM, gradiente proyectado y métodos auxiliares. |
| Resultados gráficos | [`tarea1/resultados_graficos/`](./tarea1/resultados_graficos/) | Figuras y tablas generadas para los distintos experimentos. |

## Mapa de la tarea

### Problema 1: integración numérica de EDO

Se implementan:

- Euler progresivo;
- Euler implícito;
- Heun;
- Crank–Nicolson;
- Runge–Kutta de orden 4.

Los experimentos consideran el sistema de Lotka–Volterra y el oscilador rígido de Van der Pol con \(\mu=1000\).

Archivos principales:

- [`tarea1/src/integradores.py`](./tarea1/src/integradores.py): implementación de los métodos;
- [`tarea1/src/validacion_problema1.py`](./tarea1/src/validacion_problema1.py): experimentos de Lotka–Volterra y Van der Pol;
- [`tarea1/src/utils/visualizacion.py`](./tarea1/src/utils/visualizacion.py): gráficos y tablas;
- [`tarea1/resultados_graficos/1_lotka_volterra/`](./tarea1/resultados_graficos/1_lotka_volterra/);
- [`tarea1/resultados_graficos/1_van_der_pol/`](./tarea1/resultados_graficos/1_van_der_pol/).

### Problema 2: clase para problemas de control óptimo

Se implementa una clase reutilizable para problemas de Bolza, incluyendo:

- evaluación del funcional de costo;
- Hamiltoniano;
- sistema adjunto;
- condición de transversalidad;
- controles sin restricciones o con restricciones de caja;
- minimización puntual del Hamiltoniano.

Archivo principal:

- [`tarea1/src/problemas_control.py`](./tarea1/src/problemas_control.py).

### Problema 3: resolución mediante el principio de Pontryagin

Se implementa el método de barrido hacia adelante y hacia atrás (**FBSM**) y se aplica a:

- un problema LQR escalar, comparado con la solución obtenida mediante la ecuación de Riccati;
- un modelo SIR con control de vacunación.

Archivos principales:

- [`tarea1/src/problemas_control.py`](./tarea1/src/problemas_control.py): implementación del FBSM;
- [`tarea1/src/validacion_problema3.py`](./tarea1/src/validacion_problema3.py): definición de los problemas y referencias;
- [`tarea1/src/reporte_problema3.py`](./tarea1/src/reporte_problema3.py): generación de resultados;
- [`tarea1/resultados_graficos/3_fbsm/`](./tarea1/resultados_graficos/3_fbsm/).

### Problema 4: optimización directa

Se implementa el método de gradiente proyectado usando el sistema adjunto, junto con:

- cálculo del gradiente;
- proyección sobre el conjunto admisible;
- paso de Barzilai–Borwein;
- búsqueda de línea por backtracking;
- comparación con FBSM en los problemas LQR y SIR.

Archivos principales:

- [`tarea1/src/problemas_control.py`](./tarea1/src/problemas_control.py): implementación del método;
- [`tarea1/src/reporte_problema4.py`](./tarea1/src/reporte_problema4.py): comparación numérica y generación de figuras;
- [`tarea1/resultados_graficos/4_gradiente_proyectado/`](./tarea1/resultados_graficos/4_gradiente_proyectado/).

## Estructura principal

```text
tarea1/
├── informe/
│   └── informe.tex
├── notebooks/
│   ├── ejecucion_tarea1.ipynb
│   └── ejecucion_tarea1.py
├── src/
│   ├── integradores.py
│   ├── problemas_control.py
│   ├── validacion_problema1.py
│   ├── validacion_problema3.py
│   ├── reporte_problema3.py
│   ├── reporte_problema4.py
│   └── utils/
├── resultados_graficos/
│   ├── 1_lotka_volterra/
│   ├── 1_van_der_pol/
│   ├── 3_fbsm/
│   └── 4_gradiente_proyectado/
└── tests/
```

El archivo `ejecucion_tarea1.py` contiene la representación en texto del notebook y se mantiene junto al archivo `.ipynb`.

## Ejecución

Desde la raíz del repositorio, el recorrido recomendado es abrir y ejecutar:

```bash
jupyter notebook tarea1/notebooks/ejecucion_tarea1.ipynb
```

También puede ejecutarse la versión en Python:

```bash
PYTHONPATH=tarea1/src:tarea1 python tarea1/notebooks/ejecucion_tarea1.py
```

Los resultados se guardan en:

```text
tarea1/resultados_graficos/
```

## Compilación del informe

```bash
cd tarea1/informe
pdflatex -interaction=nonstopmode -halt-on-error informe.tex
pdflatex -interaction=nonstopmode -halt-on-error informe.tex
```

El archivo resultante se genera como `tarea1/informe/informe.pdf`.

## Pruebas

Las pruebas automáticas se encuentran en [`tarea1/tests/`](./tarea1/tests/) y pueden ejecutarse desde la raíz mediante:

```bash
PYTHONPATH=tarea1/src:tarea1 pytest -q tarea1/tests
```
