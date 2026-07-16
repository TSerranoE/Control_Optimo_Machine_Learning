# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: tarea1/notebooks///ipynb,tarea1/notebooks///py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook de ejecución - Tarea 1
#
# Este notebook agrupa los experimentos numéricos solicitados en la Tarea 1:
# validación de integradores, resolución de problemas de control óptimo y
# generación de figuras comparativas.

# %% [markdown]
# ## Configuración común

# %%
import os
import sys
import warnings
from pathlib import Path

# Permite imports absolutos tanto en Jupyter como al ejecutar el script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
import numpy as np

# Backend no interactivo para ejecución por lotes.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# El warning de fsolve sobre "no making good progress" es benigno para
# campos lineales con pasos finos; la solución sigue convergiendo.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*not making good progress.*",
)

from src.integradores import EDOSolver
from src.validacion_problema1 import (
    campo_lotka_volterra,
    campo_van_der_pol,
    crear_referencia,
    ejecutar_experimento,
)
from utils.visualizacion import (
    graficar_diagrama_fase,
    graficar_error_vs_h,
    graficar_referencia_vs_aproximada,
    tabla_resultados,
)

RUTA_BASE = Path("tarea1/resultados_graficos")
RUTA_LOTKA = RUTA_BASE / "1_lotka_volterra"
RUTA_VDP = RUTA_BASE / "1_van_der_pol"

for ruta in [RUTA_LOTKA, RUTA_VDP]:
    ruta.mkdir(parents=True, exist_ok=True)

hs = [0.1, 0.05, 0.025, 0.0125, 0.00625]
metodos = ["euler_progresivo", "euler_implicito", "heun", "crank_nicolson", "rk4"]

print("Configuración lista.")
print(f"Pasos temporales: {hs}")
print(f"Métodos: {metodos}")

# %% [markdown]
# ## Problema 1b - Lotka-Volterra

# %%
parametros_lotka = {
    "alpha": 1.0,
    "beta": 0.1,
    "delta": 0.075,
    "gamma": 1.5,
}
x0_lotka = np.array([10.0, 5.0])
t0_lotka, tf_lotka = 0.0, 15.0

resultados_lotka = ejecutar_experimento(
    campo_lotka_volterra,
    x0_lotka,
    t0_lotka,
    tf_lotka,
    parametros_lotka,
    hs,
    metodos,
)

print("Tabla de resultados - Lotka-Volterra")
print(tabla_resultados(resultados_lotka).to_string(index=False))

# %% [markdown]
# ### Figuras de Lotka-Volterra

# %%
caption_error_lotka = (
    "Referencia: Crank-Nicolson con h=1e-4. "
    "Parámetros: α=1.0, β=0.1, δ=0.075, γ=1.5, "
    "x₀=[10, 5], t∈[0, 15]"
)

fig_error_lotka = graficar_error_vs_h(
    resultados_lotka,
    ruta_salida=RUTA_LOTKA / "error_vs_h.png",
    titulo="Lotka-Volterra: error vs paso temporal",
    metodo_referencia="Crank-Nicolson",
    h_referencia=1e-4,
    caption=caption_error_lotka,
)

# %%
t_ref_lotka, x_ref_lotka = crear_referencia(
    campo_lotka_volterra,
    x0_lotka,
    t0_lotka,
    tf_lotka,
    parametros_lotka,
    h=1e-4,
    metodo="crank_nicolson",
)

resolutor_lotka = EDOSolver()
solucion_lotka = resolutor_lotka.solve(
    lambda t, x, u: campo_lotka_volterra(t, x, u, parametros_lotka),
    x0_lotka,
    (t0_lotka, tf_lotka),
    hs[0],
    method="rk4",
)

caption_series_lotka = (
    "Referencia: Crank-Nicolson h=1e-4. Aproximación: RK4 h=0.1. "
    "Componentes: x₁(t) y x₂(t). "
    "Parámetros: α=1.0, β=0.1, δ=0.075, γ=1.5, x₀=[10, 5]"
)

fig_series_lotka = graficar_referencia_vs_aproximada(
    t_ref_lotka,
    x_ref_lotka.estados,
    solucion_lotka.tiempos,
    solucion_lotka.estados,
    "Lotka-Volterra: evolución temporal",
    ruta_salida=RUTA_LOTKA / "series_temporales.png",
    nombres_componentes=["x_1(t)", "x_2(t)"],
    descripcion_referencia="Referencia (CN h=1e-4)",
    descripcion_aproximacion="Aproximación (RK4 h=0.1)",
    caption=caption_series_lotka,
)

caption_fase_lotka = (
    "Calculado con Crank-Nicolson h=1e-4. "
    "Parámetros: α=1.0, β=0.1, δ=0.075, γ=1.5, x₀=[10, 5]"
)

fig_fase_lotka = graficar_diagrama_fase(
    x_ref_lotka.estados,
    "Lotka-Volterra: diagrama de fase",
    ruta_salida=RUTA_LOTKA / "diagrama_fase.png",
    caption=caption_fase_lotka,
)

# %% [markdown]
# ## Problema 1b - Van der Pol ($\mu = 0.1$)

# %%
parametros_vdp_suave = {"mu": 0.1}
x0_vdp = np.array([2.0, 0.0])
t0_vdp, tf_vdp_suave = 0.0, 20.0

resultados_vdp_suave = ejecutar_experimento(
    campo_van_der_pol,
    x0_vdp,
    t0_vdp,
    tf_vdp_suave,
    parametros_vdp_suave,
    hs,
    metodos,
)

print("Tabla de resultados - Van der Pol (mu=0.1)")
print(tabla_resultados(resultados_vdp_suave).to_string(index=False))

# %%
caption_error_vdp_suave = (
    "Referencia: Crank-Nicolson con h=1e-4. "
    "Parámetros: μ=0.1, x₀=[2, 0], t∈[0, 20]"
)

fig_error_vdp_suave = graficar_error_vs_h(
    resultados_vdp_suave,
    ruta_salida=RUTA_VDP / "error_vs_h_mu_0_1.png",
    titulo="Van der Pol (μ=0.1): error vs paso temporal",
    metodo_referencia="Crank-Nicolson",
    h_referencia=1e-4,
    caption=caption_error_vdp_suave,
)

# %%
t_ref_vdp_suave, x_ref_vdp_suave = crear_referencia(
    campo_van_der_pol,
    x0_vdp,
    t0_vdp,
    tf_vdp_suave,
    parametros_vdp_suave,
    h=1e-4,
    metodo="crank_nicolson",
)

caption_fase_vdp_suave = (
    "Calculado con Crank-Nicolson h=1e-4. "
    "Parámetros: μ=0.1, x₀=[2, 0]"
)

fig_fase_vdp_suave = graficar_diagrama_fase(
    x_ref_vdp_suave.estados,
    "Van der Pol (μ=0.1): diagrama de fase",
    ruta_salida=RUTA_VDP / "diagrama_fase_mu_0_1.png",
    caption=caption_fase_vdp_suave,
)

# %% [markdown]
# ## Problema 1b - Van der Pol ($\mu = 1000$): estabilidad vs precisión
#
# El objetivo es mostrar que los métodos explícitos requieren pasos
# extremadamente pequeños para estabilidad, mientras que los implícitos
# admiten pasos mucho mayores.
#
# Configuración stiff:
# - `tf = 10` para los experimentos de error (explícitos e implícitos),
#   suficiente para medir la estabilidad sin exceder el tiempo de cómputo.
# - `tf = 1000` para el diagrama de fase, donde se necesita un horizonte
#   largo para observar parte del ciclo límite de relajación
#   (período ~1.6·μ ≈ 1600 para μ=1000).
# - `h_referencia = 1e-5` para el caso stiff, calculado con los argumentos
#   por defecto de fsolve para reducir el tiempo de cómputo.

# %%
parametros_vdp_stiff = {"mu": 1000.0}
tf_vdp_stiff = 10.0
tf_vdp_stiff_fase = 3000.0
h_referencia_stiff = 1e-5
argumentos_fsolve_stiff = {"xtol": 1e-10, "maxfev": 200}

# Referencia única para todos los experimentos stiff con tf=10.
print("Calculando referencia stiff (esto puede tardar unos minutos)...")
_, solucion_ref_vdp_stiff = crear_referencia(
    campo_van_der_pol,
    x0_vdp,
    t0_vdp,
    tf_vdp_stiff,
    parametros_vdp_stiff,
    h=h_referencia_stiff,
    metodo="crank_nicolson",
    argumentos_fsolve=argumentos_fsolve_stiff,
)

# Experimento A: métodos explícitos con pasos grandes (inestables)
hs_expl_grandes = [0.01, 0.005, 0.0025, 0.00125]
metodos_expl = ["euler_progresivo", "heun", "rk4"]

resultados_vdp_expl_grandes = ejecutar_experimento(
    campo_van_der_pol,
    x0_vdp,
    t0_vdp,
    tf_vdp_stiff,
    parametros_vdp_stiff,
    hs_expl_grandes,
    metodos_expl,
    h_referencia=h_referencia_stiff,
    solucion_ref=solucion_ref_vdp_stiff,
)

print("Experimento A - Explícitos con pasos grandes")
print(tabla_resultados(resultados_vdp_expl_grandes).to_string(index=False))

# Experimento B: métodos implícitos con pasos grandes (estables)
hs_impl_grandes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
metodos_impl = ["euler_implicito", "crank_nicolson"]

resultados_vdp_impl_grandes = ejecutar_experimento(
    campo_van_der_pol,
    x0_vdp,
    t0_vdp,
    tf_vdp_stiff,
    parametros_vdp_stiff,
    hs_impl_grandes,
    metodos_impl,
    h_referencia=h_referencia_stiff,
    solucion_ref=solucion_ref_vdp_stiff,
)

print("\nExperimento B - Implícitos con pasos grandes")
print(tabla_resultados(resultados_vdp_impl_grandes).to_string(index=False))

# Experimento C: métodos explícitos con pasos pequeños (costoso)
# Se usan h=1e-5 y h=5e-6 para poder dibujar líneas en el gráfico log-log.
# Heun se omite aquí para no exceder el tiempo de cómputo.
hs_expl_pequenos = [1e-5, 5e-6]
metodos_expl_pequenos = ["euler_progresivo", "rk4"]

resultados_vdp_expl_pequenos = ejecutar_experimento(
    campo_van_der_pol,
    x0_vdp,
    t0_vdp,
    tf_vdp_stiff,
    parametros_vdp_stiff,
    hs_expl_pequenos,
    metodos_expl_pequenos,
    h_referencia=h_referencia_stiff,
    solucion_ref=solucion_ref_vdp_stiff,
)

print("\nExperimento C - Explícitos con paso pequeño")
print(tabla_resultados(resultados_vdp_expl_pequenos).to_string(index=False))

# Tabla comparativa completa
resultados_vdp_stiff = (
    resultados_vdp_expl_grandes
    + resultados_vdp_impl_grandes
    + resultados_vdp_expl_pequenos
)

print("\nTabla comparativa - Van der Pol (mu=1000)")
print(tabla_resultados(resultados_vdp_stiff).to_string(index=False))

# %% [markdown]
# **Observaciones de los experimentos stiff:**
#
# - Los métodos explícitos con pasos grandes (`h >= 0.00125`) divergen
#   (`error_inf = NaN`), lo que confirma la inestabilidad para $\mu = 1000$.
# - Los métodos implícitos con pasos grandes son estables y producen errores
#   pequeños, incluso con `h = 0.1`.
# - Los métodos explícitos con paso pequeño (`h = 1e-5`) logran errores
#   razonables, pero requieren mucho más tiempo que los implícitos con
#   pasos grandes. RK4 alcanza un piso de error alrededor de `1.8e-8`,
#   impuesto por la referencia (`h_ref = 1e-5`): al usar pasos RK4 más
#   pequeños que la referencia, el error de la propia referencia domina.
#   Refinar más la referencia (p. ej. `h_ref = 1e-6`) mejoraría la medida,
#   pero el costo de cómputo se vuelve prohibitivo.

# %%
caption_vdp_stiff = (
    "Referencia: Crank-Nicolson con h=1e-5. "
    "Parámetros: μ=1000, x₀=[2, 0], t∈[0, 10]"
)

fig_error_vdp_expl_grandes = graficar_error_vs_h(
    resultados_vdp_expl_grandes,
    ruta_salida=RUTA_VDP / "error_vs_h_mu_1000_explicitos_pasos_grandes.png",
    titulo="Van der Pol (μ=1000): explícitos con pasos grandes",
    metodo_referencia="Crank-Nicolson",
    h_referencia=h_referencia_stiff,
    caption=caption_vdp_stiff,
)

fig_error_vdp_impl_grandes = graficar_error_vs_h(
    resultados_vdp_impl_grandes,
    ruta_salida=RUTA_VDP / "error_vs_h_mu_1000_implicitos_pasos_grandes.png",
    titulo="Van der Pol (μ=1000): implícitos con pasos grandes",
    metodo_referencia="Crank-Nicolson",
    h_referencia=h_referencia_stiff,
    caption=caption_vdp_stiff,
)

fig_error_vdp_expl_pequenos = graficar_error_vs_h(
    resultados_vdp_expl_pequenos,
    ruta_salida=RUTA_VDP / "error_vs_h_mu_1000_explicitos_pasos_pequenos.png",
    titulo="Van der Pol (μ=1000): explícitos con paso pequeño",
    metodo_referencia="Crank-Nicolson",
    h_referencia=h_referencia_stiff,
    caption=caption_vdp_stiff,
)

# %% [markdown]
# ### Diagrama de fase de Van der Pol ($\mu = 1000$)
#
# Se usa un horizonte largo (`tf=1000`) y un paso moderado (`h=0.05`)
# con Crank-Nicolson para visualizar parte del ciclo límite de relajación.
# El período de relajación para μ=1000 es del orden de ~1.6·μ ≈ 1600,
# por lo que tf=1000 captura una fracción significativa del ciclo sin
# exceder el tiempo de cómputo.

# %%
tf_vdp_stiff_fase = 1000.0
h_fase_stiff = 0.05

resolutor_fase_stiff = EDOSolver()
solucion_fase_stiff = resolutor_fase_stiff.solve(
    lambda t, x, u: campo_van_der_pol(t, x, u, parametros_vdp_stiff),
    x0_vdp,
    (t0_vdp, tf_vdp_stiff_fase),
    h_fase_stiff,
    method="crank_nicolson",
)

caption_fase_vdp_stiff = (
    f"Calculado con Crank-Nicolson h={h_fase_stiff}, tf={tf_vdp_stiff_fase}. "
    "Parámetros: μ=1000, x₀=[2, 0]"
)

fig_fase_vdp_stiff = graficar_diagrama_fase(
    solucion_fase_stiff.estados,
    "Van der Pol (μ=1000): diagrama de fase",
    ruta_salida=RUTA_VDP / "diagrama_fase_mu_1000.png",
    caption=caption_fase_vdp_stiff,
)

# %% [markdown]
# ## Resumen de archivos generados

# %%
for ruta in sorted(RUTA_LOTKA.glob("*.png")):
    print(ruta)
for ruta in sorted(RUTA_VDP.glob("*.png")):
    print(ruta)

# %% [markdown]
# ## Problema 3a - Barrido hacia adelante y hacia atrás
#
# El método FBSM parte de $u^{(0)}=0$ y alterna la integración directa del
# estado, la integración hacia atrás del adjunto y la minimización puntual del
# Hamiltoniano. La actualización relajada usa
# $u^{(k+1)}=(1-\omega)u^{(k)}+\omega\widetilde u$ y se detiene cuando el cambio
# relativo de $J$ cae bajo la tolerancia. Ejecutamos la API implementada con el
# LQR escalar y mostramos $x(t)$, $u^*(t)$ y la historia de costo.

# %% [markdown]
# ## Problema 3b - Validación independiente mediante Riccati
#
# Para $\dot x=ax+bu$ y
# $J=\frac12 s x(T)^2+\frac12\int_0^T(qx^2+ru^2)\,dt$, la condición de
# estacionariedad entrega $u^*=-(b/r)p$. Al proponer $p(t)=P(t)x(t)$ se obtiene
#
# $$-\dot{P}=2aP-\frac{b^2}{r}P^2+q,\qquad P(T)=s.$$
#
# Si $d=\sqrt{a^2+b^2q/r}$ y
# $P_\pm=\frac{r}{b^2}(a\pm d)$, la solución analítica queda determinada por
#
# $$\frac{P(t)-P_+}{P(t)-P_-}
# =\frac{s-P_+}{s-P_-}\exp\!\left(2d(t-T)\right).$$
#
# La referencia `ProblemaLQR` integra esta Riccati independientemente. El lado
# FBSM usa un `ControlProblem` genérico cuyo minimizador depende del adjunto;
# así la comparación no reutiliza la solución de referencia. Se reporta el
# error en norma $L^2$ para varios pasos $h$ y se grafica en escala log-log.

# %% [markdown]
# ## Problema 3c - Vacunación óptima en el modelo SIR
#
# Se resuelve $\dot S=-\beta SI-uS$, $\dot I=\beta SI-\gamma I$ con
# $u\in[0,0.4]$ y costo $\int(AI+Bu^2/2)dt$. Para este problema no lineal se
# usa Crank-Nicolson y relajación fija `omega=0.2`, valor validado para evitar
# el ciclo de dos puntos que presenta la relajación general 0.99. Comparamos
# $A/B=10$ y $A/B=1$: penalizar más las infecciones debe aumentar la vacunación.

# %%
from src.reporte_problema3 import generar_reporte_problema3

RUTA_PROBLEMA3 = RUTA_BASE / "3_fbsm"
MODO_RAPIDO_PROBLEMA3 = os.getenv("TAREA1_REPORTE_RAPIDO") == "1"
resumen_problema3 = generar_reporte_problema3(
    RUTA_PROBLEMA3, modo_rapido=MODO_RAPIDO_PROBLEMA3
)

print("\nResumen del Problema 3")
print(f"3a: {resumen_problema3['3a']}")
print(f"3b: error L²(h=0.01) = {resumen_problema3['3b']['error_h_001']:.8e}")
print(
    "3c: control medio A/B=10 vs A/B=1 = "
    f"{resumen_problema3['3c']['control_medio_alto']:.6f} vs "
    f"{resumen_problema3['3c']['control_medio_bajo']:.6f}"
)

figuras_problema3 = resumen_problema3["figuras"]
if "ipykernel" in sys.modules:
    plt.show(block=False)
for figura in figuras_problema3:
    plt.close(figura)

# %%
for ruta in sorted(RUTA_PROBLEMA3.glob("*.png")):
    print(ruta)
