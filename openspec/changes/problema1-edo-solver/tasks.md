# Tasks: Problema 1 — EDO Solver

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 380–450 |
| 400-line budget risk | Medium |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 → PR 2 → PR 3 |
| Delivery strategy | auto-chain |
| Chain strategy | stacked-to-main |

Decision needed before apply: No
Chained PRs recommended: Yes
Chain strategy: stacked-to-main
400-line budget risk: Medium

### Suggested Work Units

| Unit | Goal | Likely PR | Focused test command | Runtime harness | Rollback boundary |
|------|------|-----------|----------------------|-----------------|-------------------|
| 1 | EDOSolution + validación + grilla + Euler progresivo | PR 1 | `uv run pytest tests/funcionalidad/test_integradores.py -k "edosolution or validacion or h_escalar or euler_solucion"` | `uv run python -c "from tarea1.integradores import EDOSolver, EDOSolution"` | `tarea1/src/integradores.py`, `tests/funcionalidad/test_integradores.py` (solo tests R1-R5) |
| 2 | Métodos explícitos (Heun, RK4) + control dual + intermediates | PR 2 | `uv run pytest tests/funcionalidad/test_integradores.py -k "heun or rk4 or control"` | `uv run python -c "from tarea1.integradores import EDOSolver; s=EDOSolver(); print(s.solve(lambda t,x,u: -x, [1.0], (0,1), 0.1, 'rk4').estados[-1])"` | `tests/funcionalidad/test_integradores.py` (tests R4, Heun/RK4) |
| 3 | Métodos implícitos + convergencia de órdenes | PR 3 | `uv run pytest tests/validacion/test_convergencia.py` | `uv run python -c "from tarea1.integradores import EDOSolver; s=EDOSolver(); print(s.solve(lambda t,x,u: -x, [1.0], (0,1), 0.1, 'euler_implicito').estados[-1])"` | `tests/validacion/test_convergencia.py`, implícitos en `integradores.py` |

## Phase 1: EDOSolution + Validación + Grilla

- [x] 1.1 **RED**: Crear `tests/funcionalidad/test_integradores.py`. Escribir `test_edosolution_atributos`: resolver con stub mock, verificar `.tiempos`, `.estados`, `.t_span`. Debe fallar (ImportError o atributos inexistentes).
- [x] 1.2 **RED**: Escribir `test_intermedios_desactivados` y `test_intermedios_activados`. Verificar `None` vs lista de dicts. Debe fallar.
- [x] 1.3 **GREEN**: En `tarea1/src/integradores.py`, crear `EDOSolution` dataclass con `tiempos`, `estados`, `t_span`, `intermedios=None`. Agregar `EDOSolver` con `METODOS` y `solve()` stub que valide y retorne `EDOSolution`.
- [x] 1.4 **RED**: Escribir parametrizado de 6 casos de validación inválida (R5): `test_validacion_x0_string`, `test_validacion_t_span_desordenado`, `test_validacion_t_span_unitario`, `test_validacion_h_cero`, `test_validacion_h_negativo`, `test_validacion_h_len_incorrecta`. Deben fallar.
- [x] 1.5 **GREEN**: Implementar `_validar_entradas()` con todas las reglas de R5. Despacho `ValueError` descriptivo.
- [x] 1.6 **RED**: Escribir `test_h_escalar` y `test_h_arreglo`. Verificar pasos uniformes y subintervalos. Deben fallar.
- [x] 1.7 **GREEN**: Implementar `_construir_grilla()` que expande `h` escalar a ndarray, genera tiempos y pasos. Integrar en `solve()`.
- [x] 1.8 **GREEN**: Implementar `_euler()` (Euler progresivo). Evaluar `f(t_k, x_k, u_k)`, calcular `x_{k+1} = x_k + h_k * f(...)`. Integrar dispatch en `solve()`.
- [x] 1.9 **RED**: Escribir `test_euler_solucion_analitica` (R1) y `test_metodo_invalido_valueerror` (R1). Deben fallar.
- [x] 1.10 **GREEN**: Ajustar Euler para pasar tests. Verificar `|x[-1] - exp(-1)| < 0.05`.
- [x] 1.11 **REFACTOR**: Extraer helper `_resolver_integracion()` para dispatch con dict de métodos. Extraer `_preprocesar_control()`.

## Phase 2: Métodos Explícitos + Control Dual + Intermedios

- [ ] 2.1 **RED**: Escribir parametrizado de control dual (R4): 4 combos `method × u_type` (RK4+callable, RK4+arreglo→ValueError, Euler+callable→UserWarning, Euler+arreglo→OK). Debe fallar.
- [ ] 2.2 **GREEN**: Implementar `_preprocesar_control()`: RK4 exige callable (ValueError si ndarray), otros pre-evalúan callable con `UserWarning`, ndarray se valida contra `len(tiempos)`.
- [ ] 2.3 **RED**: Escribir `test_heun_solucion_analitica` y `test_heun_intermedios`. Verificar precisión y almacenamiento de `z`. Deben fallar.
- [ ] 2.4 **GREEN**: Implementar `_heun()`: predictor `z = x_k + h_k * f(t_k, x_k, u_k)`, corrector con promedio de `f`. Almacenar `z` si `guardar_intermedios=True`.
- [ ] 2.5 **RED**: Escribir `test_rk4_alta_precision` (R1) y `test_rk4_intermedios`. Verificar `|x[-1] - exp(-1)| < 1e-10` y `k1..k4`. Deben fallar.
- [ ] 2.6 **GREEN**: Implementar `_rk4()`: 4 etapas `k1..k4`, evaluar `u(t_k)`, `u(t_k+h/2)`, `u(t_k+h)`. Almacenar `k1..k4` si `guardar_intermedios=True`.
- [ ] 2.7 **REFACTOR**: Unificar patrón de intermediates: cada método retorna `(estados, intermedios_paso)` tuple. Centralizar ensamblaje en `_resolver_integracion()`.

## Phase 3: Métodos Implícitos

- [ ] 3.1 **RED**: Escribir `test_euler_implicito_convergencia` (R6) y `test_argumentos_fsolve_custom`. Verificar convergencia y tolerancia custom. Deben fallar.
- [ ] 3.2 **GREEN**: Implementar `_euler_implicito()`: `fsolve(g, x0=x_k)` con `g(z) = z - x_k - h_k * f(t_{k+1}, z, u_{k+1})`. Usar `argumentos_fsolve` (default `{"xtol": 1e-8}`).
- [ ] 3.3 **RED**: Escribir `test_crank_nicolson_convergencia` (R6). Debe fallar.
- [ ] 3.4 **GREEN**: Implementar `_crank_nicolson()`: `fsolve(g, x0=x_k)` con `g(z) = z - x_k - (h_k/2) * [f(t_k, x_k, u_k) + f(t_{k+1}, z, u_{k+1})]`.
- [ ] 3.5 **REFACTOR**: Unificar implícitos: extraer helper `_resolver_implícito(g_residual, guess_inicial, argumentos_fsolve)` que ambos métodos reutilizan.

## Phase 4: Convergencia de Órdenes

- [ ] 4.1 **RED**: Crear `tests/validacion/test_convergencia.py`. Escribir `test_convergencia_euler_oh`: resolver `dx/dt=-x` con `h` y `h/2`, verificar `error_ratio ≈ 2 (±0.5)`. Debe fallar.
- [ ] 4.2 **GREEN**: Verificar que Euler pasa. Si no, ajustar tolerancias del test o implementación.
- [ ] 4.3 **RED**: Escribir `test_convergencia_heun_oh2`, `test_convergencia_cn_oh2`, `test_convergencia_rk4_oh4` con ratios esperados 4±1, 4±1, 16±4. Deben fallar.
- [ ] 4.4 **GREEN**: Verificar que cada método pasa su test de convergencia.
- [ ] 4.5 **REFACTOR**: Extraer helper `helper_error_ratio(f, x0, t_span, h_base, method, u, solucion_analitica)` compartido entre los 4 tests. Limpiar duplicación.
- [ ] 4.6 **REFACTOR**: Revisión final: verificar que todos los tests pasan con `uv run pytest`, revisar docstrings estilo NumPy, nombres declarativos, comentarios clarificadores. Confirmar que `tarea1/src/integradores.py` no supera ~220 líneas.
