# Apply Progress: Problema 1 — EDO Solver (PR 2)

## Estado de tareas

### Phase 1: EDOSolution + Validación + Grilla

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

### Phase 2: Métodos Explícitos + Control Dual + Intermedios

- [x] 2.1 **RED**: Escribir parametrizado de control dual (R4): 4 combos `method × u_type` (RK4+callable, RK4+arreglo→ValueError, Euler+callable→UserWarning, Euler+arreglo→OK). Debe fallar.
- [x] 2.2 **GREEN**: Implementar `_preprocesar_control()`: RK4 exige callable (ValueError si ndarray), otros pre-evalúan callable con `UserWarning`, ndarray se valida contra `len(tiempos)`.
- [x] 2.3 **RED**: Escribir `test_heun_solucion_analitica` y `test_heun_intermedios`. Verificar precisión y almacenamiento de `z`. Deben fallar.
- [x] 2.4 **GREEN**: Implementar `_heun()`: predictor `z = x_k + h_k * f(t_k, x_k, u_k)`, corrector con promedio de `f`. Almacenar `z` si `guardar_intermedios=True`.
- [x] 2.5 **RED**: Escribir `test_rk4_alta_precision` (R1) y `test_rk4_intermedios`. Verificar `|x[-1] - exp(-1)| < 1e-10` y `k1..k4`. Deben fallar.
- [x] 2.6 **GREEN**: Implementar `_rk4()`: 4 etapas `k1..k4`, evaluar `u(t_k)`, `u(t_k+h/2)`, `u(t_k+h)`. Almacenar `k1..k4` si `guardar_intermedios=True`.
- [x] 2.7 **REFACTOR**: Unificar patrón de intermediates: cada método retorna `(estados, intermedios_paso)` tuple. Centralizar ensamblaje en `_resolver_integracion()`.

## TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| 1.1 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | Written | Passed | 1 case | Clean |
| 1.2 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | Written | Passed | 2 cases | Clean |
| 1.3 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | Written | Passed | structural | Clean |
| 1.4 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | Written | Passed | 6 cases | Clean |
| 1.5 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | Written | Passed | 6 cases | Clean |
| 1.6 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | Written | Passed | 3 cases | Clean |
| 1.7 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | Written | Passed | 3 cases | Clean |
| 1.8 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | Written | Passed | single algorithm | Clean |
| 1.9 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | Written | Passed | 2 cases | Clean |
| 1.10 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | Written | Passed | tolerance check | Clean |
| 1.11 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | N/A (new) | N/A (refactor) | Passed | N/A | Clean |
| 2.1 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | ✅ 14/14 | Written | Passed | 4 combos | Clean |
| 2.2 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | ✅ 14/14 | Written | Passed | 4 combos | Clean |
| 2.3 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | ✅ 14/14 | Written | Passed | 2 cases | Clean |
| 2.4 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | ✅ 14/14 | Written | Passed | 2 cases | Clean |
| 2.5 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | ✅ 14/14 | Written | Passed | 2 cases | Clean |
| 2.6 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | ✅ 14/14 | Written | Passed | 2 cases | Clean |
| 2.7 | `tarea1/tests/funcionalidad/test_integradores.py` | Unit | ✅ 14/14 | N/A (refactor) | Passed | N/A | Clean |

### Test Summary

- **Total tests written**: 8 nuevos en PR 2 (4 control dual + 2 Heun + 2 RK4).
- **Total tests passing**: 22/22 en `tarea1/tests/funcionalidad/test_integradores.py`.
- **Layers used**: Unit (22).
- **Approval tests**: None — no refactoring tasks de código existente.
- **Pure functions created**: `_preprocesar_control`, `_heun`, `_rk4` operan sobre argumentos; la recurrencia temporal inevitablemente requiere estado interno.

## Work Unit Evidence

| Evidence | Required value |
|---|---|
| Focused test command | `uv run pytest tarea1/tests/funcionalidad/test_integradores.py` |
| Exact result | `22 passed in ~0.15s` |
| Runtime harness command | `PYTHONPATH=tarea1/src uv run python -c "from integradores import EDOSolver; import numpy as np; s=EDOSolver(); print(s.solve(lambda t,x,u: -np.asarray(x), np.array([1.0]), (0,1), 0.1, 'rk4').estados[-1])"` |
| Exact result | `[0.36787977]` (error vs `exp(-1)` ≈ `3.3e-7`) |
| Rollback boundary | Revertir los commits de PR 2, o eliminar `_preprocesar_control`, `_heun`, `_rk4` y su despacho en `tarea1/src/integradores.py`, más los tests de Phase 2 y `tarea1/tests/funcionalidad/conftest.py` |

## Files Changed

| File | Action | What Was Done |
|------|--------|---------------|
| `tarea1/src/integradores.py` | Modified | Implementar `_preprocesar_control` dual, `_heun`, `_rk4`; actualizar despacho en `_resolver_integracion`; simplificar `_euler` tras preprocesamiento |
| `tarea1/tests/funcionalidad/test_integradores.py` | Modified | Tests R4 control dual, Heun y RK4 (precisión e intermedios); refactor de tests existentes para usar `campo_lineal` |
| `tarea1/tests/funcionalidad/conftest.py` | Created | Fixture compartida `campo_lineal` |
| `openspec/changes/problema1-edo-solver/tasks.md` | Modified | Marcar tareas 2.1–2.7 como completadas |
| `openspec/changes/problema1-edo-solver/apply-progress.md` | Modified | Consolidar progreso de PR 1 y PR 2 |

## Deviations from Design

- `test_rk4_alta_precision` usa `h=0.01` en lugar del `h=0.1` del escenario R1 del spec para cumplir `|x[-1] - exp(-1)| < 1e-10`. Con `h=0.1` el error de RK4 estándar sobre `dx/dt=-x` es ~`3.3e-7`, por lo que la tolerancia `<1e-10` no es alcanzable. La implementación RK4 es la estándar de 4 etapas.
- `_euler` ahora recibe `control: np.ndarray | None` en su firma porque `_preprocesar_control` ya normaliza callable a ndarray para métodos no-RK4. El comportamiento es equivalente al diseño.

## Issues Found

None.

## Workload / PR Boundary

- Mode: stacked PR slice (PR 2 of 3)
- Current work unit: Unit 2 — Métodos explícitos (Heun, RK4) + control dual + intermedios
- Branch: `feat/problema1-edo-solver-pr2`
- Boundary: Phase 2 tasks only; métodos implícitos y convergencia de órdenes out of scope
- Estimated review budget impact: ~320 changed lines (within 400-line budget)

## Status

7/7 Phase 2 tasks complete. 18/18 total tasks tracked across PR 1 y PR 2 complete.
Ready for next batch (PR 3) or verify.
