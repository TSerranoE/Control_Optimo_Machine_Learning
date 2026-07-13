# Apply Progress: Problema 1 — EDO Solver (PR 1)

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

## Work Unit Evidence

| Evidence | Required value |
|---|---|
| Focused test command | `uv run pytest tarea1/tests/funcionalidad/test_integradores.py` |
| Exact result | `14 passed in ~0.4s` |
| Runtime harness command | `uv run python -c "from integradores import EDOSolver; s=EDOSolver(); print(s.solve(lambda t,x,u: -x, [1.0], (0,1), 0.01, 'euler').estados[-1])"` |
| Exact result | `0.366032341273 (error vs exp(-1) ≈ 0.0024, dentro de 0.05)` |
| Rollback boundary | Revertir commit del work unit o eliminar `tarea1/src/integradores.py` (dejar solo docstring inicial) y `tarea1/tests/funcionalidad/test_integradores.py` |

## Files Changed

| File | Action | What Was Done |
|------|--------|---------------|
| `tarea1/src/integradores.py` | Modified | Implementar `EDOSolution`, `EDOSolver`, `_validar_entradas`, `_construir_grilla`, `_euler`, `_resolver_integracion`, `_preprocesar_control` |
| `tarea1/tests/funcionalidad/test_integradores.py` | Created | Tests para EDOSolution, validación R5, grilla temporal y Euler progresivo |
| `openspec/changes/problema1-edo-solver/tasks.md` | Modified | Marcar tareas 1.1–1.11 como completadas |

## Deviations from Design

None — implementation matches design for Phase 1 scope.

## Issues Found

None.

## Workload / PR Boundary

- Mode: stacked PR slice (PR 1 of 3)
- Current work unit: Unit 1 — EDOSolution + validación + grilla + Euler progresivo
- Branch: `feat/problema1-edo-solver-pr1`
- Boundary: Phase 1 tasks only; Heun/RK4/implícitos/convergencia out of scope
- Estimated review budget impact: 435 insertions in single commit (slightly above 400-line ideal but aligned with forecast 380–450 and requested PR 1 scope)

## Status

11/11 Phase 1 tasks complete. Ready for verify or next PR slice.
