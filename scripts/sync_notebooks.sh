#!/usr/bin/env bash
set -euo pipefail

# Sincroniza los notebooks emparejados (.py con formato py:percent y .ipynb)
# usando jupytext. Este script debe ejecutarse antes de commitear cambios en
# notebooks para garantizar que ambas representaciones estén alineadas.

echo "[sync-notebooks] Sincronizando notebooks emparejados..."

uv run jupytext --sync ma6914_tarea1/notebooks/*.py
uv run jupytext --sync ma6914_tarea2/notebooks/*.py

echo "[sync-notebooks] Sincronización completada."
