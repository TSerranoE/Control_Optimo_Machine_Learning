#!/usr/bin/env bash
set -euo pipefail

# Instala el hook de pre-commit que sincroniza notebooks antes de cada commit.
# Ejecutar una vez por clon del repositorio.

HOOK_SOURCE="scripts/pre-commit-hook"
HOOK_TARGET=".git/hooks/pre-commit"

if [ -f "$HOOK_TARGET" ]; then
    echo "[install-hook] Ya existe un pre-commit hook. No se sobrescribe."
    exit 0
fi

cp "$HOOK_SOURCE" "$HOOK_TARGET"
chmod +x "$HOOK_TARGET"

echo "[install-hook] Hook de pre-commit instalado en $HOOK_TARGET"
