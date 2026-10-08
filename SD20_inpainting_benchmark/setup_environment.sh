#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export PYTHONNOUSERSITE=1
mkdir -p "$ROOT/cache" "$ROOT/logs"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$PYTHON" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/requirements.txt" -c "$ROOT/requirements.lock.txt"
"$ROOT/.venv/bin/python" -m pip check
"$ROOT/.venv/bin/python" -m pip freeze > "$ROOT/requirements.installed.txt"
"$ROOT/.venv/bin/python" "$ROOT/check_environment.py"
