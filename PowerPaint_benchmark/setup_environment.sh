#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REVISION=5b4c3d52291709fcec2a1870d987da693fd3549c
if [[ ! -d "$ROOT/vendor/PowerPaint/.git" ]]; then
  git clone https://github.com/open-mmlab/PowerPaint.git "$ROOT/vendor/PowerPaint"
  git -C "$ROOT/vendor/PowerPaint" checkout --detach "$REVISION"
fi
[[ "$(git -C "$ROOT/vendor/PowerPaint" rev-parse HEAD)" == "$REVISION" ]] || { printf '%s\n' 'Unexpected vendor revision; refusing to alter existing checkout.' >&2; exit 1; }
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/requirements.txt"
"$ROOT/.venv/bin/python" -m pip check
"$ROOT/.venv/bin/python" -m pip freeze --all > "$ROOT/requirements.lock.txt"
