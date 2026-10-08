#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install pip==24.3.1 wheel==0.45.1 setuptools==69.5.1
.venv/bin/python -m pip install -r requirements.txt
if [ ! -d vendor/BrushNet/.git ]; then
    git clone https://github.com/TencentARC/BrushNet.git vendor/BrushNet
    git -C vendor/BrushNet checkout --detach 0f9d9e54ca85c40a11a8f0504b4b5b2e7e8fd14d
fi
revision="$(git -C vendor/BrushNet rev-parse HEAD)"
if [ "$revision" != 0f9d9e54ca85c40a11a8f0504b4b5b2e7e8fd14d ]; then
    printf '%s\n' 'BrushNet revision mismatch; not overwriting existing checkout.' >&2
    exit 1
fi
.venv/bin/python -m pip install --no-deps --no-build-isolation -e vendor/BrushNet
.venv/bin/python -m pip check
