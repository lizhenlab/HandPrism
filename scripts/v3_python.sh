#!/usr/bin/env bash
# Use the shared interpreter without changing its editable package install.
set -euo pipefail
v3_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
export PYTHONPATH="$v3_root/src:$v3_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd -- "$v3_root"
exec "$v3_root/.venv/bin/python" -B "$@"
