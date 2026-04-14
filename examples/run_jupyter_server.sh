#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Start JupyterLab for triton_inference_server_benchmark.ipynb
# (or any notebook in this directory).
#
# Usage (activate a Python environment with jupyterlab first):
#   cd /path/to/triton_benchmark
#   ./run_jupyter_server.sh
#
# The server listens on 127.0.0.1 (localhost only) by default.
# For remote access over SSH port-forwarding:
#   ssh -L 8888:localhost:8888 user@node
# Then open the printed http://127.0.0.1:8888/lab?token=... URL in a browser.
#
# Overrides:
#   JUPYTER_PORT=8890 ./run_jupyter_server.sh   # use a different port
#   JUPYTER_IP=0.0.0.0 ./run_jupyter_server.sh  # bind on all interfaces (use with care)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! python -c "import jupyterlab" 2>/dev/null; then
  echo "Installing jupyterlab into the current Python environment..."
  python -m pip install jupyterlab
fi

PORT="${JUPYTER_PORT:-8888}"
IP="${JUPYTER_IP:-127.0.0.1}"

if [[ "$IP" == "0.0.0.0" ]]; then
  echo "WARNING: JUPYTER_IP=0.0.0.0 binds JupyterLab on ALL network interfaces."
  echo "         This exposes the server beyond localhost, which is risky on shared"
  echo "         machines or when --network=host is in use."
  echo "         Prefer the default 127.0.0.1 and use SSH -L for remote access."
  echo ""
fi

echo "Starting JupyterLab from: $ROOT"
echo "  URL: http://127.0.0.1:${PORT}/lab (use SSH -L if remote)"
echo "  Stop: Ctrl+C"
echo ""

exec python -m jupyter lab \
  --no-browser \
  --ip="$IP" \
  --port="$PORT" \
  --notebook-dir="$ROOT" \
  "$@"
