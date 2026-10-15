#!/usr/bin/env bash
set -e

HOST_ADDRESS="${1:-127.0.0.1}"
PORT="${2:-8001}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$PROJECT_ROOT/backend"
FRONTEND_ROOT="$PROJECT_ROOT/frontend"

# Determine Python executable
if [ -f "$BACKEND_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$BACKEND_ROOT/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Error: Python 3 not found." >&2
    exit 1
fi

# Ensure CUDA and TensorRT library paths are set for GPU acceleration
export LD_LIBRARY_PATH="/home/arctec/cuda_libs/dist/usr/local/cuda-11.4/targets/aarch64-linux/lib:/home/arctec/cuda_libs/dist/usr/lib/aarch64-linux-gnu:/home/arctec/cuda_libs/dist/usr/lib/aarch64-linux-gnu/openblas-pthread:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}"

# Ensure snapshots symlink exists
export OPENCV_FFMPEG_LOGLEVEL="-8"
export OPENCV_LOG_LEVEL="SILENT"
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000|probesize;262144|analyzeduration;1000000"
mkdir -p "$BACKEND_ROOT/snapshots"
if [ ! -e "$PROJECT_ROOT/snapshots" ]; then
    ln -s "$BACKEND_ROOT/snapshots" "$PROJECT_ROOT/snapshots" 2>/dev/null || true
fi

# Check frontend build
if [ ! -f "$FRONTEND_ROOT/dist/gate/browser/index.html" ]; then
    if command -v npm &>/dev/null; then
        echo "Building Angular frontend..."
        (cd "$FRONTEND_ROOT" && npm run build)
    fi
fi

# Clean up any previous uvicorn running for this backend
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 0.5

# Find available port
check_port() {
    "$PYTHON_BIN" -c "import socket; s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('$HOST_ADDRESS', $1)); s.close()" 2>/dev/null
}

CURRENT_PORT=$PORT
while ! check_port $CURRENT_PORT; do
    echo "Port $CURRENT_PORT is in use. Trying $((CURRENT_PORT + 1))..."
    CURRENT_PORT=$((CURRENT_PORT + 1))
    if [ $CURRENT_PORT -gt 8050 ]; then
        echo "Error: Could not find an open port." >&2
        exit 1
    fi
done
PORT=$CURRENT_PORT

echo ""
echo "======================================================================"
echo "                     GateVision AI is Starting                       "
echo "======================================================================"
echo "  Web Dashboard : http://$HOST_ADDRESS:$PORT"
echo "  API Docs      : http://$HOST_ADDRESS:$PORT/docs"
echo "======================================================================"
echo "Press Ctrl+C in this terminal to shut down the server."
echo ""

cd "$BACKEND_ROOT"
exec "$PYTHON_BIN" -m uvicorn app.main:app --host "$HOST_ADDRESS" --port "$PORT" --workers 1 --no-access-log
