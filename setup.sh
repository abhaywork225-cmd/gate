#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$PROJECT_ROOT/backend"
FRONTEND_ROOT="$PROJECT_ROOT/frontend"

echo "======================================================================"
echo "          GateVision AI - Jetson AGX / Linux Setup                    "
echo "======================================================================"

# 1. Check Python
PYTHON_CMD="python3"
if ! command -v "$PYTHON_CMD" &>/dev/null; then
    echo "Error: python3 is not installed." >&2
    exit 1
fi

PY_VER=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python: $PY_VER"

# 2. Virtual Environment
VENV_DIR="$BACKEND_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

echo "Upgrading pip, setuptools, wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

# 3. Jetson hardware check
if [ -f "/etc/nv_tegra_release" ]; then
    TEGRA_REL=$(cat /etc/nv_tegra_release | head -n 1)
    echo "NVIDIA Jetson platform detected: $TEGRA_REL"
    if [ -f "/proc/device-tree/model" ]; then
        echo "Device: $(cat /proc/device-tree/model | tr -d '\0')"
    fi
fi

# 4. Install base requirements
echo "Installing backend requirements..."
"$VENV_PIP" install -r "$BACKEND_ROOT/requirements.txt"

# 5. Create directories & symlinks
mkdir -p "$BACKEND_ROOT/data"
mkdir -p "$BACKEND_ROOT/snapshots"
mkdir -p "$BACKEND_ROOT/uploads"
mkdir -p "$BACKEND_ROOT/logs"

if [ ! -e "$PROJECT_ROOT/snapshots" ]; then
    ln -s "$BACKEND_ROOT/snapshots" "$PROJECT_ROOT/snapshots" 2>/dev/null || true
fi

# 6. Initialize .env if missing
if [ ! -f "$BACKEND_ROOT/.env" ]; then
    echo "Creating backend/.env from .env.example..."
    cp "$BACKEND_ROOT/.env.example" "$BACKEND_ROOT/.env"
fi

# 7. Initialize Database
echo "Initializing database..."
(cd "$BACKEND_ROOT" && "$VENV_PYTHON" -m app.database.initialize)

# 8. Frontend build (if npm available)
if command -v npm &>/dev/null; then
    if [ ! -f "$FRONTEND_ROOT/dist/gate/browser/index.html" ]; then
        echo "Building Angular frontend..."
        (cd "$FRONTEND_ROOT" && npm ci && npm run build)
    fi
fi

echo ""
echo "======================================================================"
echo "                  Setup Complete!                                     "
echo "======================================================================"
echo "To run the application:"
echo "  ./start.sh"
echo ""
echo "To optimize AI inference for Jetson AGX Orin:"
echo "  1. Export YOLO model to TensorRT engine:"
echo "     $VENV_PYTHON backend/tools/export_tensorrt.py --model backend/models/best.pt"
echo "  2. Update backend/.env:"
echo "     MODEL_BACKEND=tensorrt"
echo "     MODEL_PATH=models/best.engine"
echo "======================================================================"
