#!/usr/bin/env bash
set -Eeuo pipefail

# One-command Jetson/Ubuntu setup for GateVision.
# Run from the repository root: bash setup_edge.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo; echo "==> $*"; }

[[ "$(uname -s)" == "Linux" ]] || die "This script is for Ubuntu/Linux only."
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python3 is required."
command -v apt-get >/dev/null 2>&1 || die "apt-get is required on the edge device."
"$PYTHON_BIN" - <<'PY' || { echo "ERROR: Python 3.11+ is required for GateVision." >&2; exit 1; }
import sys
if sys.version_info < (3, 11):
    raise SystemExit(1)
print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
PY

if [[ ! -f /etc/nv_tegra_release ]]; then
  echo "WARNING: /etc/nv_tegra_release was not found. Continuing as generic Ubuntu."
else
  echo "Jetson platform: $(head -n 1 /etc/nv_tegra_release)"
fi

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages."
  SUDO=sudo
fi

note "Installing system video/runtime packages"
$SUDO apt-get update
$SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3-venv python3-dev python3-opencv \
  gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

note "Creating the project virtual environment"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  # system-site-packages is intentional: Ubuntu's python3-opencv is normally
  # compiled with NVIDIA/GStreamer support, unlike pip's headless wheel.
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
PIP="$PY -m pip"
$PIP install --upgrade pip setuptools wheel

note "Installing Python dependencies without replacing system OpenCV"
REQ_TMP="$(mktemp)"
trap 'rm -f "$REQ_TMP"' EXIT
grep -v -E '^opencv-python(-headless)?([<>=!~]|$)' "$BACKEND_DIR/requirements.txt" > "$REQ_TMP"
$PIP install -r "$REQ_TMP"

if [[ -f "$BACKEND_DIR/requirements-ai.txt" ]]; then
  echo "AI packages are not installed blindly: Jetson PyTorch/Paddle wheels must match JetPack."
  echo "Set TORCH_WHEEL_URL and/or PADDLE_WHEEL_URL if your device vendor supplied wheels."
  if [[ -n "${TORCH_WHEEL_URL:-}" ]]; then
    $PIP install "$TORCH_WHEEL_URL"
  fi
  if [[ -n "${PADDLE_WHEEL_URL:-}" ]]; then
    $PIP install "$PADDLE_WHEEL_URL"
  fi
  $PIP install "ultralytics>=8.3,<9" "paddleocr>=3.7,<3.8" || \
    echo "WARNING: AI Python packages need Jetson-specific wheels; base setup continues."
fi

note "Verifying the actual interpreter used by start.sh"
"$PY" - <<'PY'
import importlib.util
import sys

print("Python:", sys.executable)
print("OpenCV:", end=" ")
try:
    import cv2
    print(cv2.__version__)
    info = cv2.getBuildInformation()
    gst = "GStreamer" in info and "YES" in info.split("GStreamer:", 1)[1].splitlines()[0]
    print("OpenCV GStreamer:", "YES" if gst else "NO")
    if not gst:
        print("WARNING: OpenCV has no GStreamer support; RTSP will use CPU/FFmpeg.")
except Exception as exc:
    print("MISSING", type(exc).__name__, exc)

print("Torch:", end=" ")
try:
    import torch
    print(torch.__version__, "CUDA=" + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("MISSING", type(exc).__name__, exc)

for name in ("ultralytics", "paddleocr"):
    print(name + ":", "installed" if importlib.util.find_spec(name) else "MISSING")
PY

note "Preparing runtime directories and environment"
mkdir -p "$BACKEND_DIR/data" "$BACKEND_DIR/snapshots" "$BACKEND_DIR/uploads" "$BACKEND_DIR/logs"
if [[ ! -f "$BACKEND_DIR/.env" && -f "$BACKEND_DIR/.env.example" ]]; then
  cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
  echo "Created backend/.env from backend/.env.example"
fi

if [[ -f "$BACKEND_DIR/models/best.engine" ]]; then
  echo "TensorRT engine found: backend/models/best.engine"
  echo "Set MODEL_BACKEND=tensorrt and MODEL_PATH=models/best.engine in backend/.env"
else
  echo "WARNING: no TensorRT engine found. Do not expect production FPS until one is built on this Jetson."
fi

note "Setup complete"
echo "Start the application with:"
echo "  ./start.sh"
echo "or:"
echo "  cd backend && .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1 --no-access-log"
