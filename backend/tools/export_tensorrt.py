"""Export YOLO model to TensorRT engine optimized for Jetson AGX Orin."""
import argparse
from pathlib import Path
import sys


def export_model(model_path: str = 'models/best.pt', imgsz: int = 640, half: bool = True, device: int = 0):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Please install requirements first:")
        print("  pip install ultralytics")
        sys.exit(1)

    path = Path(model_path)
    if not path.is_file():
        print(f"Error: Model file '{path}' not found.")
        sys.exit(1)

    print(f"Loading model '{path}'...")
    model = YOLO(str(path))

    print(f"Exporting to TensorRT engine (imgsz={imgsz}, half={half}, device={device})...")
    print("Note: On Jetson AGX Orin, building the TensorRT engine may take several minutes.")
    engine_path = model.export(
        format='engine',
        imgsz=imgsz,
        half=half,
        device=device,
        workspace=4,  # GB
    )
    print(f"\nSuccessfully exported TensorRT engine: {engine_path}")
    print("To use this engine, update your .env file:")
    print("  MODEL_BACKEND=tensorrt")
    print(f"  MODEL_PATH={engine_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export YOLO PyTorch model to TensorRT engine for Jetson AGX")
    parser.add_argument('--model', default='models/best.pt', help="Path to input .pt model")
    parser.add_argument('--imgsz', type=int, default=640, help="Inference image size")
    parser.add_argument('--half', action='store_true', default=True, help="Use FP16 precision (recommended on Orin)")
    parser.add_argument('--device', type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    export_model(args.model, args.imgsz, args.half, args.device)
