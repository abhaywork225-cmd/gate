from __future__ import annotations
import math
from app.domain import Detection, Frame


class YoloDetector:
    def __init__(self, settings, model=None):
        self.settings = settings
        if model is None:
            expected = {'pytorch': '.pt', 'onnx': '.onnx', 'tensorrt': '.engine'}[settings.model_backend]
            if settings.model_path.suffix != expected or not settings.model_path.is_file():
                raise ValueError('Model file missing or backend does not match extension')
            from ultralytics import YOLO
            model = YOLO(str(settings.model_path), task='detect')
        self.model = model
        self.vehicle_model = None

    def detect_vehicles(self, frame):
        """Local auxiliary truck model for calibrated multi-lane views only."""
        if self.vehicle_model is None:
            from pathlib import Path
            path = Path(self.settings.passage_vehicle_model_path)
            if not path.is_file():
                for alt in (Path('models/yolov8n.pt'), Path('backend/models/yolov8n.pt')):
                    if alt.is_file():
                        path = alt
                        break
            if not path.is_file():
                return []  # No implicit model downloads on the edge device.
            from ultralytics import YOLO
            self.vehicle_model = YOLO(str(path), task='detect')
        model = self.vehicle_model
        classes = [i for i,name in model.names.items() if name in ('truck', 'trailer')]
        if not classes:
            return []
        results = model.predict(source=frame.image,classes=classes,
            conf=max(self.settings.confidence_threshold,self.settings.vehicle_presence_min_confidence),imgsz=self.settings.model_image_size,
            device=self.settings.model_device,verbose=False)
        output=[]
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                coords = tuple(float(v) for v in box.xyxy[0].tolist())
                if all(math.isfinite(v) for v in coords):
                    output.append(Detection(frame.camera_id,frame.timestamp,
                        str(result.names[int(box.cls.item())]),float(box.conf.item()),coords))
        return output

    def detect(self, frame: Frame) -> list[Detection]:
        device = self.settings.model_device
        if device != 'cpu':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('GPU detection requested, but CUDA is unavailable')
        results = self.model.predict(source=frame.image, conf=self.settings.confidence_threshold,
                                     imgsz=self.settings.model_image_size, device=device,
                                     verbose=False, rect=False)
        height, width = frame.image.shape[:2]
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                raw_name = str(result.names[int(box.cls.item())]).lower()
                name = self.settings.class_aliases.get(raw_name, raw_name)
                if name in ('class0', '0', 'container_object'):
                    name = 'container'
                elif name in ('class1', '1', 'cointainer'):
                    name = 'container_number'
                elif name in ('class2', '2', 'feet'):
                    name = 'iso_code'
                allowed = set(self.settings.target_classes) | {
                    'container', 'container_number', 'iso_code', 'truck', 'trailer', 'vehicle'
                }
                if name not in allowed:
                    continue
                confidence = float(box.conf.item())
                coords = [float(v) for v in box.xyxy[0].tolist()]
                if not all(math.isfinite(v) for v in coords + [confidence]):
                    continue
                x1, y1, x2, y2 = coords
                bbox = (max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2))
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    # Reject tiny slivers while retaining legitimate vertical
                    # number markings for rotated OCR and strict validation.
                    if name == 'container_number':
                        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        if bw < 32.0 or bh < 10.0:
                            continue
                    detections.append(Detection(frame.camera_id, frame.timestamp, name, confidence, bbox))
        return detections
