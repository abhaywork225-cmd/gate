from __future__ import annotations
from typing import Protocol, Any
from multiprocessing import get_context
import os
import queue as queue_module
import sys
import threading
import types
import logging
from app.domain import OCRRead

log = logging.getLogger('gate')


class OCREngine(Protocol):
    def read(self, image: Any) -> OCRRead: ...
    def read_iso(self, image: Any) -> OCRRead: ...


def _paddle_text(payload: dict) -> tuple[str, float]:
    """Put Paddle's independently detected text boxes back into readable rows."""
    result = (payload or {}).get('res', payload or {})
    texts, scores, boxes = result.get('rec_texts') or [], result.get('rec_scores') or [], result.get('rec_boxes') or []
    rows: list[list[tuple[float, str, float, float]]] = []
    for index, text in enumerate(texts):
        value = str(text or '').strip()
        if not value:
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        box = boxes[index] if index < len(boxes) else [index * 100, index * 24, 0, 0]
        x = float(box[0]) if len(box) else float(index * 100)
        y = (float(box[1]) + float(box[3])) / 2 if len(box) >= 4 else float(index * 24)
        row = next((candidate for candidate in rows if abs(candidate[0][0] - y) <= 24), None)
        if row is None:
            rows.append([(y, value, score, x)])
        else:
            row.append((y, value, score, x))
    lines, values = [], []
    for row in sorted(rows, key=lambda candidate: candidate[0][0]):
        ordered = sorted(row, key=lambda item: item[3])
        lines.append(' '.join(item[1] for item in ordered))
        values.extend(item[2] for item in ordered)
    return '\n'.join(lines), (sum(values) / len(values) if values else 0.0)


def _paddle_worker(requests, responses, use_gpu: bool, detection_model: str, recognition_model: str):
    """Dedicated Paddle/CUDA process.

    PyTorch YOLO and Paddle link different CUDA/cuDNN builds on Windows. Keeping
    Paddle in this child process preserves GPU acceleration without destabilising
    the live PyTorch detector.
    """
    try:
        import site
        for package_dir in site.getsitepackages():
            dll_dir = os.path.join(package_dir, 'nvidia', 'cu13', 'bin', 'x86_64')
            if os.path.isdir(dll_dir):
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(dll_dir)
                os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
        import paddle
        # PaddleX imports ModelScope only to discover optional remote model
        # sources. Loading its PyTorch helpers would reintroduce the DLL clash.
        sys.modules['modelscope'] = types.ModuleType('modelscope')
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        from paddleocr import PaddleOCR
        device = 'gpu:0' if use_gpu and paddle.is_compiled_with_cuda() else 'cpu'
        paddle.set_device(device)
        engine = PaddleOCR(
            text_detection_model_name=detection_model,
            text_recognition_model_name=recognition_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        responses.put(('ready', True, device))
    except BaseException as exc:
        responses.put(('ready', False, f'{type(exc).__name__}: {exc}'))
        return
    while True:
        request = requests.get()
        if request is None:
            return
        request_id, image = request
        try:
            output = list(engine.predict(image))
            payload = output[0].json if output else {}
            text, confidence = _paddle_text(payload)
            # Handle vertical text crops (e.g. ECMU 5580439 on vertical corner post)
            if hasattr(image, 'shape') and len(image.shape) >= 2:
                h, w = image.shape[:2]
                if h > 1.25 * w or (not text.strip() and h > w):
                    try:
                        import cv2
                        rot_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        rot_out = list(engine.predict(rot_img))
                        rot_payload = rot_out[0].json if rot_out else {}
                        rot_text, rot_conf = _paddle_text(rot_payload)
                        if (not text.strip() and rot_text.strip()) or rot_conf > confidence:
                            text, confidence = rot_text, rot_conf
                    except Exception:
                        pass
            responses.put(('result', request_id, text, confidence, None))
        except BaseException as exc:
            responses.put(('result', request_id, '', 0.0, f'{type(exc).__name__}: {exc}'))


class PaddleOCREngine:
    """PP-OCRv6 primary reader backed by a persistent, GPU-only worker."""
    def __init__(self, settings):
        self._lock, self._sequence = threading.Lock(), 0
        self._context = get_context('spawn')
        self._requests = self._context.Queue(maxsize=1)
        self._responses = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=_paddle_worker,
            args=(self._requests, self._responses, bool(settings.ocr_gpu),
                  settings.paddle_ocr_detection_model, settings.paddle_ocr_recognition_model),
            name='gate-paddle-ocr', daemon=True,
        )
        self._process.start()
        try:
            kind, ready, detail = self._responses.get(timeout=90)
        except queue_module.Empty as exc:
            self.close()
            raise RuntimeError('PaddleOCR GPU worker did not start within 90 seconds') from exc
        if kind != 'ready' or not ready:
            self.close()
            raise RuntimeError(f'PaddleOCR GPU worker failed: {detail}')
        self.device = detail
        log.info('paddle_ocr_ready', extra={'device': self.device})

    def read(self, image) -> OCRRead:
        if image is None or getattr(image, 'size', 0) == 0 or not self._process.is_alive():
            return OCRRead('', 0.0)
        with self._lock:
            self._sequence += 1
            request_id = self._sequence
            try:
                self._requests.put((request_id, image), timeout=2)
                kind, returned_id, text, confidence, error = self._responses.get(timeout=20)
            except queue_module.Empty:
                log.warning('paddle_ocr_timeout')
                return OCRRead('', 0.0)
            if kind != 'result' or returned_id != request_id or error:
                log.warning('paddle_ocr_read_failed', extra={'error': error or 'unexpected response'})
                return OCRRead('', 0.0)
            return OCRRead(text[:4096], float(confidence))

    def read_iso(self, image) -> OCRRead:
        import re
        from app.ocr.validator import extract_iso_size_code
        result = self.read(image)
        code, _ = extract_iso_size_code(result.raw_text)
        if code:
            return OCRRead(code, result.confidence)
        compact = re.sub(r'[^A-Z0-9]', '', (result.raw_text or '').upper())
        match = re.search(r'(?<![A-Z0-9])([1-4LMN][0-9][GVRHBUTP][0-9A-Z])(?![A-Z0-9])', compact)
        return OCRRead(match.group(1) if match else '', result.confidence if match else 0.0)

    def close(self):
        if not getattr(self, '_process', None):
            return
        if self._process.is_alive():
            try:
                self._requests.put_nowait(None)
                self._process.join(timeout=5)
            except Exception:
                pass
        if self._process.is_alive():
            self._process.terminate()


class FallbackOCREngine:
    """Use the VLM only when the fast, deterministic reader has no useful text."""
    def __init__(self, primary: OCREngine, fallback_factory):
        self.primary, self._fallback_factory, self._fallback = primary, fallback_factory, None
        self._fallback_lock = threading.Lock()

    @staticmethod
    def _usable(result: OCRRead) -> bool:
        import re
        text = result.raw_text.upper()
        return bool(re.search(r'[A-Z]{3}[UJZ][\s-]*(?:[0-9][\s-]*){6}', text) or
                    re.search(r'[1-4LMN][\s-]*[0-9][\s-]*[GVRHBUTP][\s-]*[0-9A-Z]', text))

    def _secondary(self):
        with self._fallback_lock:
            if self._fallback is False:
                return None
            if self._fallback is None:
                try:
                    self._fallback = self._fallback_factory()
                except Exception as exc:
                    log.warning('ocr_fallback_unavailable', extra={'error_type': type(exc).__name__, 'error': str(exc)})
                    self._fallback = False
            return self._fallback if self._fallback is not False else None

    def read(self, image) -> OCRRead:
        primary = self.primary.read(image)
        if self._usable(primary):
            return primary
        secondary = self._secondary()
        return secondary.read(image) if secondary else primary

    def read_iso(self, image) -> OCRRead:
        primary = self.primary.read_iso(image)
        if primary.raw_text:
            return primary
        secondary = self._secondary()
        return secondary.read_iso(image) if secondary else primary

    def close(self):
        for engine in (self.primary, self._fallback):
            closer = getattr(engine, 'close', None)
            if closer:
                closer()


def _extract_text_from_items(items):
    if not items:
        return '', 0.0
    # Check vertical columns first
    cols = {}
    for it in items:
        xc = (it[0][0][0] + it[0][1][0]) / 2
        yc = (it[0][0][1] + it[0][2][1]) / 2
        matched_c = next((c for c in cols if abs(c - xc) < 55), None)
        if matched_c is None:
            matched_c = xc
            cols[matched_c] = []
        cols[matched_c].append(it)

    import re
    SIX_PREFIX = re.compile(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){6})(?![A-Z0-9])')
    from app.ocr.validator import CANDIDATE, STRUCTURE, correct_feet_size_codes
    for c in sorted(cols.keys()):
        chars = sorted(cols[c], key=lambda it: it[0][0][1])
        col_text = ''.join(str(it[1]) for it in chars)
        if STRUCTURE.search(col_text) or CANDIDATE.search(col_text) or SIX_PREFIX.search(col_text):
            combined = ' '.join(str(it[1]) for it in chars)
            min_conf = float(min(it[2] for it in chars)) if chars else 0.0
            return correct_feet_size_codes(combined)[:4096], min_conf

    # Standard horizontal reading order
    h_items = sorted(items, key=lambda it: (it[0][0][1] // 30, it[0][0][0]))
    combined = ' '.join(str(it[1]) for it in h_items)
    min_conf = float(min(it[2] for it in h_items)) if h_items else 0.0
    return correct_feet_size_codes(combined)[:4096], min_conf


class EasyOCREngine:
    def __init__(self, settings):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=settings.ocr_gpu,
                                     model_storage_directory=str(settings.ocr_model_directory),
                                     download_enabled=settings.ocr_download_enabled, verbose=False)

    def read(self, image) -> OCRRead:
        import re
        import cv2
        from app.ocr.validator import CANDIDATE, STRUCTURE
        SIX_PREFIX = re.compile(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){6})(?![A-Z0-9])')
        DIGIT_PREFIX = re.compile(r'(?<![A-Z0-9])([A-Z0-9][\s-]*[A-Z0-9][\s-]*[A-Z0-9][\s-]*[UJZ0-9](?:[\s-]*[0-9]){6,7})(?![A-Z0-9])')
        if image is None or getattr(image, 'size', 0) == 0:
            return OCRRead('', 0.0)

        h, w = image.shape[:2]
        # Avoid processing massive full 4K/1080p crops without downscaling
        proc_img = image
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            proc_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = proc_img.shape[:2]

        is_vertical = h > 1.5 * w

        if is_vertical:
            if w < 100:
                scale = min(3.0, 120.0 / max(1, w))
                proc_img = cv2.resize(proc_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            proc_img = cv2.copyMakeBorder(proc_img, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # 1. Fast standard 0-degree pass
        items = self.reader.readtext(
            proc_img, detail=1, paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            rotation_info=None,
            link_threshold=0.9 if is_vertical else 0.2
        )

        txt, conf = _extract_text_from_items(items)
        if STRUCTURE.search(txt) or CANDIDATE.search(txt) or SIX_PREFIX.search(txt) or DIGIT_PREFIX.search(txt):
            return OCRRead(txt, conf)

        # If not vertical and no valid code found yet, check vertical post strips
        if not is_vertical and w > 200:
            strips = [
                proc_img[:, int(w * 0.60):],
                proc_img[int(h * 0.08):int(h * 0.90), int(w * 0.35):int(w * 0.80)]
            ]
            for strip in strips:
                if strip.shape[0] < 50 or strip.shape[1] < 50:
                    continue
                items_strip = self.reader.readtext(
                    strip, detail=1, paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    link_threshold=0.9
                )
                txt_strip, conf_strip = _extract_text_from_items(items_strip)
                if STRUCTURE.search(txt_strip) or CANDIDATE.search(txt_strip) or SIX_PREFIX.search(txt_strip) or DIGIT_PREFIX.search(txt_strip):
                    return OCRRead(txt_strip, conf_strip)

        # For vertical crops: if padded read didn't find a candidate, retry without padding.
        if is_vertical:
            base_img = image
            if w < 100:
                scale = min(3.0, 120.0 / max(1, w))
                base_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            items2 = self.reader.readtext(
                base_img, detail=1, paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                link_threshold=0.9
            )
            txt2, conf2 = _extract_text_from_items(items2)
            if STRUCTURE.search(txt2) or CANDIDATE.search(txt2) or SIX_PREFIX.search(txt2) or DIGIT_PREFIX.search(txt2):
                return OCRRead(txt2, conf2)
            if len(txt2) > len(txt):
                txt, conf = txt2, conf2

        if not items and is_vertical:
            rot_cw = cv2.rotate(proc_img, cv2.ROTATE_90_CLOCKWISE)
            items = self.reader.readtext(
                rot_cw, detail=1, paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            txt, conf = _extract_text_from_items(items)

        return OCRRead(txt, conf)

    def read_iso(self, image) -> OCRRead:
        """Read only the four-character ISO size/type marking."""
        import re
        from app.ocr.validator import extract_iso_size_code
        result = self.read(image)
        code, _ = extract_iso_size_code(result.raw_text)
        if code:
            return OCRRead(code, result.confidence)
        compact = re.sub(r'[^A-Z0-9]', '', (result.raw_text or '').upper())
        match = re.search(r'(?<![A-Z0-9])([1-4LMN][0-9][GVRHBUTP][0-9A-Z])(?![A-Z0-9])', compact)
        return OCRRead(match.group(1) if match else '', result.confidence if match else 0.0)


class OlmOCREngine:
    """
    oLmOCR / Qwen2-VL Vision-Language OCR Engine.
    Specialized for container code extraction across complex orientations
    (vertical text, stacked characters, low-contrast stamps).
    """
    def __init__(self, settings):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model_name = getattr(settings, 'ocr_model_name', 'Qwen/Qwen2-VL-2B-Instruct')
        self.device = 'cuda' if (getattr(settings, 'ocr_gpu', True) and torch.cuda.is_available()) else 'cpu'
        self.torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32

        log.info(f'Loading oLmOCR (Qwen2-VL) model {self.model_name} on {self.device}...')
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        self.prompt = (
            "You are an industrial container OCR system. Read all container markings on this image: "
            "the 4-letter owner code (ending in U, J, or Z), 6-digit serial number, 1-digit check digit, and 4-character ISO size-type code (e.g. 45G1, 22G1). "
            "Output ONLY the letters and digits visible on the container separated by spaces. Do NOT invent or guess codes. If no container markings are clearly visible, output empty."
        )
        self.iso_prompt = (
            "Read only the four-character ISO 6346 container size/type code in this crop. "
            "Valid examples are 22G1, 45G1, 42G1, L5G1, and 22R1. "
            "Return exactly four visible letters/digits or EMPTY. Do not read the container number, "
            "timestamps, watermarks, truck numbers, or background text. Never guess hidden characters."
        )
        log.info(f'oLmOCR model loaded successfully on {self.device}.')

    def read(self, image, target: str = 'container') -> OCRRead:
        import cv2
        import re
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        if image is None or getattr(image, 'size', 0) == 0:
            return OCRRead('', 0.0)

        try:
            h, w = image.shape[:2]
            if max(h, w) > 768:
                scale = 768.0 / max(h, w)
                proc_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                proc_img = image

            rgb_image = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            prompt = self.iso_prompt if target == 'iso' else self.prompt
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': pil_image},
                        {'type': 'text', 'text': prompt}
                    ]
                }
            ]

            text_template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_template],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

            if not output_text:
                return OCRRead('', 0.0)

            # Strip conversational text if model produced extra narrative
            cleaned = output_text.replace('\n', ' ').replace('-', ' ')
            code_match = re.search(r'([A-Z]{3,4}(?:[\s-]*[0-9]){6,7}(?:[\s-]*[1-4LMN][0-9][A-Z0-9]{2})?)', cleaned.upper())
            if code_match:
                output_text = code_match.group(1).strip()
            else:
                output_text = cleaned.strip()

            if target == 'iso':
                # Do not synthesize missing characters or silently convert C to G.
                code = re.sub(r'\s+', '', output_text.upper())
                valid = bool(re.fullmatch(r'[1-4LMN][0-9][GVRHBUTP][0-9A-Z]', code))
                # This is an acceptance weight, not a calibrated probability.
                return OCRRead(code if valid else '', 0.6 if valid else 0.0)

            from app.ocr.validator import correct_feet_size_codes
            output_text = correct_feet_size_codes(output_text)

            has_letters = any(c.isalpha() for c in output_text)
            has_digits = any(c.isdigit() for c in output_text)
            if has_letters and has_digits:
                confidence = 0.95
            elif has_digits:
                confidence = 0.85
            else:
                confidence = 0.70

            return OCRRead(output_text[:4096], confidence)

        except Exception as exc:
            log.error(f'oLmOCR read error: {exc}', exc_info=True)
            return OCRRead('', 0.0)

    def read_iso(self, image) -> OCRRead:
        return self.read(image, target='iso')


def create_ocr_engine(settings) -> OCREngine:
    engine_type = getattr(settings, 'ocr_engine', 'easyocr').lower()
    if engine_type in ('paddleocr', 'paddle', 'ppocr'):
        primary = PaddleOCREngine(settings)
        fallback_engine = getattr(settings, 'ocr_fallback_engine', 'none').lower()
        if fallback_engine in ('olmocr', 'qwen2_vl', 'qwen2vl', 'qwen'):
            return FallbackOCREngine(primary, lambda: OlmOCREngine(settings))
        if fallback_engine == 'easyocr':
            return FallbackOCREngine(primary, lambda: EasyOCREngine(settings))
        return primary
    if engine_type in ('olmocr', 'qwen2_vl', 'qwen2vl', 'qwen'):
        return OlmOCREngine(settings)
    return EasyOCREngine(settings)
