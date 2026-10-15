# Local field OCR regression images

These are unmodified pixel crops from the user's retained camera frames. They
are separate test fixtures so clearing application records does not erase the
regression cases. No credentials or camera connection information is included.

| Fixture | Source camera / sequence | Expected ID | Purpose |
| --- | --- | --- | --- |
| hlhu_boxed_digit.png | gate-in-3 / 18597 | HLHU8516751 | Slanted printed rectangle confused check-digit recognition |
| oneu_vertical.png | gate-out-2 / 16329 | ONEU7355689 | Perspective-compressed vertical text beside ISO markings |
| hlhu_door_bar.png | gate-in-2 / 18335 | HLHU8516751 | Door bar incorrectly included in owner/serial text |
| hlhu_distant_negative.png | gate-in-2 / 18365 | HLHU8516751 or unreadable | Reject a different checksum-valid identity from blurred text |

Run `GATE_GPU_REPLAY=1 .venv/bin/pytest tests/test_ocr_geometry.py -q` from
backend with the GPU library paths used by start.sh. GPU tests require existing
local EasyOCR weights and disallow model downloads. Confidence assertions are
recognizer outputs, not claims of overall system accuracy.
