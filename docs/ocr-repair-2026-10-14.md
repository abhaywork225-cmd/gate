# OCR and short-dropout repair — 14 October 2026

## Implemented

- Rectify a measured printed check-digit rectangle before recognition. The previous fixed inset left skewed border strokes in the crop. Corner ordering now also handles narrow, steeply slanted rectangles.
- Select a unique vertical 11-component number column, excluding adjacent ISO markings. When the ordinary assembled line fails, deskew measured character contours and normalize their width before reading.
- Find complete horizontal owner/serial/check-digit layouts with padding. Exclude a tall door bar only when it spans the crop and is substantially taller than the characters. Reject incomplete layouts and multiple baselines.
- Require at least 0.8 measured segment confidence for the new horizontal layout path. A broader first implementation produced the wrong checksum-valid HLHU0516760 on a blurry crop; that regression was rejected and is now a GPU negative test. The algorithm does not calculate missing digits or insert an expected container ID.
- Preserve the recent measured body geometry during a brief number-only detector dropout. Reassociation requires containment, recent body evidence and appearance agreement. Text cannot renew the body timeout. This fixes a specific duplicate-visit mechanism; it does not replace the outstanding gate-wide association repair.

## Saved-image results

Offline GPU replay, original source pixels with 10-pixel crop padding:

| Source | Before | After |
| --- | --- | --- |
| gate-in-3 / 18597 | HLHU8516751, 6.7% | HLHU8516751, 96.1% |
| gate-in-2 / 18335 | HLHUV851675, invalid | HLHU8516751, 98.3% |
| gate-out-2 / 16329 | Garbled vertical text, invalid | ONEU7355689, 58.1% |
| gate-in-2 / 18365 | Invalid/blurred reading | Remains unaccepted; wrong checksum-valid alternative rejected |

At the deployed 0.4 acceptance threshold, 3 of 11 retained job crops pass syntax, checksum and confidence after repair, versus zero before, on this padded single-pass replay. The other eight remain unsuccessful. These are job crops, not independent truck counts or an overall accuracy estimate. Some source frames contain multiple number regions. Unpadded variants were also replayed, and the blurry negative case was checked specifically.

The two corrected HLHU readings come from different IN cameras, which is the required basis for two-camera number proof. This replay did **not** rewrite historical event records, reassign their folders, or upload images.

## Verification

- Explicit GPU geometry/recognition suite: 12 passed, including three positive field crops and the blurry negative field crop.
- Full backend suite: 233 passed, 5 skipped in 38.98 seconds. Four skips are the explicit GPU cases, which passed separately; one is an existing skip.
- Existing torchvision image-extension warning remains in the GPU environment; these tests use OpenCV image decoding and completed successfully.

Regression fixtures are in `backend/tests/fixtures/ocr/`, with source camera/sequence provenance. `backend/scripts/replay_ocr_jobs.py` replays retained source jobs read-only and writes only an explicit report path. Before/after raw outputs are in `backend/data/capture-audit/ocr-before-box-repair.json` and `ocr-after-box-repair.json`.

## Still outstanding

The full gate-level visit owner, reliable container/body tracking on the actual camera views, following-truck separation, and complete multi-camera capture association are not yet validated or complete. The new short-dropout guard is deliberately bounded by `gate_observation_gap_seconds`; it cannot solve long missing-body intervals or all text-only tracks.

A full-frame optical-flow experiment on sparse saved frames tracked static background features rather than container motion, so that approach was not added as a supposed fix.

Update, 15 October: the user clarified that these are live streams, without recordings. All cameras at a gate view the same passage from different sides, and their calibrated ROIs cover that shared area. Validate continuity and immediate followers using those live streams; recordings are not a prerequisite. Isolated historical snapshots cannot establish movement continuity.

Application remains stopped. No cloud data sent, no application records cleared, and no original source/proof images modified.
