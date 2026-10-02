# GateVision — Project Change Log

## 2026-10-02 — Grace period before cloud publication

- Confirmed events now wait 10 seconds before the cloud POST so late OCR from
  another calibrated camera can be merged into the same visit (especially for
  twin containers). The delay is configurable with
  `CLOUD_DISPATCH_GRACE_SECONDS`.

## 2026-10-02 — Identity-matched cloud proof

- Cloud proof selection now uploads only a snapshot tied to the validated OCR
  identity, camera and capture timestamp. Generic side/rear companion frames are
  excluded when they do not show the detected container.

## 2026-10-02 — Prevent false twins from one body and its text box

- A `container` detection and its nested `container_number` detection are now
  treated as one physical container. Text-only twin candidates require two
  separate, sufficiently large and confident `container_number` detections.

## 2026-10-01 15:35 IST — Fix YOLO container_object & Text Class Aliases

- **Model Label Normalization in YoloDetector**:
  - Identified root cause of missed container body detections on cameras (specifically `gate-out-2`): `runs/detect/gate_detector/weights/best.pt` uses label `container_object` for the physical container body and `Cointainer` for the text serial box.
  - In `backend/app/detection/yolo_detector.py`, added automatic label normalization mapping `container_object` and `cointainer` directly to `container`, and `feet` to `iso_code`.
  - Added explicit union with container and vehicle classes to ensure detections are never dropped due to class name alias mismatches.
- **Environment Configuration Alignment**:
  - Updated `CLASS_ALIASES={"cointainer":"container","container_object":"container","feet":"iso_code"}` in `backend/.env`.
  - Expanded `TARGET_CLASSES` to include `container_object`, `cointainer`, `container_number`, `feet`, and `iso_code`.
- **Verification**: Verified that a live container frame previously discarded is now immediately detected with 57.9% confidence as `Class: container`. All 24 backend test suites pass 100%.

## 2026-10-01 15:15 IST — Prevent Duplicate Passage Events and Preserve Valid Container Numbers

- **Deduplication Identity Preservation**:
  - In `backend/app/database/repositories/gate.py`, updated `number` extraction so valid container numbers with verified check digits are never blanked out to `None` during unconfirmed passes:
    `number = obs.ocr.normalized_text if (obs.confirmed or getattr(obs.ocr, 'valid_check_digit', False)) else None`
  - Ensures the 15-minute `SAME_CONTAINER_EVENT_WINDOW_SECONDS` (900s) query reliably matches existing passage records, eliminating duplicate `NEEDS_REVIEW` events for dwell tracks or stationary trucks.
- **Unconfirmed Dwell Upgrade Support**:
  - In `backend/app/events/event_manager.py`, updated `observe()` to allow incoming confirmed reads on existing dwell tracks to upgrade the database record from `NEEDS_REVIEW` to `CONFIRMED`.
  - Configured sample gating to await confirmation votes when a valid check digit is identified (`count >= 1`) rather than prematurely firing an unconfirmed event on sample 1.
- **Verification**: Cleaned and merged orphaned duplicate event `50fa6f38` into confirmed event `db286664` (`EGHU3934143`), verified 24 backend tests passing 100%.

## 2026-10-01 15:10 IST — Dual-Application Coexistence, Port 8002 Alignment, and Snapshot Publishing Fixes

- **Multi-Application Concurrent Execution**:
  - Configured `frontend/proxy.conf.json` to proxy `/api` requests to `http://127.0.0.1:8002` so the Angular UI connects directly to this workspace rather than pointing to the parallel project on port 8001.
  - Set default server port in `start.ps1` and `backend/start.ps1` to `8002` to avoid initial port collisions with other developers' applications running on port 8001.
  - Expanded `CORS_ORIGINS` in `backend/.env` to include ports `4201`, `4202`, and `8002`.
- **Snapshot Directory Junction Automation**:
  - Added automatic Windows directory junction creation in `start.ps1` linking `e:\project\snapshots` directly to `e:\project\backend\snapshots`, ensuring snapshots are visible from both root and backend workspace directories.
- **Occupancy & Unconfirmed Event Snapshot Publishing**:
  - Lowered minimum container bounding box area in `backend/app/events/occupancy.py` from `0.015` to `0.005` so wide-angle cameras and smaller lane perspective containers trigger occupancy candidates reliably.
  - Removed redundant early return in `backend/app/events/event_manager.py` that previously prevented unconfirmed passages (`status: NEEDS_REVIEW`) from saving snapshots when attempts were below max attempts. Now passages with completed entrances immediately record the event and write multi-camera snapshots.
- **Verification**: All 24 backend test suites passed 100%. Verified live database event generation, PaddleOCR output (`SUNIL TRANSPORT ALL INDIA PERALY`), and multi-camera JPEG snapshots written to disk.



- **OCR Delimiter-Tolerant Parsing & Multiline Stacking**:
  - Solved container validation failure where delimiter characters (`|`, `/`, `-`, `1`) between the 4-letter owner prefix and 6-digit serial number (e.g. `TRHU1 1545895`) caused ISO 6346 rejection. Added delimiter-tolerant verification that checks checksum after delimiter detachment, correctly identifying `TRHU1545895`.
  - Added multiline single-line normalization to parse vertically stacked container markings on doors and corner posts (e.g. `CSQU\n305438\n3`).
- **Eliminated False Size Code Overwrites**:
  - Removed faulty fallback that intercepted raw text containing size markings (`22G1`/`45G1`) and hijacked the normalized container text to `'22G'`, forcing valid reads into `VALID_SIZE_CODE` (`NEEDS_REVIEW`).
- **Fast-Passage Confirmation & Spool Path Resilience**:
  - In `EventManager`, added confirmation promotion when attempts are exhausted and at least one valid check-digit read exists (`count >= 1` and `not tied`), preventing fast-moving trucks with single sharp frames from degrading into empty `NEEDS_REVIEW` events.
  - Shortened temporary spool image filenames in `OCRJobStore` to prevent Windows `MAX_PATH` (260 char) path overflow errors.
- **Verification**: All 14 validator tests passed including new `test_delimiter_tolerant_container_read`. Verified on database sample `TRHU1 1545895` now correctly validating as `TRHU1545895` (`VALID`).

## 2026-10-01 13:10 IST — Multi-Instance Launcher, Dynamic Port Resolution, and OCR Pipeline Hardening

- **Dynamic Port Conflict Resolution & Multi-Instance Safety**:
  - Replaced aggressive process termination (`taskkill`, `Stop-Process`) in `start.ps1`, `setup_and_run.bat`, and `setup_and_run.ps1` with non-destructive port availability probing.
  - Automatically probes port `8001` (and scans up to `8010` if occupied) to dynamically select an open port, allowing parallel instances (e.g. secondary project runs on different drives) without killing existing workloads.
  - Added root launcher `start.bat` for streamlined Windows execution with integrated batch syntax error fixes (`setup_and_run.bat` unescaped parentheses fix).
- **OCR Engine Resilience & Offline Performance**:
  - In `backend/app/ocr/ocr_engine.py`, injected `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` to eliminate network lookup latency and timeouts on startup when the system is offline or air-gapped.
  - Hardened `FallbackOCREngine` against missing optional modules (e.g., `ModuleNotFoundError: No module named 'transformers'` when `OCR_FALLBACK_ENGINE=olmocr`), ensuring graceful fallback without crashing active workers or discarding evidence.
- **Lane Calibration & Gate Role Rectification**:
  - Corrected `gate-in-3` camera role assignment to `REAR` in camera selection and role configs, ensuring rear container ISO code recognition triggers properly on entry lanes.
  - Reset `occupancy_roi` in `backend/lane_calibration.json` to full-frame `[0.0, 0.0, 1.0, 1.0]` to eliminate restrictive bounding thresholds that prevented entry vehicle tracking.
- **Verification**: Tested `start.bat` launcher, verified smooth startup across concurrent port allocations, validated full-frame detection pipeline and OCR worker recovery.

## 2026-09-11 22:43 IST — Stationary occupancy and proof preservation

- Added viewpoint occupancy identity independent of tracker IDs, normalized drift/waiting classification and strict IoU reassociation (>0.45).
- Suppressed further OCR/event writes after confirmation, including queued samples, until measured clear time (default seven seconds) or a calibrated full-box exit crossing.
- Added active lane-event association beyond timed deduplication and separated verified new same-view passages from previous dwell events.
- Ranked saved proof candidates by sharpness/exposure; retained their corresponding coordinates/timestamps and prevented later stationary snapshots from replacing definitive proof.
- Protected active/pending/unconfirmed evidence from the existing completed-image cleanup. Exposed occupancy status through camera APIs and added configuration examples.
- Verification: **84 backend tests passed, 1 skipped**, including ten focused tests for long waits/ID churn, detection gaps, offline gaps, motion, source changes, departure, new passages, cross-view association, proof quality and queued OCR suppression. One existing test-client deprecation warning remains.
- Scope: runtime-local lock (not restored across service restart); calibrated single-zone viewpoint behavior. Hardware/video acceptance testing remains necessary. Existing unrelated working-tree edits were preserved.
- Timestamp basis: local implementation clock, rounded to the minute; uncommitted changes.

All times use **IST (Asia/Calcutta, UTC+05:30)**. Newest entries come first.

This is a manually maintained log, not an automatic file monitor. Add an entry with every future change: date/time, what changed, affected areas, verification, and commit ID when available. Git timestamps below are author timestamps, not proof of the exact time each file was edited. Earlier history is summarized from the available commits, not reconstructed minute by minute.

## 2026-09-10 15:45 IST — ISO Feet Size Code 'C' to 'G' Correction and UI Page Alignment Overhaul

- **ISO 6346 Size-Type Correction**: OCR models (EasyOCR, Qwen-VL) frequently misread the standard dry freight container type code 'G' as 'C' (e.g. `45C1` instead of `45G1`, `22C1` instead of `22G1`, `42C1` instead of `42G1`). Since ISO 6346 defines no container type 'C', added deterministic pattern correction `correct_feet_size_codes()` in `validator.py` and integrated into OCR reading pipeline.
- **Feet Size Extraction & Validation**: Added `parse_feet_size()` supporting ISO size-type codes (`45G1` -> `40 FT HC`, `42G1` -> `40 FT`, `22G1` -> `20 FT`, `L5G1` -> `45 FT HC`) and direct feet text (`40FT` -> `40 FT`). Marked validated size codes with `VALID_SIZE_CODE` status.
- **API & Database Integration**: Updated `GateRepository.event()` and `events()` to parse and surface `container_size` and `size_code` directly in event payloads.
- **UI Page Alignment & Layout Polish**:
  - Camera card dropdowns (`Camera source`, `Position`, `Gate direction`) given fixed label width guides (`105px`) so all inputs and dropdown arrows align on the identical vertical axis.
  - Standardized event filter row (`CONTAINER NUMBER`, `MOVEMENT`, and `Apply filters` button) to a uniform 36px control height with aligned baselines.
  - Added dedicated `Size / Feet` badge column to Recent Gate Events table (`📏 40 FT HC (45G1)`).
  - Enhanced Event Detail view with container size badge and evidence badges.
- **Verification**: All 13 validator tests passed (`test_feet_size_code_c_to_g_correction`), full backend suite passed, Angular production build succeeded, and browser subagent visually verified alignment across Overview, Events Table, and Detail pages.

## 2026-09-10 15:19 IST — Store OCR images in snapshot folders

- Changed pre-OCR storage from database image blobs to `snapshots/ocr-pending/<evidence-id>/crop.png` and `frame.png`.
- SQLite now stores only paths, metadata, OCR results and job status. Images are saved before the pending job is registered.
- Added startup migration for existing BLOB jobs, preserving IDs, validation, retries and status. Export failures roll back database changes; matching partial exports are reused on retry.
- Completed images remain on disk. Backups/retention must cover both folders and the metadata index; orphan files after interrupted registration are retained.
- Verification: **71 backend tests passed, 1 skipped**, including nine storage tests covering migration, rollback, partial writes, recovery and path containment. One existing test-client deprecation warning remains.
- Timestamp basis: local clock at review start, rounded to the minute. Migration runs on updated application startup; this entry does not claim the running application's data was already migrated.

## 2026-09-10 15:11 IST — Durable save-first OCR

- Added a transactional local SQLite OCR spool containing lossless crop/full-frame images and original camera/role/lane/timestamp metadata before OCR execution.
- Added a separate disk writer, disk-backed pending-job recovery, persisted validation/confirmation votes and retry-safe event completion.
- Allowed interval-selected crops to save independently of OCR speed, bounded by the configured per-track sample limit.
- Preserved historical job identity across source switches and restarts; stopped attaching unrelated current-time companion frames to delayed jobs.
- Added configurable retained-payload capacity (`OCR_SPOOL_MAX_MB`, default 10240 MB), spool counts and error reporting. Existing evidence is not automatically deleted.
- Limits: pre-commit staging remains volatile; queue saturation can drop unsaved samples; shared-GPU contention remains. Pending jobs require models and event database availability to finish.
- Verification: **68 passed, 1 skipped** in the backend suite, including recovery, pixel fidelity, capacity, failed-save retry, OCR retry, confirmation and post-event-commit crash tests. One existing test-client deprecation warning remains. Physical-camera and actual power-cut tests were not performed.
- Timestamp basis: local clock during implementation; uncommitted working-tree changes.

## 2026-09-10 14:50 IST — Latest-frame streaming and YOLO scheduling

- Replaced dashboard MJPEG connections with demand-driven WebSocket JPEG previews (one frame in flight per view); retained older preview endpoints for compatibility.
- Added origin validation, reconnect/cleanup behavior and development-proxy WebSocket support. Added and installed `websockets` 15.0.1 within the declared dependency range.
- CPU OCR no longer holds the YOLO accelerator lock. Scheduler leaves the latest frame pending while GPU OCR is busy, instead of processing a frame selected before the wait.
- New preview overlays use only the latest source-matched detection result with a 300 ms age limit, including empty results to clear old boxes.
- Removed buffer size from mandatory FFmpeg open parameters; retained best-effort post-open tuning.
- Documented that OCR reads in-memory crops **before** event snapshots are saved; GPU OCR can still block YOLO on a shared accelerator.
- Verification: full regression run 60 passed, 1 skipped; Angular build passed. All three targeted live-preview tests then passed, including eight simultaneous test connections with health still accessible. No physical-camera latency or throughput benchmark was possible.
- Timestamp basis: local clock during implementation, rounded to the minute; uncommitted changes.

## 2026-09-10 14:10 IST — Project documentation refresh

- Updated `brain.md` throughout to match eight views, MJPEG streaming, automatic ONVIF discovery, source preservation, saved operator roles, current APIs and configuration.
- Clarified that site/admin/cloud labels are static UI text, not authentication or delivery evidence.
- Clarified that the Prosper dispatcher is memory-only, not a durable outbox.
- Added this change log and linked it from the project brain.
- Scope: documentation only. Existing application changes were preserved.
- Verification: compared documentation with current source and Git history; checked Markdown changes for whitespace errors. Application tests were not rerun for this documentation-only update.
- Timestamp basis: local clock at the start of this documentation review, rounded to the minute. Status: uncommitted working-tree changes.

## 2026-09-10 — Automatic camera discovery and operator assignment

Exact edit start/end times were not recorded. Final frontend build completed at **14:02:01 IST** (tool timestamp `2026-09-10T08:32:01.939Z`); this is a build time, not a commit time.

- Added background ONVIF discovery at application startup and periodic retry (default: 60 seconds after a scan completes).
- Added ONVIF Media1 profile/RTSP URI negotiation using optional shared camera credentials, bounded discovery and same-host endpoint checks.
- Added discovered sources to camera dropdowns and automatically filled unused enabled views without overwriting saved source choices, configured RTSP inputs or active uploaded videos.
- Added saved position and IN/OUT/Unknown controls. Newly auto-assigned views remain Unassigned/Unknown until the operator chooses their role.
- Added discovery status, Search again, and camera setup warnings to the dashboard.
- Fixed MJPEG preview to follow the current worker after switching sources.
- Added configuration examples, role persistence and discovery tests.
- Main areas: `backend/app/camera/discovery.py`, `backend/app/runtime.py`, camera APIs/settings, Angular camera controls and backend tests.
- Verification: **58 tests passed, 1 skipped**; Angular production build passed. One existing test-client deprecation warning remained. Local ONVIF scan returned **0 cameras**, so live hardware behavior was not verified.
- Status at log creation: uncommitted working-tree changes; no commit ID yet.

## 2026-09-09 18:31:27 IST — Gate Module consolidation

- Commit: `5e5ab70`.
- Git subject: `feat: complete GateVision with container OCR, feet detection, multi-page UI, and event persistence`.
- Reviewed state includes eight IN/OUT views, MJPEG detection previews, event-detail navigation, OCR engine options, container-size labels, evidence persistence and Prosper cloud dispatch.
- Added the one-command `start.ps1` startup path and retired legacy application files in this commit.
- Timestamp basis: Git author date. Historical test results are not recorded here; the commit subject is not evidence of production readiness.

## 2026-08-21 11:10:58 IST — Model and training artifacts

- Commit: `e17e8f5`.
- Git subject: `feat: add trained YOLO models, TensorRT engines, and detection training runs`.
- Scope summarized from the commit subject; this does not imply all historical artifacts remain in the current tree.
- Timestamp basis: Git author date. Verification results not reconstructed.

## 2026-08-20 16:07:21 IST — Earlier four-camera baseline

- Commit: `5a17b62`.
- Git subject: `feat: complete GateVision 4-camera live AI surveillance & container OCR`.
- Historical four-camera baseline, superseded by the current eight-view configuration.
- Timestamp basis: Git author date. Verification results not reconstructed.

## Template for future entries

```markdown
## YYYY-MM-DD HH:mm:ss IST — Short change title

- Changed: what was added, fixed or removed and why.
- Areas: affected modules or documents.
- Verification: checks run, outcomes and untested limitations.
- Timestamp basis: actual edit/review time or Git author date.
- Status / commit: uncommitted, or the verified commit ID.
```
