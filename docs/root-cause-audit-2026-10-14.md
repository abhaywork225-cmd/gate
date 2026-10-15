# Field evidence audit — 14 October 2026

## Verdict

The application is not production ready. The requested gate-wide capture session is only partially implemented: per-camera tracks still own event creation. Folder sharing depends on those tracks already agreeing. Tests of isolated components did not establish that agreement on real footage.

The application was stopped during this audit. No cameras were started, records deleted, or cloud requests made. Offline GPU inference used existing local models and retained snapshots.

## Observed data

The current dataset contains nine events/visit roots: eight UNREADABLE and one CAPTURED. Seven ENTRY records were created from 10:58:01.795677 to 10:58:08.911101 UTC. Their evidence includes repeated views/readings of blue container HLHU8516751. Two EXIT records were created 0.413 seconds apart at 10:56:11 UTC.

There are 12 persisted OCR jobs; 11 source job images remain. Two retained job images are the same camera/sequence, so this is ten distinct retained source frames. One completed surplus frame was removed; a retained audit row does not mean its source image still exists.

## Reproduced failures

1. **Body detection is unreliable on these views.** GPU replay of all 11 retained job images produced zero `container` body detections. Auxiliary yolov8n produced a truck box in only two. The larger configured auxiliary weight is absent and the implementation falls back to the local nano weight. The original best.pt gives the same pattern as best.engine; switching backend alone is not the remedy. This sample does not measure all-day recall, but it directly exercises the failed captures.

2. **A number marking becomes a passage.** `PassageCollector.update` adds unmatched `container_number` boxes to the list of bodies. The IoU/colour tracker then tracks that small text rectangle. Movement, a second marking on the same container, and body/number class changes fragment one physical passage. In gate-in-1 sequence 18329, the horizontal and vertical numbers on one visible blue container received different passage IDs and folders: `15547d43-…` and `2df7375b-…`. Gate-in-3 independently created three visits as the same container moved across its view. Earlier body observations can be replaced by text-only geometry, losing useful tracking continuity.

3. **Cross-camera folder sharing happens too late.** It borrows a visit ID only when exactly one eligible other-camera visit exists. Fragmented local tracks make that ambiguous. `_collect_gate_camera_context` then explicitly returns when more than one active visit exists. Thus the first failure creates both the separate-folder symptom and missing companion captures.

4. **Dumping a camera image is not the same as OCR processing it.** Shared camera context uses three rotating JPEG slots per camera. Those images are context only; they do not automatically enter OCR. Number proofs still require that camera's independently assigned detection job. Therefore the implemented behaviour does not fulfil “all relevant camera snapshots go to one visit and are processed together.” Simply showing those context images as validated proof would conceal this defect.

5. **OCR has genuine recognition failures as well as an early-return defect.** On gate-in-3 sequence 18597, the GPU reader reproduced `HLHU 851675 1` at 0.066802 confidence. The segmented reader uses the weakest segment score. The enhanced image returned `HLHU 851675` at 0.996012, missing the check digit. High confidence on incomplete text is not high confidence in the complete ID. Standard reading without segmentation also missed the digit at multiple crop paddings. Gate-in-2's distant text and gate-in-1's vertical/angled markings failed in additional replay samples. Raising displayed confidence or manufacturing the check digit would not fix recognition.

6. **Fragmentation makes the workload worse.** Each new passage creates independent region budgets, image writes, and OCR work. The live implementation has one inference loop and one OCR worker; a powerful GPU does not remove duplicated work or guarantee sufficient per-camera cadence. Offline warm detector+auxiliary calls took roughly 0.09–0.12 seconds on these images; OCR variants took roughly 0.09–2.28 seconds. These are isolated replay timings, not concurrent live performance measurements.

## Change made in this audit

`GateRuntime._read_ocr` now attempts enhancement when the original complete/checksum-valid read is below the configured acceptance confidence. Previously it returned immediately, even though downstream code would reject that confidence. Candidate selection preserves the complete original if enhancement loses the check digit; it does not inflate confidence.

Targeted regression suite: 25 passed. Full backend suite: 223 passed, 1 skipped (39.08 seconds). This corrects retry eligibility, **not** the underlying low-confidence example or duplicate-visit architecture. GPU replay confirmed that simple enhancement alone did not repair that example.

## Required correction and validation

- Establish a gate-level visit owner before per-camera OCR jobs/folders, while maintaining distinct container evidence within a visit.
- Use measured container/body tracks with continuity through brief missed detections. Text boxes are evidence regions, not independent vehicle identities. Do not replace this with unconditional time-window merging, which would combine immediately following trucks.
- Validate or improve the container detector on these actual camera views. Current body detection cannot support the requested tracking guarantees.
- Associate each camera observation to a visit/container, then queue bounded best candidate images for that owner. Ambiguous observations must remain available for association instead of silently disabling shared capture.
- Final proof selection must preserve the exact camera/frame/number relationship and provide the same selected bytes to UI and cloud.
- Run synchronized sequence replays covering immediate followers, two visible trucks, repeated markings, stopping/reversing, twins, occlusion, and missing/low-confidence OCR. The retained stills prove the current defects but cannot establish continuous tracking correctness. No recorded video files were found in the inspected project/upload paths.

The existing mock-based tests remain useful regression checks, but are not evidence of production recognition accuracy. Several test defaults explicitly disable the active visit-sharing workflow. The next release requires field sequence acceptance results, not another “tests pass” readiness claim.

## Reproduction artifacts

- `backend/scripts/audit_live_evidence.py`: read-only source DB/snapshot detector replay; report path is explicit.
- `backend/data/capture-audit/field-audit-2026-10-14.json`: retained-image TensorRT and auxiliary results from this audit.
- `backend/data/capture-audit/ocr-field-audit-2026-10-14.json`: original/enhanced GPU OCR samples.

Camera on-screen dates differ from application timestamps. Association currently uses application capture timestamps; overlay dates alone do not prove a synchronisation defect.
