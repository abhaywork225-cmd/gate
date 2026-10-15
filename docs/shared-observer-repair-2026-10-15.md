# Shared live-camera observer repair — 15 October 2026

The configured live cameras observe the same gate passage from different sides. They do not require recorded videos to participate in a visit.

## Changes

When one unambiguous live visit is active, fresh companion frames reserve camera membership in that visit before their own detector or OCR succeeds. This reservation has the same visit/folder ID, and cannot keep the visit alive after its measured members depart. A later single-target detection adopts the reservation. Multiple simultaneous targets do not adopt it or borrow a cross-camera visit by timing alone.

Companion ROI frames now enter the persisted background OCR job pipeline, with at most three independent attempts per observer. Previously they were only rotating context JPEGs and never entered OCR. Each attempt first detects a unique number region inside that camera's calibrated ROI. Multiple regions remain unresolved rather than selecting a confidence winner. Only successful independent recognition supplies number proof; context images do not inherit another camera's number. A failed context retry alongside actual number evidence does not create another unreadable container record.

Pre-visit cached frames are rejected, even when inside the normal freshness window. Existing rejection of stale frames, unrelated gate/direction cameras and independent file timelines remains. Explicit multi-lane views require a measured owner instead of automatic reservation.

## Validation and limits

The full backend suite passed after the implementation: 237 passed, 5 skipped. Added coverage includes pre-detection membership, later adoption, bounded retries, departure closure, simultaneous-target rejection and pre-visit stale-frame rejection. The separate proof-pipeline regression uses mocked recognizer output to exercise persistence and final payload building, not measure OCR accuracy.

This is a repair to shared capture and OCR scheduling, not completion of reliable physical visit tracking. Same-camera text-track fragmentation when container/body detection is absent, immediate followers, and reversal continuity still require further work and live validation. The historical detector audit found missing body detections on the retained frames; this change does not fix the detector itself. The three-attempt limit bounds work but cannot guarantee readable number capture.

No application restart, cloud upload, historical event rewrite or evidence deletion was performed. Existing GPU OCR geometry repairs remain unchanged.
