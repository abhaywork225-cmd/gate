# Validation — 14 October 2026

Backend remains stopped. No cloud events were sent; existing evidence was preserved.

Inventory: 11 UNREADABLE events, 17 snapshot directories, 53 image files. Stored captures are dated 8 October, not a new live test on 14 October.

Inspected proof: `1c535df5-91f6-537c-9884-3666530e7731/gate-in-2_de3f6cd1-2f71-43ea-9c63-1319569f5829_proof.jpg`. Container lettering is visible. Its capture metadata labels the job as truck context, not a number detection. Runtime deliberately assigns UNREADABLE to context-only captures without OCR. This detector/targeted-recovery gap remains unresolved; passing unit tests is not a production accuracy claim.

Storage repair: deleting discarded/surplus OCR captures now removes empty staging ancestors without removing another camera's files. Regression covers two cameras sharing one staged visit.

Validation: 205 tests passed, 1 skipped. Live cross-camera identity, missed number detections, close-following vehicles and actual GPU OCR replay remain necessary before production sign-off.

Further repairs: a context-only job now gets one saved-vehicle detector retry; exactly one ROI-eligible number detection is cropped and persisted before normal checksum/confidence OCR. Multiple number boxes remain unreadable rather than selecting an unrelated winner. Real TensorRT GPU replay ran against a retained source: the stored vehicle box was 118x22 pixels and detected no number. Vehicle-first capture now rejects boxes smaller than 0.5% of the frame or 2% of frame height. This does not establish recovery of the visible container number in that historical example. Application remains stopped; no data was sent or historical events rewritten.
