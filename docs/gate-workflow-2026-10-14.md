# Flat shared visit folder

A container or number inside the configured ROI opens a visit. Fresh, enabled cameras of the same gate and direction contribute context frames to that single visit even without independent local detections. Offline/stale cameras and independent video-file clocks are excluded. Multiple ambiguous active visits are not populated with shared camera context automatically.

```
backend/snapshots/<visit-id>/
  gate-in-1_context_0.jpg
  gate-in-1_context_0.json
  gate-in-2_context_0.jpg
  gate-in-3_context_0.jpg
  gate-in-1_<sequence>_<sample>_frame.png
  gate-in-1_<sequence>_<sample>_crop.png
  gate-in-2_<proof-id>_proof.jpg
  event_<container-record-id>.json
  evidence_<container-record-id>.json
  visit.json
```

No camera or container subdirectories are created for new captures. Context snapshots use a three-frame rolling buffer per camera sampled at most once per second. Context metadata records the actual source, capture time and sequence. Number crops remain bounded by the OCR candidate pipeline. Retained container proofs stay isolated through container records, not folders.

Folder membership alone is not proof of a number. Only independently validated, owned OCR supplies cloud proof and confirmation. Independent frames of the same number across fragmented text regions in one visit now aggregate using unique camera/source/sequence keys. Repeated boxes from the same frame cannot inflate that vote count.

Restart preserves storage visit IDs. Proven late associations consolidate referenced files and context into the visit root without changing job states. Existing finalized historical folder layouts remain readable and are not reorganized automatically. No cloud sending or live backend startup is authorized by these implementation changes.

Number proof policy: production defaults require two distinct camera proofs per known container number. Early finalization waits for this coverage; upload building rejects a known number with fewer proofs. Final-capture preview uses the exact same source-image selection and bytes, even when sending is held, and the UI shows camera coverage. Unreadable departures preserve the explicit best-image fallback previously requested. Tests verify exact JPEG byte equality and one-camera hold. Validation: 222 backend tests passed, 1 skipped; Angular production build passed.
