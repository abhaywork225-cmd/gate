# Duplicate label-fragment reconciliation — 15 October 2026

## Implemented

Text-only passages from the same camera can rejoin before finalization when independent accepted OCR frames identify the same complete number, the camera source/gate/lane/direction agree, and observations form a continuous chain within the configured observation-gap limit. A known cleared visit cannot rejoin a later one. Conflicting identities, measured vehicle/container tracks, same-frame-only readings and changed camera generations do not qualify.

The canonical visit ID is applied after OCR results so subsequent companion capture and jobs use the shared visit. Existing staging files are joined through the existing finalization path after pending OCR completes. Physical-body provenance is persisted conservatively for restart recovery. This is evidence-based reconciliation, not a global container-number/time deduplication rule.

## Verified

A full pipeline regression with two number markings, no detected body, and independent readings finishes with one container record and one folder. Negative tests cover different IDs, actual measured trucks, cleared/later visits, changed sources and duplicate observations of the same frame. A chain test covers continuous fragments whose total duration exceeds one observation-gap interval.

## Remaining field failure

A read-only metadata replay of the retained 36 OCR jobs reconstructed 25 intermediate passages. The conservative grouping still produced 11 groups, including unknown fragments. Some intermediate tracks contain both ECNU2257909 and TSSU2473645. Therefore this change does not establish that the original 15 saved records would become exactly two correct visits. Historical metadata lacks complete physical-body provenance; replay inferred text-only ownership only where the stored owner box equals the number box and no chassis was recorded.

A complete repair still needs physical track/identity transition handling and association of unreadable fragments without guessing their container ownership. The companion three-attempt limit also remains unchanged in this patch. Do not present this patch or synthetic tests as production readiness.

Existing event records, snapshots and cloud state were not modified. Application was not started.
