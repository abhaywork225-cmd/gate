# Gate capture audit — 7 October 2026

Gate confirmation now requires a stationary truck inside the calibrated gate ROI, repeated matching container readings, the configured OCR confidence threshold, and a valid ISO checksum. A visible number alone does not establish a gate visit.

## Coverage and findings

- Audited all 341 saved folders, decoded all 10,345 images: zero corrupt images.
- Revalidated all 3,262 stored OCR jobs. 875 previously accepted readings lacked complete checksum-valid observed text under strict validation. Older logic could synthesize missing characters or check digits.
- Compared historical boxes with the current calibration: 830 jobs outside the current ROI; 1,262 jobs marked nonstationary. Historical calibration may have differed, so these counts are diagnostic rather than labeled ground truth.
- Ran fresh GPU OCR on 854 representative crops across every folder, plus bounded context replay across every folder. Context expansion recovered 33 additional valid crop reads. 40 sampled folders had a stationary, in-ROI reading meeting the threshold; 18 had two matching such readings. These are sample evidence counts, not an accuracy percentage.
- Ran primary GPU detector on one saved source frame per folder: all 341 frames produced detections; 263 had an in-ROI number detection. This replay excludes the runtime auxiliary truck detector.
- Visually reviewed contact sheets covering every folder. Common failures include partial crops, container ribs, check-digit borders, blur, glare, and branding mistaken for identity text.

## Changes

- Gate ROI filtering before tracking and stationary-only OCR admission; removed outside-box association margin and whole-frame companion identity OCR.
- Strict observed-text ISO validation; no synthesized characters/check digits and no artificial confidence floor.
- Repeated matching reads required for confirmation; review candidates remain visible with reasons.
- Original crop OCR first, segmented owner/serial/printed-check-digit OCR, and bounded expansion clipped to gate ROI. Accepted expanded crop saved as proof while preserving the original.
- GPU requests fail explicitly if GPU is unavailable, rather than silently switching OCR/detection to CPU.
- Removed size-only twin inference; improved review upgrades and visit separation.
- Cloud payload identity is revalidated against actual OCR evidence; number confidence remains separate from detector/size confidence.

## Corrected requested event

`e0ac0b97-2ab3-4fa8-b79b-4bf1e04e9d6b`: fresh replay of all 29 saved jobs recovered `SEKU1486328` from three independent stationary, in-ROI frames. The local event is now CONFIRMED. Original failed observations were preserved. This correction was not sent to cloud.

## Validation and limits

Full backend regression suite: 153 passed, 1 skipped. Final targeted ROI/OCR storage suite: 19 passed. Frontend production build passed. Diff whitespace check passed.

Backend remains stopped for the user's restart; automatic cloud dispatch remains paused. A new live gate passage has not been tested. Saved images lack manually labeled ground truth, so this audit does not establish production accuracy or guarantee detection of every truck. Other historical events were not bulk rewritten or sent to cloud.

Machine-readable results: `backend/data/capture-audit/final-summary.json`, `replay-context.jsonl`, and `requested-event-correction.json`.


## Follow-up: premature stationary decisions and upload proof

Live dwell configuration previously used 0.5 seconds with 0.02 normalized centre drift (about 51 pixels at 2560 width). Updated local configuration to 1.5 seconds and 0.004 drift. Slot timers now reset after observation gaps; cumulative box scale changes over 5% also restart the stop timer. These thresholds still require a live gate trial and physical ROI calibration.

Cloud proof now requires the same camera and capture timestamp (50 ms tolerance) as a complete checksum-valid OCR observation meeting the confidence threshold. Unread companion frames are excluded before quality selection, independently for each container identity. Cloud preview uses the exact payload images. PNG source evidence is encoded as JPEG when labelled JPEG; original files remain intact.

The latest event 209496e2-ff60-4b39-bfaf-5436ce651dec exposed a narrow-glyph failure: the isolated printed 8 was read as 0. Correcting the narrow crop proportions recovered CMAU0623258 in all 17 saved crop reads on GPU; 16 met the 0.6 confidence threshold. This is a targeted regression result, not overall accuracy. Historical event was not automatically promoted or sent. Report: backend/data/capture-audit/latest-check-digit-replay.json.

Exposure API returned camera offline, so shutter settings were not changed. A sharper source may help blur, but it does not fix early stationary admission, incorrect ROI or narrow-glyph normalization.


## Implemented 15-second best-available policy

A per-visit monotonic timer is registered when a tracked object becomes eligible in the ROI. It is independent of OCR completion. Confirmed single-container evidence is eligible immediately; otherwise the timer saves the best observed slot reading and source frame at 15 seconds, including incomplete or absent numbers. It records BEST_AVAILABLE in an additive database column and queues delivery with no grace delay. The existing dispatcher enabled/automatic-dispatch controls remain authoritative; local cloud sending stays disabled.

Deadline payloads use the saved deadline observation and exact same camera/capture frame, actual OCR confidence and validation status. A missing digit is never synthesized. No OCR means an empty number with captured proof. Waiting visits are queued once; persistent cloud-synced flags and the BEST_AVAILABLE policy protect delivery retries/backlog. Local statuses distinguish CAPTURED from OCR CONFIRMED.

Validation: 163 backend tests passed, 1 skipped; frontend build passed. Dedicated deadline regressions cover partial OCR, missing OCR, matching proof and once-only queueing. Existing captures were backed up before the additive schema migration at backend/data/gate-before-deadline-policy.db. No cloud POST was performed. The 15 seconds is the local capture/queue deadline; remote receipt still depends on connectivity and the site's acceptance of partial/empty container numbers, which has not been tested while cloud is disabled.


## Follow-up: same-truck duplicate visits

Two Gate OUT 2 captures split one apparent orange-container passage while switching between a vertical number marking and an ISO-code box. Neither visit queued OCR because of stationary-only admission. Text markings can no longer trigger the separate-container handoff; raw truck presence intersecting the tracking ROI prevents text loss from clearing the visit. Cross-camera provisional visit association now runs once for each new view occupancy even though that view already generated a provisional ID, and can join a still-active confirmed view. Moving frames now enter OCR inside the ROI. Capture deadlines retain best observations per camera and slot; the payload selects matching, nonconflicting source proof per camera. Existing duplicate records were preserved rather than relabelled as a new successful passage.

Regression coverage includes two-camera provisional association, retained occupancy during lost text, moving-frame OCR, once-only deadline queueing and per-camera proof. A new live passage is still required; this does not establish production accuracy.

## Accuracy workflow update

The user superseded the 15-second dispatch requirement. Elapsed time no longer
finalizes an unreadable or partial visit. Source-bound OCR improvements are
persisted throughout an unsent visit; a valid number is retained when later reads
are worse. Automatic publication requires the event's independent confirmation
and matching validated source evidence. Lane-passage capture continues to use its
own passage lifecycle. Cloud testing remains disabled; backend restart is needed
to load the changes. Event 7d3909d0-f854-42bc-a05f-0002856e92ca was repaired from
saved job 6 (raw OCR T C K U 6 8 3 6 5 2 9, confidence 0.94261), preserving the
source evidence and a database backup. This is not a production accuracy claim.

## Visit association recheck

Event 9a0aa479-1861-4e06-a7c7-014920434523 incorrectly acquired MSKU7519800
from gate-in-1 at 12:01:02 UTC, after its FCIU8060114 capture at 11:44:17 UTC.
The evidence frame shows a different truck. A database backup was retained and
the unsupported second identity removed from the event fields; audit evidence
was retained. Boundary-filtered detections now update the observed departure
position so the next number can start a fresh visit. An unconfirmed rear-labelled
read cannot promote a single event into a twin. Independent confirmation is
required for repository twin fusion. Live acceptance remains pending; port 8001
was not listening during this recheck.
