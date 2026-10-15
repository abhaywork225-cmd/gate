# Gate workflow and overnight diagnosis — 8 October 2026

The overnight audit found 45 events, 14 confirmed and 31 needing review. Fifteen events have more than one checksum-valid OCR candidate. A different checksum-valid candidate alone does not prove a different truck; OCR can also produce checksum-valid errors. In event 22977344-736c-4030-96cb-319e8eb325d6, visual inspection of all 19 images shows a red TGBU truck and a brown TCKU truck captured about 14 minutes apart. They must not share one event. All 2,738 saved image files across the 45 events passed decoding/integrity verification; this is not a claim that all text is readable.

## Workflow

1. Capture live frames from each enabled camera. Camera loss is an outage, never proof of truck departure.
2. Run vehicle/container/number detection. Restrict eligible text and occupancy to the calibrated gate ROI. The ROI must exclude parked and queued vehicles as far as the physical view allows.
3. Start a camera visit when eligible evidence enters the ROI. Assign a provisional event ID. Keep the visit while the same truck waits or creeps.
4. Associate companion cameras only when visit arrival times and recent measurements are concurrent. Continuous activity on an old camera visit is not a new arrival. The current association window is configurable (default 8 seconds); it requires physical-site validation. Timing alone cannot distinguish two trucks simultaneously visible in an overly broad ROI.
5. Admit one unfinished OCR capture per camera visit. Persist the source frame and text crop before OCR. Other detector frames update presence without saving additional OCR images or repeatedly scoring/hydrating the event.
6. Run configured GPU OCR on the detected text crop. Failed/partial reads may retry enhanced, expanded or segmented crops. These retries remain a measurable source of latency; the overnight median was 8.3 seconds, p95 28.3 seconds.
7. Validate observed text format and ISO 6346 checksum. Accumulate independent frame reads using the configured confidence and confirmation count. A checksum-valid read can still be a recognition error; do not invent digits or increase confidence.
8. Select a proof from actual source evidence supporting the chosen identity. Keep that proof and its number crop. After confirmation, cancel extra pending work and delete surplus completed captures. Retain audit metadata. Unresolved images remain available for further OCR.
9. Stop OCR capture for a completed camera visit, while detector presence/clearance tracking continues. A separate aligned candidate or observed boundary handoff can start a new visit without requiring an entirely empty broad scene. Two number markings require a common measured vehicle before text-based twin detection; sequential arrivals alone are not twin evidence.
10. Build UI proof preview and cloud payload from the same selection logic. Include a camera only when its own evidence supports the identity. Do not substitute an unrelated companion frame to fill a missing viewpoint.
11. Queue only confirmed, supported events under the current implementation. Local testing keeps cloud disabled. Confirmed capture readiness and successful cloud delivery are separate states; network retries must reuse the existing event ID. There is no elapsed 15-second timer converting incomplete OCR to a confirmed result. Incomplete reads remain review events; changing the cloud contract to submit explicitly unverified evidence requires a deliberate implementation, not relabelling the number as confirmed.

## Changes in this repair

- Compare visit start times as well as last measured activity during provisional camera association.
- Preserve capture-time job event IDs during delayed OCR processing; remove unrestricted current-live twin association.
- Prevent a rejected borrowed event ID from rejoining the same event through a primary-key collision fallback. Preserve repeated evidence from the same base origin.
- Allow completed text visits to hand off to spatially separate, aligned arrivals with a configured movement axis.
- Require common vehicle support for text-only twin candidates; sequential boundary arrivals start a fresh visit.
- Remove per-detector-frame evidence image scoring and database hydration.
- Release completed in-memory proof arrays and skip repeated database hydration for completed dispatch-controller visits.

## Validation limits

The historical mixed records are preserved for diagnosis and have not been automatically split or uploaded. The CSV audit lists every stored event and candidate identities. The user disconnected the cameras; that explains the present camera outage. Live capture accuracy, the concurrency window, next-truck separation and throughput must be verified with reconnected cameras. The repair does not recover trucks for which no source images were saved and is not production certification.

## Implemented vehicle workflow (subsequent update)

`VEHICLE_VISIT_TRACKING=true` is now the default. Truck/trailer detection and per-camera vehicle tracking feed a passage collector for ordinary ROIs as well as explicit lane polygons. OCR-admitted number regions retain capture-time ownership and three candidate frames each; one job per region is in flight. The tracker uses class/IoU continuity with a lightweight appearance gate to distinguish abrupt vehicle changes. This is not a trained multi-camera re-identification model and must be validated for similar-looking trucks and occlusions.

Matching identity evidence and concurrent unambiguous camera visits combine viewpoints. Different numbers are separated into deterministic container event IDs unless one measured truck supports both and both have supported 20-foot codes. Each region's spool jobs are linked only to its resulting record. Final records retain one supporting source proof per camera; cloud and local preview use the same final builder. The legacy global number/time duplicate guard cannot merge new tracked visits.

Complete OCR from all enabled participating gate cameras can finalize while the truck waits. Tracking then continues without retaining image arrays or admitting OCR work. Otherwise observed departure drains the saved evidence and finalizes. No detected number still retains a source context image and produces `UNREADABLE_CAPTURE`: number null, confidence zero and an explicit unreadable recognition/validation status. A checksum-valid final candidate with insufficient confirmation votes is explicitly unconfirmed (`CAPTURED`); it is not silently upgraded to confirmed. Remote acceptance of null container numbers must be checked against the actual cloud API before production sending; local tests only verify our payload contract.

`REVERSAL_GRACE_SECONDS=30` permits an unpublished closed passage to reopen. A published truck with measured reversing movement can also return during that window without another event. Direction comes from the calibrated clearance axis. A new truck can start independently while that grace window runs. This finite window is a documented operational limit, not a guarantee for arbitrary-length disappearance/re-entry. Camera outages do not advance measured departure.

Durable region IDs and visit IDs recover finalization after a restart. Reprocessing the same camera/source/frame cannot add another OCR vote. Event IDs are deterministic, and delivery retries reuse them.

Historical mixed events are preserved and are not converted to the new final policies or automatically uploaded. Production acceptance remains blocked on representative live traffic, night/occlusion/reversal tests and the cloud's unreadable schema acceptance. Camera disconnection prevents measuring live throughput in this session.

Final local validation: 194 backend tests passed, one skipped; Angular production build passed. The new number-only fallback captures eligible container markings even when vehicle detection misses, while prohibiting twin inference without chassis support. A later vehicle detection can retain that existing capture visit. Unreadable companion views of an otherwise identified single-container visit are not extra containers and are finalized without resurrection on restart.

The camera arrival window is separately configurable (`CROSS_CAMERA_ARRIVAL_WINDOW_SECONDS=30`); the recent-measurement window remains 8 seconds by default. Both start-time proximity and a recent active single observer are required for a provisional association. File-camera clocks are not implicitly fused. A vehicle measured outside the ROI cannot keep a visit alive solely because it remains elsewhere in the full camera image. These settings need calibration against real transport between viewpoints.


Capture storage repair: runtime now stages unfinished OCR captures under `backend/snapshots/_processing/<visit>/<camera>/`. Finalized retained source frames and crops move to `<event>/evidence/<camera>/`; selected upload proofs and manifests remain in the event directory. Recovery does not promote unfinished passage jobs or change PENDING to DONE. Older capture paths remain readable and are not deleted automatically.
