# Harsh test report — 2026-10-02

## Test suites added

`backend/tests/test_harsh_conditions.py` covers ten release-gate checks:

- invalid/partial OCR cannot create a twin;
- ROI boundary and tracker-ID churn;
- two validated camera identities merge into one event;
- blank proof images fall back safely;
- cloud dispatch grace period is actually enforced;
- corrupted container numbers are rejected or corrected without being trusted as-is.

Run it from the backend directory:

```powershell
cd D:\ai-video-based-inventory\ai-video-based-inventory\backend
..\backend\.venv\Scripts\python.exe -m pytest tests/test_harsh_conditions.py -q
```

Result: **10 passed** (one pytest cache-permission warning only).

## Full regression result

Command:

```powershell
..\backend\.venv\Scripts\python.exe -m pytest tests -q
```

Result: **5 failures, 119 passed, 1 skipped**.

The failures are real production risks, not test infrastructure failures:

1. **Lane does not unlock after the configured clear interval** under forced
   tracker churn. A new truck after the clear window still produced only one
   event. This needs an explicit post-clear state transition test and fix.
2. **OCR proof cleanup changes the stored image bytes**. The sharp proof is
   selected, but the retained image is not byte-identical to the original
   frame. The proof pipeline needs a documented crop/encoding contract.
3. **Confirmation proof is not the exact sharp first frame** when a later
   frame is processed. This can explain degraded cloud proof images and must
   be fixed before production.

The earlier repository defect found by the new suite was fixed: `:front` no
longer implies twin, and a `:rear` origin with invalid/partial OCR cannot set
`is_twin`.

## Release gate recommendation

Do not call the system production-ready until the three regression failures
above are resolved and the complete suite is green. The newly added harsh
suite should run on every deployment build.
