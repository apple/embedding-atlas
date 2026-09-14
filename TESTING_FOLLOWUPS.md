# Testing follow-ups

No known functional blocker remains for the 73.6M-row Overture Places workflow. The following scenarios are useful future hardening work, but are not required for the current handoff:

- Add a proxy-based E2E test that closes a fixed-length Arrow response halfway through its body. The current recovery test resets the request before its response body starts, while the production connector now also validates and retries mid-body truncation.
- Run the large-dataset soak on a memory-constrained 16 GB machine or CI runner. The current 73.6M-row tests ran on a 128 GB development machine.
- Add the confidence-color → geocoder navigation sequence to the checked-in opt-in large-data Playwright suite. It is currently covered by the manual automated Chrome run against `places_08_2026.parquet`.
- Exercise the packaged desktop/Electron sidecar with this exact 73.6M-row file after the next desktop binary rebuild. The source server and Python wheel are covered now.
- Periodically verify the live Photon geocoder service separately; deterministic UI tests should continue mocking it to avoid third-party outages and rate limits.
