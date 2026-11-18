# Data spike findings (2026-11-18)

Source details and URLs: [data/raw/README.md](../../data/raw/README.md).
Source letters: **A** Socrata airport-month T-100 · **B** TranStats T-100 Segment (route-level, manual) ·
**C** Delay Cause · **D** FAA enplanements · **E** OurAirports · **F** NAS Status.

## Headline
- **Only B has route-level rows** (origin, dest, distance, scheduled/performed departures, class).
  Socrata (A) is airport × month only. B needs a one-time manual download (steps in the README).
- Everything else is scriptable today: A (API), C (form + link), D, E, F (direct).
- Latest data: B and C → **Aug 2026**; A → **Jul 2026**; D → **CY2025**.

## KPI feasibility

| KPI | Feasible? | Source / fields | Gap → proposed fix |
|---|---|---|---|
| **Load factor** | ✅ Yes | A: `total_passengers / total_seats` (all-cargo flights have 0 seats and 0 passengers, so the ratio is not distorted). Better: B `PASSENGERS / SEATS` filtered to passenger service. | None. Use B when available so service class is explicit; A as fallback. |
| **Passenger growth** | ✅ Yes | D: `CY 25 Enplanements` vs `CY 24 Enplanements` (official). Or A/B monthly passengers → trailing-12 vs prior-12. | D is calendar-year, not trailing-12. **Fix:** use T-12 YoY from A/B as the scored metric, and show FAA CY growth as an official cross-check. |
| **Delay rate** | ✅ Yes | C: `arr_del15 / arr_flights` | Reporting carriers + domestic arrivals only (e.g. ANC international/cargo not covered). State it in caveats. |
| **NAS delay share** | ✅ Yes | C: `nas_delay / arr_delay` (five cause columns sum exactly to `arr_delay`) | Same scope caveat as delay rate. |
| **Completion gap** | ⚠️ Only with B | B: `1 − DEPARTURES_PERFORMED / DEPARTURES_SCHEDULED` | A has no scheduled departures. **Fix:** (1) do the B download; (2) fallback/cross-check = cancellation rate `arr_cancelled / arr_flights` from C. Verify after download that `DEPARTURES_SCHEDULED` is populated for all carriers (it may be 0 for non-scheduled service, so filter to scheduled passenger service). |
| **Long-haul share** | ❌ Blocked until B | B: `DISTANCE` per segment + `DEPARTURES_PERFORMED` and `SEATS` → share ≥ 3,000 mi, seat-weighted share, sensitivity at 2,500 / 3,000 / 4,000 mi | A only gives an *average* distance per flight, which cannot produce a share. **Fix:** the B download. No public scriptable alternative found. |
| **Unmet demand proxy** | ✅ Once components exist | LF (A/B) + delay rate & NAS share (C) + growth (A/B/D) + completion gap (B or C) | Not directly observable (always say so). Works with C cancellation rate if B is not used. |
| **Investment score** | ✅ Yes | Components above + scale = D enplanements | None. Hub size from D `Hub`, state/region from D `ST` or E `iso_region`. |

Extra context available (not in KPIs yet): runway count / longest runway (E), FAA hub class (D),
live ground stops / delays (F, context only).

## Other findings that affect design
1. **Cargo at ANC.** A counts all-cargo departures (ANC is a major cargo hub), so departure-based metrics
   from A are misleading for ANC. For question 3 we need B filtered to passenger service.
2. **Different windows.** B/C end Aug 2026, A ends Jul 2026, D is CY2025. Proposal: standard window
   **Sep 2025–Aug 2026** (B + C). If B is unavailable, use **Aug 2025–Jul 2026** for A and C, and state it.
3. **A and D agree.** 2025 passengers from A are within ±1% of FAA CY2025 enplanements for all 5 test airports,
   so A is a valid fallback for passengers/seats.
4. **C coverage differs from T-100.** Delay data covers only large reporting carriers, so arrival counts are
   lower than T-100 departures (SNA: ~46k arrivals in C vs ~51k departures/yr in A).
5. **Fragile endpoints.** C (form behaviour) and F (no published API docs) may change. The ETL should fail
   with a clear message and point to the manual fallback in data/raw/README.md.

## To verify after the B download
- `CLASS` and `AIRCRAFT_CONFIG` code meanings (use the term-definitions file from the download;
  expected: passenger service classes vs all-cargo).
- `DEPARTURES_SCHEDULED` populated for the test airports.
- File size per year (to decide whether to commit a filtered extract or keep it local only).

## Decision needed from you
- **Do the B manual download** (3 zips, steps in data/raw/README.md). Without it, long-haul share
  (question 3) cannot be computed, and the completion gap falls back to the cancellation rate.
