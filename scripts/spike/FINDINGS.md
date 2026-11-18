# Data spike findings (2026-11-18)

Source details and URLs: [data/raw/README.md](../../data/raw/README.md).
Source letters: **A** Socrata airport-month T-100 · **B** TranStats T-100 Segment (route-level, manual) ·
**C** Delay Cause · **D** FAA enplanements · **E** OurAirports · **F** NAS Status.

## Headline
- **Only B has route-level rows** (origin, dest, distance, scheduled/performed departures, class).
  Socrata (A) is airport × month only. B needs a manual download (steps in the README) –
  2024, 2025 and 2026 (Jan–Aug) are downloaded and verified.
- Everything else is scriptable today: A (API), C (form + link), D, E, F (direct).
- Latest data: B and C → **Aug 2026**; A → **Jul 2026**; D → **CY2025**.

## KPI feasibility

| KPI | Feasible? | Source / fields | Gap → proposed fix |
|---|---|---|---|
| **Load factor** | ✅ Yes | A: `total_passengers / total_seats` (all-cargo flights have 0 seats and 0 passengers, so the ratio is not distorted). Better: B `PASSENGERS / SEATS` filtered to passenger service. | None. Use B when available so service class is explicit; A as fallback. |
| **Passenger growth** | ✅ Yes | D: `CY 25 Enplanements` vs `CY 24 Enplanements` (official). Or A/B monthly passengers → trailing-12 vs prior-12. | D is calendar-year, not trailing-12. **Fix:** use T-12 YoY from A/B as the scored metric, and show FAA CY growth as an official cross-check. |
| **Delay rate** | ✅ Yes | C: `arr_del15 / arr_flights` | Reporting carriers + domestic arrivals only (e.g. ANC international/cargo not covered). State it in caveats. |
| **NAS delay share** | ✅ Yes | C: `nas_delay / arr_delay` (five cause columns sum exactly to `arr_delay`) | Same scope caveat as delay rate. |
| **Completion gap** | ✅ Yes (with filter) | B: `1 − DEPARTURES_PERFORMED / DEPARTURES_SCHEDULED`, CLASS `F`, **rows with `DEPARTURES_SCHEDULED > 0` only** | Foreign carriers report `DEPARTURES_SCHEDULED = 0` (see below), so the naive ratio goes negative. With the filter the metric covers US carriers only – state it. Cross-check: cancellation rate from C. |
| **Long-haul share** | ✅ Yes | B: `DISTANCE` (statute miles, per segment) + `DEPARTURES_PERFORMED` and `SEATS`, passenger service only | Verified for ANC (below). |
| **Unmet demand proxy** | ✅ Once components exist | LF (A/B) + delay rate & NAS share (C) + growth (A/B/D) + completion gap (B or C) | Not directly observable (always say so). Works with C cancellation rate if B is not used. |
| **Investment score** | ✅ Yes | Components above + scale = D enplanements | None. Hub size from D `Hub`, state/region from D `ST` or E `iso_region`. |

Extra context available (not in KPIs yet): runway count / longest runway (E), FAA hub class (D),
live ground stops / delays (F, context only).

## Other findings that affect design
1. **Cargo at ANC.** A counts all-cargo departures (ANC is a major cargo hub), so departure-based metrics
   from A are misleading for ANC. For question 3 we need B filtered to passenger service.
2. **Different windows.** B/C end Aug 2026, A ends Jul 2026, D is CY2025. Agreed standard window:
   **Sep 2025–Aug 2026** (B + C). If B is unavailable, use **Aug 2025–Jul 2026** for A and C, and state it.
3. **A and D agree.** 2025 passengers from A are within ±1% of FAA CY2025 enplanements for all 5 test airports,
   so A is a valid fallback for passengers/seats.
4. **C coverage differs from T-100.** Delay data covers only large reporting carriers, so arrival counts are
   lower than T-100 departures (SNA: ~46k arrivals in C vs ~51k departures/yr in A).
5. **Fragile endpoints.** C (form behaviour) and F (no published API docs) may change. The ETL should fail
   with a clear message and point to the manual fallback in data/raw/README.md.

## T-100 Segment (B) verified – 2026-11-18
Files in `data/raw/t100_segment/` (git-ignored, all 50 form fields, one zip per year):

| Year | Rows | Months | Zip size |
|---|---|---|---|
| 2024 | 549,799 | 2024-01 .. 2024-12 | 18.5 MB |
| 2025 | 571,262 | 2025-01 .. 2025-12 | 19.6 MB |
| 2026 | 381,593 | 2026-01 .. 2026-08 | 13.4 MB |

The standard window **Sep 2025 – Aug 2026** is fully covered, plus the prior 12 months for YoY.

**Service class (`CLASS`)** – no term-definitions file was in the zip, so meanings are confirmed from
the data pattern (consistent with BTS service-class definitions):

| CLASS | Meaning | Evidence (2025) |
|---|---|---|
| F | Scheduled passenger / cargo service | 9.9M of 10.0M departures on passenger config; has scheduled departures |
| G | Scheduled all-cargo | 566k of 568k departures on freighter config (`AIRCRAFT_CONFIG = 2`) |
| L | Non-scheduled passenger / cargo (charter) | mostly passenger config; `DEPARTURES_SCHEDULED` always 0 |
| P | Non-scheduled all-cargo | mostly freighter config; `DEPARTURES_SCHEDULED` always 0 |

`AIRCRAFT_CONFIG`: 1 = passenger, 2 = freighter (≈0 seats), 3 = combi, 4 = seaplane (same evidence).

**Scheduled vs performed departures:** `DEPARTURES_SCHEDULED` is populated only for scheduled classes
(F, G) **and only for US carriers**: foreign carriers (e.g. Air Canada, Volaris, Air France) report 0.
In 2025, 66,778 class-F rows have scheduled = 0 but performed > 0 (838,659 departures).
Naive gap for LAX = −18.6% (wrong); with `DEPARTURES_SCHEDULED > 0`:

| Airport (CY2025, class F, US carriers) | Scheduled | Performed | Completion gap |
|---|---|---|---|
| ANC | 35,735 | 34,494 | 3.47% |
| BOS | 173,320 | 170,403 | 1.68% |
| LAX | 217,078 | 214,854 | 1.02% |
| SFO | 167,178 | 165,832 | 0.81% |
| SNA | 47,753 | 47,141 | 1.28% |

**Distance:** `DISTANCE` = segment distance in statute miles (2025: median 763, max 10,192); also `DISTANCE_GROUP`.

**Months covered:** 2024-01 .. 2026-08 (matches the form's "Latest Available Data: August 2026").

### ANC long-haul sanity check (`scripts/spike/anc_long_haul_check.py`)
Window **Sep 2025 – Aug 2026** (12 months). Passenger service only (CLASS F/L, not freighter).
All-cargo/freighter departures = **52.9%** of all ANC departures and are excluded – this is why
A (Socrata) alone would give a wrong answer for ANC.

| Threshold | Departures share | Seat-weighted share | Long-haul departures |
|---|---|---|---|
| ≥ 2,500 mi | 9.50% | 18.30% | 3,818 |
| **≥ 3,000 mi (default)** | **2.67%** | **5.17%** | 1,072 |
| ≥ 4,000 mi | 0.13% | 0.21% | 54 |

40,198 passenger departures performed. Top destinations ≥ 2,500 mi: MSP (2,519 mi), ORD (2,846),
PHX (2,552), HNL (2,777), DFW (3,043), ATL (3,417), IAH (3,266), IAD (3,356) – all US domestic.
The 3,000 mi line sits between MSP/ORD and DFW, so the answer is **very sensitive to the threshold**
(2.7% vs 9.5% at 2,500 mi). Earlier calendar-year runs for comparison: CY2024 2.34%, CY2025 2.78% at 3,000 mi.

## Agreed decisions (2026-11-18)
1. **Passenger service** = `CLASS in (F, L)` and `AIRCRAFT_CONFIG != 2`.
2. **Completion gap** = `1 − DEPARTURES_PERFORMED / DEPARTURES_SCHEDULED`, computed **only on rows with
   `DEPARTURES_SCHEDULED > 0`** (in practice US carriers). Every answer that uses it must state this limitation.
3. **Long-haul share** default threshold = **3,000 statute miles**, always shown with the
   **2,500 / 4,000 mi sensitivity table** and **both departure-weighted and seat-weighted** shares.
4. **Standard window** = trailing 12 months **Sep 2025 – Aug 2026** (B + C); stated in every answer.
