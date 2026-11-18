# Raw data provenance

Raw files in this folder are **git-ignored** (only this README is committed).
All sources below were verified on **2026-11-18** with the probe scripts in `scripts/spike/`.
Test airports used for every probe: BOS, LAX, SNA, ANC, SFO.

## Summary

| ID | Source | URL | Access | Grain | Coverage (latest) | Works? |
|---|---|---|---|---|---|---|
| A | BTS Socrata: *AFF - T100 Segment Summary By Origin Airport* (`r495-tyji`) | https://data.bts.gov/resource/r495-tyji.json | API (SODA, no key) | origin airport × month | 2014-01 → **2026-07** | Yes |
| B | BTS TranStats: *T-100 Segment (All Carriers)* | https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4 | **Manual download** (ASP.NET postback form) | carrier × origin × dest × aircraft × class × month | 1990 → **2026-08** | Yes – 2024, 2025, 2026 (Jan–Aug) downloaded 2026-11-18 |
| C | BTS TranStats: *Airline On-Time Delay Cause* | https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp | Scripted: form POST (period) → GET of the page's "Download Raw Data" link → zip | carrier × arrival airport × month | 2003-06 → **2026-08** | Yes |
| D | FAA Passenger Boarding (Enplanements), commercial service airports | CY2025: https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/arp-cy2025-commercial-service-enplanements.xlsx  ·  CY2024: https://www.faa.gov/sites/faa.gov/files/2026-09/arp-cy2024-commercial-service-enplanements.xlsx | Direct download (.xlsx) | airport × calendar year | **CY2025** (522 airports) | Yes |
| E | OurAirports | https://davidmegginson.github.io/ourairports-data/airports.csv  ·  `.../runways.csv` | Direct download (CSV) | airport / runway | updated nightly (Last-Modified 2026-11-18) | Yes |
| F | FAA NAS Status (live) | https://nasstatus.faa.gov/api/airport-events (JSON)  ·  https://nasstatus.faa.gov/api/airport-status-information (XML) | Live HTTP GET, no auth | active event per airport | real time | Yes |

## Details per source

### A. BTS Socrata (data.bts.gov)
- Catalog search (`q=T100`) returns only **aggregated** T-100 datasets:
  `bu82-4pwz` (national), `jqx4-4iha` (national monthly), `q4tb-tbff` (by carrier),
  `56rv-9p75` (by country), `r495-tyji` (by origin airport), `3xj5-daif` (preliminary national estimates).
  `https://data.bts.gov/data.json` (420 datasets) lists nothing more granular.
- **No route-level dataset exists on Socrata.** The finest grain is origin airport × month (`r495-tyji`).
- `r495-tyji` fields: `origin_airport_code, reporting_month, total_departures, total_passengers,
  total_seats, total_load_factor, total_distance_flight_sm` (average), `total_freight_lbs`, plus
  `domestic_*`, `outbound_international*`, `inbound_international*` splits.
- **Missing:** destination, per-route distance, departures *scheduled*, service class, aircraft config.
  Includes all-cargo flights (ANC: 520M lbs freight in Dec 2025) – departures are inflated at cargo hubs;
  passengers/seats are not affected.
- Sanity check: 2025 passenger sums match FAA CY2025 enplanements within ±1% for all 5 test airports.

### B. BTS T-100 Segment (All Carriers) – MANUAL DOWNLOAD
- Fields offered include `YEAR, MONTH, ORIGIN, DEST, DISTANCE, SEATS, PASSENGERS,
  DEPARTURES_SCHEDULED, DEPARTURES_PERFORMED, CLASS, AIRCRAFT_CONFIG, FREIGHT, ...`.
- No static file exists (checked `https://transtats.bts.gov/PREZIP/` – it has On-Time and DB1B files, no T-100).
  Download is an ASP.NET postback with `__VIEWSTATE`/`__EVENTVALIDATION`; per project rules we do not scrape it.

**Steps for you (3 downloads, one per year):**
1. Open https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4
2. Filter Geography: **All**. Filter Year: **2024**. Filter Period: **All Months**.
3. Tick these fields (untick anything else to keep the file small):
   `YEAR, MONTH, UNIQUE_CARRIER, CARRIER_NAME, CLASS, ORIGIN, ORIGIN_STATE_ABR, ORIGIN_COUNTRY,
   DEST, DEST_STATE_ABR, DEST_COUNTRY, DISTANCE, AIRCRAFT_CONFIG, DEPARTURES_SCHEDULED,
   DEPARTURES_PERFORMED, SEATS, PASSENGERS`
4. Tick **Download as zip** (and optionally *Term definitions*). Click **Download**.
5. Save the zip as `data/raw/t100_segment/T100_SEGMENT_2024.zip`.
6. Repeat for **2025** and **2026** (2026 = Jan–Aug, the latest available).
7. Run `python scripts/spike/probe_b_t100_segment.py` – it prints columns, months and sample rows.

Downloaded files (2026-11-18): `T_T100_SEGMENT_ALL_CARRIER_<timestamp>.zip`, each containing
`T_T100_SEGMENT_ALL_CARRIER.csv` with all 50 fields. No term-definitions file was included.

| Zip | Year | Rows | Months |
|---|---|---|---|
| `T_T100_SEGMENT_ALL_CARRIER_20261118_090154.zip` | 2024 | 549,799 | 01–12 |
| `T_T100_SEGMENT_ALL_CARRIER_20261118_085332.zip` | 2025 | 571,262 | 01–12 |
| `T_T100_SEGMENT_ALL_CARRIER_20261118_091016.zip` | 2026 | 381,593 | 01–08 |

Why 2024–2026: trailing 12 months (Sep 2025–Aug 2026) plus the prior 12 months for YoY.

### C. BTS Airline On-Time Delay Cause
- No Socrata dataset (searched data.bts.gov, data.transportation.gov, datahub.transportation.gov).
- The TranStats page has `PeriodFrom`/`PeriodTo` selects (period id = year×12 + month, e.g. Aug 2026 = 24320).
  A POST of the page's own form, then a GET of the `ot_delaycause1_DL.aspx?...` link it renders,
  returns `Airline_Delay_Cause_<timestamp>.zip` (~0.9 MB for 12 months) containing
  `Airline_Delay_Cause.csv` and `Download_Column_Definitions.xlsx`.
- Fields: `year, month, carrier, carrier_name, airport, airport_name, arr_flights, arr_del15,
  carrier_ct, weather_ct, nas_ct, security_ct, late_aircraft_ct, arr_cancelled, arr_diverted,
  arr_delay, carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay` (delay in minutes).
- Verified: the five cause columns sum exactly to `arr_delay` (BOS, 12 months).
- Scope: reporting (large US) carriers only, domestic arrivals only.
- Fragility: relies on the page's form behaviour, not a documented API. Manual fallback: open the page,
  pick From/To month, Submit, click *Download Raw Data*. Second fallback: the flight-level
  On-Time Reporting zips in `https://transtats.bts.gov/PREZIP/` (direct URLs, latest
  `On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2026_8.zip`, ~30 MB/month).

### D. FAA Passenger Boarding (Enplanements)
- Index pages: https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger (current year)
  and `.../passenger/previous_years`.
- Columns: `Rank, RO, ST, Locid, City, Airport Name, S/L, Hub, CY 25 Enplanements, CY 24 Enplanements, % Change`
  (the CY2024 file has `CY 24` / `CY 23`). One sheet, header in row 1. `Hub` = L/M/S/N.
- Each file already contains two years, so CY2025 alone gives CY24→CY25 growth.
- Calendar-year data – does not align with a trailing-12-month window.

### E. OurAirports
- `airports.csv` (86,371 rows): `ident, type, name, latitude_deg, longitude_deg, iso_country,
  iso_region (US-MA), municipality, scheduled_service, icao_code, iata_code, ...`
- `runways.csv` (48,433 rows): `airport_ident, length_ft, width_ft, surface, closed, le_ident, he_ident, ...`
- Join on `iata_code` (airports) and `ident` ↔ `airport_ident` (runways). Community-maintained data.

### F. FAA NAS Status
- `api/airport-events` (JSON): list of airports with an active event; keys
  `airportId, groundStop, groundDelay, airportClosure, freeForm, arrivalDelay, departureDelay,
  airportConfig, deicing, airportLongName, latitude, longitude`. Airports with no event are absent.
- `api/airport-status-information` (XML): same information grouped by `Delay_type`.
- Not formally documented as an API – context only, never used in scoring.
