# Processed data (committed)

Compact, cleaned copies of every source so the app builds without the manual TranStats
downloads or network access (`python -m data.etl --offline`). Regenerated only by
`python -m data.etl --from-raw`. Do not edit by hand; this file is generated from `manifest.json`.

| File | Source URL | Access | Period | Rows | Built |
|---|---|---|---|---|---|
| `delay_cause.parquet` | https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp | public web form (scripted) | 2024-09..2026-08 | 8,600 | 2026-11-18 |
| `faa_enplanements.parquet` | https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/arp-cy2025-commercial-service-enplanements.xlsx<br>https://www.faa.gov/sites/faa.gov/files/2026-09/arp-cy2024-commercial-service-enplanements.xlsx | direct download | 2023..2025 | 1,552 | 2026-11-18 |
| `ourairports_us.parquet` | https://davidmegginson.github.io/ourairports-data/airports.csv<br>https://davidmegginson.github.io/ourairports-data/runways.csv | direct download | n/a | 679 | 2026-11-18 |
| `runways_us.parquet` | https://davidmegginson.github.io/ourairports-data/runways.csv | direct download | n/a | 1,269 | 2026-11-18 |
| `socrata_airport_months.parquet` | https://data.bts.gov/resource/r495-tyji.json | public API | 2024-01..2026-07 | 28,035 | 2026-11-18 |
| `t100_routes.parquet` | https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4 | manual download | 2024-01..2026-08 | 419,215 | 2026-11-18 |

Notes:
- `t100_routes.parquet`: passenger service only (`CLASS in (F, L)` and `AIRCRAFT_CONFIG != 2`), US origins,
  aggregated to origin × dest × month. `dep_*_us` columns only count rows with scheduled departures > 0.
- `delay_cause.parquet`: reporting carriers summed per arrival airport × month.
- `socrata_airport_months.parquet`: all carriers incl. all-cargo flights (cross-check only).
- `runways_us.parquet`: open runways at US commercial airports; scoring applies the minimum length.
