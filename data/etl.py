"""ETL entry point (python -m data.etl): fetch, clean and load sources into data/airports.duckdb.

  python -m data.etl             live sources (fallback to cache), T-100 from cache
  python -m data.etl --offline   everything from data/processed/ (no network)
  python -m data.etl --from-raw  rebuild T-100 from data/raw zips and rewrite all processed caches
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import duckdb
import pandas as pd

from data import quality
from data.db import DB_PATH, PROCESSED_DIR, RAW_DIR, write_table
from data.sources import delay_cause, faa, ourairports, socrata, t100
from data.sources.common import Provenance

log = logging.getLogger("etl")
MANIFEST = "manifest.json"


def build_airports(oa: pd.DataFrame, faa_df: pd.DataFrame) -> pd.DataFrame:
    """OurAirports metadata + latest FAA hub class."""
    latest = faa_df[faa_df["hub"].notna()].sort_values("year").drop_duplicates("airport", keep="last")
    hubs = latest[["airport", "hub"]].rename(columns={"airport": "iata", "hub": "faa_hub"})
    return oa.merge(hubs, on="iata", how="left")


def update_manifest(processed_dir: Path, runs: list[Provenance], files: dict[str, str]) -> None:
    """Record provenance of every rewritten cache file and regenerate processed/README.md."""
    path = processed_dir / MANIFEST
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for p in runs:
        if p.cache_written:
            manifest[files[p.source]] = {
                "source": p.source, "url": p.url, "access": p.access,
                "period": p.data_period, "rows": p.rows, "built": p.built,
            }
    path.write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Processed data (committed)",
        "",
        "Compact, cleaned copies of every source so the app builds without the manual TranStats",
        "downloads or network access (`python -m data.etl --offline`). Regenerated only by",
        "`python -m data.etl --from-raw`. Do not edit by hand; this file is generated from `manifest.json`.",
        "",
        "| File | Source URL | Access | Period | Rows | Built |",
        "|---|---|---|---|---|---|",
    ]
    for fname, m in sorted(manifest.items()):
        urls = "<br>".join(m["url"].split(" + "))
        lines.append(f"| `{fname}` | {urls} | {m['access']} | {m['period']} | {m['rows']:,} | {m['built']} |")
    lines += [
        "",
        "Notes:",
        "- `t100_routes.parquet`: passenger service only (`CLASS in (F, L)` and `AIRCRAFT_CONFIG != 2`), US origins,",
        "  aggregated to origin × dest × month. `dep_*_us` columns only count rows with scheduled departures > 0.",
        "- `delay_cause.parquet`: reporting carriers summed per arrival airport × month.",
        "- `socrata_airport_months.parquet`: all carriers incl. all-cargo flights (cross-check only).",
    ]
    (processed_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(mode: str = "live", processed_dir: Path = PROCESSED_DIR,
          raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> tuple[dict[str, pd.DataFrame], list[Provenance], int]:
    """Build the DuckDB file from scratch. Returns (tables, provenance, number_of_quality_flags)."""
    routes, p_t100 = t100.load(mode, processed_dir, raw_dir / "t100_segment")
    delay, p_delay = delay_cause.load(mode, processed_dir)
    soc, p_soc = socrata.load(mode, processed_dir)
    enpl, p_faa = faa.load(mode, processed_dir)
    oa, p_oa = ourairports.load(mode, processed_dir)
    runs = [p_t100, p_delay, p_soc, p_faa, p_oa]

    tables = {
        "t100_routes": routes,
        "delay_cause": delay,
        "socrata_airport_months": soc,
        "faa_enplanements": enpl,
        "airports": build_airports(oa, enpl),
    }
    empty = [name for name, df in tables.items() if df.empty]
    if empty:
        raise RuntimeError(f"empty tables: {empty}")

    db_path.unlink(missing_ok=True)
    with duckdb.connect(str(db_path)) as con:
        for name, df in tables.items():
            write_table(con, name, df)
        write_table(con, "etl_runs", pd.DataFrame([asdict(p) for p in runs]))

    if mode == "from_raw":
        files = {t100.SOURCE: t100.CACHE_FILE, delay_cause.SOURCE: delay_cause.CACHE_FILE,
                 socrata.SOURCE: socrata.CACHE_FILE, faa.SOURCE: faa.CACHE_FILE,
                 ourairports.SOURCE: ourairports.CACHE_FILE}
        update_manifest(processed_dir, runs, files)

    lines, flags = quality.report(tables)
    source_lines = ["Sources (mode: live / cache / raw):"]
    for p in runs:
        note = f"  WARNING: {p.warning}" if p.warning else ""
        source_lines.append(f"  {p.source:<24} {p.mode:<6} {p.data_period:<18}{note}")
    lines[2:2] = source_lines
    print("\n".join(lines))
    return tables, runs, flags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/airports.duckdb")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--offline", action="store_true", help="use data/processed only (no network)")
    group.add_argument("--from-raw", action="store_true", help="rebuild T-100 from data/raw and rewrite caches")
    args = parser.parse_args(argv)
    mode = "offline" if args.offline else "from_raw" if args.from_raw else "live"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("building %s (mode=%s)", DB_PATH, mode)
    try:
        build(mode)
    except Exception as exc:
        log.error("ETL failed: %s", exc)
        return 1
    log.info("done: %s", DB_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
