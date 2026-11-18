"""Cache fallback, quality checks and an offline end-to-end build into a temp DuckDB."""

import logging

import duckdb
import pandas as pd
import pytest

from data import quality
from data.etl import build
from data.sources import delay_cause, faa, ourairports, socrata, t100
from data.sources.common import load_with_fallback
from data.sources.t100 import aggregate_routes, filter_passenger_service

# ---------- fallback ----------

def test_failed_fetch_falls_back_to_cache_with_warning(tmp_path, caplog):
    cache = tmp_path / "x.parquet"
    pd.DataFrame({"year": [2025], "month": [1], "v": [1]}).to_parquet(cache)

    def broken():
        raise ConnectionError("source down")

    with caplog.at_level(logging.WARNING, logger="etl"):
        df, prov = load_with_fallback("x", "http://x", "public API", broken, cache, mode="live")
    assert len(df) == 1 and prov.mode == "cache"
    assert "source down" in prov.warning and "source down" in caplog.text


def test_offline_never_calls_fetch(tmp_path):
    cache = tmp_path / "x.parquet"
    pd.DataFrame({"year": [2025]}).to_parquet(cache)

    def must_not_run():
        raise AssertionError("fetch called in offline mode")

    _, prov = load_with_fallback("x", "u", "a", must_not_run, cache, mode="offline")
    assert prov.mode == "cache"


def test_live_success_does_not_rewrite_cache_but_from_raw_does(tmp_path):
    cache = tmp_path / "x.parquet"
    fresh = lambda: pd.DataFrame({"year": [2026]})
    load_with_fallback("x", "u", "a", fresh, cache, mode="live")
    assert not cache.exists()
    _, prov = load_with_fallback("x", "u", "a", fresh, cache, mode="from_raw")
    assert cache.exists() and prov.cache_written


def test_missing_cache_and_failed_fetch_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_with_fallback("x", "u", "a", lambda: pd.DataFrame(), tmp_path / "none.parquet", mode="live")

# ---------- quality ----------

def test_cross_check_flags_more_than_five_percent():
    t100_df = pd.DataFrame({"year": [2025, 2025], "origin": ["AAA", "BBB"], "passengers": [104, 106]})
    soc = pd.DataFrame({"year": [2025, 2025], "airport": ["AAA", "BBB"], "passengers": [100, 100]})
    faa_df = pd.DataFrame({"year": [2025, 2025], "airport": ["AAA", "BBB"], "enplanements": [100, 100]})
    out = quality.passenger_cross_check(t100_df, soc, faa_df, airports=["AAA", "BBB"])
    assert not out.loc["AAA", "flag"]       # 4% -> ok
    assert out.loc["BBB", "flag"]           # 6% -> flag


def test_window_complete_lists_missing_months():
    df = pd.DataFrame({"year": [2025, 2025], "month": [11, 12]})
    assert quality.window_complete(df, (2025, 11), (2025, 12)) == []
    assert quality.window_complete(df, (2025, 11), (2026, 1)) == [(2026, 1)]

# ---------- offline end-to-end ----------

@pytest.fixture
def processed_dir(tmp_path, t100_raw, delay_raw):
    d = tmp_path / "processed"
    d.mkdir()
    routes = aggregate_routes(filter_passenger_service(t100_raw))
    routes.to_parquet(d / t100.CACHE_FILE)
    delay_cause.aggregate_to_airport_month(delay_raw).to_parquet(d / delay_cause.CACHE_FILE)
    pd.DataFrame({"year": [2025], "month": [9], "airport": ["ANC"], "departures": [20],
                  "passengers": [2550], "seats": [3000], "freight_lbs": [0]}).to_parquet(d / socrata.CACHE_FILE)
    pd.DataFrame({"airport": ["ANC"], "year": [2025], "enplanements": [2600], "hub": ["M"],
                  "state": ["AK"], "rank": [59]}).to_parquet(d / faa.CACHE_FILE)
    pd.DataFrame({"iata": ["ANC"], "icao": ["PANC"], "name": ["Ted Stevens"], "city": ["Anchorage"],
                  "state": ["AK"], "lat": [61.2], "lon": [-150.0], "type": ["large_airport"],
                  "runway_count": [3], "longest_runway_ft": [12400]}).to_parquet(d / ourairports.CACHE_FILE)
    pd.DataFrame({"iata": ["ANC"], "runway": ["07R/25L"], "length_ft": [12400.0], "width_ft": [200.0],
                  "surface": ["ASP"]}).to_parquet(d / ourairports.RUNWAYS_CACHE_FILE)
    return d


def test_offline_build_creates_all_tables(processed_dir, tmp_path):
    db = tmp_path / "test.duckdb"
    tables, runs, _ = build("offline", processed_dir=processed_dir, raw_dir=tmp_path / "raw", db_path=db)
    assert all(p.mode == "cache" for p in runs)
    with duckdb.connect(str(db), read_only=True) as con:
        names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        assert names == {"t100_routes", "delay_cause", "socrata_airport_months",
                         "faa_enplanements", "airports", "runways", "etl_runs"}
        assert con.execute("SELECT count(*) FROM t100_routes").fetchone()[0] == 1
        assert con.execute("SELECT faa_hub FROM airports WHERE iata='ANC'").fetchone()[0] == "M"
