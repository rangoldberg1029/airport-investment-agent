"""Golden checks against the real DuckDB (skipped when data/airports.duckdb has not been built).

Expected values come from the data validation (docs/data_validation.md), computed independently.
"""

import pytest

from data.db import DB_PATH

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="run `python -m data.etl --offline` first")


@pytest.fixture(scope="module")
def real_frames():
    from data.repository import load_frames
    return load_frames()


def test_anc_long_haul_matches_spike(real_frames):
    from scoring.metrics import long_haul_share
    d = long_haul_share(real_frames, "ANC")["data"]
    assert d["total_passenger_departures"] == 40_198
    assert d["long_haul_departure_share_pct"] == 2.67
    assert d["long_haul_seat_share_pct"] == 5.17
    by_t = {s["threshold_mi"]: s for s in d["sensitivity"]}
    assert by_t[2500]["departure_share_pct"] == 9.50
    assert by_t[4000]["departure_share_pct"] == 0.13


def test_sfo_delay_matches_spike(real_frames):
    from scoring.metrics import delay_profile
    d = delay_profile(real_frames, "SFO")["data"]
    assert d["delay_rate_pct"] == pytest.approx(33.4, abs=0.1)
    assert d["nas_share_pct"] == pytest.approx(48.0, abs=0.1)


def test_ranking_universe_is_faa_hubs(real_frames):
    from scoring.metrics import kpi_table
    k = kpi_table(real_frames)
    assert set(k.loc[k["in_universe"], "hub"]) == {"L", "M", "S"}
    assert 130 <= int(k["in_universe"].sum()) <= 150
