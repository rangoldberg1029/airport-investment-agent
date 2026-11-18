"""KPI functions against the hand-checked fixture in conftest.py (airport AAA)."""

import json

import pytest

from scoring import metrics
from scoring.common import in_window, shift_month, window_label

REL = 1e-6


@pytest.fixture
def aaa(frames):
    return metrics.kpi_table(frames).loc["AAA"]


def test_window_helpers():
    assert shift_month((2026, 7), -11) == (2025, 8)
    assert shift_month((2025, 12), 1) == (2026, 1)
    assert window_label(((2025, 9), (2026, 8))) == "Sep 2025 - Aug 2026"


def test_window_filter_excludes_other_months(frames):
    t = frames.t100
    assert in_window(t, ((2025, 9), (2026, 8))).sum() == 4 * 3 * 12   # 4 airports x 3 routes x 12 months


def test_traffic_and_load_factor(aaa):
    assert aaa["departures"] == 156 and aaa["seats"] == 20_400 and aaa["passengers"] == 16_950
    assert aaa["lf_avg"] == pytest.approx(16_950 / 20_400, rel=REL)
    assert aaa["lf_peak"] == pytest.approx(1550 / 1700, rel=REL)
    assert aaa["lf_peak_month"] == "2026-07"


def test_growth_vs_prior_window(aaa):
    assert aaa["passengers_prior"] == 15_600
    assert aaa["growth_yoy"] == pytest.approx(16_950 / 15_600 - 1, rel=REL)


def test_delay_rate_nas_share_uses_cause_sum_and_cancellations(aaa):
    assert aaa["delay_rate"] == pytest.approx(0.20)
    assert aaa["nas_share"] == pytest.approx(50 / 200)        # NOT 50 / 210 (arr_delay)
    assert aaa["cancellation_rate"] == pytest.approx(0.02)


def test_completion_gap_uses_us_columns_only(aaa):
    assert aaa["completion_gap"] == pytest.approx(1 - 120 / 132, rel=REL)


def test_long_haul_thresholds_are_inclusive_and_seat_weighted(aaa):
    assert aaa["lh_dep_share_3000"] == pytest.approx(36 / 156, rel=REL)   # 3000-mi route counts
    assert aaa["lh_seat_share_3000"] == pytest.approx(8_400 / 20_400, rel=REL)
    assert aaa["lh_dep_share_2500"] == pytest.approx(36 / 156, rel=REL)
    assert aaa["lh_dep_share_4000"] == pytest.approx(12 / 156, rel=REL)


def test_ops_per_runway_uses_latest_12_socrata_months(frames, aaa):
    assert metrics.socrata_window(frames.socrata, ((2025, 9), (2026, 8))) == ((2025, 8), (2026, 7))
    assert aaa["ops_per_runway"] == pytest.approx(2 * 3_600 / 2)


def test_missing_delay_data_stays_nan(frames):
    frames.delay = frames.delay[frames.delay["airport"] != "BBB"]
    k = metrics.kpi_table(frames)
    assert k.loc["BBB", "delay_rate"] != k.loc["BBB", "delay_rate"]    # NaN, not 0
    env = metrics.delay_profile(frames, "BBB")
    assert any("No delay data" in c for c in env["caveats"])


def test_long_haul_share_envelope(frames):
    env = metrics.long_haul_share(frames, "AAA")
    d = env["data"]
    assert d["long_haul_departure_share_pct"] == round(100 * 36 / 156, 2)
    assert [s["threshold_mi"] for s in d["sensitivity"]] == [2500, 3000, 4000]
    assert [r["dest"] for r in d["top_routes_at_or_above_3000_mi"]] == ["L1", "L2"]


@pytest.mark.parametrize("fn", [metrics.airport_profile, metrics.delay_profile, metrics.completion_gap,
                                metrics.long_haul_share, metrics.ops_per_runway])
def test_every_metric_returns_json_envelope(frames, fn):
    env = fn(frames, "AAA")
    assert set(env) == {"data", "sources", "data_period", "assumptions", "caveats"}
    assert env["sources"] and env["data_period"]
    json.dumps(env)


def test_completion_gap_always_states_us_limitation(frames):
    assert any("US carriers only" in c for c in metrics.completion_gap(frames, "AAA")["caveats"])
