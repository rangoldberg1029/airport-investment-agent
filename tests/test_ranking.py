"""Normalisation, weighting, indices, capacity gap, rank stability and name resolution."""

import json

import pandas as pd
import pytest

from scoring import ranking
from scoring.common import resolve_airport, resolve_region
from scoring.config import INVESTMENT_WEIGHTS, REGIONS


# ---------------------------------------------------------------- normalisation / weights

def test_percentile_midrank_with_ties_and_nan():
    ref = pd.Series([1.0, 2.0, 2.0, 4.0])
    p = ranking.percentile_against(pd.Series([2.0, 4.0, 0.0, None]), ref)
    assert p.iloc[0] == pytest.approx((1 + 0.5 * 2) / 4 * 100)   # 50
    assert p.iloc[1] == pytest.approx((3 + 0.5) / 4 * 100)       # 87.5
    assert p.iloc[2] == 0
    assert pd.isna(p.iloc[3])


def test_weighted_index_contributions_sum_to_score_and_renormalise():
    pcts = pd.DataFrame({"a": [100.0, 50.0], "b": [0.0, None]}, index=["X", "Y"])
    score, contrib, renorm = ranking.weighted_index(pcts, {"a": 0.5, "b": 0.5})
    assert score["X"] == pytest.approx(50.0)
    assert contrib.loc["X"].sum() == pytest.approx(score["X"])
    assert score["Y"] == pytest.approx(50.0) and renorm["Y"] and not renorm["X"]   # b missing -> a gets 100%


def test_validate_weights_override_and_errors():
    w, notes = ranking.validate_weights({"growth": 0.5}, INVESTMENT_WEIGHTS)
    assert sum(w.values()) == pytest.approx(1) and w["growth"] == pytest.approx(0.5 / 1.25)
    assert notes and "User-overridden" in notes[0]
    with pytest.raises(ValueError):
        ranking.validate_weights({"price": 1}, INVESTMENT_WEIGHTS)
    with pytest.raises(ValueError):
        ranking.validate_weights({"growth": -1}, INVESTMENT_WEIGHTS)
    assert ranking.validate_weights(None, INVESTMENT_WEIGHTS) == (INVESTMENT_WEIGHTS, [])


# ---------------------------------------------------------------- capacity gap

def test_capacity_gap_by_hand():
    g = ranking.capacity_gap(passengers=800, seats=900, departures=9, target_lf=0.8)
    assert g["extra_seats"] == 100 and g["extra_departures"] == 1
    assert ranking.capacity_gap(700, 1000, 10, 0.8)["extra_seats"] == 0     # already below target


# ---------------------------------------------------------------- indices on the fixture

def test_congestion_index_components_add_up(frames):
    env = ranking.congestion_index(frames, ["AAA", "BBB"])
    for row in env["data"]["airports"]:
        assert sum(c["contribution"] for c in row["components"]) == pytest.approx(row["congestion_index"], abs=0.05)
    json.dumps(env)


def test_unmet_demand_drivers_are_above_hub_median(frames):
    d = ranking.unmet_demand(frames, "AAA")["data"]
    # AAA vs its large-hub peers (AAA, BBB): higher LF, peak LF, growth and delay rate than BBB
    names = {x["component"] for x in d["drivers_above_peer_median"]}
    assert {"lf_avg", "lf_peak", "growth_yoy", "delay_rate"} <= names
    assert "nas_share" not in names                      # identical for all airports -> not above median
    for x in d["drivers_above_peer_median"]:
        assert x["value"] > x["peer_median"]
    assert set(d["capacity_gap"]) == {"lf_0.85", "lf_0.80"}


def test_investment_scores_rank_only_universe_and_list_others(frames):
    d = ranking.investment_scores(frames, states=["MA"])["data"]
    assert [r["airport"] for r in d["ranking"]] and "DDD" not in [r["airport"] for r in d["ranking"]]
    assert [o["airport"] for o in d["outside_ranking_universe"]] == []   # DDD has < 50k passengers
    top = d["ranking"][0]
    assert sum(c["contribution"] for c in top["components"]) == pytest.approx(top["investment_score"], abs=0.05)


def test_rank_stability_detects_swap(frames, monkeypatch):
    monkeypatch.setattr(ranking, "WEIGHT_PRESETS", {
        "lf": {"demand_pressure": 1, "growth": 0, "congestion": 0, "scale": 0},
        "growth": {"demand_pressure": 0, "growth": 1, "congestion": 0, "scale": 0},
    })
    d = ranking.rank_stability(frames, states=["MA", "CT"], top_n=1)["data"]
    # CCC has the highest LF; AAA the highest growth -> the #1 changes
    assert d["rankings"]["lf"] == ["CCC"] and d["rankings"]["growth"] == ["AAA"]
    assert d["same_top_n_set"] is False


# ---------------------------------------------------------------- name / region resolution

def test_resolve_airport(frames):
    assert resolve_airport("lax", frames.airports) == ("LAX", None)
    iata, note = resolve_airport("LA", frames.airports)
    assert iata == "LAX" and "interpreted as LAX" in note and "SNA" in note
    assert resolve_airport("John Wayne", frames.airports)[0] == "SNA"
    assert resolve_airport("Anchorage", frames.airports)[0] == "ANC"
    with pytest.raises(ValueError, match="unknown airport"):
        resolve_airport("Atlantis", frames.airports)


def test_new_england_definition():
    assert set(REGIONS["new england"]) == {"CT", "ME", "MA", "NH", "RI", "VT"}
    states, note = resolve_region("New England")
    assert len(states) == 6 and "CT, ME, MA, NH, RI, VT" in note
