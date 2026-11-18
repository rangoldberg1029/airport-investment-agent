"""Agent tools on the synthetic fixture (no network)."""

import json

import pytest

from agent.tools import TOOL_SCHEMAS, TOOLS, execute_tool, shorten


def view(record) -> dict:
    return json.loads(record.llm_view)


def test_schemas_match_dispatch_table():
    assert {s["function"]["name"] for s in TOOL_SCHEMAS} == set(TOOLS)


def test_profile_bundles_kpis_and_precomputed_differences(frames):
    r = execute_tool("get_airport_profile", {"airport": "AAA"}, frames)
    d = view(r)["data"]
    assert r.error is None and d["airport"] == "AAA"
    assert d["delay_rate_pct"] == 20.0
    assert d["delay_rate_minus_national_pp"] == round(20.0 - d["national"]["delay_rate_pct"], 2)
    assert d["commercial_runways"] == 2 and d["investment_score"] is not None
    assert set(r.envelope) == {"data", "sources", "data_period", "assumptions", "caveats"}


def test_compare_congestion_has_differences_for_every_metric(frames):
    r = execute_tool("compare_congestion", {"airports": ["AAA", "BBB"]}, frames)
    d = view(r)["data"]
    diffs = d["differences"]["AAA_minus_BBB"]
    v = d["values"]
    assert diffs["delay_rate_pct"] == [round(v["delay_rate_pct"]["AAA"] - v["delay_rate_pct"]["BBB"], 3), "pp"]
    assert diffs["commercial_runways"][0] == 2 - 4
    assert set(d["contribution_differences"]["AAA_minus_BBB"]) == {"delay_rate", "nas_share", "ops_per_runway",
                                                                   "cancellation_rate"}
    assert d["runway_lengths_ft_excluded_short"]["AAA"] == [2500]


def test_compare_congestion_needs_two_airports(frames):
    r = execute_tool("compare_congestion", {"airports": ["AAA"]}, frames)
    assert r.error and "2 to 5" in r.error


def test_rank_airports_weights_override_and_stability(frames):
    r = execute_tool("rank_airports", {"states": ["MA", "CT"], "weights": {"growth": 0.6}}, frames)
    d = view(r)["data"]
    assert d["weights"]["growth"] == pytest.approx(0.6 / 1.35, abs=1e-3)
    assert d["rows"][0][-1] is not None                      # gap_to_next precomputed
    assert "stability_top3" in d
    assert any("User-overridden" in a for a in r.envelope["assumptions"])


def test_name_resolution_is_reported_as_assumption(frames):
    frames.airports.loc[frames.airports.iata == "AAA", "city"] = "Springfield"
    r = execute_tool("get_airport_profile", {"airport": "Springfield"}, frames)
    assert any("interpreted as AAA" in a for a in view(r)["assumptions"])


def test_unknown_airport_returns_error_payload_not_exception(frames):
    r = execute_tool("unmet_demand", {"airport": "Atlantis"}, frames)
    assert r.error and "unknown airport" in view(r)["error"]


def test_bad_arguments_and_unknown_tool(frames):
    assert "invalid JSON" in execute_tool("long_haul_share", "{not json", frames).error
    assert "unknown tool" in execute_tool("fly_me_to_the_moon", {}, frames).error
    assert execute_tool("long_haul_share", {"airprt": "AAA"}, frames).error   # wrong argument name


def test_long_haul_custom_threshold(frames):
    d = view(execute_tool("long_haul_share", {"airport": "AAA", "threshold_mi": 4000}, frames))["data"]
    assert d["threshold_mi"] == 4000 and d["departure_share_pct"] == round(100 * 12 / 156, 2)


def test_unmet_demand_drivers_include_difference_to_median(frames):
    d = view(execute_tool("unmet_demand", {"airport": "AAA"}, frames))["data"]
    for comp, value, median, delta, _ in d["drivers"]:
        assert delta == round(value - median, 3)
    assert {row[0] for row in d["capacity_gap"]} == {0.85, 0.8}


def test_live_status_no_events_is_not_an_error(frames):
    fake = lambda: ([{"airportId": "BBB", "groundStop": {"reason": "weather"}}], "Wed Nov 18 12:00:00 2026 GMT")
    r = execute_tool("live_airport_status", {"airport": "AAA"}, frames, overrides=None) if False else None
    from agent.tools import live_airport_status
    env, data = live_airport_status(frames, "AAA", fetch=fake)
    assert data["status"] == "no active FAA delay events reported right now"
    assert data["feed_timestamp"] == "Wed Nov 18 12:00:00 2026 GMT" and "feed updated" in env["data_period"]
    env, data = live_airport_status(frames, "BBB", fetch=fake)
    assert data["events"] == [{"type": "groundStop", "reason": "weather"}]


def test_live_status_feed_failure(frames):
    from agent.tools import live_airport_status

    def broken():
        raise ConnectionError("down")
    _, data = live_airport_status(frames, "AAA", fetch=broken)
    assert data["status"] == "live FAA status unavailable right now"


def test_methodology_data_sources_states_access_and_kpis(frames):
    d = view(execute_tool("explain_methodology", {"topic": "data_sources"}, frames))["data"]["data_sources"]
    access = {s["id"]: s["access"] for s in d}
    assert access["T100"].startswith("manual download")
    assert access["SOC"].startswith("public API")
    assert access["FAA"].startswith("public file download")
    assert all(s["feeds"] for s in d)
    assert execute_tool("explain_methodology", {"topic": "astrology"}, frames).error


def test_shorten_keeps_unknown_and_resolution_texts():
    texts = ["'LA' interpreted as LAX (other airports in the area: BUR)", "Higher score = better",
             "Delay data covers large US carriers ..."]
    assert shorten(texts) == [texts[0], "DELAY covers reporting carriers, domestic arrivals only"]


@pytest.mark.parametrize("name,args", [
    ("get_airport_profile", {"airport": "AAA"}), ("rank_airports", {"states": ["MA", "CT"]}),
    ("compare_congestion", {"airports": ["AAA", "BBB"]}), ("long_haul_share", {"airport": "AAA"}),
    ("unmet_demand", {"airport": "AAA"}), ("explain_methodology", {"topic": "all"}),
])
def test_llm_view_size_cap(frames, name, args):
    r = execute_tool(name, args, frames)
    cap = 1100 if name == "explain_methodology" else 700          # tokens (~4 chars each)
    assert len(r.llm_view) / 4 <= cap, f"{name} view too large: {len(r.llm_view) // 4} tokens"


def test_prompt_and_schemas_size_cap():
    from agent.prompts import SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT) / 4 <= 450
    assert len(json.dumps(TOOL_SCHEMAS, separators=(",", ":"))) / 4 <= 850
