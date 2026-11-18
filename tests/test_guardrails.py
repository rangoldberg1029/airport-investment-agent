"""Number traceability check."""

from agent.guardrails import extract_numbers, untraceable_numbers

TOOL = [{"data": {"delay_rate_pct": 33.39, "lf": 0.817, "passengers": 36_497_303, "growth_pct": -1.88,
                  "note": "national rate 22.03%"}}]


def values(text):
    return [(n.value, n.decimals, n.is_percent) for n in extract_numbers(text)]


def test_extract_formats():
    assert values("1,234,567 and 12.5% and +2.3% and 0.817") == [
        (1234567, 0, False), (12.5, 1, True), (2.3, 1, True), (0.817, 3, False)]
    assert values("36.5 million")[0][0] == 36_500_000
    assert values("2.1M passengers, 67k seats")[:2] == [(2_100_000, 1, False), (67_000, 0, False)]
    assert values("−1.9%") == [(-1.9, 1, True)]


def test_ordinals_and_decimals_are_parsed_whole():
    assert values("63.7th percentile, 99th, 2nd") == [(63.7, 1, False), (99, 0, False), (2, 0, False)]
    assert untraceable_numbers("peak LF at the 63.7th percentile", [{"pctile": 63.7}]) == []


def test_codes_and_identifiers_are_not_numbers():
    assert values("runway 07L/25R, BTS T-100, CY2025, r495-tyji, Q1") == []


def test_matches_with_rounding_and_percent_conversion():
    answer = "Delay rate 33.4% (vs 22.0% nationally), load factor 81.7%, about 36.5 million passengers, growth -1.9%."
    assert untraceable_numbers(answer, TOOL) == []


def test_flags_invented_numbers():
    flagged = untraceable_numbers("Delay rate 33.39% and the airport needs 12,345 more gates.", TOOL)
    assert [f["value"] for f in flagged] == ["12,345"]
    assert "gates" in flagged[0]["context"]


def test_ignores_years_small_integers_config_and_user_numbers():
    answer = ("In 2025 the top 3 airports; long-haul >= 3,000 mi (sensitivity 2,500 / 4,000); target LF 0.85; "
              "growth weight 35%; you asked for 40%.")
    assert untraceable_numbers(answer, TOOL, user_text="weight growth at 40%") == []


def test_precision_tolerance_is_strict_for_exact_figures():
    assert untraceable_numbers("delay rate 33.5%", TOOL)          # 33.39 would round to 33.4, not 33.5
