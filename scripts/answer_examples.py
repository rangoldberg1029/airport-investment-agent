"""Answer the 4 example questions (see README.md) with deterministic Python only (no LLM).

Every number printed below comes from scoring/* envelopes; this file only formats them.
Run:  python scripts/answer_examples.py   (needs data/airports.duckdb: python -m data.etl --offline)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.repository import load_frames  # noqa: E402
from scoring import metrics, ranking  # noqa: E402
from scoring.common import window_label  # noqa: E402
from scoring.config import MIN_COMMERCIAL_RUNWAY_FT, WINDOW  # noqa: E402


def header(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def footer(*envelopes: dict, periods: dict[str, str] | None = None) -> None:
    """Print the sources, period(s), assumptions and caveats of all envelopes used (deduplicated).

    `periods` (label -> period) prints one labeled line per data period when sources differ.
    """
    merged = {"sources": [], "assumptions": [], "caveats": []}
    for env in envelopes:
        for key in merged:
            merged[key] += [x for x in env[key] if x not in merged[key]]
    if periods:
        print("\n  Data periods:")
        for label, period in periods.items():
            print(f"    - {label}: {period}")
    else:
        print("\n  Data period: " + envelopes[0]["data_period"])
    for key in ("sources", "assumptions", "caveats"):
        print(f"  {key.capitalize()}:")
        for item in merged[key]:
            print(f"    - {item}")


def components(rows: list[dict], indent: str = "    ") -> None:
    print(f"{indent}{'component':<18}{'raw value':>14}{'percentile':>12}{'weight':>8}{'contribution':>14}")
    for c in rows:
        print(f"{indent}{c['component']:<18}{str(c['raw_value']):>14}{c['percentile']:>12}"
              f"{c['weight']:>8.2f}{c['contribution']:>14}")


# ------------------------------------------------------------------------------------------ Q1
def q1(frames) -> None:
    header("Q1. Which airports in New England are strong candidates for terminal expansion?")
    inv = ranking.investment_scores(frames, region="New England")
    stab = ranking.rank_stability(frames, region="New England", top_n=3)
    d = inv["data"]
    print(f"  Investment score (0-100), weights {d['weights']}; national rank among "
          f"{d['ranked_nationally']} ranked airports\n")
    print(f"  {'#':<3}{'airport':<9}{'state':<7}{'hub':<13}{'score':>7}{'nat. rank':>11}"
          f"{'LF':>8}{'growth %':>10}{'congest.':>10}{'passengers':>14}")
    for r in d["ranking"]:
        c = {x["component"]: x["raw_value"] for x in r["components"]}
        print(f"  {r['rank']:<3}{r['airport']:<9}{r['state']:<7}{r['hub']:<13}{r['investment_score']:>7}"
              f"{r['national_rank']:>11}{c['demand_pressure']:>8}{c['growth']:>10}{c['congestion']:>10}"
              f"{c['scale']:>14,.0f}")
    top = d["ranking"][0]
    print(f"\n  Score breakdown for #1 {top['airport']}:")
    components(top["components"])
    print("\n  Outside the ranking universe (FAA non-hubs, shown for completeness):")
    for o in d["outside_ranking_universe"]:
        print(f"    {o['airport']} ({o['state']}): {o['passengers']:,} passengers")
    s = stab["data"]
    print(f"\n  Rank stability (top {s['top_n']}): same set under all presets = {s['same_top_n_set']}, "
          f"same order = {s['same_top_n_order']}")
    for preset, top3 in s["rankings"].items():
        print(f"    {preset:<16} {', '.join(top3)}")
    footer(inv, stab)


# ------------------------------------------------------------------------------------------ Q2
def q2(frames) -> None:
    header("Q2. Compare LA (LAX) and Santa Ana (SNA) airport congestion levels.")
    cong = ranking.congestion_index(frames, ["LA", "Santa Ana"])
    delays = [metrics.delay_profile(frames, a) for a in ("LAX", "SNA")]
    ops = [metrics.ops_per_runway(frames, a) for a in ("LAX", "SNA")]
    profiles = [metrics.airport_profile(frames, a) for a in ("LAX", "SNA")]
    for row in cong["data"]["airports"]:
        print(f"\n  {row['airport']} ({row['hub']}): congestion index {row['congestion_index']} / 100, "
              f"national rank {row['national_rank']} of {row['ranked_airports']}")
        components(row["components"])
    print(f"\n  {'metric':<34}{'LAX':>14}{'SNA':>14}{'national':>12}")
    dl, ds = delays[0]["data"], delays[1]["data"]
    for key, label in [("delay_rate_pct", "arrival delay rate (>=15 min) %"),
                       ("nas_share_pct", "NAS share of delay minutes %"),
                       ("cancellation_rate_pct", "cancellation rate %")]:
        print(f"  {label:<34}{dl[key]:>14}{ds[key]:>14}{dl['national'][key]:>12}")
    ol, os_ = ops[0]["data"], ops[1]["data"]
    print(f"  {'operations per runway':<34}{ol['operations_per_runway']:>14,.0f}{os_['operations_per_runway']:>14,.0f}")
    print(f"  {'commercial runways (>= 5,000 ft)':<34}{ol['runway_count']:>14.0f}{os_['runway_count']:>14.0f}")
    pl, ps = profiles[0]["data"], profiles[1]["data"]
    print(f"  {'load factor (avg)':<34}{pl['load_factor_avg']:>14}{ps['load_factor_avg']:>14}")
    print(f"  {'passengers':<34}{pl['passengers']:>14,}{ps['passengers']:>14,}")
    print(f"\n  Runways counted (>= {MIN_COMMERCIAL_RUNWAY_FT:,} ft):")
    for o in ops:
        od = o["data"]
        excluded = f"; excluded (short): {', '.join(od['runways_excluded_short'])}" if od["runways_excluded_short"] else ""
        print(f"    {od['airport']}: {', '.join(od['runways_counted'])}{excluded}")
    footer(cong, *delays, *ops, *profiles, periods={
        "Delay & traffic (BTS delay cause, T-100)": f"{window_label(WINDOW)} (trailing 12 months)",
        "Operations (Socrata)": ops[0]["data"]["period"],
    })


# ------------------------------------------------------------------------------------------ Q3
def q3(frames) -> None:
    header("Q3. What is the percentage of long-haul flights out of Anchorage (ANC)?")
    lh = metrics.long_haul_share(frames, "Anchorage")
    d = lh["data"]
    print(f"  {d['long_haul_departure_share_pct']}% of {d['total_passenger_departures']:,} passenger departures "
          f"from {d['airport']} fly >= {d['default_threshold_mi']:,} statute miles "
          f"({d['long_haul_seat_share_pct']}% of seats).\n")
    print(f"  {'threshold (mi)':<16}{'departures %':>14}{'seats %':>10}")
    for s in d["sensitivity"]:
        print(f"  {s['threshold_mi']:<16,}{s['departure_share_pct']:>14}{s['seat_share_pct']:>10}")
    key = f"top_routes_at_or_above_{d['default_threshold_mi']}_mi"
    print(f"\n  Top routes >= {d['default_threshold_mi']:,} mi:")
    for r in d[key]:
        print(f"    ANC-{r['dest']} ({r['dest_country']}) {r['distance_mi']:,.0f} mi: "
              f"{r['departures']:,} departures, {r['seats']:,} seats")
    footer(lh)


# ------------------------------------------------------------------------------------------ Q4
def q4(frames) -> None:
    header("Q4. What is the unmet flight demand at SFO and why?")
    u = ranking.unmet_demand(frames, "San Francisco")
    dp = metrics.delay_profile(frames, "SFO")
    d = u["data"]
    print(f"  Unmet-demand proxy score: {d['unmet_demand_score']} / 100 ({d['hub']})\n")
    components(d["components"], "  ")
    print(f"\n  Drivers (above the median of {d['peer_group']}):")
    for x in d["drivers_above_peer_median"]:
        print(f"    {x['component']:<16} SFO {x['value']:>8}  vs peer median {x['peer_median']:>8}  "
              f"(percentile {x['percentile']})")
    dd = dp["data"]
    print("\n  Plausibility check - SFO vs national vs large-hub median:")
    print(f"    {'metric':<26}{'SFO':>8}{'national':>10}{'large-hub med.':>16}")
    for key, label in [("delay_rate_pct", "delay rate %"), ("nas_share_pct", "NAS share of delay min %"),
                       ("cancellation_rate_pct", "cancellation rate %")]:
        print(f"    {label:<26}{dd[key]:>8}{dd['national'][key]:>10}{dd['hub_peer_median'][key]:>16}")
    print(f"    (national = {dd['national']['arrivals']:,.0f} reporting-carrier arrivals at all airports)")
    print("\n  Capacity gap estimate (extra seats needed to reach the target load factor):")
    for target, entry in d["capacity_gap"].items():
        for period in ("window", "peak_month"):
            g = entry[period]
            label = "12-month window" if period == "window" else f"peak month {g['month']}"
            print(f"    target {g['target_load_factor']:.2f} | {label:<20} current LF {g['current_load_factor']:.3f}"
                  f" -> +{g['extra_seats']:,} seats ({g['extra_seats_pct']}%), ~{g['extra_departures']:,} departures")
        print(f"      -> {entry['interpretation']}")
    footer(u, dp)


if __name__ == "__main__":
    frames = load_frames()
    for question in (q1, q2, q3, q4):
        question(frames)
