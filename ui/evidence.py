"""Evidence: turn a tool envelope into tables (DataFrames) and one Altair chart. Pure functions."""

import json

import altair as alt
import pandas as pd

from agent.tools import ToolCallRecord

Table = tuple[str, pd.DataFrame]


def _components_long(rows: list[dict], label_key: str = "airport") -> pd.DataFrame:
    """[{airport, components:[...]}] -> long table airport, component, value, percentile, weight, contribution."""
    out = []
    for r in rows:
        for c in r["components"]:
            out.append({label_key: r[label_key], "component": c["component"], "value": c["raw_value"],
                        "percentile": c["percentile"], "weight": c["weight"], "contribution": c["contribution"]})
    return pd.DataFrame(out)


def tables_rank(d: dict) -> list[Table]:
    ranking = d["ranking"]
    rows = []
    for r in ranking["ranking"]:
        raw = {c["component"]: c["raw_value"] for c in r["components"]}
        rows.append({"rank": r["rank"], "airport": r["airport"], "state": r["state"], "hub": r["hub"],
                     "investment score": r["investment_score"], "national rank": r["national_rank"],
                     "load factor": raw["demand_pressure"], "growth %": raw["growth"],
                     "congestion index": raw["congestion"], "passengers": raw["scale"]})
    tables = [("Investment score ranking", pd.DataFrame(rows))]
    tables.append(("Score components (percentile vs US hubs, weight, contribution)",
                   _components_long(ranking["ranking"])))
    stab = d["stability"]
    tables.append(("Rank stability: top 3 per weight preset",
                   pd.DataFrame([{"preset": k, "weights": json.dumps(stab["presets"][k]), "top 3": ", ".join(v)}
                                 for k, v in stab["rankings"].items()])))
    if ranking["outside_ranking_universe"]:
        tables.append(("Outside the ranking universe (FAA non-hubs)", pd.DataFrame(ranking["outside_ranking_universe"])))
    return tables


def tables_compare(d: dict) -> list[Table]:
    raw = pd.DataFrame(d["raw_values"])
    raw.index.name = "airport"
    return [("Metrics side by side", raw.T.reset_index().rename(columns={"index": "metric"})),
            ("Congestion index components", _components_long(d["congestion"]["airports"]))]


def tables_long_haul(d: dict) -> list[Table]:
    sens = pd.DataFrame(d["sensitivity"]).rename(columns={
        "threshold_mi": "threshold (mi)", "departure_share_pct": "departures %", "seat_share_pct": "seats %"})
    key = next(k for k in d if k.startswith("top_routes"))
    return [("Long-haul share by distance threshold", sens),
            (f"Top routes >= {d['default_threshold_mi']:,} mi", pd.DataFrame(d[key]))]


def tables_unmet(d: dict) -> list[Table]:
    u = d["unmet_demand"]
    comps = pd.DataFrame(u["components"]).rename(columns={"raw_value": "value"})
    drivers = pd.DataFrame(u["drivers_above_peer_median"])
    gaps = []
    for target, entry in u["capacity_gap"].items():
        for period in ("window", "peak_month"):
            g = entry[period]
            gaps.append({"target LF": g["target_load_factor"],
                         "period": "12-month window" if period == "window" else f"peak month {g['month']}",
                         "current LF": g["current_load_factor"], "extra seats": g["extra_seats"],
                         "extra seats %": g["extra_seats_pct"], "extra departures": g["extra_departures"]})
    dp = d["delay_profile"]
    vs = pd.DataFrame([
        {"metric": m, u["airport"]: dp[f"{m}_pct"], "national": dp["national"][f"{m}_pct"],
         "hub peer median": dp["hub_peer_median"][f"{m}_pct"]}
        for m in ("delay_rate", "nas_share", "cancellation_rate")])
    tables = [("Unmet-demand components", comps)]
    if not drivers.empty:
        tables.append((f"Drivers above the median of {u['peer_group']}", drivers))
    tables += [("Capacity-gap estimate (demand held fixed)", pd.DataFrame(gaps)),
               ("Delays vs national and peers (%)", vs)]
    return tables


def tables_profile(d: dict) -> list[Table]:
    p, dl, g, o = d["profile"], d["delay"], d["completion_gap"], d["ops"]
    rows = [
        ("passengers", p["passengers"]), ("seats", p["seats"]), ("departures", p["departures"]),
        ("load factor (avg)", p["load_factor_avg"]), (f"load factor (peak, {p['peak_month']})", p["load_factor_peak_month"]),
        ("passenger growth vs prior 12 months %", p["growth_yoy_pct"]),
        ("delay rate %", dl["delay_rate_pct"]), ("NAS share of delay minutes %", dl["nas_share_pct"]),
        ("cancellation rate %", dl["cancellation_rate_pct"]),
        ("completion gap (US carriers) %", g["completion_gap_pct"]),
        ("operations per commercial runway", o["operations_per_runway"]), ("commercial runways", o["runway_count"]),
    ]
    return [(f"{p['airport']} profile", pd.DataFrame(rows, columns=["metric", "value"]))]


def tables_live(d: dict) -> list[Table]:
    return [("Active FAA events", pd.DataFrame(d.get("events") or [{"status": d.get("status")}]))]


TABLES = {"rank_airports": tables_rank, "compare_congestion": tables_compare, "long_haul_share": tables_long_haul,
          "unmet_demand": tables_unmet, "get_airport_profile": tables_profile, "live_airport_status": tables_live}


def tables(record: ToolCallRecord) -> list[Table]:
    fn = TABLES.get(record.name)
    if record.error or fn is None:
        return []
    try:
        return fn(record.envelope["data"])
    except (KeyError, TypeError, ValueError, StopIteration):
        return []


# ---------------------------------------------------------------- charts

def _stacked(df: pd.DataFrame, y: str, title: str, order: list[str]) -> alt.Chart:
    return (alt.Chart(df, title=title)
            .mark_bar()
            .encode(x=alt.X("contribution:Q", title="contribution (points of 100)", stack="zero"),
                    y=alt.Y(f"{y}:N", sort=order, title=None),
                    color=alt.Color("component:N", legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=[y, "component", "value", "percentile", "weight", "contribution"])
            .properties(height=max(120, 34 * len(order))))


def chart(record: ToolCallRecord) -> alt.Chart | None:
    """One chart per tool call where it helps; None otherwise."""
    if record.error:
        return None
    d = record.envelope["data"]
    try:
        if record.name == "rank_airports":
            rows = d["ranking"]["ranking"]
            order = [r["airport"] for r in rows]
            return _stacked(_components_long(rows), "airport", "Investment score by component", order)
        if record.name == "compare_congestion":
            rows = d["congestion"]["airports"]
            return _stacked(_components_long(rows), "airport", "Congestion index by component",
                            [r["airport"] for r in rows])
        if record.name == "unmet_demand":
            u = d["unmet_demand"]
            df = _components_long([{"airport": u["airport"], "components": u["components"]}])
            return _stacked(df, "airport", f"Unmet-demand proxy by component ({u['unmet_demand_score']} / 100)",
                            [u["airport"]])
        if record.name == "long_haul_share":
            df = pd.DataFrame(d["sensitivity"]).melt("threshold_mi", var_name="measure", value_name="share %")
            df["measure"] = df["measure"].map({"departure_share_pct": "departures", "seat_share_pct": "seats"})
            return (alt.Chart(df, title=f"{d['airport']}: long-haul share by threshold")
                    .mark_bar()
                    .encode(x=alt.X("threshold_mi:O", title="threshold (statute miles)"),
                            xOffset="measure:N", y=alt.Y("share %:Q"),
                            color=alt.Color("measure:N", legend=alt.Legend(orient="bottom", title=None)),
                            tooltip=["threshold_mi", "measure", "share %"])
                    .properties(height=240))
    except (KeyError, TypeError, ValueError):
        return None
    return None
