"""KPI functions. Pure: input is a Frames object (DataFrames), output is the standard envelope.

All per-airport numbers come from one table, `kpi_table()`, so the same KPI is identical everywhere.
"""

import pandas as pd

from data.repository import Frames
from scoring.common import (
    Window, envelope, in_window, pct, resolve_airport, rnd, shift_month, window_label,
)
from scoring.config import (
    DEFAULT_LONG_HAUL_MI, HUB_NAMES, LONG_HAUL_THRESHOLDS_MI, PRIOR_WINDOW,
    RANKING_UNIVERSE_HUBS, SOURCE_LABELS, WINDOW,
)

CAUSE_COLUMNS = ["carrier_delay_min", "weather_delay_min", "nas_delay_min",
                 "security_delay_min", "late_aircraft_delay_min"]

PERIOD = f"{window_label(WINDOW)} (trailing 12 months); prior window {window_label(PRIOR_WINDOW)}"
COMPLETION_CAVEAT = ("Completion gap covers US carriers only: foreign carriers report 0 scheduled "
                     "departures in T-100, so their flights are excluded from this metric.")
NAS_ASSUMPTION = ("NAS share = NAS delay minutes / sum of the five delay-cause minutes "
                  "(NAS = airport/airspace capacity, ATC, non-extreme weather)")
OPS_ASSUMPTION = "Operations = 2 x departures (arrivals assumed equal to departures), all carriers incl. cargo"
OPS_CAVEAT = "Operations per runway is a rough proxy: ignores runway configuration, length, simultaneous use and general aviation"
DELAY_CAVEAT = ("Delay data covers large US 'reporting carriers' and domestic arrivals only "
                "(international and small-carrier flights are not included).")


def _safe_div(a, b):
    """a / b with NaN where b <= 0 (works for scalars and Series)."""
    if isinstance(b, pd.Series):
        return a / b.where(b > 0)
    return a / b if b and b > 0 else float("nan")


# ---------------------------------------------------------------- building blocks

def traffic_kpis(t100: pd.DataFrame, window: Window, prior: Window) -> pd.DataFrame:
    """Passengers, seats, departures, LF (avg + peak month) and YoY growth per origin airport."""
    tw = t100[in_window(t100, window)]
    g = tw.groupby("origin")
    k = pd.DataFrame({
        "state_t100": g["origin_state"].first(),
        "passengers": g["passengers"].sum(),
        "seats": g["seats"].sum(),
        "departures": g["dep_performed"].sum(),
        "dep_scheduled_us": g["dep_scheduled_us"].sum(),
        "dep_performed_us": g["dep_performed_us"].sum(),
    })
    k["lf_avg"] = _safe_div(k["passengers"], k["seats"])

    monthly = tw.groupby(["origin", "year", "month"])[["passengers", "seats", "dep_performed"]].sum()
    monthly = monthly[monthly["seats"] > 0]
    monthly["lf"] = monthly["passengers"] / monthly["seats"]
    peak = monthly.loc[monthly.groupby(level="origin")["lf"].idxmax()].reset_index().set_index("origin")
    k["lf_peak"] = peak["lf"]
    k["lf_peak_month"] = peak["year"].astype(str) + "-" + peak["month"].map("{:02d}".format)
    k["peak_passengers"], k["peak_seats"], k["peak_departures"] = (
        peak["passengers"], peak["seats"], peak["dep_performed"])

    prior_pax = t100[in_window(t100, prior)].groupby("origin")["passengers"].sum()
    k["passengers_prior"] = prior_pax
    k["growth_yoy"] = k["passengers"] / prior_pax.where(prior_pax > 0) - 1
    k["completion_gap"] = 1 - _safe_div(k["dep_performed_us"], k["dep_scheduled_us"])
    return k


def long_haul_kpis(t100: pd.DataFrame, window: Window, thresholds=LONG_HAUL_THRESHOLDS_MI) -> pd.DataFrame:
    """Departure- and seat-weighted share of segments with distance >= threshold, per origin."""
    tw = t100[in_window(t100, window)]
    total = tw.groupby("origin")[["dep_performed", "seats"]].sum()
    out = pd.DataFrame(index=total.index)
    for t in thresholds:
        lh = tw[tw["distance_mi"] >= t].groupby("origin")[["dep_performed", "seats"]].sum()
        lh = lh.reindex(total.index, fill_value=0)
        out[f"lh_dep_share_{t}"] = _safe_div(lh["dep_performed"], total["dep_performed"])
        out[f"lh_seat_share_{t}"] = _safe_div(lh["seats"], total["seats"])
    return out


def delay_kpis(delay: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Delay rate, NAS share (of the five cause columns - agreed denominator) and cancellation rate."""
    d = delay[in_window(delay, window)].groupby("airport")[
        ["arr_flights", "arr_del15", "arr_cancelled"] + CAUSE_COLUMNS].sum()
    causes = d[CAUSE_COLUMNS].sum(axis=1)
    return pd.DataFrame({
        "arr_flights": d["arr_flights"],
        "delay_rate": _safe_div(d["arr_del15"], d["arr_flights"]),
        "nas_share": _safe_div(d["nas_delay_min"], causes),
        "cancellation_rate": _safe_div(d["arr_cancelled"], d["arr_flights"]),
    })


def national_delay(delay: pd.DataFrame, window: Window) -> dict:
    """Delay rate / NAS share / cancellation rate over all airports in the delay data."""
    d = delay[in_window(delay, window)]
    causes = d[CAUSE_COLUMNS].sum().sum()
    return {
        "delay_rate": d["arr_del15"].sum() / d["arr_flights"].sum(),
        "nas_share": d["nas_delay_min"].sum() / causes,
        "cancellation_rate": d["arr_cancelled"].sum() / d["arr_flights"].sum(),
        "arr_flights": d["arr_flights"].sum(),
    }


def socrata_window(socrata: pd.DataFrame, window: Window) -> Window:
    """Latest 12 months available in Socrata that end no later than `window` end."""
    ym = socrata["year"] * 100 + socrata["month"]
    end_ym = int(ym[ym <= window[1][0] * 100 + window[1][1]].max())
    end = (end_ym // 100, end_ym % 100)
    return shift_month(end, -11), end


def ops_kpis(socrata: pd.DataFrame, airports: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Operations per open runway: 2 x all-carrier departures (arrivals ~ departures) / runways."""
    sw = socrata_window(socrata, window)
    deps = socrata[in_window(socrata, sw)].groupby("airport")["departures"].sum()
    runways = airports.set_index("iata")["runway_count"]
    out = pd.DataFrame({"socrata_departures": deps, "runway_count": runways})
    out["ops_per_runway"] = _safe_div(2 * out["socrata_departures"], out["runway_count"])
    return out


def faa_kpis(faa: pd.DataFrame) -> pd.DataFrame:
    """Latest FAA calendar year: hub class, state, enplanements and CY growth (cross-check)."""
    latest = int(faa["year"].max())
    cur = faa[faa["year"] == latest].set_index("airport")
    prev = faa[faa["year"] == latest - 1].set_index("airport")["enplanements"]
    return pd.DataFrame({
        "hub": cur["hub"], "state_faa": cur["state"],
        "faa_enplanements": cur["enplanements"],
        "faa_growth": cur["enplanements"] / prev.where(prev > 0) - 1,
        "faa_year": latest,
    })


def kpi_table(frames: Frames, window: Window = WINDOW, prior: Window = PRIOR_WINDOW) -> pd.DataFrame:
    """One row per airport (index = IATA) with every KPI. Cached on the Frames object."""
    key = ("kpi", window, prior)
    if key in frames.cache:
        return frames.cache[key]
    k = traffic_kpis(frames.t100, window, prior)
    k = k.join(long_haul_kpis(frames.t100, window), how="left")
    k = k.join(delay_kpis(frames.delay, window), how="left")
    k = k.join(ops_kpis(frames.socrata, frames.airports, window), how="left")
    k = k.join(faa_kpis(frames.faa), how="left")
    k["state"] = k["state_faa"].fillna(k["state_t100"])
    k["in_universe"] = k["hub"].isin(RANKING_UNIVERSE_HUBS)
    k.index.name = "airport"
    frames.cache[key] = k
    return k


def hub_label(hub) -> str:
    return HUB_NAMES.get(hub, "not in FAA commercial-service list") if isinstance(hub, str) else "not in FAA list"


def _row(frames: Frames, name: str) -> tuple[pd.Series, list[str]]:
    """Resolve a name and return its KPI row plus any name-resolution assumption."""
    iata, assumption = resolve_airport(name, frames.airports)
    k = kpi_table(frames)
    if iata not in k.index:
        raise ValueError(f"{iata} has no passenger service in T-100 for {window_label(WINDOW)}")
    return k.loc[iata], [assumption] if assumption else []


# ---------------------------------------------------------------- public functions (envelopes)

def airport_profile(frames: Frames, name: str) -> dict:
    """Traffic, load factor (average and peak month) and YoY passenger growth."""
    r, assumptions = _row(frames, name)
    data = {
        "airport": r.name, "hub": hub_label(r["hub"]), "state": r["state"],
        "passengers": int(r["passengers"]), "seats": int(r["seats"]), "departures": int(r["departures"]),
        "load_factor_avg": rnd(r["lf_avg"]),
        "load_factor_peak_month": rnd(r["lf_peak"]), "peak_month": r["lf_peak_month"],
        "passengers_prior_window": int(r["passengers_prior"]) if r["passengers_prior"] == r["passengers_prior"] else None,
        "growth_yoy_pct": pct(r["growth_yoy"]),
        f"faa_cy{int(r['faa_year']) if r['faa_year'] == r['faa_year'] else ''}_enplanement_growth_pct": pct(r["faa_growth"]),
    }
    return envelope(
        data, [SOURCE_LABELS["t100"], SOURCE_LABELS["faa"]], PERIOD,
        assumptions + ["Load factor = passengers / seats on departing passenger segments",
                       "Growth = passengers in window vs the prior 12 months (T-100); FAA calendar-year growth shown as cross-check"],
        ["T-100 counts passengers per flight segment (connecting passengers count at each departure)"],
    )


def delay_profile(frames: Frames, name: str) -> dict:
    """Delay rate, NAS share and cancellation rate, with national and hub-peer comparisons."""
    r, assumptions = _row(frames, name)
    k = kpi_table(frames)
    nat = national_delay(frames.delay, WINDOW)
    peers = k[(k["hub"] == r["hub"]) & k["in_universe"]]
    cols = ["delay_rate", "nas_share", "cancellation_rate"]
    data = {
        "airport": r.name, "hub": hub_label(r["hub"]), "arrivals": r["arr_flights"],
        **{f"{c}_pct": pct(r[c]) for c in cols},
        "national": {f"{c}_pct": pct(nat[c]) for c in cols} | {"arrivals": nat["arr_flights"]},
        "hub_peer_median": {f"{c}_pct": pct(peers[c].median()) for c in cols}
        | {"peer_group": hub_label(r["hub"]), "n_peers": len(peers)},
    }
    caveats = [DELAY_CAVEAT]
    if r["arr_flights"] != r["arr_flights"]:
        caveats.append(f"No delay data for {r.name} (no reporting-carrier arrivals in the window)")
    return envelope(
        data, [SOURCE_LABELS["delay"]], PERIOD,
        assumptions + ["Delay rate = arrivals delayed >= 15 min / arrivals",
                       NAS_ASSUMPTION],
        caveats,
    )


def completion_gap(frames: Frames, name: str) -> dict:
    r, assumptions = _row(frames, name)
    data = {"airport": r.name, "completion_gap_pct": pct(r["completion_gap"]),
            "departures_scheduled_us": int(r["dep_scheduled_us"]),
            "departures_performed_us": int(r["dep_performed_us"])}
    return envelope(data, [SOURCE_LABELS["t100"]], PERIOD,
                    assumptions + ["Completion gap = 1 - departures performed / departures scheduled"],
                    [COMPLETION_CAVEAT])


def long_haul_share(frames: Frames, name: str, thresholds=LONG_HAUL_THRESHOLDS_MI,
                    default: int = DEFAULT_LONG_HAUL_MI, top_n: int = 5) -> dict:
    """Long-haul share at every threshold (departure- and seat-weighted) plus top long-haul routes."""
    r, assumptions = _row(frames, name)
    thresholds = tuple(sorted(set(thresholds) | {default}))
    lh = long_haul_kpis(frames.t100, WINDOW, thresholds).loc[r.name]
    tw = frames.t100[in_window(frames.t100, WINDOW) & (frames.t100["origin"] == r.name)]
    routes = (tw[tw["distance_mi"] >= default]
              .groupby(["dest", "dest_country"])
              .agg(distance_mi=("distance_mi", "max"), departures=("dep_performed", "sum"), seats=("seats", "sum"))
              .sort_values("departures", ascending=False).head(top_n).reset_index())
    data = {
        "airport": r.name,
        "default_threshold_mi": default,
        "long_haul_departure_share_pct": pct(lh[f"lh_dep_share_{default}"]),
        "long_haul_seat_share_pct": pct(lh[f"lh_seat_share_{default}"]),
        "total_passenger_departures": int(r["departures"]),
        "sensitivity": [
            {"threshold_mi": t, "departure_share_pct": pct(lh[f"lh_dep_share_{t}"]),
             "seat_share_pct": pct(lh[f"lh_seat_share_{t}"])} for t in thresholds
        ],
        f"top_routes_at_or_above_{default}_mi": routes.to_dict("records"),
    }
    return envelope(
        data, [SOURCE_LABELS["t100"]], PERIOD,
        assumptions + [f"Long-haul = segment distance >= {default:,} statute miles (assumption; see sensitivity)",
                       "Passenger service only: CLASS F/L, freighters excluded"],
        ["Distance is the flight segment, not the passenger's full journey",
         "All-cargo flights are excluded (relevant at cargo hubs such as ANC)"],
    )


def ops_per_runway(frames: Frames, name: str) -> dict:
    r, assumptions = _row(frames, name)
    sw = socrata_window(frames.socrata, WINDOW)
    data = {"airport": r.name, "operations_per_runway": rnd(r["ops_per_runway"], 0),
            "runway_count": r["runway_count"], "all_carrier_departures": r["socrata_departures"],
            "period": window_label(sw)}
    return envelope(
        data, [SOURCE_LABELS["socrata"], SOURCE_LABELS["ourairports"]], window_label(sw),
        assumptions + [OPS_ASSUMPTION],
        [OPS_CAVEAT,
         f"Uses Socrata's latest 12 months ({window_label(sw)}), one month earlier than the standard window"],
    )
