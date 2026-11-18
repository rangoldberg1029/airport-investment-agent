"""Small hand-made fixtures for ETL tests (no network)."""

import pandas as pd
import pytest


def t100_raw_row(**overrides) -> dict:
    row = {
        "YEAR": 2025, "MONTH": 9, "ORIGIN": "ANC", "DEST": "ORD",
        "ORIGIN_STATE_ABR": "AK", "DEST_STATE_ABR": "IL", "ORIGIN_COUNTRY": "US", "DEST_COUNTRY": "US",
        "CLASS": "F", "AIRCRAFT_CONFIG": 1, "DISTANCE": 2846.0,
        "DEPARTURES_SCHEDULED": 10.0, "DEPARTURES_PERFORMED": 9.0, "SEATS": 1800.0, "PASSENGERS": 1500.0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def t100_raw() -> pd.DataFrame:
    return pd.DataFrame([
        t100_raw_row(),                                                          # kept
        t100_raw_row(DEPARTURES_SCHEDULED=5.0, DEPARTURES_PERFORMED=5.0,
                     SEATS=900.0, PASSENGERS=800.0),                             # kept, same route-month
        t100_raw_row(CLASS="L", DEPARTURES_SCHEDULED=0.0, DEPARTURES_PERFORMED=2.0,
                     SEATS=300.0, PASSENGERS=250.0),                             # charter: kept, no schedule
        t100_raw_row(CLASS="G", AIRCRAFT_CONFIG=2, SEATS=0.0, PASSENGERS=0.0),   # scheduled all-cargo: dropped
        t100_raw_row(CLASS="P", AIRCRAFT_CONFIG=2, SEATS=0.0, PASSENGERS=0.0),   # charter cargo: dropped
        t100_raw_row(CLASS="F", AIRCRAFT_CONFIG=2, SEATS=0.0, PASSENGERS=0.0),   # freighter in class F: dropped
        t100_raw_row(ORIGIN="YVR", ORIGIN_COUNTRY="CA", ORIGIN_STATE_ABR=None),  # foreign origin: dropped
    ])


@pytest.fixture
def delay_raw() -> pd.DataFrame:
    base = {"year": 2025, "month": 9, "airport": "SFO", "arr_diverted": 0, "security_delay": 0}
    return pd.DataFrame([
        {**base, "carrier": "UA", "arr_flights": 100, "arr_del15": 30, "arr_cancelled": 2, "arr_delay": 1000,
         "carrier_delay": 300, "weather_delay": 100, "nas_delay": 400, "late_aircraft_delay": 200},
        {**base, "carrier": "AS", "arr_flights": 50, "arr_del15": 10, "arr_cancelled": 1, "arr_delay": 500,
         "carrier_delay": 100, "weather_delay": 0, "nas_delay": 300, "late_aircraft_delay": 100},
    ])


# ---------------------------------------------------------------- scoring fixtures
# Four airports over 24 months (Sep 2024 - Aug 2026). Hand-checked values for AAA, window Sep 2025 - Aug 2026:
#   routes per month: short 500 mi (10 dep, 1000 seats, 800 pax; 950 pax in 2026-07; 700 pax in prior window),
#                     3000 mi (2 dep, 400 seats, 360 pax), 4500 mi (1 dep, 300 seats, 240 pax)
#   window: departures 156, seats 20,400, passengers 16,950 -> LF 0.830882; peak 2026-07 LF 1550/1700
#   prior passengers 15,600 -> growth 16,950/15,600 - 1 = 0.086538
#   long-haul >=3000: departures 36/156, seats 8,400/20,400; >=4000: 12/156
#   completion gap (short route only has a US schedule: 11 scheduled, 10 performed) = 1 - 120/132
#   delay: per month 100 arrivals, 20 delayed, 2 cancelled; causes 200 min (NAS 50) but arr_delay 210
#   socrata: 300 departures/month, latest month 2026-07; 2 commercial runways (+1 short GA runway,
#   not counted) -> 2*3600/2 = 3600 ops per runway

def _months(start=(2024, 9), n=24):
    y, m = start
    for _ in range(n):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def make_t100(origin: str, state: str, short_pax: int, peak_pax: int, prior_pax: int) -> list[dict]:
    rows = []
    for y, m in _months():
        in_window = (y, m) >= (2025, 9)
        pax = (peak_pax if (y, m) == (2026, 7) else short_pax) if in_window else prior_pax
        base = {"year": y, "month": m, "origin": origin, "origin_state": state, "origin_country": "US",
                "dest_state": "XX", "dest_country": "US"}
        rows += [
            {**base, "dest": "S1", "distance_mi": 500.0, "dep_performed": 10, "dep_scheduled_us": 11,
             "dep_performed_us": 10, "seats": 1000, "passengers": pax},
            {**base, "dest": "L1", "distance_mi": 3000.0, "dep_performed": 2, "dep_scheduled_us": 0,
             "dep_performed_us": 0, "seats": 400, "passengers": 360},
            {**base, "dest": "L2", "distance_mi": 4500.0, "dep_performed": 1, "dep_scheduled_us": 0,
             "dep_performed_us": 0, "seats": 300, "passengers": 240},
        ]
    return rows


AIRPORT_SPECS = {  # iata: (state, hub, short_pax, peak_pax, prior_pax, delayed_per_month, runways)
    "AAA": ("MA", "L", 800, 950, 700, 20, 2),
    "BBB": ("MA", "L", 700, 720, 700, 10, 4),
    "CCC": ("CT", "M", 900, 990, 800, 30, 1),
    "DDD": ("MA", "N", 500, 520, 500, 5, 1),
}


@pytest.fixture
def frames():
    from data.repository import Frames

    t100, delay, socrata, faa, airports = [], [], [], [], []
    for iata, (state, hub, short, peak, prior, delayed, runways) in AIRPORT_SPECS.items():
        t100 += make_t100(iata, state, short, peak, prior)
        for y, m in _months():
            delay.append({"year": y, "month": m, "airport": iata, "arr_flights": 100, "arr_del15": delayed,
                          "arr_cancelled": 2, "arr_diverted": 0, "arr_delay_min": 210,
                          "carrier_delay_min": 100, "weather_delay_min": 10, "nas_delay_min": 50,
                          "security_delay_min": 0, "late_aircraft_delay_min": 40, "n_carriers": 1})
            if (y, m) <= (2026, 7):
                socrata.append({"year": y, "month": m, "airport": iata, "departures": 300,
                                "passengers": 0, "seats": 0, "freight_lbs": 0})
        faa += [{"airport": iata, "year": 2025, "enplanements": 20_000, "hub": hub, "state": state, "rank": 1},
                {"airport": iata, "year": 2024, "enplanements": 16_000, "hub": None, "state": state, "rank": None}]
        airports.append({"iata": iata, "icao": "K" + iata, "name": f"{iata} Intl", "city": f"City {iata}",
                         "state": state, "lat": 0.0, "lon": 0.0, "type": "large_airport",
                         "runway_count": runways, "longest_runway_ft": 10_000, "faa_hub": hub})
    for iata, city in [("LAX", "Los Angeles"), ("SNA", "Santa Ana"), ("ANC", "Anchorage"), ("SFO", "San Francisco")]:
        airports.append({"iata": iata, "icao": "K" + iata, "name": f"{city} Intl", "city": city, "state": "CA",
                         "lat": 0.0, "lon": 0.0, "type": "large_airport", "runway_count": 2,
                         "longest_runway_ft": 10_000, "faa_hub": "L"})
    # commercial runways: `runways` x 8,000 ft, plus one 4,999-ft and one exactly 5,000-ft runway rule check
    # below: every airport also gets a short GA runway (2,500 ft) that must NOT be counted.
    runway_rows = []
    for iata, spec in AIRPORT_SPECS.items():
        for i in range(spec[6]):
            runway_rows.append({"iata": iata, "runway": f"{i + 1}L/{i + 1}R", "length_ft": 8000.0,
                                "width_ft": 150.0, "surface": "ASP"})
        runway_rows.append({"iata": iata, "runway": "GA1/GA2", "length_ft": 2500.0, "width_ft": 75.0, "surface": "ASP"})
    return Frames(pd.DataFrame(t100), pd.DataFrame(delay), pd.DataFrame(socrata),
                  pd.DataFrame(faa), pd.DataFrame(airports), pd.DataFrame(runway_rows))
