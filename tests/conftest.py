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
