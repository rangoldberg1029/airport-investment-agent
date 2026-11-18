"""Pure transforms of the online sources: delay cause, Socrata, FAA, OurAirports."""

import pandas as pd

from data.quality import delay_cause_sum_violations
from data.sources.delay_cause import aggregate_to_airport_month
from data.sources.faa import combine_years, parse_enplanements
from data.sources.ourairports import build as build_ourairports
from data.sources.ourairports import build_runways
from data.sources.socrata import parse_records


def test_delay_cause_sums_carriers(delay_raw):
    out = aggregate_to_airport_month(delay_raw)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.arr_flights == 150 and row.arr_del15 == 40 and row.arr_cancelled == 3
    assert row.arr_delay_min == 1500 and row.nas_delay_min == 700
    assert row.n_carriers == 2


def test_delay_cause_sum_check_catches_broken_row(delay_raw):
    good = aggregate_to_airport_month(delay_raw)
    assert delay_cause_sum_violations(good) == 0
    bad = good.assign(nas_delay_min=good.nas_delay_min + 200)  # causes now 13% above arr_delay
    assert delay_cause_sum_violations(bad) == 1


def test_socrata_parse_records_types_and_grain():
    records = [
        {"origin_airport_code": "BOS", "reporting_month": "2026-07-01T00:00:00.000",
         "total_departures": "17676", "total_passengers": "2013353", "total_seats": "2363679",
         "total_freight_lbs": "15936034"},
        {"origin_airport_code": "SNA", "reporting_month": "2026-06-01T00:00:00.000",
         "total_departures": "4528", "total_passengers": "486619", "total_seats": "627364"},  # freight missing
    ]
    out = parse_records(records)
    assert list(out.columns) == ["year", "month", "airport", "departures", "passengers", "seats", "freight_lbs"]
    assert out.loc[0, "passengers"] == 2013353 and out.loc[0, "month"] == 7
    assert out.loc[1, "freight_lbs"] == 0


def faa_sheet(new: int, old: int, rows: list[tuple]) -> pd.DataFrame:
    cols = ["Rank", "RO", "ST", "Locid", "City", "Airport Name", "S/L", "Hub",
            f"CY {new} Enplanements", f"CY {old} Enplanements", "% Change"]
    return pd.DataFrame(rows, columns=cols)


def test_faa_parses_both_header_variants_and_prefers_primary_year():
    cy25 = faa_sheet(25, 24, [(1, "NE", "MA", "BOS", "Boston", "Logan", "P", "L", 21_021_153, 21_090_721, -0.003),
                              (None, None, None, None, "footnote", None, None, None, None, None, None)])
    cy24 = faa_sheet(24, 23, [(1, "NE", "MA", "BOS", "Boston", "Logan", "P", "L", 21_090_000, 19_962_678, 0.056)])
    out = combine_years([parse_enplanements(cy25), parse_enplanements(cy24)]).set_index("year")
    assert sorted(out.index) == [2023, 2024, 2025]
    assert out.loc[2025, "enplanements"] == 21_021_153
    assert out.loc[2024, "enplanements"] == 21_090_000      # primary row of the CY2024 file wins
    assert out.loc[2024, "hub"] == "L"
    assert pd.isna(out.loc[2023, "hub"])                    # only known as a secondary column


def test_ourairports_filters_us_commercial_and_ignores_closed_runways():
    airports = pd.DataFrame([
        {"ident": "KBOS", "iata_code": "BOS", "icao_code": "KBOS", "iso_country": "US", "iso_region": "US-MA",
         "scheduled_service": "yes", "name": "Logan", "municipality": "Boston",
         "latitude_deg": 42.36, "longitude_deg": -71.0, "type": "large_airport"},
        {"ident": "KXYZ", "iata_code": None, "icao_code": None, "iso_country": "US", "iso_region": "US-MA",
         "scheduled_service": "yes", "name": "No IATA", "municipality": "X",
         "latitude_deg": 0, "longitude_deg": 0, "type": "small_airport"},
        {"ident": "CYVR", "iata_code": "YVR", "icao_code": "CYVR", "iso_country": "CA", "iso_region": "CA-BC",
         "scheduled_service": "yes", "name": "Vancouver", "municipality": "Vancouver",
         "latitude_deg": 49.2, "longitude_deg": -123.2, "type": "large_airport"},
    ])
    runways = pd.DataFrame([
        {"id": 1, "airport_ident": "KBOS", "length_ft": 10083, "closed": 0},
        {"id": 2, "airport_ident": "KBOS", "length_ft": 7000, "closed": 0},
        {"id": 3, "airport_ident": "KBOS", "length_ft": 12000, "closed": 1},
    ])
    out = build_ourairports(airports, runways)
    assert list(out["iata"]) == ["BOS"]
    assert out.loc[0, "state"] == "MA"
    assert out.loc[0, "runway_count"] == 2 and out.loc[0, "longest_runway_ft"] == 10083


def test_ourairports_runways_table_keeps_open_runways_of_us_commercial_airports():
    airports = pd.DataFrame([
        {"ident": "KSNA", "iata_code": "SNA", "icao_code": "KSNA", "iso_country": "US", "iso_region": "US-CA",
         "scheduled_service": "yes", "name": "John Wayne", "municipality": "Santa Ana",
         "latitude_deg": 33.7, "longitude_deg": -117.9, "type": "large_airport"},
    ])
    runways = pd.DataFrame([
        {"id": 1, "airport_ident": "KSNA", "le_ident": "02L", "he_ident": "20R", "length_ft": 5700,
         "width_ft": 150, "surface": "ASP", "closed": 0},
        {"id": 2, "airport_ident": "KSNA", "le_ident": "02R", "he_ident": "20L", "length_ft": 2886,
         "width_ft": 75, "surface": "ASP", "closed": 0},
        {"id": 3, "airport_ident": "KSNA", "le_ident": "H1", "he_ident": None, "length_ft": 60,
         "width_ft": 60, "surface": "CON", "closed": 1},
        {"id": 4, "airport_ident": "KXXX", "le_ident": "09", "he_ident": "27", "length_ft": 9000,
         "width_ft": 150, "surface": "ASP", "closed": 0},
    ])
    out = build_runways(airports, runways)
    assert list(out["runway"]) == ["02L/20R", "02R/20L"]          # closed + other airports dropped
    assert list(out["iata"]) == ["SNA", "SNA"]                    # length filter is applied in scoring, not here
