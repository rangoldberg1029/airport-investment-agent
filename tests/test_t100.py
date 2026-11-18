"""T-100 passenger filter and route aggregation."""

from data.sources.t100 import aggregate_routes, filter_passenger_service


def test_filter_keeps_only_passenger_service_from_us(t100_raw):
    kept = filter_passenger_service(t100_raw)
    assert sorted(kept["CLASS"]) == ["F", "F", "L"]
    assert (kept["AIRCRAFT_CONFIG"] != 2).all()
    assert (kept["ORIGIN_COUNTRY"] == "US").all()


def test_aggregate_sums_into_one_route_month(t100_raw):
    out = aggregate_routes(filter_passenger_service(t100_raw))
    assert len(out) == 1
    row = out.iloc[0]
    assert row.dep_performed == 9 + 5 + 2
    assert row.seats == 1800 + 900 + 300
    assert row.passengers == 1500 + 800 + 250
    assert row.distance_mi == 2846


def test_us_schedule_columns_exclude_rows_without_schedule(t100_raw):
    row = aggregate_routes(filter_passenger_service(t100_raw)).iloc[0]
    # the charter row (scheduled = 0, performed = 2) must not enter the completion-gap columns
    assert row.dep_scheduled_us == 10 + 5
    assert row.dep_performed_us == 9 + 5


def test_distance_uses_max_when_values_differ(t100_raw):
    t100_raw.loc[1, "DISTANCE"] = 2850.0
    row = aggregate_routes(filter_passenger_service(t100_raw)).iloc[0]
    assert row.distance_mi == 2850
