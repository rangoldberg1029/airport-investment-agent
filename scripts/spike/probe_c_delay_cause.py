"""Probe C: BTS Airline On-Time Delay Cause (carrier x airport x month).

The TranStats page OT_DelayCause1.asp has a period selector (POST) and then shows a
"Download Raw Data" link (GET) that returns a zip with Airline_Delay_Cause.csv.
Period ids are year*12 + month (e.g. Aug 2026 = 24320) - read from the page's <select>.

Run:  python scripts/spike/probe_c_delay_cause.py
"""

import io
import re
import zipfile

import pandas as pd
import requests

TEST_AIRPORTS = ["BOS", "LAX", "SNA", "ANC", "SFO"]
BASE = "https://www.transtats.bts.gov/OT_Delay/"


def latest_period_id(html: str) -> int:
    """The first option of the PeriodFrom select is the latest available month."""
    block = html[html.find("NAME='PeriodFrom'"):]
    return int(re.search(r"VALUE ='(\d+)'", block).group(1))


def download(months: int = 12) -> pd.DataFrame:
    """Download the trailing `months` months for all carriers and all airports."""
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    first = session.get(BASE + "OT_DelayCause1.asp", timeout=60).text
    end = latest_period_id(first)
    page = session.post(
        BASE + "OT_DelayCause1.asp?20=E",
        data={"Carrier": "All", "Airport": "All", "PeriodFrom": end - months + 1, "PeriodTo": end},
        timeout=60,
    ).text
    link = re.search(r"href='(ot_delaycause1_DL\.aspx\?[^']+)'", page).group(1)
    resp = session.get(BASE + link, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        return pd.read_csv(zf.open("Airline_Delay_Cause.csv"))


if __name__ == "__main__":
    df = download()
    print(f"rows={len(df):,}  columns={list(df.columns)}")
    print("months:", sorted({(y, m) for y, m in zip(df.year, df.month)}))
    cols = ["arr_flights", "arr_del15", "arr_cancelled", "arr_delay", "carrier_delay",
            "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]
    print("\nTrailing-12-month sums for test airports (all reporting carriers):")
    print(df[df.airport.isin(TEST_AIRPORTS)].groupby("airport")[cols].sum().to_string())
    print("\nSample raw rows:")
    print(df[df.airport.isin(TEST_AIRPORTS)].head(5).to_string())
