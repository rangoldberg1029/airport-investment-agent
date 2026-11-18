"""BTS Airline On-Time Delay Cause -> arrival airport x month table (carriers summed).

TranStats has no documented API: we POST the page's own period form, then GET the
"Download Raw Data" link it renders (verified in the data spike, scripts/spike/probe_c_delay_cause.py).
"""

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from data.sources.common import Provenance, load_with_fallback

SOURCE = "delay_cause"
URL = "https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp"
ACCESS = "public web form (scripted)"
CACHE_FILE = "delay_cause.parquet"
MONTHS = 24

SUM_COLUMNS = {
    "arr_flights": "arr_flights",
    "arr_del15": "arr_del15",
    "arr_cancelled": "arr_cancelled",
    "arr_diverted": "arr_diverted",
    "arr_delay": "arr_delay_min",
    "carrier_delay": "carrier_delay_min",
    "weather_delay": "weather_delay_min",
    "nas_delay": "nas_delay_min",
    "security_delay": "security_delay_min",
    "late_aircraft_delay": "late_aircraft_delay_min",
}


def _latest_period_id(html: str) -> int:
    """First option of the PeriodFrom select = latest month; id = year*12 + month."""
    block = html[html.find("NAME='PeriodFrom'"):]
    return int(re.search(r"VALUE ='(\d+)'", block).group(1))


def download_raw(months: int = MONTHS) -> pd.DataFrame:
    """Download the trailing `months` months (all carriers, all airports) as carrier-level rows."""
    base = URL.rsplit("/", 1)[0] + "/"
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    end = _latest_period_id(session.get(URL, timeout=60).text)
    page = session.post(
        base + "OT_DelayCause1.asp?20=E",
        data={"Carrier": "All", "Airport": "All", "PeriodFrom": end - months + 1, "PeriodTo": end},
        timeout=60,
    ).text
    link = re.search(r"href='(ot_delaycause1_DL\.aspx\?[^']+)'", page).group(1)
    resp = session.get(base + link, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        return pd.read_csv(zf.open("Airline_Delay_Cause.csv"))


def aggregate_to_airport_month(raw: pd.DataFrame) -> pd.DataFrame:
    """Sum carrier rows into airport x month; keep the number of reporting carriers."""
    grouped = raw.groupby(["year", "month", "airport"])
    out = grouped[list(SUM_COLUMNS)].sum().rename(columns=SUM_COLUMNS)
    out["n_carriers"] = grouped["carrier"].nunique()
    out = out.reset_index()
    out[["year", "month"]] = out[["year", "month"]].astype(int)
    return out


def load(mode: str, processed_dir: Path) -> tuple[pd.DataFrame, Provenance]:
    return load_with_fallback(
        SOURCE, URL, ACCESS,
        fetch=lambda: aggregate_to_airport_month(download_raw()),
        cache_path=processed_dir / CACHE_FILE,
        mode=mode,
    )
