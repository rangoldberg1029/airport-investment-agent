"""Probe B: BTS T-100 Segment (All Carriers) on TranStats - route-level data.

The download is an ASP.NET postback form (no direct URL), so it is done MANUALLY in a
browser (steps in data/raw/README.md). This script only:
  1. checks the download page is reachable and prints "Latest Available Data",
  2. if manually downloaded files exist in data/raw/t100_segment/, prints sample rows.

Run:  python scripts/spike/probe_b_t100_segment.py
"""

import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

TEST_AIRPORTS = ["BOS", "LAX", "SNA", "ANC", "SFO"]
PAGE = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "t100_segment"


def check_page() -> None:
    """Print latest available month and the field names offered on the download form."""
    html = requests.get(PAGE, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).text
    latest = re.search(r"Latest Available Data:\s*([^<]+)", html)
    print("Latest available:", latest.group(1).strip() if latest else "not found")
    fields = sorted(set(re.findall(r'name="([A-Z_]+)"', html)))
    print("Fields on form:", fields)


def read_manual_files() -> None:
    """Read any CSV (or zipped CSV) the user downloaded and print rows for test airports."""
    files = sorted(RAW_DIR.glob("*.zip")) + sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print(f"\nNo files in {RAW_DIR} yet - do the manual download in data/raw/README.md.")
        return
    for path in files:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                df = pd.read_csv(zf.open(csv_name), low_memory=False)
        else:
            df = pd.read_csv(path, low_memory=False)
        print(f"\n=== {path.name}: {df.shape[0]:,} rows")
        print("columns:", list(df.columns))
        print("year/month:", sorted(df.groupby(["YEAR", "MONTH"]).groups))
        sample = df[df["ORIGIN"].isin(TEST_AIRPORTS)].head(10)
        print(sample.to_string())


if __name__ == "__main__":
    check_page()
    read_manual_files()
