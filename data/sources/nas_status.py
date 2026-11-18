"""FAA NAS Status (nasstatus.faa.gov) - live airport events, fetched at query time.

Context only: never stored, never used in scoring. The endpoints are public but not formally
documented as an API (verified in the data spike), so failures are expected and handled.
"""

import re
from datetime import datetime, timezone

import requests

EVENTS_URL = "https://nasstatus.faa.gov/api/airport-events"
STATUS_XML_URL = "https://nasstatus.faa.gov/api/airport-status-information"
EVENT_TYPES = ("groundStop", "groundDelay", "airportClosure", "arrivalDelay", "departureDelay",
               "deicing", "airportConfig", "freeForm")
EVENT_FIELDS = ("simpleText", "text", "reason", "avgDelay", "maxDelay", "minDelay", "trend",
                "startTime", "endTime", "updatedAt")


def fetch_feed(timeout: int = 15) -> tuple[list[dict], str]:
    """Return (events, feed timestamp). Timestamp from the XML feed, else retrieval time (UTC)."""
    events = requests.get(EVENTS_URL, timeout=timeout).json()
    try:
        xml = requests.get(STATUS_XML_URL, timeout=timeout).text
        stamp = re.search(r"<Update_Time>([^<]+)</Update_Time>", xml).group(1)
    except Exception:
        stamp = datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S %Y GMT") + " (retrieval time)"
    return events, stamp


def parse_airport_events(events: list[dict], iata: str) -> list[dict]:
    """Active events for one airport, keeping only non-empty event types and useful fields."""
    out = []
    for event in events:
        if event.get("airportId") != iata:
            continue
        for kind in EVENT_TYPES:
            detail = event.get(kind)
            if detail:
                fields = {k: detail[k] for k in EVENT_FIELDS if isinstance(detail, dict) and detail.get(k)}
                out.append({"type": kind, **fields})
    return out
