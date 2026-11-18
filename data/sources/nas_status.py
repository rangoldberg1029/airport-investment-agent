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
# Fields kept per event (names as sent by the feed; verified against the live feed).
EVENT_FIELDS = ("impactingCondition", "reason", "avgDelay", "maxDelay", "averageDelay", "arrivalDeparture",
                "trend", "startTime", "endTime", "programExpirationTime", "eventTime", "expTime", "text",
                "arrivalRunwayConfig", "departureRunwayConfig", "arrivalRate")
EVENT_NAMES = {"groundStop": "Ground stop", "groundDelay": "Ground delay program", "airportClosure": "Closure",
               "arrivalDelay": "Arrival delays", "departureDelay": "Departure delays", "deicing": "De-icing",
               "airportConfig": "Airport configuration", "freeForm": "Notice"}
# FAA reason prefixes, e.g. "WX:Thunderstorms" -> "Weather: thunderstorms".
REASON_PREFIXES = {"WX": "Weather", "VOL": "Volume", "RWY": "Runway", "EQ": "Equipment", "TM": "Traffic management",
                   "OTHER": "Other", "STAFF": "Staffing"}


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


# ---------------------------------------------------------------- readable descriptions (pure)

def readable_reason(reason) -> str | None:
    """'WX:Thunderstorms' -> 'Weather: thunderstorms'; 'wind' -> 'wind'."""
    if not reason:
        return None
    text = str(reason).strip()
    prefix, sep, rest = text.partition(":")
    if sep and prefix.upper() in REASON_PREFIXES:
        return f"{REASON_PREFIXES[prefix.upper()]}: {rest.strip().lower()}"
    return text.lower()


def readable_time(stamp, tz_name: str | None, today=None) -> str | None:
    """'2026-11-18T20:00:00Z' -> '20:00 UTC (12:00 PST)'; the date is added when it is not `today` (UTC)."""
    if not stamp:
        return None
    from zoneinfo import ZoneInfo

    try:
        utc = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return str(stamp)
    day = "" if today is not None and utc.date() == today else f"{utc:%b} {utc.day} "
    text = f"{day}{utc:%H:%M} UTC"
    if tz_name:
        local = utc.astimezone(ZoneInfo(tz_name))
        text += f" ({local:%H:%M %Z})"
    return text


FEED_CODE_RE = re.compile(r"\s*\(\s*\.?\d+(?:\.\d+)?\s*\)")      # raw feed codes such as "(.308)"


def clean_config(value) -> str | None:
    """Runway configuration without raw feed codes: '28L/28R (.308)' -> '28L/28R'."""
    if not value:
        return None
    text = re.sub(r"\s+", " ", FEED_CODE_RE.sub("", str(value))).strip(" ,;")
    return text or None


def _minutes(value) -> str | None:
    """92.0 -> '92 min'; '16 minutes' -> '16 min'."""
    if value in (None, ""):
        return None
    number = re.search(r"\d+(?:\.\d+)?", str(value))
    return f"{float(number.group()):.0f} min" if number else None


def describe_event(event: dict, tz_name: str | None, today=None) -> str | None:
    """One readable line per event, e.g. 'Ground stop until 20:00 UTC (12:00 PST); reason: wind'.
    None when the event carries nothing worth showing."""
    kind = event.get("type")
    name = EVENT_NAMES.get(kind, kind or "Event")
    reason = readable_reason(event.get("impactingCondition") or event.get("reason"))
    parts = []
    if kind == "groundStop":
        until = readable_time(event.get("programExpirationTime") or event.get("endTime"), tz_name, today)
        head = f"{name} until {until}" if until else name
    elif kind == "groundDelay":
        stats = [f"avg {_minutes(event.get('avgDelay'))}" if _minutes(event.get("avgDelay")) else None,
                 f"max {_minutes(event.get('maxDelay'))}" if _minutes(event.get("maxDelay")) else None]
        head = name + (f": {', '.join(s for s in stats if s)}" if any(stats) else "")
        until = readable_time(event.get("endTime"), tz_name, today)
        if until:
            parts.append(f"until {until}")
    elif kind in ("arrivalDelay", "departureDelay"):
        window = event.get("arrivalDeparture") if isinstance(event.get("arrivalDeparture"), dict) else {}
        span = [_minutes(window.get("min")), _minutes(window.get("max"))]
        head = name + (f": {span[0]} to {span[1]}" if all(span) else
                       f": avg {_minutes(event.get('averageDelay'))}" if _minutes(event.get("averageDelay")) else "")
        trend = event.get("trend") or window.get("trend")
        if trend:
            parts.append(f"trend {str(trend).lower()}")
    elif kind == "airportClosure" or kind == "freeForm":
        if not event.get("text"):
            return None
        head = f"{name}: {event['text']}"
        window = [readable_time(event.get("startTime"), tz_name, today), readable_time(event.get("endTime"), tz_name, today)]
        if all(window):
            parts.append(f"from {window[0]} to {window[1]}")
    elif kind == "deicing":
        until = readable_time(event.get("expTime"), tz_name, today)
        head = f"{name} in effect" + (f" until {until}" if until else "")
    elif kind == "airportConfig":
        arrivals, departures = clean_config(event.get("arrivalRunwayConfig")), clean_config(event.get("departureRunwayConfig"))
        config = [f"arrival runways {arrivals}" if arrivals else None,
                  f"departure runways {departures}" if departures else None,
                  f"arrival rate {event['arrivalRate']}/hour" if event.get("arrivalRate") else None]
        config = [c for c in config if c]
        if not config:
            return None                      # never show an empty row
        head = f"{name}: {', '.join(config)}"
    else:
        return None
    if reason:
        parts.append(reason if ":" in reason else f"reason: {reason}")   # "Weather: wind" / "reason: wind"
    return "; ".join([head, *parts])


def describe_events(events: list[dict], tz_name: str | None, today=None) -> list[str]:
    """Readable lines for an airport's events, most disruptive first, empty events dropped."""
    order = {kind: i for i, kind in enumerate(EVENT_TYPES)}
    lines = [describe_event(e, tz_name, today) for e in sorted(events, key=lambda e: order.get(e.get("type"), 99))]
    return [line for line in lines if line]
