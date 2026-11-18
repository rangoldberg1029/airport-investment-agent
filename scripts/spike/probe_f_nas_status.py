"""Probe F: FAA NAS Status live airport events (context only, not used in scoring).

Two public endpoints on nasstatus.faa.gov (verified responding, no auth). They are not
formally documented as an API, so treat the format as subject to change.

Run:  python scripts/spike/probe_f_nas_status.py
"""

import json

import requests

TEST_AIRPORTS = ["BOS", "LAX", "SNA", "ANC", "SFO"]
EVENTS_JSON = "https://nasstatus.faa.gov/api/airport-events"
STATUS_XML = "https://nasstatus.faa.gov/api/airport-status-information"

if __name__ == "__main__":
    events = requests.get(EVENTS_JSON, timeout=30).json()
    print(f"{len(events)} airports with active events. Keys per event: {list(events[0].keys()) if events else '-'}")
    for event in events:
        if event["airportId"] in TEST_AIRPORTS:
            print(json.dumps(event, indent=2)[:1500])
    active = [e["airportId"] for e in events]
    print("Test airports with NO active event:", [a for a in TEST_AIRPORTS if a not in active])

    xml = requests.get(STATUS_XML, timeout=30).text
    print("\nXML feed (first 600 chars):\n", xml[:600])
