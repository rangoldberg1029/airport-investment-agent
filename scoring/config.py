"""Scoring configuration: every assumption used by metrics and rankings, as named constants.

Change a value here and every answer changes consistently. Answers quote these values.
"""

# ---------------------------------------------------------------- time windows
# Trailing 12 months (inclusive (year, month)); agreed in scripts/spike/FINDINGS.md.
WINDOW = ((2025, 9), (2026, 8))
PRIOR_WINDOW = ((2024, 9), (2025, 8))   # the 12 months before WINDOW, for YoY growth

# ---------------------------------------------------------------- long-haul
# Segment distance in statute miles. 3,000 mi is an ASSUMPTION; answers always show all thresholds.
LONG_HAUL_THRESHOLDS_MI = (2500, 3000, 4000)
DEFAULT_LONG_HAUL_MI = 3000

# ---------------------------------------------------------------- capacity gap
# Load factor at which an airport's flights are considered "effectively full" (industry heuristic,
# conservative for investors). The gap = extra seats needed to bring LF down to this level.
TARGET_LOAD_FACTOR = 0.85
# Sensitivity line shown in the same answer: 0.80 = median LF of FAA large hubs, Sep 2025 - Aug 2026.
LOAD_FACTOR_SENSITIVITY = (0.80,)

# ---------------------------------------------------------------- ranking universe
# Only FAA large (L), medium (M) and small (S) hubs are ranked (~144 airports, >= ~0.5M enplanements).
# Non-hubs are shown in regional answers but marked "outside ranking universe".
RANKING_UNIVERSE_HUBS = ("L", "M", "S")
HUB_NAMES = {"L": "large hub", "M": "medium hub", "S": "small hub", "N": "non-hub"}

# ---------------------------------------------------------------- runways
# Only open runways at least this long (feet, inclusive) count for operations per runway, i.e. runways
# usable by commercial passenger jets. Drops short general-aviation runways such as SNA 02R/20L
# (2,886 ft) and BOS 15L/33R (2,557 ft); keeps SNA 02L/20R (5,700 ft), the only jet runway at SNA.
# Checked on all 384 open runways at FAA L/M/S hubs: no hub is left with zero runways at 5,000 ft
# (at 5,500 ft one hub would be, at 6,000 ft three).
MIN_COMMERCIAL_RUNWAY_FT = 5000

# ---------------------------------------------------------------- index weights (each sums to 1)
# Investment score components: demand pressure = load factor, growth = YoY passengers,
# congestion = congestion index, scale = passengers. All percentile-normalised vs the universe.
INVESTMENT_WEIGHTS = {"demand_pressure": 0.35, "growth": 0.25, "congestion": 0.25, "scale": 0.15}

# Named weight presets. A request like "weight growth higher" maps to a preset, so the same question always
# gives the same ranking; custom weights are used only when the user gives explicit numbers.
# All presets are also used to test rank stability.
DEFAULT_PRESET = "balanced"
WEIGHT_PRESETS = {
    "balanced": INVESTMENT_WEIGHTS,
    "demand-led": {"demand_pressure": 0.50, "growth": 0.20, "congestion": 0.15, "scale": 0.15},
    "congestion-led": {"demand_pressure": 0.25, "growth": 0.15, "congestion": 0.45, "scale": 0.15},
    "growth-led": {"demand_pressure": 0.25, "growth": 0.45, "congestion": 0.15, "scale": 0.15},
    "scale-led": {"demand_pressure": 0.25, "growth": 0.15, "congestion": 0.15, "scale": 0.45},
}

# Congestion index components. ops_per_runway is a rough proxy (see metrics.kpi_table).
CONGESTION_WEIGHTS = {"delay_rate": 0.35, "nas_share": 0.35, "ops_per_runway": 0.20, "cancellation_rate": 0.10}

# Unmet-demand proxy components (higher = more pressure for all of them).
UNMET_DEMAND_WEIGHTS = {
    "lf_avg": 0.25, "lf_peak": 0.20, "growth_yoy": 0.20,
    "delay_rate": 0.10, "nas_share": 0.15, "completion_gap": 0.10,
}

# ---------------------------------------------------------------- geography
REGIONS = {
    "new england": ("CT", "ME", "MA", "NH", "RI", "VT"),
    "west coast": ("CA", "OR", "WA"),
    "pacific northwest": ("OR", "WA", "ID"),
}

# name -> (IATA, alternatives). Alternatives are other commercial airports the name could mean;
# when present, the interpretation is reported as an assumption.
AIRPORT_ALIASES = {
    "la": ("LAX", ("BUR", "LGB", "ONT", "SNA")),
    "los angeles": ("LAX", ("BUR", "LGB", "ONT", "SNA")),
    "santa ana": ("SNA", ()),
    "john wayne": ("SNA", ()),
    "orange county": ("SNA", ()),
    "anchorage": ("ANC", ()),
    "san francisco": ("SFO", ()),
    "sf": ("SFO", ("OAK", "SJC")),
    "bay area": ("SFO", ("OAK", "SJC")),
    "boston": ("BOS", ()),
    "logan": ("BOS", ()),
    "new york": ("JFK", ("LGA", "EWR")),
    "nyc": ("JFK", ("LGA", "EWR")),
    "chicago": ("ORD", ("MDW",)),
    "washington": ("DCA", ("IAD", "BWI")),
    "dc": ("DCA", ("IAD", "BWI")),
    "hartford": ("BDL", ()),
    "providence": ("PVD", ()),
    "portland maine": ("PWM", ()),
    "portland": ("PDX", ("PWM",)),
}

# Non-US airports a user is likely to ask about (the 40 largest non-US destinations in the T-100 data, plus a
# few major hubs): IATA -> (name, city). Asking about them gets a clear "outside the US scope" answer
# instead of "unknown airport".
NON_US_AIRPORTS = {
    "LHR": ("London Heathrow Airport", "london"), "LGW": ("London Gatwick Airport", "london"),
    "YYZ": ("Toronto Pearson International Airport", "toronto"),
    "CUN": ("Cancun International Airport", "cancun"), "MEX": ("Mexico City International Airport", "mexico city"),
    "CDG": ("Paris Charles de Gaulle Airport", "paris"), "FRA": ("Frankfurt Airport", "frankfurt"),
    "ICN": ("Seoul Incheon International Airport", "seoul"), "YVR": ("Vancouver International Airport", "vancouver"),
    "AMS": ("Amsterdam Airport Schiphol", "amsterdam"), "GDL": ("Guadalajara International Airport", "guadalajara"),
    "HND": ("Tokyo Haneda Airport", "tokyo"), "NRT": ("Tokyo Narita International Airport", "tokyo"),
    "PUJ": ("Punta Cana International Airport", "punta cana"),
    "YUL": ("Montreal-Trudeau International Airport", "montreal"),
    "PTY": ("Tocumen International Airport", "panama city"), "SJD": ("Los Cabos International Airport", "los cabos"),
    "TPE": ("Taiwan Taoyuan International Airport", "taipei"), "IST": ("Istanbul Airport", "istanbul"),
    "YYC": ("Calgary International Airport", "calgary"), "DUB": ("Dublin Airport", "dublin"),
    "SAL": ("El Salvador International Airport", "san salvador"), "MAD": ("Madrid-Barajas Airport", "madrid"),
    "FCO": ("Rome Fiumicino Airport", "rome"), "MUC": ("Munich Airport", "munich"),
    "BOG": ("El Dorado International Airport", "bogota"), "MBJ": ("Sangster International Airport", "montego bay"),
    "SDQ": ("Las Americas International Airport", "santo domingo"), "DOH": ("Hamad International Airport", "doha"),
    "GRU": ("Sao Paulo/Guarulhos International Airport", "sao paulo"),
    "HKG": ("Hong Kong International Airport", "hong kong"), "SJO": ("Juan Santamaria International Airport",
                                                                     "san jose costa rica"),
    "PVR": ("Puerto Vallarta International Airport", "puerto vallarta"),
    "DXB": ("Dubai International Airport", "dubai"), "ZRH": ("Zurich Airport", "zurich"),
    "GUA": ("La Aurora International Airport", "guatemala city"), "AUA": ("Queen Beatrix International Airport",
                                                                         "aruba"),
    "BCN": ("Barcelona-El Prat Airport", "barcelona"), "LIS": ("Lisbon Airport", "lisbon"),
    "SIN": ("Singapore Changi Airport", "singapore"), "SYD": ("Sydney Airport", "sydney"),
    "PEK": ("Beijing Capital International Airport", "beijing"),
    "PVG": ("Shanghai Pudong International Airport", "shanghai"),
}

# Local time zone per state (for live FAA status times). States spanning two zones use the zone of their
# main commercial airports.
STATE_TIMEZONES = {
    **dict.fromkeys(("CT", "DE", "DC", "FL", "GA", "IN", "KY", "ME", "MD", "MA", "MI", "NH", "NJ", "NY", "NC",
                     "OH", "PA", "RI", "SC", "VT", "VA", "WV"), "America/New_York"),
    **dict.fromkeys(("AL", "AR", "IL", "IA", "KS", "LA", "MN", "MS", "MO", "NE", "ND", "OK", "SD", "TN", "TX",
                     "WI"), "America/Chicago"),
    **dict.fromkeys(("CO", "ID", "MT", "NM", "UT", "WY"), "America/Denver"),
    "AZ": "America/Phoenix",
    **dict.fromkeys(("CA", "NV", "OR", "WA"), "America/Los_Angeles"),
    "AK": "America/Anchorage", "HI": "Pacific/Honolulu", "PR": "America/Puerto_Rico", "VI": "America/St_Thomas",
    "GU": "Pacific/Guam",
}

# ---------------------------------------------------------------- citations
SOURCE_LABELS = {
    "t100": "BTS T-100 Segment (All Carriers), TranStats - passenger service (CLASS F/L, non-freighter)",
    "delay": "BTS Airline On-Time Delay Cause, TranStats - reporting carriers, domestic arrivals",
    "socrata": "BTS T-100 Segment Summary by Origin Airport (data.bts.gov r495-tyji) - all carriers incl. cargo",
    "faa": "FAA Passenger Boarding (Enplanement) Data, commercial service airports",
    "ourairports": "OurAirports airports.csv / runways.csv (open runways)",
}
