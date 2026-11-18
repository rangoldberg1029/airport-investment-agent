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

# ---------------------------------------------------------------- index weights (each sums to 1)
# Investment score components: demand pressure = load factor, growth = YoY passengers,
# congestion = congestion index, scale = passengers. All percentile-normalised vs the universe.
INVESTMENT_WEIGHTS = {"demand_pressure": 0.35, "growth": 0.25, "congestion": 0.25, "scale": 0.15}

WEIGHT_PRESETS = {
    "balanced": INVESTMENT_WEIGHTS,
    "demand-led": {"demand_pressure": 0.50, "growth": 0.20, "congestion": 0.15, "scale": 0.15},
    "congestion-led": {"demand_pressure": 0.25, "growth": 0.15, "congestion": 0.45, "scale": 0.15},
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

# ---------------------------------------------------------------- citations
SOURCE_LABELS = {
    "t100": "BTS T-100 Segment (All Carriers), TranStats - passenger service (CLASS F/L, non-freighter)",
    "delay": "BTS Airline On-Time Delay Cause, TranStats - reporting carriers, domestic arrivals",
    "socrata": "BTS T-100 Segment Summary by Origin Airport (data.bts.gov r495-tyji) - all carriers incl. cargo",
    "faa": "FAA Passenger Boarding (Enplanement) Data, commercial service airports",
    "ourairports": "OurAirports airports.csv / runways.csv (open runways)",
}
