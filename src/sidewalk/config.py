"""Project-wide constants for the pavement-width pipeline.

Framing note (used throughout the report):
    "Multi-source geospatial pavement-width measurement with computer-vision validation."
Computer vision / street-level imagery is a VALIDATION layer only. Every number
reported as a width is derived from surveyed OS MasterMap geometry.
"""
from __future__ import annotations

import os
import pathlib

# --------------------------------------------------------------------------
# Paths. Resolved against the repository root, not the working directory, so
# the library behaves the same from a script, a notebook or a container.
# Override with SIDEWALK_DATA_DIR.
# --------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get("SIDEWALK_DATA_DIR", ROOT / "data"))
OUTPUTS = pathlib.Path(os.environ.get("SIDEWALK_OUTPUT_DIR", ROOT / "outputs"))

TOPO_GPKG = str(DATA / "mastermap-topo_6447205/mastermap-topo_6447205.gpkg")
HWY_GPKG = str(DATA / "MasterMap Highways Network_rami_6447203/mastermap-highways-area_6447203.gpkg")
PATHS_GPKG = str(DATA / "MasterMap Highways Network - Paths_6447204/mastermap-highways-paths-area_6447204.gpkg")
LSOA_SHP = str(DATA / "LB_LSOA_shp/Tower Hamlets.shp")

CRS = "EPSG:27700"  # British National Grid. Metres. All measurement happens here.

# Supplied Digimap order extent (BNG), from contents_order_3008173.txt
EXTENT = (530600.0, 180500.0, 534000.0, 182400.0)

# --------------------------------------------------------------------------
# Roads in scope
# --------------------------------------------------------------------------
REQUIRED_ROADS = [
    "Whitechapel High Street",
    "Commercial Street",
    "Brushfield Street",
    "Brick Lane",
    "Scarborough Street",
    "St Mark Street",
    "Gunthorpe Street",
]

# Three chosen by us. Each is picked to stress a different part of the method.
CHOSEN_ROADS = {
    "Leman Street": (
        "Wide A-road with bus stops, loading bays and a segregated cycle track - "
        "tests the top of the band scale and whether cycle infrastructure is "
        "wrongly absorbed into the footway."
    ),
    "Wentworth Street": (
        "Petticoat Lane street market, partly pedestrianised - the hard case where "
        "'carriageway' and 'footway' genuinely blur, so the method has to say so "
        "rather than invent a clean answer."
    ),
    "Fournier Street": (
        "Intact Georgian terrace with constant frontage-to-frontage width - a "
        "control road where the true width is uniform, so any variation we report "
        "is our own measurement error, not the street."
    ),
}
ALL_ROADS = REQUIRED_ROADS + list(CHOSEN_ROADS)

# --------------------------------------------------------------------------
# UK footway width guidance (from the assessment brief)
# --------------------------------------------------------------------------
BANDS = [
    ("below minimum",      0.00, 1.00, "#8c1d18"),
    ("substandard",        1.00, 1.50, "#d1495b"),
    ("minimum acceptable", 1.50, 2.00, "#edae49"),
    ("good",               2.00, 3.30, "#66a182"),
    ("high capacity",      3.30, 1e9, "#2e6f95"),
]
BAND_NAMES = [b[0] for b in BANDS]
BAND_COLOURS = {b[0]: b[3] for b in BANDS}


def band_of(width: float) -> str:
    """Classify a footway width (m) against the brief's UK guidance bands."""
    if width is None or width != width:  # NaN
        return "no data"
    for name, lo, hi, _ in BANDS:
        if lo <= width < hi:
            return name
    return "no data"


# --------------------------------------------------------------------------
# Measurement parameters (all are swept in the sensitivity analysis)
# --------------------------------------------------------------------------
SAMPLE_SPACING = 1.0     # m between transects along the street axis
ENSEMBLE_HALF_WINDOW = 2.0   # m: a reported width is the robust centre of every
                             # transect within +/- this distance, not one ray
TANGENT_WINDOW = 8.0     # m half-window for the smoothed centreline tangent
KERB_FIT_RADIUS = 4.0    # m radius used to fit the local kerb direction
KERB_CORNER_TOL = 0.10   # m RMS residual to a local quadratic; above this the kerb
                         # has a corner/break and the fan fallback is used
FAN_HALF_ANGLE = 30.0    # deg, fallback ray fan half-width
FAN_STEP = 5.0           # deg
MAX_RAY = 45.0           # m, how far a transect ray is allowed to travel
EPS = 0.02               # m, nudge used to step across a polygon boundary

# --------------------------------------------------------------------------
# QA thresholds. These are OUR thresholds, set by sensitivity analysis.
# They are NOT published UK standards and are not presented as such.
# --------------------------------------------------------------------------
CLOSURE_GOOD = 0.20        # m  |residual| below this -> GOOD
CLOSURE_INVESTIGATE = 0.50 # m  |residual| below this -> INVESTIGATE, above -> AMBIGUOUS

MAX_PLAUSIBLE_HALF_CARRIAGEWAY = 30.0  # m. On a dual carriageway the ray crosses
                                       # the near carriageway, the median and the
                                       # far carriageway before it reaches the far
                                       # kerb, so this has to be generous. The real
                                       # leak guard is the other-road crossing test.
MAX_PLAUSIBLE_FOOTWAY = 15.0           # m, beyond this it is a plaza, not a footway
NEGLIGIBLE_FOOTWAY = 0.25              # m, below this there is effectively no footway
MAX_CYCLETRACK = 8.0                   # m. A cross-section cannot contain more than
                                       # this much cycle track; more means the ray is
                                       # running ALONG a track at a junction, not across it.
MAX_FRONTAGE_GAP = 12.0                # m, how far past the footway we still accept a
                                       # building as 'the frontage' for the closure test
MAX_NORMAL_DISAGREE = 20.0             # deg between the two kerb normals before the
                                       # cross-section is treated as geometrically unsafe
NEIGHBOUR_WINDOW = 6.0                 # m, half-window for the neighbour consistency check
NEIGHBOUR_TOL = 0.50                   # m, absolute tolerance
NEIGHBOUR_MAD_K = 3.5                  # robust z-score cut

# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
MIN_SEGMENT_LEN = 10.0   # m, no segment shorter than this
MIN_MEAN_SHIFT = 0.25    # m, a change smaller than this is not "meaningful"
PELT_PENALTY_K = 3.0     # penalty multiplier on sigma^2 * log(n)
MAX_SEGMENT_GAP = 6.0    # m: a rejected stretch shorter than this is bridged rather
                         # than used to force a segment boundary

# MIN_MEAN_SHIFT is derived, not guessed. Two adjacent segment means each carry
# about SIGMA_SURVEY of uncertainty, so the smallest difference we can defend is
# roughly sqrt(2) * SIGMA_SURVEY = 0.21 m. We round up to 0.25 m and sweep it.

# --------------------------------------------------------------------------
# Effective walkable width (optional accessibility derivative - NOT a substitute
# for the geometric footway width the assessment asks for)
# --------------------------------------------------------------------------
OBSTRUCTION_BUFFER = {
    # OS MasterMap Topographic Point classes
    "Post": 0.25, "Tree": 0.75, "Structure": 0.50,
    # OSM street-furniture classes (effective radius of the swept obstruction, m)
    "tree": 0.75, "bollard": 0.20, "street_lamp": 0.25, "bench": 0.60,
    "waste_basket": 0.30, "bicycle_parking": 0.60, "telephone": 0.45,
    "post_box": 0.35, "street_cabinet": 0.45, "utility_pole": 0.25,
    "traffic_signals": 0.25, "advertising": 0.40, "fire_hydrant": 0.20,
    "drinking_fountain": 0.35, "flagpole": 0.20, "monitoring_station": 0.30,
}
DEFAULT_OBSTRUCTION_RADIUS = 0.30

# --------------------------------------------------------------------------
# Uncertainty budget
# --------------------------------------------------------------------------
# OS MasterMap Topography is specified at 0.5 m RMSE *absolute* accuracy in urban
# areas. We are not measuring an absolute position though - we measure the
# distance between two edges captured in the same survey, a few metres apart,
# where relative accuracy is considerably better. SIGMA_SURVEY is our assumed
# floor on that relative accuracy and is swept in the sensitivity analysis.
SIGMA_SURVEY = 0.15      # m, 1-sigma
CI_Z = 1.96              # 95% interval

# Confidence model weights (geometric mean of evidence terms)
CONFIDENCE_WEIGHTS = {
    "method": 1.0,
    "estimator_agreement": 1.5,
    "ensemble_stability": 1.5,
    "closure": 1.0,
    "back_edge": 1.0,
    "topology": 1.0,
    "source": 0.5,
}
JUNCTION_INFLUENCE = 12.0   # m: topology confidence recovers over this distance

ARTEFACT_WEIGHTS = {
    "width_extremity": 0.26,
    "low_confidence": 0.20,
    "ribbon_mismatch": 0.18,
    "instability": 0.13,
    "open_back_edge": 0.10,
    "closure_ambiguous": 0.08,
    "junction_proximity": 0.05,
}
