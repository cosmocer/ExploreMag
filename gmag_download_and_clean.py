#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download, clean, cadence-convert, and package multi-network GMAG data.

The command-line program discovers stations geographically, retrieves data
through GMAG plus network-specific compatibility paths, validates cadence and
coverage, cleans N/E/Z vectors, applies the requested baseline, calculates a
Pi2-band product, and writes viewer-compatible NetCDF plus station reports.

Supported sources include THEMIS-hosted station data, direct CARISMA F01
files, IMAGE archive fallback loading, and MagStar/Gannon CDF products with
source-specific timestamp and component handling. Downloads are concurrent
where safe, while each station is accepted or rejected independently.

Output choices
--------------
The program asks for these choices when run interactively unless the matching
command-line options are supplied.

Cadence:
    --cadence-mode original
        Preserve each station's native cadence. The single NetCDF4 file uses
        nested groups: /stations/<code>/time, nez, raw_nez, pi2_nez, and
        data_quality. This is valid NetCDF4, but readers that require one
        root-level station/time array cannot use this layout.

    --cadence-mode 1s
        Put all accepted stations on a common 1-second grid. Native 0.5-second
        data are averaged into populated output bins. Slower native data are
        interpolated only across intervals allowed by the short-gap limit.

    --cadence-mode 10s
        Put all accepted stations on a common 10-second grid. Faster native
        samples are averaged within populated bins, avoiding false gaps from
        station timestamp phase offsets.

    --cadence-mode 2hz
        Put all accepted stations on a common 0.5-second (2-Hz) grid. Native
        1-second data are linearly interpolated onto intervening half seconds.

Magnetic values:
    --data-mode perturbation
        If --quiet-start and --quiet-end are both supplied, subtract the quiet
        rolling-mean baseline. If no quiet interval is supplied, warn the user,
        subtract the first complete N/E/Z sample for each station, and then run
        cleaning, gap filling, and despiking. The station report records the
        baseline timestamp and the N/E/Z values subtracted.

    --data-mode actual
        Retain cleaned absolute magnetic measurements. No baseline is
        subtracted; subtracted_baseline is [0, 0, 0].

NetCDF variables
----------------
In common-cadence modes, the viewer-compatible root variables are:
    time(time)
    station(station)
    component(component)
    nez(station, time, component)
    raw_nez(station, time, component)
    pi2_nez(station, time, component)
    data_quality(station, time, component)

``nez`` contains the selected output: perturbations or actual measurements.
``raw_nez`` always contains cleaned absolute measurements. ``pi2_nez`` is the
linearly detrended, zero-phase Butterworth Pi2-band product. Quality values are
0=observed/aligned, 1=interpolated or repaired, and 2=missing.

Cleaning uses vector-consistent rolling-median/MAD spike detection, optional
near-zero dropout detection, bounded repair of short transients, and limited
interior-gap interpolation. Sustained excursions and long gaps are preserved.

Per-station root metadata include native and output cadence, coverage, baseline
method, baseline timestamp, and subtracted_baseline. ``quiet_baseline`` is kept
as a legacy alias of subtracted_baseline. Native-cadence mode stores separate
station time axes in NetCDF4 groups; common-cadence modes use rectangular root
arrays. A CSV station report records accepted and rejected stations and their
processing details, and the completed NetCDF is validated before exit.

Examples
--------
Common 2-Hz perturbations using a quiet interval:

    python gmag_download_and_clean.py \
        --cadence-mode 2hz --data-mode perturbation \
        --start "2024-05-12 03:00:00" --end "2024-05-12 06:00:00" \
        --quiet-start "2024-05-09 03:00:00" \
        --quiet-end "2024-05-09 06:00:00"

Common 1-second perturbations using each station's first active data point:

    python gmag_download_and_clean.py \
        --cadence-mode 1s --data-mode perturbation --yes

Native-cadence actual measurements:

    python gmag_download_and_clean.py \
        --cadence-mode original --data-mode actual

Dependencies
------------
    numpy pandas scipy netCDF4 matplotlib
    cartopy or basemap for the accepted-station map
    GMAG dependencies including cdflib, requests, and wget
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num
from scipy import signal


# =============================================================================
# Defaults
# =============================================================================

PROJECT_DIR = Path("/home/ceren/Documents/GitHub/substorm_analysis/gmag_data_download_2Hz")
DEFAULT_OUTPUT = PROJECT_DIR / "gmag_data_download.nc"

DEFAULT_START = "2010-02-16 07:00:00"
DEFAULT_END = "2010-02-16 08:00:00"
DEFAULT_QUIET_START: Optional[str] = None
DEFAULT_QUIET_END: Optional[str] = None

DEFAULT_LAT_MIN = 30.0
DEFAULT_LAT_MAX = 80.0
DEFAULT_LON_MIN = -160.0
DEFAULT_LON_MAX = -50.0

DEFAULT_MAX_NATIVE_CADENCE_SECONDS = 1.10
DEFAULT_OUTPUT_CADENCE_SECONDS = 0.5
DEFAULT_CADENCE_MODE = "2hz"
DEFAULT_DATA_MODE = "perturbation"
DEFAULT_SHORT_GAP_SECONDS = 2
DEFAULT_MIN_COVERAGE = 0.80

# Quiet-reference data are reduced to one background vector. A centred
# rolling mean suppresses small fluctuations; the median of the valid
# rolling-background samples in the requested quiet interval is retained.
DEFAULT_QUIET_BASELINE_ROLLING_WINDOW = "30min"

# Vector despiking defaults adapted from
# pyspedas_ground_magnetometers_cleaning_despike.py. The rolling-MAD test is
# applied per component, but the union of all component masks is evaluated
# as one N/E/Z vector. Only short transient runs are repaired; sustained
# excursions are preserved so storm-time signatures are not deleted.
DEFAULT_DESPIKE_ROLLING_WINDOW = "5min"
DEFAULT_DESPIKE_MAD_THRESHOLD = 12.0
DEFAULT_DESPIKE_MIN_ABS_DEVIATION_NT = 50.0
DEFAULT_ZERO_DROPOUT_THRESHOLD_NT = 1.0
# Only short rolling-MAD excursions are treated as instrumental spikes.
# Longer runs are preserved because they are more likely to be real
# storm/substorm magnetic variations than isolated bad samples.
DEFAULT_MAX_DESPIKE_DURATION_SECONDS = 2.0

DEFAULT_PI2_MIN_PERIOD_SECONDS = 40.0
DEFAULT_PI2_MAX_PERIOD_SECONDS = 150.0
DEFAULT_FILTER_ORDER = 4
DEFAULT_FILTER_PADDING_SECONDS = 900
DEFAULT_CARISMA_DOWNLOAD_WORKERS = 6
DEFAULT_CARISMA_REQUEST_TIMEOUT_SECONDS = 5.0
DEFAULT_CARISMA_FALLBACK_MODE = "fast"

# A very conservative physical-range check. Common CDF fill values are much
# larger than this. Ordinary ground magnetic-field values are far below it.
ABSOLUTE_VALUE_LIMIT_NT = 1.0e7

QUALITY_OBSERVED = np.uint8(0)
QUALITY_INTERPOLATED = np.uint8(1)
QUALITY_MISSING = np.uint8(2)

# The official THEMIS GMAG availability table identifies these stations as the
# MagStar group. They are hosted in the normal THEMIS daily-CDF hierarchy, for
# example:
#   .../data/themis/thg/l2/mag/col/2024/thg_l2_mag_col_20240512_v01.cdf
#
# Longitudes are stored here in the [-180, 180) convention used by this script.
# The list intentionally reflects the stations visible in the THEMIS archive;
# GMAG's local station tables may not yet contain them.
MAGSTAR_CATALOGUE_SOURCE = (
    "https://themis.ssl.berkeley.edu/gmag/gmag_list.php "
    "(group=MagStar; checked 2026-07-20)"
)
MAGSTAR_DATA_SOURCE = "THEMIS GMAG daily CDF archive"
MAGSTAR_PROVENANCE = (
    "MagStar operational magnetometer array; project investigators Jennifer "
    "Gannon (Computational Physics, Inc.) and Delores Knipp (University of "
    "Colorado Boulder); station metadata supplemented from the THEMIS GMAG "
    "availability table."
)
MAGSTAR_STATION_ROWS: tuple[tuple[str, str, float, float], ...] = (
    ("COL", "Pawnee, CO", 40.02, -103.70),
    ("DAT", "Lewis, NY (ATLAS)", 44.27, -73.60),
    ("DBO", "Boulder, CO (University of Colorado)", 40.20, -105.20),
    ("DCT", "New Britain, CT", 41.66, -72.80),
    ("DHE", "Hennepin, MN", 44.90, -93.70),
    ("DMA", "MIT Haystack Observatory, MA", 42.60, -71.50),
    ("DME", "Augusta, ME", 44.31, -69.80),
    ("DMO", "University of Missouri, MO", 38.90, -92.32),
    ("DOH", "Blue Skies, Wauseon, OH", 41.55, -84.10),
    ("DSH", "Sugar Hills, MN", 47.12, -93.70),
    ("DTX", "Odessa, TX", 31.84, -102.40),
    ("DVA", "Warrenton, VA", 38.70, -77.80),
)
MAGSTAR_STATION_CODES = frozenset(row[0] for row in MAGSTAR_STATION_ROWS)

# CARISMA canonical output codes and archive aliases. Direct CARISMA F01
# downloads use the canonical four-character code first. Aliases are then tried
# through the THEMIS daily-CDF archive. The Cxx codes are SuperMAG-designated
# identifiers for CARISMA sites without an official IAGA code.
CARISMA_CODE_ALIASES: dict[str, tuple[str, ...]] = {
    "ISLL": ("ISL",),
    "GULL": ("C04",),
    "OXFO": ("C09",),
    "RABB": ("RAL",),
    "BRD": (),
    "LGRR": ("C05",),
    "GILL": ("GIM",),
    "SACH": ("SAH",),
    "TALO": ("TAL",),
    "CONT": ("CNL",),
    "DAWS": ("DAW",),
    "RANK": ("RAN",),
    "FSIM": ("FSP",),
    "ESKI": ("EKP",),
    "FSMI": ("SMI",),
    "FCHP": ("C03",),
    "FCHU": (),
    "MCMU": ("FMC",),
    "PINA": ("PIN",),
    "BACK": ("C02",),
    "WGRY": ("C13",),
    "VULC": ("T03",),
    "MSTK": ("C06",),
    "NORM": ("C07",),
    "OSAK": ("C08",),
    "POLS": ("C10",),
    "THRF": ("C11",),
    "WEYB": ("C12",),
    "ANNA": ("C01",),
}
CARISMA_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: canonical
    for canonical, aliases in CARISMA_CODE_ALIASES.items()
    for alias in aliases
}
CARISMA_CANONICAL_CODES = frozenset(CARISMA_CODE_ALIASES)
CARISMA_RECOGNIZED_CODES = frozenset(
    set(CARISMA_CANONICAL_CODES) | set(CARISMA_ALIAS_TO_CANONICAL)
)

CARISMA_STATION_NAMES: dict[str, str] = {
    "ISLL": "Island Lake",
    "GULL": "Gull Lake",
    "OXFO": "Oxford House",
    "RABB": "Rabbit Lake",
    "BRD": "Brandon",
    "LGRR": "Little Grand Rapids",
    "GILL": "Gillam",
    "SACH": "Sachs Harbour",
    "TALO": "Taloyoak",
    "CONT": "Contwoyto Lake",
    "DAWS": "Dawson City",
    "RANK": "Rankin Inlet",
    "FSIM": "Fort Simpson",
    "ESKI": "Eskimo Point",
    "FSMI": "Fort Smith",
    "FCHP": "Fort Chipewyan",
    "FCHU": "Fort Churchill",
    "MCMU": "Fort McMurray",
    "PINA": "Pinawa",
    "BACK": "Back Lake",
    "WGRY": "Wells Gray",
    "VULC": "Vulcan",
    "MSTK": "Ministik Lake",
    "NORM": "Norman Wells",
    "OSAK": "Osakis",
    "POLS": "Polson",
    "THRF": "Thief River Falls",
    "WEYB": "Weyburn",
    "ANNA": "Ann Arbor",
}

# Supplemental rows ensure canonical CARISMA sites remain selectable even when
# an older GMAG station table contains only a short/SuperMAG alias. Coordinates
# are geographic and use the [-180, 180) longitude convention.
CARISMA_STATION_ROWS: tuple[tuple[str, str, float, float], ...] = (
    ("SACH", "Sachs Harbour", 71.98, -125.23),
    ("TALO", "Taloyoak", 69.54, -93.55),
    ("CONT", "Contwoyto Lake", 65.754, -111.25),
    ("NORM", "Norman Wells", 65.257, -126.689),
    ("DAWS", "Dawson City", 64.048, -139.11),
    ("RANK", "Rankin Inlet", 62.824, -92.11),
    ("FSIM", "Fort Simpson", 61.756, -121.23),
    ("ESKI", "Eskimo Point", 61.106, -94.05),
    ("FSMI", "Fort Smith", 60.017, -111.95),
    ("FCHP", "Fort Chipewyan", 58.769, -111.106),
    ("FCHU", "Fort Churchill", 58.763, -94.080),
    ("MCMU", "Fort McMurray", 56.657, -111.21),
    ("GILL", "Gillam", 56.376, -94.64),
    ("RABB", "Rabbit Lake", 58.222, -103.68),
    ("BACK", "Back Lake", 57.707, -94.206),
    ("OXFO", "Oxford House", 54.929, -95.287),
    ("ISLL", "Island Lake", 53.856, -94.66),
    ("MSTK", "Ministik Lake", 53.351, -112.974),
    ("LGRR", "Little Grand Rapids", 52.035, -95.463),
    ("WGRY", "Wells Gray", 51.883, -120.026),
    ("VULC", "Vulcan", 50.367, -112.98),
    ("PINA", "Pinawa", 50.199, -96.04),
    ("GULL", "Gull Lake", 50.061, -108.261),
    ("BRD", "Brandon", 49.87, -99.974),
    ("WEYB", "Weyburn", 49.693, -103.80),
    ("THRF", "Thief River Falls", 48.027, -96.365),
    ("POLS", "Polson", 47.664, -114.209),
    ("OSAK", "Osakis", 45.871, -95.083),
    ("ANNA", "Ann Arbor", 42.417, -83.902),
)

SUPERMAG_STATION_LIST_SOURCE = (
    "SuperMAG station list updated 2025-02-27; downloaded 2025-07-12"
)
SUPERMAG_REFERENCE = (
    "Gjerloev, J. W. (2012), The SuperMAG data processing technique, "
    "J. Geophys. Res., 117, A09213, doi:10.1029/2012JA017683"
)
CARISMA_ACKNOWLEDGEMENT = (
    "I.R. Mann, D.K. Milling and the rest of the CARISMA team for use of "
    "GMAG data. CARISMA is operated by the University of Alberta, funded by "
    "the Canadian Space Agency."
)

# Generic archive-code fallbacks are attempted after the catalogue code. Each
# tuple is (archive code, physical array, loader module).
STATION_CODE_FALLBACKS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "SNK": (("SNKQ", "THEMIS", "THEMIS"),),
    "NAN": (("NAIN", "MACCS", "THEMIS"),),
}

# Add Nain even when it is absent from an older local GMAG catalogue.
GENERAL_STATION_SUPPLEMENTS: tuple[tuple[str, str, str, float, float], ...] = (
    ("NAN", "Nain", "MACCS", 56.40, -61.70),
)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class CatalogueStation:
    code: str
    station_name: str
    array_name: str
    glat: float
    glon: float


@dataclass(frozen=True)
class StationLoadCandidate:
    load_station: CatalogueStation
    output_station: CatalogueStation
    loader: str
    description: str


@dataclass(frozen=True)
class DataWindow:
    """A requested data window and the UTC daily files needed to cover it."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def day_start(self) -> pd.Timestamp:
        return self.start.floor("D")

    @property
    def day_end(self) -> pd.Timestamp:
        return self.end.floor("D")

    @property
    def ndays(self) -> int:
        return int((self.day_end - self.day_start) / pd.Timedelta(days=1)) + 1


@dataclass
class LoadedStation:
    code: str
    station_name: str
    array_name: str
    glat: float
    glon: float
    times: pd.DatetimeIndex
    output_cadence_seconds: float
    native_cadence_seconds: float
    quiet_native_cadence_seconds: float
    raw_nez: np.ndarray
    output_nez: np.ndarray
    pi2_nez: np.ndarray
    subtracted_baseline: np.ndarray
    baseline_method: str
    baseline_timestamp: str
    data_mode: str
    quality: np.ndarray
    coverage_fraction: float
    quiet_coverage_fraction: float
    source_coordinate_system: str
    source_component_mapping: str
    source_columns: tuple[str, str, str]
    metadata_text: str
    output_resampling: str


@dataclass
class StationAttempt:
    code: str
    array_name: str
    glat: float
    glon: float
    status: str
    reason: str
    native_cadence_seconds: float = np.nan
    quiet_native_cadence_seconds: float = np.nan
    output_cadence_seconds: float = np.nan
    coverage_fraction: float = np.nan
    quiet_coverage_fraction: float = np.nan
    baseline_method: str = ""
    baseline_timestamp: str = ""
    subtracted_n_nt: float = np.nan
    subtracted_e_nt: float = np.nan
    subtracted_z_nt: float = np.nan
    output_resampling: str = ""


# =============================================================================
# General helpers
# =============================================================================

def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_longitude(value: float) -> float:
    """Convert a longitude to [-180, 180)."""
    return float(((float(value) + 180.0) % 360.0) - 180.0)


def parse_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def merged_daily_spans(windows: Sequence[DataWindow]) -> list[tuple[pd.Timestamp, int]]:
    """Return non-overlapping contiguous UTC-day spans for several windows."""
    days: set[pd.Timestamp] = set()
    for window in windows:
        days.update(pd.date_range(window.day_start, window.day_end, freq="1D"))
    ordered = sorted(days)
    if not ordered:
        return []

    spans: list[tuple[pd.Timestamp, int]] = []
    span_start = ordered[0]
    previous = ordered[0]
    for day in ordered[1:]:
        if day - previous != pd.Timedelta(days=1):
            spans.append(
                (span_start, int((previous - span_start) / pd.Timedelta(days=1)) + 1)
            )
            span_start = day
        previous = day
    spans.append(
        (span_start, int((previous - span_start) / pd.Timedelta(days=1)) + 1)
    )
    return spans


def utc_index(index: Iterable[Any]) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(index, utc=True, errors="coerce")
    return pd.DatetimeIndex(parsed)


def flatten_column_name(column: Any) -> str:
    if isinstance(column, tuple):
        return "_".join(str(item) for item in column if str(item) != "")
    return str(column)


def find_column(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    contains: Sequence[str] = (),
) -> Optional[str]:
    normalized = {str(column): normalize_name(column) for column in frame.columns}
    alias_set = {normalize_name(alias) for alias in aliases}

    for column, name in normalized.items():
        if name in alias_set:
            return column

    for column, name in normalized.items():
        if any(normalize_name(token) in name for token in contains):
            return column
    return None


def first_finite_numeric(series: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return None
    return float(numeric.iloc[0])


def safe_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return ",".join(safe_string(item) for item in value.reshape(-1))
    return str(value)


def dataframe_metadata_text(meta: Any, code: str) -> str:
    """Flatten a station's metadata into readable text for provenance."""
    if meta is None:
        return ""
    try:
        if isinstance(meta, pd.Series):
            rows = meta.to_frame().T
        elif isinstance(meta, pd.DataFrame):
            rows = meta.copy()
        else:
            return safe_string(meta)

        code_col = find_column(rows, aliases=("code", "station", "iaga"))
        if code_col is not None:
            mask = rows[code_col].astype(str).str.upper() == code.upper()
            if mask.any():
                rows = rows.loc[mask]
        if rows.empty:
            return ""

        row = rows.iloc[0]
        return " | ".join(
            f"{column}={safe_string(row[column])}" for column in rows.columns
        )
    except Exception:
        return safe_string(meta)


# =============================================================================
# Import and station catalogue
# =============================================================================

def import_gmag():
    try:
        from gmag import utils  # type: ignore
        import gmag.arrays.themis as themis  # type: ignore
        import gmag.arrays.carisma as carisma  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Could not import GMAG. Clone the official kylermurphy/gmag "
            "repository, configure gmagrc/data_dir including ca_http, and "
            "install it with 'pip install -e .'. Original import error: "
            + repr(exc)
        ) from exc
    return utils, themis, carisma


def load_station_catalogue(utils_module) -> pd.DataFrame:
    """Load GMAG geographic station information across supported arrays."""
    errors: list[str] = []
    for all_token in ("ALL", "*", "all"):
        try:
            catalogue = utils_module.load_station_geo(param=all_token)
            if isinstance(catalogue, pd.DataFrame) and not catalogue.empty:
                return catalogue.copy()
        except Exception as exc:
            errors.append(f"param={all_token!r}: {exc!r}")

    raise RuntimeError(
        "GMAG utils.load_station_geo() did not return a station catalogue. "
        + " ; ".join(errors)
    )


def standardize_catalogue(catalogue: pd.DataFrame) -> list[CatalogueStation]:
    frame = catalogue.copy()
    frame.columns = [flatten_column_name(column) for column in frame.columns]

    code_col = find_column(
        frame,
        aliases=("code", "stationcode", "iaga", "station"),
        contains=("code", "iaga"),
    )
    lat_col = find_column(
        frame,
        aliases=("glat", "geolat", "latitude", "geographiclatitude", "lat"),
        contains=("latitude", "glat"),
    )
    lon_col = find_column(
        frame,
        aliases=("glon", "geolon", "longitude", "geographiclongitude", "lon"),
        contains=("longitude", "glon"),
    )
    array_col = find_column(
        frame,
        aliases=("array", "arrayname", "network", "magnetometerarray"),
        contains=("array", "network"),
    )
    name_col = find_column(
        frame,
        aliases=("name", "stationname", "sitename"),
        contains=("stationname", "sitename"),
    )

    if code_col is None or lat_col is None or lon_col is None:
        raise KeyError(
            "Could not identify code/latitude/longitude columns in the GMAG "
            f"station catalogue. Columns are: {list(frame.columns)}"
        )

    output: list[CatalogueStation] = []
    for _, row in frame.iterrows():
        code = safe_string(row[code_col]).strip().upper()
        if not code or code in {"NAN", "NONE"}:
            continue
        try:
            glat = float(row[lat_col])
            glon = normalize_longitude(float(row[lon_col]))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(glat) or not np.isfinite(glon):
            continue

        array_name = (
            safe_string(row[array_col]).strip() if array_col is not None else ""
        )
        station_name = (
            safe_string(row[name_col]).strip() if name_col is not None else code
        )
        output.append(
            CatalogueStation(
                code=code,
                station_name=station_name or code,
                array_name=array_name,
                glat=glat,
                glon=glon,
            )
        )

    # A code may occur more than once in catalogue tables. Retain the first
    # occurrence because GMAG's THEMIS loader addresses stations by code.
    unique: dict[str, CatalogueStation] = {}
    for station in output:
        unique.setdefault(station.code, station)
    return list(unique.values())


def built_in_magstar_catalogue() -> list[CatalogueStation]:
    """Return MagStar stations missing from older GMAG station tables."""
    return [
        CatalogueStation(
            code=code,
            station_name=name,
            array_name="MagStar (Gannon/CPI)",
            glat=float(glat),
            glon=normalize_longitude(float(glon)),
        )
        for code, name, glat, glon in MAGSTAR_STATION_ROWS
    ]


def built_in_carisma_catalogue() -> list[CatalogueStation]:
    """Return canonical CARISMA rows used to repair older catalogues."""
    return [
        CatalogueStation(
            code=code,
            station_name=name,
            array_name="CARISMA",
            glat=float(glat),
            glon=normalize_longitude(float(glon)),
        )
        for code, name, glat, glon in CARISMA_STATION_ROWS
    ]


def built_in_general_supplements() -> list[CatalogueStation]:
    return [
        CatalogueStation(
            code=code, station_name=name, array_name=array_name,
            glat=float(glat), glon=normalize_longitude(float(glon)),
        )
        for code, name, array_name, glat, glon in GENERAL_STATION_SUPPLEMENTS
    ]


def merge_missing_station_catalogues(
    primary: Sequence[CatalogueStation],
    supplements: Sequence[CatalogueStation],
) -> list[CatalogueStation]:
    """Add only station codes absent from the primary catalogue."""
    merged = {station.code.upper(): station for station in primary}
    for station in supplements:
        merged.setdefault(station.code.upper(), station)
    return list(merged.values())


def canonical_carisma_code(code: str) -> str:
    upper = str(code).strip().upper()
    return CARISMA_ALIAS_TO_CANONICAL.get(upper, upper)


def canonicalize_carisma_alias_rows(
    stations: Sequence[CatalogueStation],
) -> list[CatalogueStation]:
    """Collapse short CARISMA aliases onto canonical output station codes."""
    normalized: dict[str, CatalogueStation] = {}
    for station in stations:
        original = station.code.upper()
        canonical = canonical_carisma_code(original)
        if original in CARISMA_ALIAS_TO_CANONICAL:
            station = CatalogueStation(
                code=canonical,
                station_name=CARISMA_STATION_NAMES.get(canonical, station.station_name),
                array_name="CARISMA",
                glat=station.glat,
                glon=station.glon,
            )
        # Prefer an existing canonical row over a converted alias row.
        if canonical not in normalized or original == canonical:
            normalized[canonical] = station
    return list(normalized.values())


def merge_station_catalogues(
    primary: Sequence[CatalogueStation],
    supplements: Sequence[CatalogueStation],
) -> list[CatalogueStation]:
    """Merge catalogues by code, allowing explicit supplements to update rows."""
    merged = {station.code.upper(): station for station in primary}
    for station in supplements:
        merged[station.code.upper()] = station
    return list(merged.values())


def is_magstar_station(station: CatalogueStation | LoadedStation) -> bool:
    """Return True for a built-in MagStar code or a MagStar array label."""
    return (
        station.code.upper() in MAGSTAR_STATION_CODES
        or "magstar" in normalize_name(station.array_name)
    )


def is_carisma_station(station: CatalogueStation | LoadedStation) -> bool:
    code = canonical_carisma_code(station.code)
    return (
        code in CARISMA_CANONICAL_CODES
        or "carisma" in normalize_name(station.array_name)
        or "canopus" in normalize_name(station.array_name)
    )


def station_load_candidates(
    station: CatalogueStation,
    allow_fallback: bool = True,
    carisma_fallback_mode: str = DEFAULT_CARISMA_FALLBACK_MODE,
) -> list[StationLoadCandidate]:
    """Build ordered direct-array and bounded fallback loading attempts.

    ``fast`` tries one direct CARISMA file and then the canonical THEMIS code.
    ``exhaustive`` additionally tries the catalogue spelling and every known
    short/SuperMAG alias. ``none`` disables THEMIS fallback for CARISMA.
    """
    candidates: list[StationLoadCandidate] = []
    seen: set[tuple[str, str, str]] = set()

    def add_candidate(
        load_code: str,
        load_array: str,
        loader: str,
        output_station: CatalogueStation,
        description: str,
    ) -> None:
        load_code = load_code.upper()
        key = (loader.upper(), load_code, output_station.code.upper())
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            StationLoadCandidate(
                load_station=CatalogueStation(
                    code=load_code,
                    station_name=output_station.station_name,
                    array_name=load_array,
                    glat=station.glat,
                    glon=station.glon,
                ),
                output_station=output_station,
                loader=loader.upper(),
                description=description,
            )
        )

    if is_carisma_station(station):
        canonical = canonical_carisma_code(station.code)
        output_station = CatalogueStation(
            code=canonical,
            station_name=CARISMA_STATION_NAMES.get(canonical, station.station_name),
            array_name="CARISMA",
            glat=station.glat,
            glon=station.glon,
        )

        add_candidate(
            canonical, "CARISMA", "CARISMA", output_station,
            f"direct CARISMA F01 code {canonical}",
        )

        if allow_fallback and carisma_fallback_mode != "none":
            if carisma_fallback_mode == "fast":
                archive_codes = [canonical]
            elif carisma_fallback_mode == "exhaustive":
                archive_codes = [canonical, station.code.upper()]
                archive_codes.extend(CARISMA_CODE_ALIASES.get(canonical, ()))
            else:
                raise ValueError(
                    f"Unknown CARISMA fallback mode {carisma_fallback_mode!r}"
                )
            for archive_code in archive_codes:
                add_candidate(
                    archive_code, "CARISMA", "THEMIS", output_station,
                    f"CARISMA data hosted by THEMIS under {archive_code}",
                )
        return candidates

    primary_loader = "MAGSTAR" if is_magstar_station(station) else "THEMIS"
    add_candidate(
        station.code, station.array_name, primary_loader, station,
        f"primary {primary_loader} code {station.code}",
    )

    if allow_fallback:
        for fallback_code, fallback_array, fallback_loader in (
            STATION_CODE_FALLBACKS.get(station.code.upper(), ())
        ):
            fallback_station = CatalogueStation(
                code=fallback_code,
                station_name=(
                    "Sanikiluaq" if fallback_code == "SNKQ"
                    else "Nain" if fallback_code == "NAIN"
                    else station.station_name
                ),
                array_name=fallback_array,
                glat=station.glat,
                glon=station.glon,
            )
            add_candidate(
                fallback_code, fallback_array, fallback_loader, fallback_station,
                f"fallback {station.code}->{fallback_code} from {fallback_array}",
            )
    return candidates


def filter_catalogue(
    stations: Sequence[CatalogueStation],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    requested_codes: Optional[Sequence[str]],
) -> list[CatalogueStation]:
    requested = (
        {str(code).strip().upper() for code in requested_codes}
        if requested_codes
        else None
    )

    catalogue_requested = None if requested is None else set(requested)
    if catalogue_requested is not None:
        available_codes = {station.code.upper() for station in stations}

        # Treat CARISMA aliases as equivalent to their canonical output code.
        for requested_code in list(requested):
            canonical = canonical_carisma_code(requested_code)
            if canonical in CARISMA_CANONICAL_CODES:
                catalogue_requested.add(canonical)

        # Allow explicit requests for an archive fallback code when only the
        # original catalogue row is available.
        for original_code, fallbacks in STATION_CODE_FALLBACKS.items():
            for fallback_code, _, _ in fallbacks:
                if fallback_code in requested and fallback_code not in available_codes:
                    catalogue_requested.add(original_code)

    selected: list[CatalogueStation] = []
    for station in stations:
        in_region = (
            lat_min <= station.glat <= lat_max
            and lon_min <= station.glon <= lon_max
        )
        if not in_region:
            continue
        if (
            catalogue_requested is not None
            and station.code.upper() not in catalogue_requested
        ):
            continue
        selected.append(station)

    selected.sort(key=lambda item: (-item.glat, item.glon, item.code))
    return selected


# =============================================================================
# GMAG loading and generic DataFrame parsing
# =============================================================================

_CARISMA_COORDINATE_CACHE: dict[int, Optional[pd.DataFrame]] = {}


def _carisma_file_frame(
    carisma_module: Any,
    code: str,
    day_start: pd.Timestamp,
    ndays: int,
) -> pd.DataFrame:
    date_text = day_start.strftime("%Y-%m-%d")
    try:
        frame = carisma_module.list_files(
            code.upper(), date_text, ndays=ndays, gz=True
        )
    except TypeError:
        frame = carisma_module.list_files(
            site=code.upper(), sdate=date_text, ndays=ndays, gz=True
        )
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("gmag.arrays.carisma.list_files did not return a DataFrame")
    return frame


def _download_one_carisma_file(
    row: pd.Series,
    force_download: bool,
    download_missing: bool,
    timeout_seconds: float,
) -> tuple[str, str]:
    """Download one CARISMA file and return (status, path-or-message).

    Existing files are reused unless ``force_download`` is true. When
    ``download_missing`` is false, a missing local file is reported without
    making a network request.
    """
    import requests

    filename = Path(str(row["dir"])) / str(row["fname"])
    if filename.exists() and not force_download:
        return "cached", str(filename)
    if not download_missing:
        return "missing", f"{filename}: not cached (existing-only mode)"

    filename.parent.mkdir(parents=True, exist_ok=True)
    url = str(row["hdir"]) + str(row["fname"])
    try:
        response = requests.get(url, timeout=timeout_seconds)
    except Exception as exc:
        return "error", f"{url}: {exc}"
    if not response.ok:
        return "missing", f"{url}: HTTP {response.status_code}"

    temporary = filename.with_suffix(filename.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(filename)
    return "downloaded", str(filename)


def prefetch_carisma_files(
    carisma_module: Any,
    codes: Sequence[str],
    day_start: pd.Timestamp,
    ndays: int,
    force_download: bool,
    download_missing: bool,
    workers: int,
    timeout_seconds: float,
) -> dict[str, int]:
    """Download all selected CARISMA files concurrently.

    The original GMAG downloader checks missing files serially with a five-second
    request timeout. Concurrent prefetching keeps the same source while avoiding
    a station-by-station timeout penalty.
    """
    tasks: list[pd.Series] = []
    for code in sorted({canonical_carisma_code(value) for value in codes}):
        frame = _carisma_file_frame(carisma_module, code, day_start, ndays)
        tasks.extend(row for _, row in frame.iterrows())

    counts = {"cached": 0, "downloaded": 0, "missing": 0, "error": 0}
    if not tasks:
        return counts

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {
            executor.submit(
                _download_one_carisma_file,
                row,
                force_download,
                download_missing,
                timeout_seconds,
            ): row
            for row in tasks
        }
        for future in as_completed(future_map):
            try:
                status, _ = future.result()
            except Exception:
                status = "error"
            counts[status] = counts.get(status, 0) + 1
    return counts


def _read_carisma_f01_window(
    filename: Path,
    code: str,
    read_start: Optional[pd.Timestamp],
    read_end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """Read only the needed UTC window from a fixed-width CARISMA F01 file."""
    start_key = parse_utc(read_start).strftime("%Y%m%d%H%M%S") if read_start is not None else None
    end_key = parse_utc(read_end).strftime("%Y%m%d%H%M%S") if read_end is not None else None
    opener = gzip.open if filename.suffix.lower() == ".gz" else open

    times: list[str] = []
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    flags: list[str] = []

    with opener(filename, "rt", encoding="ascii", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            if len(line) < 44:
                continue
            stamp = line[0:14].strip()
            if len(stamp) != 14 or not stamp.isdigit():
                continue
            if start_key is not None and stamp < start_key:
                continue
            if end_key is not None and stamp > end_key:
                break
            try:
                x_value = float(line[14:24].strip())
                y_value = float(line[24:34].strip())
                z_value = float(line[34:44].strip())
            except ValueError:
                x_value = y_value = z_value = np.nan
            times.append(stamp)
            x_values.append(x_value)
            y_values.append(y_value)
            z_values.append(z_value)
            flags.append(line[44:46].strip())

    if not times:
        return pd.DataFrame()
    index = pd.to_datetime(
        pd.Series(times), format="%Y%m%d%H%M%S", utc=True, errors="coerce"
    )
    frame = pd.DataFrame(
        {
            f"{code.upper()}_X": x_values,
            f"{code.upper()}_Y": y_values,
            f"{code.upper()}_Z": z_values,
            f"{code.upper()}_flag": flags,
        },
        index=pd.DatetimeIndex(index),
    )
    frame = frame.loc[frame.index.notna()]
    components = [f"{code.upper()}_X", f"{code.upper()}_Y", f"{code.upper()}_Z"]
    good = frame[f"{code.upper()}_flag"].astype(str).str.strip().eq(".")
    vertical_positive = frame[f"{code.upper()}_Z"] >= 0
    frame.loc[~(good & vertical_positive), components] = np.nan
    return frame


def _carisma_station_metadata(
    carisma_module: Any,
    code: str,
    day_start: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    """Return cached station metadata and rotate XYZ to HDZ when possible."""
    year = int(day_start.year)
    if year not in _CARISMA_COORDINATE_CACHE:
        try:
            _CARISMA_COORDINATE_CACHE[year] = carisma_module.utils.load_station_coor(
                param="CARISMA", col="array", year=year
            )
        except Exception:
            _CARISMA_COORDINATE_CACHE[year] = None
    metadata = _CARISMA_COORDINATE_CACHE[year]
    if not isinstance(metadata, pd.DataFrame) or metadata.empty:
        return pd.DataFrame(), "CARISMA coordinate metadata unavailable"
    code_column = find_column(metadata, aliases=("code", "station", "iaga"))
    if code_column is None:
        return pd.DataFrame(), "CARISMA metadata code column unavailable"
    rows = metadata.loc[metadata[code_column].astype(str).str.upper() == code.upper()].copy()
    if rows.empty:
        return rows, f"No CARISMA metadata row for {code}"
    return rows.iloc[:1].copy(), ""


def load_carisma_f01_direct(
    carisma_module: Any,
    code: str,
    day_start: pd.Timestamp,
    ndays: int,
    force_download: bool,
    read_start: Optional[pd.Timestamp] = None,
    read_end: Optional[pd.Timestamp] = None,
    download_files: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a CARISMA F01 station while parsing only the requested window."""
    file_frame = _carisma_file_frame(carisma_module, code, day_start, ndays)
    if download_files:
        try:
            carisma_module.download(
                f_df=file_frame, force=force_download, verbose=False
            )
        except TypeError:
            carisma_module.download(f_df=file_frame, force=force_download)

    daily_frames: list[pd.DataFrame] = []
    for _, row in file_frame.iterrows():
        filename = Path(str(row["dir"])) / str(row["fname"])
        if not filename.exists():
            continue
        daily = _read_carisma_f01_window(
            filename, code, read_start=read_start, read_end=read_end
        )
        if not daily.empty:
            daily_frames.append(daily)

    if not daily_frames:
        raise FileNotFoundError(
            f"No CARISMA F01 data were available for {code} in the requested window"
        )

    raw = pd.concat(daily_frames).sort_index()
    raw = raw.groupby(level=0).mean(numeric_only=True).sort_index()
    metadata, metadata_error = _carisma_station_metadata(
        carisma_module, code, day_start
    )

    x_name = f"{code.upper()}_X"
    y_name = f"{code.upper()}_Y"
    z_name = f"{code.upper()}_Z"
    rotated = raw.copy()
    rotation_error = metadata_error
    if not metadata.empty:
        declination_column = find_column(
            metadata, aliases=("declination", "dec"), contains=("declination",)
        )
        try:
            if declination_column is None:
                raise KeyError("declination column unavailable")
            declination = float(metadata.iloc[0][declination_column])
            cosine = np.cos(np.deg2rad(declination))
            sine = np.sin(np.deg2rad(declination))
            rotated[f"{code.upper()}_H"] = raw[x_name] * cosine + raw[y_name] * sine
            rotated[f"{code.upper()}_D"] = raw[y_name] * cosine - raw[x_name] * sine
            rotation_error = ""
        except Exception as exc:
            rotation_error = repr(exc)

    resolution = data_resolution_seconds(
        pd.DataFrame(
            {
                "N": pd.to_numeric(rotated.get(f"{code.upper()}_H", rotated[x_name]), errors="coerce"),
                "E": pd.to_numeric(rotated.get(f"{code.upper()}_D", rotated[y_name]), errors="coerce"),
                "Z": pd.to_numeric(rotated[z_name], errors="coerce"),
            },
            index=rotated.index,
        )
    )
    if metadata.empty:
        metadata = pd.DataFrame([{"array": "CARISMA", "code": code.upper()}])
    metadata = metadata.copy()
    metadata["array"] = "CARISMA"
    metadata["code"] = code.upper()
    metadata["Time Resolution"] = resolution
    metadata["Coordinates"] = (
        "Geomagnetic H/D/Z" if not rotation_error
        else "Geographic X/Y/Z; rotation metadata unavailable"
    )
    metadata["PI"] = "Ian Mann"
    metadata["Institution"] = "University of Alberta"
    metadata["CARISMA file reader"] = "windowed fixed-width F01 parser"
    if rotation_error:
        metadata["rotation_error"] = rotation_error
    return rotated, metadata


def call_carisma_load(
    carisma_module: Any,
    code: str,
    day_start: pd.Timestamp,
    ndays: int,
    force_download: bool,
    read_start: Optional[pd.Timestamp] = None,
    read_end: Optional[pd.Timestamp] = None,
    download_files: bool = True,
):
    """Load CARISMA once; avoid repeated GMAG network/parser retries."""
    return load_carisma_f01_direct(
        carisma_module,
        code,
        day_start,
        ndays,
        force_download,
        read_start=read_start,
        read_end=read_end,
        download_files=download_files,
    )


def call_themis_load(
    themis_module,
    code: str,
    day_start: pd.Timestamp,
    ndays: int,
    force_download: bool,
    download_files: bool = True,
):
    """Call THEMIS while respecting the selected cache/download policy.

    ``download_files=False`` passes ``dl=False`` and never falls back to a
    signature that could silently restore GMAG's default download behavior.
    """
    date_text = day_start.strftime("%Y-%m-%d")
    if download_files:
        attempts = [
            dict(sdate=date_text, ndays=ndays, dl=True, force=force_download),
            dict(sdate=date_text, ndays=ndays, dl=True),
            dict(sdate=date_text, ndays=ndays),
        ]
        positional_attempts = [
            dict(ndays=ndays, dl=True, force=force_download),
            dict(ndays=ndays, dl=True),
            dict(ndays=ndays),
        ]
    else:
        attempts = [dict(sdate=date_text, ndays=ndays, dl=False)]
        positional_attempts = [dict(ndays=ndays, dl=False)]

    last_type_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return themis_module.load(code, **kwargs)
        except TypeError as exc:
            last_type_error = exc
            continue

    # Some versions may prefer sdate as the second positional argument.
    for kwargs in positional_attempts:
        try:
            return themis_module.load(code, date_text, **kwargs)
        except TypeError as exc:
            last_type_error = exc
            continue

    raise TypeError(
        f"Could not call gmag.arrays.themis.load for {code}. "
        f"Last signature error: {last_type_error!r}"
    )


def _cdf_attribute_data(cdf_file: Any, name: str, entry: int = 0) -> Any:
    """Return an attribute value across cdflib API versions."""
    try:
        value = cdf_file.attget(name, entry)
    except Exception:
        return None
    if hasattr(value, "Data"):
        return value.Data
    if isinstance(value, dict):
        return value.get("Data", value.get("data", value))
    return value


def _cdf_variable_type(cdf_file: Any, variable: str) -> str:
    """Return the CDF variable data-type description when available."""
    try:
        info = cdf_file.varinq(variable)
    except Exception:
        return ""
    if isinstance(info, dict):
        value = info.get("Data_Type_Description", info.get("data_type_description", ""))
    else:
        value = getattr(info, "Data_Type_Description", "")
    return safe_string(value).upper()


def _as_utc_datetime_index(values: Any) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return pd.DatetimeIndex(parsed)


def decode_cdf_time(
    raw_time: Any,
    cdf_file: Any,
    variable: str,
    expected_day: pd.Timestamp,
) -> tuple[pd.DatetimeIndex, str]:
    """Decode CDF time values without assuming Unix seconds.

    The released GMAG THEMIS reader uses ``pd.to_datetime(..., unit='s')``
    for every station. Some MagStar files use a true CDF epoch type, which
    makes that conversion overflow into an out-of-bounds year. This routine
    first obeys the variable's CDF type and then uses date-window scoring as a
    fallback for older cdflib/CDF combinations with incomplete type metadata.
    """
    import cdflib  # GMAG dependency; imported lazily for clearer startup errors.

    expected_day = parse_utc(expected_day).floor("D")
    window_start = expected_day - pd.Timedelta(days=1)
    window_end = expected_day + pd.Timedelta(days=2)
    type_description = _cdf_variable_type(cdf_file, variable)

    candidates: list[tuple[str, pd.DatetimeIndex]] = []

    def add_candidate(label: str, converted: Any) -> None:
        try:
            index = _as_utc_datetime_index(converted)
        except Exception:
            return
        if len(index) == 0 or index.notna().sum() == 0:
            return
        candidates.append((label, index))

    # Use the official CDF epoch converter whenever the declared type is an
    # epoch type. It supports CDF_EPOCH, CDF_EPOCH16, and TT2000.
    if any(token in type_description for token in ("CDF_EPOCH", "TT2000")):
        try:
            add_candidate(
                f"cdflib.cdfepoch.to_datetime ({type_description})",
                cdflib.cdfepoch.to_datetime(raw_time),
            )
        except Exception:
            pass

    raw_array = np.asarray(raw_time)
    if np.issubdtype(raw_array.dtype, np.datetime64):
        add_candidate("native datetime64", raw_array)

    # Fallbacks are scored against the day encoded in the filename. This is
    # safer than choosing solely by magnitude because TT2000 and Unix
    # nanoseconds can both be O(1e18).
    if not candidates or not type_description:
        try:
            add_candidate("cdflib.cdfepoch.to_datetime (inferred)",
                          cdflib.cdfepoch.to_datetime(raw_time))
        except Exception:
            pass
        for unit in ("s", "ms", "us", "ns"):
            try:
                add_candidate(
                    f"Unix {unit}",
                    pd.to_datetime(raw_array, unit=unit, utc=True, errors="coerce"),
                )
            except Exception:
                pass

    if not candidates:
        raise ValueError(
            f"Could not decode CDF time variable {variable!r}; "
            f"declared type={type_description or 'unknown'}"
        )

    scored: list[tuple[float, float, str, pd.DatetimeIndex]] = []
    target_mid = expected_day + pd.Timedelta(hours=12)
    for label, index in candidates:
        valid = index.notna()
        if not valid.any():
            continue
        in_window = valid & (index >= window_start) & (index < window_end)
        fraction = float(np.count_nonzero(in_window) / np.count_nonzero(valid))
        valid_values = index[valid]
        midpoint = valid_values[len(valid_values) // 2]
        distance_seconds = abs((midpoint - target_mid).total_seconds())
        scored.append((-fraction, distance_seconds, label, index))

    if not scored:
        raise ValueError(f"No finite timestamps decoded from {variable!r}")
    scored.sort(key=lambda item: (item[0], item[1]))
    neg_fraction, _, method, best = scored[0]
    in_window_fraction = -neg_fraction
    if in_window_fraction < 0.50:
        first = best[best.notna()][0] if best.notna().any() else "NaT"
        last = best[best.notna()][-1] if best.notna().any() else "NaT"
        raise ValueError(
            f"Decoded timestamps for {variable!r} do not match file day "
            f"{expected_day.date()}; method={method}, range={first}..{last}, "
            f"declared type={type_description or 'unknown'}"
        )
    return best, method


def _magstar_component_columns(labels: Any, code: str, n_component: int) -> list[str]:
    flat = np.asarray(labels, dtype=object).reshape(-1)
    output: list[str] = []
    fallback = ["H", "D", "Z"]
    for i in range(n_component):
        text = safe_string(flat[i]) if i < len(flat) else ""
        name = normalize_name(text)
        if "north" in name or name in {"h", "bh", "x", "bx"}:
            token = "H"
        elif "east" in name or "decl" in name or name in {"d", "e", "y", "by"}:
            token = "D"
        elif "vertical" in name or "down" in name or name in {"z", "bz"}:
            token = "Z"
        else:
            token = fallback[i] if i < len(fallback) else f"C{i + 1}"
        output.append(f"{code.upper()}_{token}")
    return output


def load_magstar_cdf_direct(
    themis_module: Any,
    code: str,
    day_start: pd.Timestamp,
    ndays: int,
    force_download: bool,
    download_files: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MagStar CDFs directly and decode their actual CDF epoch type.

    This intentionally bypasses only GMAG's timestamp conversion. Download
    paths and file naming still come from ``gmag.arrays.themis``.
    """
    import cdflib

    date_text = day_start.strftime("%Y-%m-%d")
    file_frame = themis_module.list_files(code.upper(), date_text, ndays=ndays)
    if not isinstance(file_frame, pd.DataFrame):
        raise TypeError("gmag.arrays.themis.list_files did not return a DataFrame")

    if download_files:
        try:
            themis_module.download(
                f_df=file_frame, force=force_download, verbose=False
            )
        except TypeError:
            themis_module.download(f_df=file_frame, force=force_download)

    station_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    base = f"thg_mag_{code.lower()}"

    for _, row in file_frame.iterrows():
        filename = Path(str(row["dir"])) / str(row["fname"])
        if not filename.exists():
            continue
        cdf_file = cdflib.CDF(str(filename))
        try:
            data = np.asarray(cdf_file.varget(base))
            labels = cdf_file.varget(base + "_labl")
            raw_time = cdf_file.varget(base + "_time")
            expected_day = parse_utc(row.get("date", day_start))
            time_index, time_method = decode_cdf_time(
                raw_time, cdf_file, base + "_time", expected_day
            )

            if data.ndim == 1:
                data = data[:, None]
            if data.ndim != 2:
                raise ValueError(f"Unexpected {base} shape {data.shape}")
            if data.shape[0] != len(time_index) and data.shape[1] == len(time_index):
                data = data.T
            if data.shape[0] != len(time_index):
                raise ValueError(
                    f"Time/data length mismatch for {code}: "
                    f"{len(time_index)} timestamps, data shape {data.shape}"
                )

            columns = _magstar_component_columns(labels, code, data.shape[1])
            daily = pd.DataFrame(data=data, index=time_index, columns=columns)
            station_frames.append(daily)

            resolution = _cdf_attribute_data(cdf_file, "Time_resolution", 0)
            pi_name = _cdf_attribute_data(cdf_file, "PI_name", 0)
            institution = _cdf_attribute_data(cdf_file, "PI_affiliation", 0)
            coordinates = ", ".join(safe_string(v) for v in np.asarray(labels).reshape(-1))
            metadata_rows.append(
                {
                    "array": "MagStar (Gannon/CPI)",
                    "code": code.upper(),
                    "Time Resolution": safe_string(resolution),
                    "Coordinates": coordinates,
                    "PI": safe_string(pi_name),
                    "Institution": safe_string(institution),
                    "CDF time variable type": _cdf_variable_type(cdf_file, base + "_time"),
                    "CDF time decoding": time_method,
                    "source_file": str(filename),
                }
            )
        finally:
            close = getattr(cdf_file, "close", None)
            if callable(close):
                close()

    if not station_frames:
        raise FileNotFoundError(f"No MagStar CDF data files were available for {code}")
    data_frame = pd.concat(station_frames).sort_index()
    data_frame = data_frame.groupby(level=0).mean().sort_index()
    metadata = pd.DataFrame(metadata_rows[:1])
    return data_frame, metadata



def load_candidate_window(
    candidate: StationLoadCandidate,
    window: DataWindow,
    themis_module: Any,
    carisma_module: Any,
    force_download: bool,
    download_missing: bool,
) -> tuple[pd.DataFrame, Any]:
    """Load one station candidate for one active or quiet-reference window."""
    load_info = candidate.load_station
    try:
        if candidate.loader == "CARISMA":
            result = call_carisma_load(
                carisma_module,
                code=load_info.code,
                day_start=window.day_start,
                ndays=window.ndays,
                force_download=force_download,
                read_start=window.start,
                read_end=window.end,
                download_files=False,
            )
            data, meta = unpack_load_result(result)
        elif candidate.loader == "MAGSTAR":
            data, meta = load_magstar_cdf_direct(
                themis_module,
                code=load_info.code,
                day_start=window.day_start,
                ndays=window.ndays,
                force_download=force_download,
                download_files=download_missing,
            )
        elif candidate.loader == "THEMIS":
            result = call_themis_load(
                themis_module,
                code=load_info.code,
                day_start=window.day_start,
                ndays=window.ndays,
                force_download=force_download,
                download_files=download_missing,
            )
            data, meta = unpack_load_result(result)
        else:
            raise ValueError(f"Unknown loader {candidate.loader!r}")
    except Exception as exc:
        raise RuntimeError(
            f"{window.name} window {window.start} to {window.end}: {exc}"
        ) from exc

    # THEMIS and MagStar loaders normally return complete daily files. Trim the
    # frame here so active and quiet data remain disjoint even when separated by
    # several days. CARISMA is already windowed, but the same trim is harmless.
    index = utc_index(data.index)
    valid = ~index.isna()
    data = data.loc[np.asarray(valid)].copy()
    data.index = index[valid]
    data = data.loc[(data.index >= window.start) & (data.index <= window.end)]
    if data.empty:
        raise RuntimeError(
            f"{window.name} window {window.start} to {window.end}: no samples returned"
        )
    return data, meta


def unpack_load_result(result: Any) -> tuple[pd.DataFrame, Any]:
    if isinstance(result, tuple) and len(result) >= 2:
        data, meta = result[0], result[1]
    else:
        data, meta = result, None
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "GMAG load result did not contain a pandas DataFrame; got "
            f"{type(data).__name__}"
        )
    return data.copy(), meta


def component_token_from_column(column: str, code: str) -> Optional[str]:
    """Infer source component label from a station DataFrame column."""
    raw = str(column).strip()
    low = raw.lower()
    code_low = code.lower()

    # Tokenized form is strongest: KAPU_H, thg_mag_kapu_h, KAPU.BZ, etc.
    tokens = [token for token in re.split(r"[^a-z0-9]+", low) if token]
    tokens_without_code = [token for token in tokens if token != code_low]

    aliases = {
        "n": "N", "north": "N", "bn": "N", "dbn": "N",
        "h": "H", "horizontal": "H", "bh": "H",
        "x": "X", "bx": "X", "dbx": "X",
        "e": "E", "east": "E", "be": "E", "dbe": "E",
        "d": "D", "decl": "D", "declination": "D",
        "y": "Y", "by": "Y", "dby": "Y",
        "z": "Z", "bz": "Z", "dbz": "Z", "vertical": "Z",
        "down": "Z", "v": "Z",
    }
    for token in reversed(tokens_without_code):
        if token in aliases:
            return aliases[token]

    # Compact names such as KAPUH or thgmagkapuh.
    compact = normalize_name(raw)
    compact = compact.replace(normalize_name(code), "")
    suffix_aliases = (
        ("dbnorth", "N"), ("north", "N"), ("dbn", "N"), ("bn", "N"),
        ("horizontal", "H"), ("bh", "H"),
        ("dbeast", "E"), ("east", "E"), ("dbe", "E"), ("be", "E"),
        ("vertical", "Z"), ("down", "Z"), ("dbz", "Z"), ("bz", "Z"),
        ("dbx", "X"), ("bx", "X"), ("dby", "Y"), ("by", "Y"),
        ("h", "H"), ("n", "N"), ("x", "X"),
        ("e", "E"), ("d", "D"), ("y", "Y"), ("z", "Z"),
    )
    for suffix, label in suffix_aliases:
        if compact.endswith(suffix):
            return label
    return None


def metadata_strings(meta: Any, code: str) -> list[str]:
    text: list[str] = []
    if meta is None:
        return text

    if isinstance(meta, pd.Series):
        frame = meta.to_frame().T
    elif isinstance(meta, pd.DataFrame):
        frame = meta.copy()
    else:
        return [safe_string(meta)]

    code_col = find_column(frame, aliases=("code", "station", "iaga"))
    if code_col is not None:
        mask = frame[code_col].astype(str).str.upper() == code.upper()
        if mask.any():
            frame = frame.loc[mask]
    if frame.empty:
        return text

    for column in frame.columns:
        text.append(f"{column}={safe_string(frame.iloc[0][column])}")
    return text


def infer_coordinate_system(meta: Any, code: str, source_labels: Sequence[str]) -> str:
    combined = " ".join(metadata_strings(meta, code)).lower()
    labels = "".join(source_labels).upper()

    if any(token in combined for token in ("geomagnetic", "magnetic", "hdz", "hez")):
        return "local geomagnetic"
    if any(token in combined for token in ("geographic", "xyz", "geo")):
        return "geographic"
    if "H" in labels and ("E" in labels or "D" in labels):
        return "local geomagnetic (inferred from H/E-or-D/Z labels)"
    if "X" in labels and "Y" in labels:
        return "geographic or instrument XYZ (inferred; verify metadata)"
    if "N" in labels and "E" in labels:
        return "NEZ (inferred from labels)"
    return "unknown; component mapping inferred from DataFrame labels"


def select_station_component_columns(
    data: pd.DataFrame,
    code: str,
    meta: Any,
) -> tuple[tuple[str, str, str], tuple[str, str, str], str]:
    """Return columns mapped to output N, E, Z and source labels.

    Mapping rules
    -------------
    N output accepts source N, H, or X.
    E output accepts source E, D, or Y.
    Z output accepts source Z.

    No sign reversal is applied. This is explicitly recorded in NetCDF
    metadata. The horizontal mapping supports the H/E/Z and H/D/Z labels used
    by several ground magnetometer products, plus XYZ and NEZ.
    """
    frame = data.copy()
    frame.columns = [flatten_column_name(column) for column in frame.columns]
    code_norm = normalize_name(code)

    candidates = []
    for column in frame.columns:
        col_norm = normalize_name(column)
        token = component_token_from_column(column, code)
        station_match = code_norm in col_norm
        candidates.append((str(column), token, station_match))

    # Prefer columns explicitly carrying the station code. If a single-station
    # load returns only three magnetic columns without code labels, permit them.
    matched = [item for item in candidates if item[2] and item[1] is not None]
    usable = matched if matched else [item for item in candidates if item[1] is not None]

    def choose(labels: Sequence[str]) -> Optional[tuple[str, str]]:
        for label in labels:
            for column, token, _ in usable:
                if token == label:
                    return column, token
        return None

    north = choose(("N", "H", "X"))
    east = choose(("E", "D", "Y"))
    vertical = choose(("Z",))

    if north is None or east is None or vertical is None:
        details = ", ".join(
            f"{column!r}->{token or '?'}{'*' if station_match else ''}"
            for column, token, station_match in candidates
        )
        meta_text = " ; ".join(metadata_strings(meta, code))
        raise ValueError(
            f"Could not identify three components for {code}. "
            f"Parsed columns: {details}. Metadata: {meta_text}"
        )

    columns = (north[0], east[0], vertical[0])
    source_labels = (north[1], east[1], vertical[1])
    mapping = f"{source_labels[0]}->N, {source_labels[1]}->E, {source_labels[2]}->Z; signs preserved"
    return columns, source_labels, mapping


def extract_station_frame(
    data: pd.DataFrame,
    code: str,
    meta: Any,
) -> tuple[pd.DataFrame, tuple[str, str, str], str, str]:
    frame = data.copy()
    frame.columns = [flatten_column_name(column) for column in frame.columns]

    index = utc_index(frame.index)
    valid_time = ~index.isna()
    frame = frame.loc[np.asarray(valid_time)].copy()
    frame.index = index[valid_time]
    frame = frame.sort_index()

    columns, source_labels, mapping = select_station_component_columns(frame, code, meta)
    out = pd.DataFrame(index=frame.index)
    out["N"] = pd.to_numeric(frame[columns[0]], errors="coerce")
    out["E"] = pd.to_numeric(frame[columns[1]], errors="coerce")
    out["Z"] = pd.to_numeric(frame[columns[2]], errors="coerce")

    values = out.to_numpy(dtype=float)
    bad = (~np.isfinite(values)) | (np.abs(values) > ABSOLUTE_VALUE_LIMIT_NT)
    values[bad] = np.nan
    out.loc[:, ["N", "E", "Z"]] = values

    # Average duplicate timestamps. This also protects against repeated CDF
    # records and overlapping daily files.
    out = out.groupby(level=0).mean().sort_index()
    coordinate_system = infer_coordinate_system(meta, code, source_labels)
    return out, columns, mapping, coordinate_system


def metadata_resolution_seconds(meta: Any, code: str) -> Optional[float]:
    if meta is None:
        return None
    if isinstance(meta, pd.Series):
        frame = meta.to_frame().T
    elif isinstance(meta, pd.DataFrame):
        frame = meta.copy()
    else:
        return None

    code_col = find_column(frame, aliases=("code", "station", "iaga"))
    if code_col is not None:
        mask = frame[code_col].astype(str).str.upper() == code.upper()
        if mask.any():
            frame = frame.loc[mask]
    if frame.empty:
        return None

    candidate_columns = [
        column for column in frame.columns
        if any(
            token in normalize_name(column)
            for token in ("resolution", "cadence", "sampling", "sampleperiod", "dt")
        )
    ]
    for column in candidate_columns:
        value = frame.iloc[0][column]
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if np.isfinite(numeric) and float(numeric) > 0:
            return float(numeric)

        text = safe_string(value).lower()
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|millisecond|s|sec|second|min|minute)", text)
        if match:
            amount = float(match.group(1))
            unit = match.group(2)
            if unit.startswith("m") and unit not in {"ms", "millisecond"}:
                return amount * 60.0
            if unit in {"ms", "millisecond"}:
                return amount / 1000.0
            return amount
    return None


def data_resolution_seconds(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    valid = frame[["N", "E", "Z"]].notna().any(axis=1)
    times = frame.index[valid]
    if len(times) < 2:
        return np.nan
    diffs = np.diff(times.asi8).astype(float) / 1.0e9
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return np.nan
    # The low quantile is more robust than the median when an otherwise
    # one-second series contains many outage gaps.
    return float(np.nanpercentile(diffs, 20.0))


# =============================================================================
# Pi2 preparation
# =============================================================================

def _window_to_samples(index: pd.DatetimeIndex, window: str | pd.Timedelta) -> int:
    """Convert a pandas time window to an approximate number of samples."""
    index = pd.DatetimeIndex(pd.to_datetime(index, utc=True, errors="coerce"))
    if len(index) < 2:
        return 1

    valid = index[~index.isna()]
    if len(valid) < 2:
        return 1
    cadence = pd.Series(valid).diff().median()
    if pd.isna(cadence) or cadence <= pd.Timedelta(0):
        return 1
    return max(1, int(np.ceil(pd.to_timedelta(window) / cadence)))


def _split_mask_by_run_duration(
    mask: pd.Series,
    max_duration_seconds: float,
) -> tuple[pd.Series, pd.Series]:
    """Split a boolean mask into short and long contiguous runs.

    A one-sample run has the duration of one native sample. Candidate runs
    separated by a short gap are grouped before classification. Short groups
    are eligible for interpolation. Long groups are returned separately so
    rolling-MAD excursions associated with real disturbances can be preserved.
    """
    mask = mask.fillna(False).astype(bool)
    short_mask = pd.Series(False, index=mask.index, dtype=bool)
    long_mask = pd.Series(False, index=mask.index, dtype=bool)
    if not mask.any():
        return short_mask, long_mask

    valid_index = pd.DatetimeIndex(mask.index)
    if len(valid_index) >= 2:
        diffs = np.diff(valid_index.asi8).astype(float) / 1.0e9
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        cadence_seconds = float(np.nanmedian(diffs)) if diffs.size else 1.0
    else:
        cadence_seconds = 1.0

    max_samples = max(
        1, int(np.floor(max_duration_seconds / cadence_seconds + 1.0e-9))
    )
    values = mask.to_numpy(dtype=bool)
    grouped = values.copy()

    # Join candidate runs separated by only a very short non-candidate gap.
    # This prevents a broad physical disturbance with a few sub-threshold
    # samples from being misclassified as several isolated instrumental spikes.
    false_starts = np.flatnonzero((~grouped) & np.r_[True, grouped[:-1]])
    false_ends = np.flatnonzero((~grouped) & np.r_[grouped[1:], True])
    for gap_start, gap_end in zip(false_starts, false_ends):
        gap_length = int(gap_end - gap_start + 1)
        bounded = gap_start > 0 and gap_end < len(grouped) - 1
        if bounded and gap_length <= max_samples:
            grouped[gap_start : gap_end + 1] = True

    starts = np.flatnonzero(grouped & np.r_[True, ~grouped[:-1]])
    ends = np.flatnonzero(grouped & np.r_[~grouped[1:], True])
    for run_start, run_end in zip(starts, ends):
        run_length = int(run_end - run_start + 1)
        destination = short_mask if run_length <= max_samples else long_mask
        # Classify only the original candidate samples; the bridged gap is used
        # solely to determine whether the surrounding excursion is sustained.
        original_slice = values[run_start : run_end + 1]
        if original_slice.any():
            positions = np.flatnonzero(original_slice) + run_start
            destination.iloc[positions] = True
    return short_mask, long_mask


def clean_magnetometer_spikes(
    frame: pd.DataFrame,
    columns: Sequence[str] = ("N", "E", "Z"),
    zero_drop_columns: Sequence[str] = ("N", "Z"),
    zero_reference_frame: Optional[pd.DataFrame] = None,
    zero_threshold: float = DEFAULT_ZERO_DROPOUT_THRESHOLD_NT,
    rolling_window: str | pd.Timedelta = DEFAULT_DESPIKE_ROLLING_WINDOW,
    mad_threshold: float = DEFAULT_DESPIKE_MAD_THRESHOLD,
    min_abs_deviation: float = DEFAULT_DESPIKE_MIN_ABS_DEVIATION_NT,
    max_spike_duration_seconds: float = DEFAULT_MAX_DESPIKE_DURATION_SECONDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repair short vector spikes without deleting sustained disturbances.

    Rolling-median/MAD candidate masks are calculated independently for N, E,
    and Z and combined into one vector mask. Only contiguous candidate runs no
    longer than ``max_spike_duration_seconds`` are treated as instrumental
    spikes. Longer rolling-MAD excursions are preserved because auroral and
    storm-time magnetic variations can legitimately remain far from a local
    five-minute median for many seconds.

    Confirmed short spikes are removed from all three components and replaced
    by time interpolation between the nearest valid vector samples. Therefore
    spike cleaning does not itself create NaN intervals. Genuine source gaps
    and long near-zero dropouts remain NaN and are handled later by the normal
    short-gap policy.
    """
    columns = [column for column in columns if column in frame.columns]
    if not columns:
        raise ValueError("No requested magnetic components are present")
    if mad_threshold <= 0:
        raise ValueError("mad_threshold must be positive")
    if min_abs_deviation < 0:
        raise ValueError("min_abs_deviation cannot be negative")
    if zero_threshold < 0:
        raise ValueError("zero_threshold cannot be negative")
    if max_spike_duration_seconds <= 0:
        raise ValueError("max_spike_duration_seconds must be positive")
    if pd.to_timedelta(rolling_window) <= pd.Timedelta(0):
        raise ValueError("rolling_window must be positive")

    original = frame.copy().sort_index()
    original.index = pd.DatetimeIndex(pd.to_datetime(original.index, utc=True))
    original.loc[:, columns] = original[columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Use a separate statistics frame so detected zero/dropout values do not
    # contaminate the local rolling median and MAD. For first-sample-baselined
    # perturbations, zero/dropout detection must use the corresponding absolute
    # measurements because valid perturbations naturally cross zero.
    statistics_frame = original.copy()
    if zero_reference_frame is None:
        zero_reference = original.copy()
    else:
        zero_reference = zero_reference_frame.copy().sort_index()
        zero_reference.index = pd.DatetimeIndex(
            pd.to_datetime(zero_reference.index, utc=True)
        )
        zero_reference = zero_reference.reindex(original.index)
        zero_reference.loc[:, columns] = zero_reference[columns].apply(
            pd.to_numeric, errors="coerce"
        )
    zero_candidates = [
        column for column in zero_drop_columns if column in zero_reference.columns
    ]
    zero_eligible: list[str] = []
    level_floor = max(10.0 * zero_threshold, min_abs_deviation)
    for column in zero_candidates:
        values = np.abs(zero_reference[column].to_numpy(dtype=float))
        typical_level = (
            float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
        )
        if np.isfinite(typical_level) and typical_level > level_floor:
            zero_eligible.append(column)

    if zero_eligible:
        zero_dropout_mask = (
            zero_reference[zero_eligible].abs() <= zero_threshold
        ).any(axis=1)
    else:
        zero_dropout_mask = pd.Series(
            False, index=statistics_frame.index, dtype=bool
        )
    statistics_frame.loc[zero_dropout_mask, columns] = np.nan

    window_samples = _window_to_samples(statistics_frame.index, rolling_window)
    min_periods = max(5, window_samples // 5)
    component_masks: dict[str, pd.Series] = {}
    component_counts: dict[str, int] = {}

    for column in columns:
        series = statistics_frame[column]
        local_median = series.rolling(
            rolling_window, center=True, min_periods=min_periods
        ).median()
        residual = (series - local_median).abs()
        local_mad = residual.rolling(
            rolling_window, center=True, min_periods=min_periods
        ).median()
        robust_sigma = 1.4826 * local_mad

        fallback_sigma = float(np.nanmedian(robust_sigma.to_numpy(dtype=float)))
        if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
            fallback_sigma = float(np.nanmedian(residual.to_numpy(dtype=float)))
        if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
            fallback_sigma = 1.0

        threshold = mad_threshold * robust_sigma
        threshold = threshold.fillna(mad_threshold * fallback_sigma)
        threshold = threshold.clip(lower=min_abs_deviation)
        mask = (residual > threshold).fillna(False)
        component_masks[column] = mask
        component_counts[column] = int(mask.sum())

    shared_candidate_mask = pd.Series(
        False, index=statistics_frame.index, dtype=bool
    )
    for mask in component_masks.values():
        shared_candidate_mask |= mask

    short_spike_mask, long_candidate_mask = _split_mask_by_run_duration(
        shared_candidate_mask,
        max_duration_seconds=max_spike_duration_seconds,
    )
    short_zero_mask, long_zero_mask = _split_mask_by_run_duration(
        zero_dropout_mask,
        max_duration_seconds=max_spike_duration_seconds,
    )

    repair_mask = short_spike_mask | short_zero_mask
    cleaned = original.copy()
    cleaned.loc[repair_mask | long_zero_mask, columns] = np.nan

    # Interpolate a complete vector only at confirmed short bad-sample rows.
    # Existing source NaNs and long zero/dropout intervals are never filled here.
    interpolated = cleaned[columns].interpolate(
        method="time", axis=0, limit_area="inside"
    )
    cleaned.loc[repair_mask, columns] = interpolated.loc[repair_mask, columns]

    repair_success = repair_mask & cleaned[columns].notna().all(axis=1)
    repair_failed = repair_mask & ~cleaned[columns].notna().all(axis=1)

    report: dict[str, Any] = {
        "zero_dropout_rows": int(zero_dropout_mask.sum()),
        "zero_dropout_components_used": tuple(zero_eligible),
        "rolling_mad_candidates_by_component": component_counts,
        "shared_candidate_rows": int(shared_candidate_mask.sum()),
        "short_spike_rows_repaired": int(short_spike_mask.sum()),
        "short_zero_rows_repaired": int(short_zero_mask.sum()),
        "long_candidate_rows_preserved": int(long_candidate_mask.sum()),
        "long_zero_rows_left_missing": int(long_zero_mask.sum()),
        "vector_rows_repaired": int(repair_success.sum()),
        "vector_rows_unrepaired": int(repair_failed.sum() + long_zero_mask.sum()),
        "_repaired_index": pd.DatetimeIndex(cleaned.index[repair_success]),
    }
    return cleaned, report


def interpolate_short_gaps(
    values: np.ndarray,
    max_gap_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill interior NaN runs no longer than ``max_gap_samples``.

    Quality flags are assigned on the selected output grid. This covers both
    cadence-conversion samples and short vector gaps introduced by despiking.
    Longer gaps remain NaN.
    """
    output = np.asarray(values, dtype=float).copy()
    if output.ndim != 2:
        raise ValueError("values must have shape (time, component)")

    quality = np.full(output.shape, QUALITY_MISSING, dtype=np.uint8)
    quality[np.isfinite(output)] = QUALITY_OBSERVED

    if max_gap_samples <= 0:
        return output, quality

    n_time, n_component = output.shape
    x = np.arange(n_time, dtype=float)
    for component in range(n_component):
        y = output[:, component].copy()
        missing = ~np.isfinite(y)
        if not missing.any():
            continue

        starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
        ends = np.flatnonzero(missing & np.r_[~missing[1:], True])
        for gap_start, gap_end in zip(starts, ends):
            length = gap_end - gap_start + 1
            if (
                length > max_gap_samples
                or gap_start == 0
                or gap_end == n_time - 1
            ):
                continue
            left = y[gap_start - 1]
            right = y[gap_end + 1]
            if not np.isfinite(left) or not np.isfinite(right):
                continue
            y[gap_start : gap_end + 1] = np.interp(
                x[gap_start : gap_end + 1],
                [x[gap_start - 1], x[gap_end + 1]],
                [left, right],
            )
            quality[gap_start : gap_end + 1, component] = QUALITY_INTERPOLATED
        output[:, component] = y
    return output, quality


def snap_cadence_seconds(cadence_seconds: float) -> float:
    """Snap measured cadence to common GMAG values while retaining odd cadences."""
    cadence = float(cadence_seconds)
    if np.isclose(cadence, 0.5, rtol=0.0, atol=0.08):
        return 0.5
    if np.isclose(cadence, 1.0, rtol=0.0, atol=0.12):
        return 1.0
    return max(1.0e-6, round(cadence, 6))


def cadence_time_grid(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cadence_seconds: float,
) -> pd.DatetimeIndex:
    cadence = snap_cadence_seconds(cadence_seconds)
    grid_start = parse_utc(start)
    grid_end = parse_utc(end)
    if np.isclose(cadence, 60.0, rtol=0.0, atol=1.0e-9):
        # The minute product is always labelled on exact UTC minute marks,
        # even when the requested interval includes non-zero seconds.
        grid_start = grid_start.ceil("1min")
        grid_end = grid_end.floor("1min")
    return pd.date_range(
        start=grid_start,
        end=grid_end,
        freq=pd.to_timedelta(cadence, unit="s"),
    )


def _align_frame_to_rule(
    frame: pd.DataFrame,
    cadence_seconds: float,
) -> pd.DataFrame:
    rule = pd.to_timedelta(snap_cadence_seconds(cadence_seconds), unit="s")
    work = frame[["N", "E", "Z"]].copy().sort_index()
    work.index = pd.DatetimeIndex(pd.to_datetime(work.index, utc=True)).round(rule)
    return work.groupby(level=0).mean().sort_index()


def resample_native_to_output_grid(
    frame: pd.DataFrame,
    target_time: pd.DatetimeIndex,
    filter_time: pd.DatetimeIndex,
    native_cadence_seconds: float,
    output_cadence_seconds: float,
    max_gap_seconds: float,
    despike: bool,
    despike_rolling_window: str | pd.Timedelta,
    despike_mad_threshold: float,
    despike_min_abs_deviation_nt: float,
    max_despike_duration_seconds: float,
    zero_dropout_threshold_nt: float,
    zero_dropout_check: bool,
    zero_reference_frame: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Clean at native cadence, then place data on the requested output grid.

    Faster data are averaged into slower output bins. Slower data are aligned
    to their native timestamps and linearly interpolated only onto the requested
    faster grid, subject to the configured maximum short-gap duration.
    """
    native_cadence_seconds = snap_cadence_seconds(native_cadence_seconds)
    output_cadence_seconds = snap_cadence_seconds(output_cadence_seconds)
    native_rule = pd.to_timedelta(native_cadence_seconds, unit="s")
    output_rule = pd.to_timedelta(output_cadence_seconds, unit="s")

    work = _align_frame_to_rule(frame, native_cadence_seconds)
    zero_reference = None
    if zero_reference_frame is not None:
        zero_reference = _align_frame_to_rule(
            zero_reference_frame, native_cadence_seconds
        ).reindex(work.index)

    repaired_index = pd.DatetimeIndex([])
    if despike:
        work, cleaning_report = clean_magnetometer_spikes(
            work,
            columns=("N", "E", "Z"),
            zero_drop_columns=("N", "Z") if zero_dropout_check else (),
            zero_reference_frame=zero_reference,
            zero_threshold=zero_dropout_threshold_nt,
            rolling_window=despike_rolling_window,
            mad_threshold=despike_mad_threshold,
            min_abs_deviation=despike_min_abs_deviation_nt,
            max_spike_duration_seconds=max_despike_duration_seconds,
        )
        repaired_index = cleaning_report["_repaired_index"]
        component_text = ",".join(
            f"{key}:{value}"
            for key, value in cleaning_report[
                "rolling_mad_candidates_by_component"
            ].items()
        )
        cleaning_text = (
            "transient vector cleaning enabled "
            f"(zero candidates={cleaning_report['zero_dropout_rows']}, "
            f"MAD candidates={component_text}, "
            f"shared candidates={cleaning_report['shared_candidate_rows']}, "
            f"short vector rows repaired={cleaning_report['vector_rows_repaired']}, "
            f"long MAD candidate rows preserved="
            f"{cleaning_report['long_candidate_rows_preserved']}, "
            f"unrepaired invalid rows={cleaning_report['vector_rows_unrepaired']}, "
            f"maximum repaired transient={max_despike_duration_seconds:g} s)"
        )
    else:
        cleaning_text = "vector cleaning disabled"

    if output_cadence_seconds > native_cadence_seconds * 1.05:
        # Average every populated slower-output bin. Selecting only samples at
        # the exact output timestamp made phase-offset native series (common in
        # IMAGE and custom feeds) look as though they had regular gaps even when
        # their high-rate input was continuous.
        work = work.resample(
            output_rule,
            origin=filter_time[0],
            label="left",
            closed="left",
        ).mean()
        work.index = work.index.round(output_rule)
        work = work.groupby(level=0).mean().sort_index()
        cadence_text = (
            f"native {native_cadence_seconds:g}-second samples averaged into "
            f"populated common {output_cadence_seconds:g}-second bins"
        )
    elif output_cadence_seconds < native_cadence_seconds * 0.95:
        cadence_text = (
            f"native {native_cadence_seconds:g}-second samples linearly "
            f"interpolated onto the common {output_cadence_seconds:g}-second grid"
        )
    else:
        work.index = work.index.round(output_rule)
        work = work.groupby(level=0).mean().sort_index()
        cadence_text = (
            f"native {native_cadence_seconds:g}-second samples retained on "
            f"their {output_cadence_seconds:g}-second output grid"
        )

    native_on_filter = work.reindex(filter_time).to_numpy(dtype=float)
    max_gap_samples = int(
        np.floor(max_gap_seconds / output_cadence_seconds + 1.0e-9)
    )
    cleaned_filter, quality_filter = interpolate_short_gaps(
        native_on_filter,
        max_gap_samples=max_gap_samples,
    )
    if despike:
        post_frame, post_report = clean_magnetometer_spikes(
            pd.DataFrame(cleaned_filter, index=filter_time, columns=("N", "E", "Z")),
            columns=("N", "E", "Z"),
            zero_drop_columns=(),
            rolling_window=despike_rolling_window,
            mad_threshold=despike_mad_threshold,
            min_abs_deviation=despike_min_abs_deviation_nt,
            max_spike_duration_seconds=max_despike_duration_seconds,
        )
        cleaned_filter = post_frame.to_numpy(dtype=float)
        post_positions = filter_time.get_indexer(post_report["_repaired_index"])
        post_positions = post_positions[post_positions >= 0]
        if post_positions.size:
            quality_filter[post_positions, :] = QUALITY_INTERPOLATED
        cleaning_text += (
            f"; post-gap-fill despike repeated "
            f"({post_report['vector_rows_repaired']} vector rows repaired)"
        )

    if len(repaired_index):
        repaired_positions = filter_time.get_indexer(repaired_index)
        repaired_positions = repaired_positions[repaired_positions >= 0]
        if repaired_positions.size:
            quality_filter[repaired_positions, :] = QUALITY_INTERPOLATED

    target_positions = filter_time.get_indexer(target_time)
    if np.any(target_positions < 0):
        raise RuntimeError("Internal target/filter time-axis mismatch")
    cleaned_target = cleaned_filter[target_positions, :]
    quality_target = quality_filter[target_positions, :]
    return (
        cleaned_filter,
        cleaned_target,
        quality_target,
        f"{cadence_text}; {cleaning_text}",
    )


def quiet_rolling_mean_baseline(
    frame: pd.DataFrame,
    quiet_start: pd.Timestamp,
    quiet_end: pd.Timestamp,
    native_cadence_seconds: float,
    max_gap_seconds: float,
    minimum_samples: int,
    rolling_window: str | pd.Timedelta,
    despike: bool,
    despike_rolling_window: str | pd.Timedelta,
    despike_mad_threshold: float,
    despike_min_abs_deviation_nt: float,
    max_despike_duration_seconds: float,
    zero_dropout_threshold_nt: float,
    zero_dropout_check: bool,
) -> tuple[np.ndarray, float, str]:
    """Reduce quiet-reference data to one smoothed N/E/Z background vector.

    The quiet interval is not converted into or retained as a full output time
    series. Instead, samples stay on their native cadence, are
    vector-cleaned, and have only short interior gaps filled. A centred rolling
    mean is then calculated for each component. The scalar baseline retained
    for each component is the median of the valid rolling-background values
    whose timestamps fall inside the requested quiet interval.

    Taking the median after the rolling mean prevents a remaining slow excursion
    or edge effect from dominating the single baseline value, while the rolling
    mean suppresses the small fluctuations that should not be included in the
    background field.
    """
    if quiet_end < quiet_start:
        raise ValueError("quiet_end precedes quiet_start")
    rolling_td = pd.to_timedelta(rolling_window)
    if rolling_td <= pd.Timedelta(0):
        raise ValueError("Quiet-baseline rolling window must be positive")
    if native_cadence_seconds <= 0:
        raise ValueError("Quiet native cadence must be positive")

    work = frame[["N", "E", "Z"]].copy().sort_index()
    native_rule = pd.to_timedelta(
        snap_cadence_seconds(native_cadence_seconds), unit="s"
    )
    work.index = work.index.round(native_rule)
    work = work.groupby(level=0).mean().sort_index()

    if despike:
        work, cleaning_report = clean_magnetometer_spikes(
            work,
            columns=("N", "E", "Z"),
            zero_drop_columns=("N", "Z") if zero_dropout_check else (),
            zero_threshold=zero_dropout_threshold_nt,
            rolling_window=despike_rolling_window,
            mad_threshold=despike_mad_threshold,
            min_abs_deviation=despike_min_abs_deviation_nt,
            max_spike_duration_seconds=max_despike_duration_seconds,
        )
        cleaning_text = (
            "transient vector cleaning enabled "
            f"(zero candidates={cleaning_report['zero_dropout_rows']}, "
            f"shared MAD candidates={cleaning_report['shared_candidate_rows']}, "
            f"short vector rows repaired={cleaning_report['vector_rows_repaired']}, "
            f"long MAD candidate rows preserved="
            f"{cleaning_report['long_candidate_rows_preserved']}, "
            f"unrepaired invalid rows={cleaning_report['vector_rows_unrepaired']}, "
            f"maximum repaired transient={max_despike_duration_seconds:g} s)"
        )
    else:
        cleaning_text = "vector cleaning disabled"

    if work.empty:
        raise ValueError("No quiet-reference samples remain after preprocessing")

    native_grid = pd.date_range(
        start=work.index.min(), end=work.index.max(), freq=native_rule
    )
    native_values = work.reindex(native_grid).to_numpy(dtype=float)
    max_gap_samples = int(
        np.floor(max_gap_seconds / native_rule.total_seconds() + 1.0e-9)
    )
    cleaned_values, _ = interpolate_short_gaps(
        native_values, max_gap_samples=max_gap_samples
    )
    cleaned = pd.DataFrame(
        cleaned_values, index=native_grid, columns=["N", "E", "Z"]
    )

    requested = cleaned.loc[(cleaned.index >= quiet_start) & (cleaned.index <= quiet_end)]
    if requested.empty:
        raise ValueError("Quiet interval does not overlap the loaded quiet data")

    component_coverage = requested.notna().mean(axis=0).to_numpy(dtype=float)
    quiet_coverage = float(np.nanmin(component_coverage))
    for component in ("N", "E", "Z"):
        finite_count = int(requested[component].notna().sum())
        if finite_count < minimum_samples:
            raise ValueError(
                f"Only {finite_count} finite native samples in quiet interval "
                f"for component {component}; require at least {minimum_samples}"
            )

    window_samples = max(
        1, int(np.ceil(rolling_td.total_seconds() / native_rule.total_seconds()))
    )
    # Require at least half a rolling window, while retaining the existing
    # minimum-sample quality criterion for short selected intervals.
    rolling_min_periods = max(
        minimum_samples, int(np.ceil(0.5 * window_samples))
    )
    rolling_background = cleaned.rolling(
        rolling_td,
        center=True,
        min_periods=rolling_min_periods,
    ).mean()
    selected_background = rolling_background.loc[
        (rolling_background.index >= quiet_start)
        & (rolling_background.index <= quiet_end)
    ]

    baseline = np.full(3, np.nan, dtype=float)
    for index, component in enumerate(("N", "E", "Z")):
        finite_background = selected_background[component].dropna().to_numpy(dtype=float)
        if finite_background.size == 0:
            raise ValueError(
                f"No valid {rolling_td} rolling-mean background values for "
                f"quiet component {component}. Increase the quiet interval, "
                "provide more padding/data, or reduce the rolling window."
            )
        baseline[index] = float(np.nanmedian(finite_background))

    processing_text = (
        f"native quiet cadence {native_rule.total_seconds():g} s; {cleaning_text}; "
        f"interior gaps <= {max_gap_seconds:g} s filled; centred {rolling_td} "
        "rolling mean; median of valid rolling-background values retained as "
        "the scalar N/E/Z quiet baseline; quiet time series not stored"
    )
    return baseline, quiet_coverage, processing_text


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0 or not mask.any():
        return []
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def bandpass_pi2(
    values: np.ndarray,
    sampling_hz: float,
    min_period_seconds: float,
    max_period_seconds: float,
    order: int,
) -> np.ndarray:
    """Band-pass complete finite runs and retain NaN across longer gaps."""
    if min_period_seconds <= 0 or max_period_seconds <= min_period_seconds:
        raise ValueError("Require 0 < min Pi2 period < max Pi2 period")
    if sampling_hz <= 0:
        raise ValueError("Sampling frequency must be positive")

    low_hz = 1.0 / max_period_seconds
    high_hz = 1.0 / min_period_seconds
    nyquist = 0.5 * sampling_hz
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Invalid filter band {low_hz:g}--{high_hz:g} Hz for "
            f"Nyquist frequency {nyquist:g} Hz"
        )

    sos = signal.butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sampling_hz,
        output="sos",
    )
    output = np.full_like(values, np.nan, dtype=float)

    # Require several longest-period cycles to reduce edge-dominated results.
    minimum_run = max(int(np.ceil(3.0 * max_period_seconds * sampling_hz)), 60)

    for component in range(values.shape[1]):
        y = np.asarray(values[:, component], dtype=float)
        finite = np.isfinite(y)
        for start, end in contiguous_true_runs(finite):
            if end - start < minimum_run:
                continue
            segment = signal.detrend(y[start:end], type="linear")
            try:
                output[start:end, component] = signal.sosfiltfilt(sos, segment)
            except ValueError:
                # A run can still be too short for scipy's internal padding.
                continue
    return output


def pi2_band_resolvable(
    sampling_hz: float,
    min_period_seconds: float,
    max_period_seconds: float,
) -> bool:
    """Return whether a sampled series can represent the full Pi2 band."""
    if sampling_hz <= 0 or min_period_seconds <= 0 or max_period_seconds <= min_period_seconds:
        return False
    low_hz = 1.0 / max_period_seconds
    high_hz = 1.0 / min_period_seconds
    return bool(0.0 < low_hz < high_hz < 0.5 * sampling_hz)


def _resolved_native_cadence(
    frame: pd.DataFrame,
    meta: Any,
    code: str,
    label: str,
    max_native_cadence_seconds: float,
) -> float:
    """Resolve and validate cadence for one independently loaded interval."""
    meta_resolution = metadata_resolution_seconds(meta, code)
    measured_resolution = data_resolution_seconds(frame)
    if meta_resolution is not None and np.isfinite(meta_resolution):
        cadence = float(meta_resolution)
        if np.isfinite(measured_resolution) and measured_resolution < cadence / 2:
            cadence = measured_resolution
    else:
        cadence = measured_resolution

    if not np.isfinite(cadence):
        raise ValueError(f"Could not determine {label} native sampling cadence")
    if cadence > max_native_cadence_seconds:
        raise ValueError(
            f"{label} native cadence is {cadence:g} s, slower than allowed "
            f"{max_native_cadence_seconds:g} s"
        )
    return float(cadence)


def first_data_point_baseline(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[np.ndarray, pd.Timestamp]:
    """Return the first complete N/E/Z vector in the requested sample."""
    requested = frame.loc[(frame.index >= start) & (frame.index <= end), ["N", "E", "Z"]]
    complete = requested.dropna(how="any")
    if complete.empty:
        raise ValueError(
            "No complete N/E/Z sample is available at the beginning of the "
            "requested interval for first-data-point baselining"
        )
    timestamp = pd.Timestamp(complete.index[0])
    return complete.iloc[0].to_numpy(dtype=float), timestamp


def prepare_station(
    catalogue_station: CatalogueStation,
    active_data: pd.DataFrame,
    active_meta: Any,
    quiet_data: Optional[pd.DataFrame],
    quiet_meta: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    filter_start: pd.Timestamp,
    filter_end: pd.Timestamp,
    cadence_mode: str,
    data_mode: str,
    baseline_method: str,
    quiet_start: Optional[pd.Timestamp],
    quiet_end: Optional[pd.Timestamp],
    max_native_cadence_seconds: float,
    max_gap_seconds: float,
    min_coverage: float,
    min_quiet_samples: int,
    quiet_baseline_rolling_window: str | pd.Timedelta,
    pi2_min_period_seconds: float,
    pi2_max_period_seconds: float,
    filter_order: int,
    despike: bool,
    despike_rolling_window: str | pd.Timedelta,
    despike_mad_threshold: float,
    despike_min_abs_deviation_nt: float,
    max_despike_duration_seconds: float,
    zero_dropout_threshold_nt: float,
    zero_dropout_check: bool,
) -> LoadedStation:
    """Prepare one station for native or common-cadence output."""
    active_frame, source_columns, mapping, coordinate_system = extract_station_frame(
        active_data, catalogue_station.code, active_meta
    )
    active_cadence = _resolved_native_cadence(
        active_frame,
        active_meta,
        catalogue_station.code,
        "active",
        max_native_cadence_seconds,
    )
    active_cadence = snap_cadence_seconds(active_cadence)

    if cadence_mode == "original":
        output_cadence_seconds = active_cadence
    elif cadence_mode == "1s":
        output_cadence_seconds = 1.0
    elif cadence_mode == "10s":
        output_cadence_seconds = 10.0
    elif cadence_mode == "1min":
        output_cadence_seconds = 60.0
    elif cadence_mode == "2hz":
        output_cadence_seconds = 0.5
    else:
        raise ValueError(f"Unknown cadence mode {cadence_mode!r}")

    target_time = cadence_time_grid(start, end, output_cadence_seconds)
    filter_time = cadence_time_grid(
        filter_start, filter_end, output_cadence_seconds
    )

    quiet_cadence = np.nan
    quiet_coverage = np.nan
    quiet_processing = "not used"
    quiet_coordinate_system = coordinate_system
    quiet_metadata_text = ""
    baseline_timestamp = ""

    if data_mode == "actual":
        baseline = np.zeros(3, dtype=float)
        processing_frame = active_frame
        zero_reference_frame = None
        baseline_method = "none"
    elif baseline_method == "first-sample":
        baseline, first_timestamp = first_data_point_baseline(
            active_frame, start=start, end=end
        )
        baseline_timestamp = first_timestamp.isoformat()
        # The requested order is respected: subtract the first data point first,
        # then perform cleaning, gap filling, and despiking. Dropout detection
        # still references the absolute measurements.
        processing_frame = active_frame - baseline[None, :]
        zero_reference_frame = active_frame
    elif baseline_method == "quiet":
        if quiet_data is None or quiet_start is None or quiet_end is None:
            raise ValueError("Quiet-baseline mode requires a complete quiet interval")
        quiet_frame, _, quiet_mapping, quiet_coordinate_system = extract_station_frame(
            quiet_data, catalogue_station.code, quiet_meta
        )
        if quiet_mapping != mapping:
            raise ValueError(
                "Active and quiet intervals use different component mappings: "
                f"active={mapping!r}, quiet={quiet_mapping!r}"
            )
        quiet_cadence = _resolved_native_cadence(
            quiet_frame,
            quiet_meta,
            catalogue_station.code,
            "quiet-reference",
            max_native_cadence_seconds,
        )
        quiet_cadence = snap_cadence_seconds(quiet_cadence)
        baseline, quiet_coverage, quiet_processing = quiet_rolling_mean_baseline(
            frame=quiet_frame,
            quiet_start=quiet_start,
            quiet_end=quiet_end,
            native_cadence_seconds=quiet_cadence,
            max_gap_seconds=max_gap_seconds,
            minimum_samples=min_quiet_samples,
            rolling_window=quiet_baseline_rolling_window,
            despike=despike,
            despike_rolling_window=despike_rolling_window,
            despike_mad_threshold=despike_mad_threshold,
            despike_min_abs_deviation_nt=despike_min_abs_deviation_nt,
            max_despike_duration_seconds=max_despike_duration_seconds,
            zero_dropout_threshold_nt=zero_dropout_threshold_nt,
            zero_dropout_check=zero_dropout_check,
        )
        processing_frame = active_frame
        zero_reference_frame = None
        quiet_metadata_text = dataframe_metadata_text(
            quiet_meta, catalogue_station.code
        )
    else:
        raise ValueError(f"Unknown baseline method {baseline_method!r}")

    cleaned_filter, cleaned_target, quality_target, active_resampling = (
        resample_native_to_output_grid(
            frame=processing_frame,
            target_time=target_time,
            filter_time=filter_time,
            native_cadence_seconds=active_cadence,
            output_cadence_seconds=output_cadence_seconds,
            max_gap_seconds=max_gap_seconds,
            despike=despike,
            despike_rolling_window=despike_rolling_window,
            despike_mad_threshold=despike_mad_threshold,
            despike_min_abs_deviation_nt=despike_min_abs_deviation_nt,
            max_despike_duration_seconds=max_despike_duration_seconds,
            zero_dropout_threshold_nt=zero_dropout_threshold_nt,
            zero_dropout_check=zero_dropout_check,
            zero_reference_frame=zero_reference_frame,
        )
    )

    if data_mode == "actual":
        raw_filter = cleaned_filter
        raw_target = cleaned_target
        output_filter = cleaned_filter
        output_target = cleaned_target
    elif baseline_method == "first-sample":
        output_filter = cleaned_filter
        output_target = cleaned_target
        raw_filter = cleaned_filter + baseline[None, :]
        raw_target = cleaned_target + baseline[None, :]
    else:
        raw_filter = cleaned_filter
        raw_target = cleaned_target
        output_filter = cleaned_filter - baseline[None, :]
        output_target = cleaned_target - baseline[None, :]

    active_horizontal = (
        np.isfinite(output_target[:, 0]) & np.isfinite(output_target[:, 1])
    )
    coverage = float(np.count_nonzero(active_horizontal) / len(target_time))
    if coverage < min_coverage:
        raise ValueError(
            f"active horizontal {1.0 / output_cadence_seconds:g}-Hz coverage is "
            f"{coverage:.1%}, below required {min_coverage:.1%}"
        )

    output_sampling_hz = 1.0 / output_cadence_seconds
    filtered_full = np.full_like(output_filter, np.nan, dtype=float)
    if pi2_band_resolvable(
        output_sampling_hz,
        pi2_min_period_seconds,
        pi2_max_period_seconds,
    ):
        try:
            filtered_full = bandpass_pi2(
                output_filter,
                sampling_hz=output_sampling_hz,
                min_period_seconds=pi2_min_period_seconds,
                max_period_seconds=pi2_max_period_seconds,
                order=filter_order,
            )
        except Exception:
            # Pi2 is an optional derived product. A filtering problem must not
            # reject otherwise valid 10-second (or other cadence) magnetic data
            # or prevent its NetCDF file from being written.
            pass
    # An unresolved band (including 1-minute output) also remains NaN. This is
    # a derived-product limitation, not a download or station-data failure.
    target_positions = filter_time.get_indexer(target_time)
    pi2 = filtered_full[target_positions, :]

    metadata_text = dataframe_metadata_text(active_meta, catalogue_station.code)
    if quiet_metadata_text and quiet_metadata_text != metadata_text:
        metadata_text = (
            f"{metadata_text} | quiet_reference_metadata={quiet_metadata_text}"
            if metadata_text else f"quiet_reference_metadata={quiet_metadata_text}"
        )
    if is_magstar_station(catalogue_station):
        supplement = (
            f"array=MagStar | data_source={MAGSTAR_DATA_SOURCE} | "
            f"catalogue_source={MAGSTAR_CATALOGUE_SOURCE} | {MAGSTAR_PROVENANCE}"
        )
        metadata_text = f"{metadata_text} | {supplement}" if metadata_text else supplement

    if data_mode == "actual":
        baseline_text = "actual measurements retained; no baseline subtracted"
    elif baseline_method == "first-sample":
        baseline_text = (
            f"first complete sample at {baseline_timestamp} subtracted before "
            "cleaning, gap filling, and despiking"
        )
    else:
        baseline_text = f"quiet-reference baseline: {quiet_processing}"

    output_resampling = f"active: {active_resampling} | baseline: {baseline_text}"
    if quiet_coordinate_system != coordinate_system:
        output_resampling += (
            f" | coordinate metadata differs: active={coordinate_system}; "
            f"quiet={quiet_coordinate_system}"
        )

    return LoadedStation(
        code=catalogue_station.code,
        station_name=catalogue_station.station_name,
        array_name=catalogue_station.array_name,
        glat=catalogue_station.glat,
        glon=catalogue_station.glon,
        times=target_time,
        output_cadence_seconds=output_cadence_seconds,
        native_cadence_seconds=active_cadence,
        quiet_native_cadence_seconds=quiet_cadence,
        raw_nez=raw_target.astype(np.float32),
        output_nez=output_target.astype(np.float32),
        pi2_nez=pi2.astype(np.float32),
        subtracted_baseline=np.asarray(baseline, dtype=np.float32),
        baseline_method=baseline_method,
        baseline_timestamp=baseline_timestamp,
        data_mode=data_mode,
        quality=quality_target,
        coverage_fraction=coverage,
        quiet_coverage_fraction=quiet_coverage,
        source_coordinate_system=coordinate_system,
        source_component_mapping=mapping,
        source_columns=source_columns,
        metadata_text=metadata_text,
        output_resampling=output_resampling,
    )


# =============================================================================
# NetCDF output
# =============================================================================

def create_string_variable(ds: Dataset, name: str, dimension: str, values: Sequence[str]):
    variable = ds.createVariable(name, str, (dimension,))
    variable[:] = np.asarray(list(values), dtype=object)
    return variable


def _write_root_station_metadata(ds: Dataset, stations: Sequence[LoadedStation]) -> None:
    n_station = len(stations)
    ds.createDimension("station", n_station)
    ds.createDimension("component", 3)
    component_var = ds.createVariable("component", "i4", ("component",))
    component_var[:] = np.asarray([0, 1, 2], dtype=np.int32)
    component_var.description = "0=N, 1=E, 2=Z"
    component_var.source_mapping_note = (
        "N accepts source N/H/X; E accepts E/D/Y; Z accepts Z. "
        "Source signs are preserved. See source_component_mapping."
    )
    create_string_variable(ds, "station", "station", [s.code for s in stations])
    create_string_variable(ds, "array_name", "station", [s.array_name for s in stations])
    create_string_variable(ds, "station_name", "station", [s.station_name for s in stations])
    create_string_variable(
        ds, "source_coordinate_system", "station",
        [s.source_coordinate_system for s in stations],
    )
    create_string_variable(
        ds, "source_component_mapping", "station",
        [s.source_component_mapping for s in stations],
    )
    create_string_variable(ds, "gmag_metadata", "station", [s.metadata_text for s in stations])
    create_string_variable(
        ds, "output_resampling", "station", [s.output_resampling for s in stations]
    )
    create_string_variable(
        ds, "baseline_method", "station", [s.baseline_method for s in stations]
    )
    create_string_variable(
        ds, "baseline_timestamp", "station", [s.baseline_timestamp for s in stations]
    )

    def vector(name: str, values, units: str = "", long_name: str = ""):
        var = ds.createVariable(name, "f4", ("station",), fill_value=np.nan)
        var[:] = np.asarray(values, dtype=np.float32)
        if units:
            var.units = units
        if long_name:
            var.long_name = long_name
        return var

    vector("glat", [s.glat for s in stations], "degrees_north")
    vector("glon", [s.glon for s in stations], "degrees_east")
    vector(
        "native_cadence_seconds", [s.native_cadence_seconds for s in stations], "s",
        "native cadence in the active interval",
    )
    vector(
        "quiet_native_cadence_seconds",
        [s.quiet_native_cadence_seconds for s in stations], "s",
        "native cadence in the quiet-reference interval",
    )
    vector(
        "output_cadence_seconds", [s.output_cadence_seconds for s in stations], "s",
        "station output cadence",
    )
    vector(
        "coverage_fraction", [s.coverage_fraction for s in stations], "1",
        "finite horizontal coverage in the active interval",
    )
    vector(
        "quiet_coverage_fraction", [s.quiet_coverage_fraction for s in stations], "1",
        "minimum finite component coverage in the quiet-reference interval",
    )
    baseline = ds.createVariable(
        "subtracted_baseline", "f4", ("station", "component"), fill_value=np.nan
    )
    baseline[:] = np.stack([s.subtracted_baseline for s in stations])
    baseline.units = "nT"
    baseline.long_name = "N/E/Z value subtracted from the active measurements"

    # Retain the previous variable name for readers already using it. It now
    # records the actual value subtracted, regardless of baseline method.
    legacy = ds.createVariable(
        "quiet_baseline", "f4", ("station", "component"), fill_value=np.nan
    )
    legacy[:] = baseline[:]
    legacy.units = "nT"
    legacy.long_name = "legacy alias of subtracted_baseline"


def _write_time_variable(container: Any, times: pd.DatetimeIndex) -> Any:
    naive_utc = times.tz_convert("UTC").tz_localize(None).to_pydatetime()
    units = "seconds since 1970-01-01 00:00:00 UTC"
    var = container.createVariable("time", "f8", ("time",))
    var[:] = date2num(list(naive_utc), units=units, calendar="standard")
    var.units = units
    var.calendar = "standard"
    var.standard_name = "time"
    var.axis = "T"
    return var


def _write_data_variables(
    container: Any,
    output_nez: np.ndarray,
    raw_nez: np.ndarray,
    pi2_nez: np.ndarray,
    quality: np.ndarray,
    dimensions: tuple[str, ...],
    data_mode: str,
    pi2_min_period_seconds: float,
    pi2_max_period_seconds: float,
    filter_order: int,
    max_gap_seconds: float,
) -> None:
    kwargs = dict(zlib=True, complevel=4)
    nez = container.createVariable("nez", "f4", dimensions, fill_value=np.nan, **kwargs)
    raw = container.createVariable("raw_nez", "f4", dimensions, fill_value=np.nan, **kwargs)
    pi2 = container.createVariable("pi2_nez", "f4", dimensions, fill_value=np.nan, **kwargs)
    geo = container.createVariable("geo", "f4", dimensions, fill_value=np.nan, **kwargs)
    quality_var = container.createVariable("data_quality", "u1", dimensions, **kwargs)
    nez[:] = output_nez
    raw[:] = raw_nez
    pi2[:] = pi2_nez
    geo[:] = np.full_like(output_nez, np.nan, dtype=np.float32)
    quality_var[:] = quality
    for var in (nez, raw, pi2, geo):
        var.units = "nT"
    nez.long_name = (
        "cleaned actual magnetic measurements" if data_mode == "actual"
        else "cleaned baseline-subtracted magnetic perturbation"
    )
    raw.long_name = "cleaned absolute magnetic measurements"
    pi2.long_name = "Pi2-band linearly detrended magnetic data"
    pi2.minimum_period_seconds = float(pi2_min_period_seconds)
    pi2.maximum_period_seconds = float(pi2_max_period_seconds)
    pi2.filter = f"Butterworth band-pass, order {filter_order}, zero phase"
    if not np.isfinite(pi2_nez).any():
        pi2.note = (
            "Unavailable when the output cadence cannot resolve the complete "
            "requested period band, or when finite runs are too short for filtering"
        )
    geo.long_name = "unavailable geographic-coordinate vector"
    geo.note = "Stored as NaN to retain the existing viewer-compatible schema"
    quality_var.long_name = "data quality flag"
    quality_var.flag_values = np.asarray(
        [QUALITY_OBSERVED, QUALITY_INTERPOLATED, QUALITY_MISSING], dtype=np.uint8
    )
    quality_var.flag_meanings = "observed_or_aligned short_gap_interpolated missing"
    quality_var.max_interpolated_gap_seconds = float(max_gap_seconds)


def write_netcdf(
    output_path: Path,
    stations: Sequence[LoadedStation],
    start: pd.Timestamp,
    end: pd.Timestamp,
    quiet_start: Optional[pd.Timestamp],
    quiet_end: Optional[pd.Timestamp],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    pi2_min_period_seconds: float,
    pi2_max_period_seconds: float,
    filter_order: int,
    max_gap_seconds: float,
    cadence_mode: str,
    data_mode: str,
    command_line: str,
    download_mode: str,
    despike: bool,
    despike_rolling_window: str,
    despike_mad_threshold: float,
    despike_min_abs_deviation_nt: float,
    max_despike_duration_seconds: float,
    zero_dropout_threshold_nt: float,
    zero_dropout_check: bool,
    quiet_baseline_rolling_window: str,
) -> None:
    if not stations:
        raise ValueError("No accepted stations to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(output_path, "w", format="NETCDF4") as ds:
        _write_root_station_metadata(ds, stations)

        if cadence_mode == "original":
            station_parent = ds.createGroup("stations")
            for station in stations:
                group = station_parent.createGroup(station.code)
                group.createDimension("time", len(station.times))
                group.createDimension("component", 3)
                component = group.createVariable("component", "i4", ("component",))
                component[:] = np.asarray([0, 1, 2], dtype=np.int32)
                component.description = "0=N, 1=E, 2=Z"
                _write_time_variable(group, station.times)
                _write_data_variables(
                    group,
                    station.output_nez,
                    station.raw_nez,
                    station.pi2_nez,
                    station.quality,
                    ("time", "component"),
                    data_mode,
                    pi2_min_period_seconds,
                    pi2_max_period_seconds,
                    filter_order,
                    max_gap_seconds,
                )
                baseline = group.createVariable(
                    "subtracted_baseline", "f4", ("component",), fill_value=np.nan
                )
                baseline[:] = station.subtracted_baseline
                baseline.units = "nT"
                group.output_cadence_seconds = float(station.output_cadence_seconds)
                group.native_cadence_seconds = float(station.native_cadence_seconds)
                group.baseline_method = station.baseline_method
                group.baseline_timestamp = station.baseline_timestamp
                group.data_mode = station.data_mode
            ds.netcdf_layout = (
                "nested station groups; each /stations/<code> group has its own "
                "time dimension and data variables"
            )
            ds.viewer_compatibility_note = (
                "Native-cadence mode cannot provide one rectangular root-level "
                "nez(station,time,component) array. Readers must support NetCDF4 groups."
            )
        else:
            reference_time = stations[0].times
            if any(not station.times.equals(reference_time) for station in stations[1:]):
                raise RuntimeError("Common-cadence stations do not share an identical time grid")
            ds.createDimension("time", len(reference_time))
            _write_time_variable(ds, reference_time)
            output = np.stack([s.output_nez for s in stations])
            raw = np.stack([s.raw_nez for s in stations])
            pi2 = np.stack([s.pi2_nez for s in stations])
            quality = np.stack([s.quality for s in stations])
            _write_data_variables(
                ds, output, raw, pi2, quality,
                ("station", "time", "component"),
                data_mode,
                pi2_min_period_seconds,
                pi2_max_period_seconds,
                filter_order,
                max_gap_seconds,
            )
            ds.netcdf_layout = "common root-level station/time/component arrays"
            ds.output_cadence_seconds = float(stations[0].output_cadence_seconds)
            ds.output_sampling_hz = float(1.0 / stations[0].output_cadence_seconds)

        ds.title = "GMAG ground magnetometers with selectable output cadence"
        ds.source = (
            "CARISMA F01 data prefetched concurrently using gmag.arrays.carisma paths; "
            "THEMIS-hosted CDF data downloaded with gmag.arrays.themis; "
            "MagStar CDF timestamps decoded directly with cdflib"
        )
        ds.station_catalogue = (
            "GMAG utils.load_station_geo plus canonical CARISMA, Nain, "
            "and built-in MagStar/Gannon supplements"
        )
        ds.magstar_catalogue_source = MAGSTAR_CATALOGUE_SOURCE
        ds.magstar_station_codes = ",".join(sorted(MAGSTAR_STATION_CODES))
        ds.magstar_provenance = MAGSTAR_PROVENANCE
        ds.carisma_acknowledgement = CARISMA_ACKNOWLEDGEMENT
        ds.station_alias_source = SUPERMAG_STATION_LIST_SOURCE
        ds.supermag_reference = SUPERMAG_REFERENCE
        ds.carisma_aliases = "; ".join(
            f"{canonical}={','.join(aliases)}"
            for canonical, aliases in CARISMA_CODE_ALIASES.items() if aliases
        )
        ds.interval_start_utc = start.isoformat()
        ds.interval_end_utc = end.isoformat()
        ds.quiet_interval_start_utc = quiet_start.isoformat() if quiet_start is not None else ""
        ds.quiet_interval_end_utc = quiet_end.isoformat() if quiet_end is not None else ""
        ds.region = (
            f"latitude {lat_min:g} to {lat_max:g} degrees north; "
            f"longitude {lon_min:g} to {lon_max:g} degrees east"
        )
        ds.cadence_mode = cadence_mode
        ds.data_mode = data_mode
        ds.source_download_mode = download_mode
        ds.baseline_definition = (
            "Per-station baseline_method identifies quiet rolling-mean, first-sample, "
            "or no subtraction. subtracted_baseline stores the N/E/Z values used."
        )
        ds.component_sign_convention = (
            "No component sign reversal was applied. Verify source metadata "
            "before interpreting vertical polarization or rotation sense."
        )
        ds.vector_despiking_enabled = int(bool(despike))
        ds.vector_despiking = (
            "Rolling-median/MAD candidates are calculated independently for N, E, "
            "and Z and combined into one vector mask. Only contiguous candidate "
            f"runs <= {max_despike_duration_seconds:g} s are repaired in all three "
            "components. Longer excursions are preserved. "
            f"window={despike_rolling_window}, MAD multiplier={despike_mad_threshold:g}, "
            f"minimum absolute deviation={despike_min_abs_deviation_nt:g} nT, "
            f"near-zero dropout check={'enabled' if zero_dropout_check else 'disabled'}, "
            f"threshold={zero_dropout_threshold_nt:g} nT."
        )
        ds.quiet_baseline_method = (
            f"centred {quiet_baseline_rolling_window} rolling mean followed by the "
            "median of valid rolling-background values when a quiet interval is supplied"
        )
        ds.pi2_processing = (
            "Data are cleaned at native cadence before cadence conversion. Faster "
            "data are averaged for slower output; slower data are linearly interpolated "
            f"only across gaps <= {max_gap_seconds:g} s. Pi2 filtering uses a "
            f"{filter_order}th-order zero-phase Butterworth band-pass for periods "
            f"{pi2_min_period_seconds:g}--{pi2_max_period_seconds:g} s."
        )
        if not pi2_band_resolvable(
            1.0 / float(stations[0].output_cadence_seconds),
            pi2_min_period_seconds,
            pi2_max_period_seconds,
        ):
            ds.pi2_processing += (
                " The selected output cadence cannot resolve the complete requested "
                "Pi2 band, so pi2_nez is stored as NaN without rejecting station data."
            )
        ds.history = (
            f"Created {datetime.now(timezone.utc).isoformat()} by "
            "gmag_download_and_clean.py"
        )
        ds.command_line = command_line

def default_station_map_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_accepted_station_map.png")


def write_station_map(
    output_path: Path,
    stations: Sequence[LoadedStation],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dpi: int = 180,
) -> None:
    """Plot accepted station locations over continents in the requested grid.

    Cartopy is used when available. A Basemap fallback is included because it
    is common in existing space-physics environments and does not require the
    rest of the downloader to fail when Cartopy is absent.
    """
    if not stations:
        raise ValueError("No accepted stations are available for the map")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("Station-map creation requires matplotlib") from exc

    width = max(9.0, min(15.0, 8.0 + (lon_max - lon_min) / 25.0))
    height = max(6.0, min(11.0, 5.0 + (lat_max - lat_min) / 18.0))
    lons = np.asarray([station.glon for station in stations], dtype=float)
    lats = np.asarray([station.glat for station in stations], dtype=float)

    fig = None
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        projection = ccrs.PlateCarree()
        fig = plt.figure(figsize=(width, height), constrained_layout=True)
        ax = plt.axes(projection=projection)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)
        ax.add_feature(cfeature.OCEAN, facecolor="0.94", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="0.86", edgecolor="none", zorder=0)
        ax.add_feature(
            cfeature.LAKES, facecolor="0.94", edgecolor="0.55", linewidth=0.4
        )
        ax.coastlines(resolution="50m", linewidth=0.7, color="0.25")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="0.35")
        try:
            ax.add_feature(cfeature.STATES, linewidth=0.35, edgecolor="0.55")
        except Exception:
            pass
        gridliner = ax.gridlines(
            crs=projection, draw_labels=True, linewidth=0.45,
            color="0.45", alpha=0.65, linestyle="--"
        )
        gridliner.top_labels = False
        gridliner.right_labels = False
        ax.scatter(
            lons, lats, transform=projection, s=42, marker="o",
            facecolor="limegreen", edgecolor="black", linewidth=0.65, zorder=5,
            label=f"Accepted stations ({len(stations)})",
        )
        for station in stations:
            ax.text(
                station.glon + 0.35, station.glat + 0.18, station.code,
                transform=projection, fontsize=7.5, zorder=6,
                bbox=dict(
                    facecolor="white", edgecolor="none", alpha=0.6, pad=0.6
                ),
            )
    except Exception:
        if fig is not None:
            plt.close(fig)
        try:
            from mpl_toolkits.basemap import Basemap
        except Exception as exc:
            raise RuntimeError(
                "Station-map creation requires Cartopy or Basemap"
            ) from exc

        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
        map_object = Basemap(
            projection="cyl", llcrnrlon=lon_min, urcrnrlon=lon_max,
            llcrnrlat=lat_min, urcrnrlat=lat_max, resolution="l", ax=ax,
        )
        map_object.drawmapboundary(fill_color="0.94")
        map_object.fillcontinents(color="0.86", lake_color="0.94", zorder=0)
        map_object.drawcoastlines(linewidth=0.7, color="0.25")
        map_object.drawcountries(linewidth=0.5, color="0.35")
        try:
            map_object.drawstates(linewidth=0.35, color="0.55")
        except Exception:
            pass
        lat_step = 5 if (lat_max - lat_min) <= 60 else 10
        lon_step = 10 if (lon_max - lon_min) <= 140 else 20
        parallels = np.arange(
            np.floor(lat_min / lat_step) * lat_step, lat_max + lat_step, lat_step
        )
        meridians = np.arange(
            np.floor(lon_min / lon_step) * lon_step, lon_max + lon_step, lon_step
        )
        map_object.drawparallels(
            parallels, labels=[1, 0, 0, 0], linewidth=0.45,
            color="0.45", dashes=[4, 4], fontsize=8
        )
        map_object.drawmeridians(
            meridians, labels=[0, 0, 0, 1], linewidth=0.45,
            color="0.45", dashes=[4, 4], fontsize=8
        )
        x, y = map_object(lons, lats)
        ax.scatter(
            x, y, s=42, marker="o", facecolor="limegreen", edgecolor="black",
            linewidth=0.65, zorder=5, label=f"Accepted stations ({len(stations)})"
        )
        for station in stations:
            sx, sy = map_object(station.glon + 0.35, station.glat + 0.18)
            ax.text(
                sx, sy, station.code, fontsize=7.5, zorder=6,
                bbox=dict(
                    facecolor="white", edgecolor="none", alpha=0.6, pad=0.6
                ),
            )

    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title(
        "Accepted GMAG stations on the requested geographic grid\n"
        f"{start.strftime('%Y-%m-%d %H:%M:%S')} to "
        f"{end.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_report(output_path: Path, attempts: Sequence[StationAttempt]) -> Path:
    report_path = output_path.with_name(output_path.stem + "_station_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "station": attempt.code,
                "array": attempt.array_name,
                "glat": attempt.glat,
                "glon": attempt.glon,
                "status": attempt.status,
                "reason": attempt.reason,
                "native_cadence_seconds": attempt.native_cadence_seconds,
                "quiet_native_cadence_seconds": attempt.quiet_native_cadence_seconds,
                "output_cadence_seconds": attempt.output_cadence_seconds,
                "coverage_fraction": attempt.coverage_fraction,
                "quiet_coverage_fraction": attempt.quiet_coverage_fraction,
                "baseline_method": attempt.baseline_method,
                "baseline_timestamp": attempt.baseline_timestamp,
                "subtracted_n_nt": attempt.subtracted_n_nt,
                "subtracted_e_nt": attempt.subtracted_e_nt,
                "subtracted_z_nt": attempt.subtracted_z_nt,
                "output_resampling": attempt.output_resampling,
            }
            for attempt in attempts
        ]
    )
    frame.to_csv(report_path, index=False)
    return report_path


def validate_netcdf(path: Path) -> None:
    with Dataset(path, "r") as ds:
        cadence_mode = safe_string(getattr(ds, "cadence_mode", "2hz"))
        required_root = {
            "station", "component", "glat", "glon", "subtracted_baseline",
            "native_cadence_seconds", "output_cadence_seconds",
            "coverage_fraction", "output_resampling", "baseline_method",
        }
        missing = required_root - set(ds.variables)
        if missing:
            raise RuntimeError(
                f"Output validation failed; missing root variables: {sorted(missing)}"
            )
        names = [safe_string(value) for value in ds["station"][:]]
        print("\nNetCDF validation")
        print("-----------------")
        print(f"File:       {path}")
        print(f"Stations:   {', '.join(names)}")

        if cadence_mode == "original":
            if "stations" not in ds.groups:
                raise RuntimeError("Native-cadence output is missing /stations groups")
            parent = ds.groups["stations"]
            for i, code in enumerate(names):
                if code not in parent.groups:
                    raise RuntimeError(f"Missing station group /stations/{code}")
                group = parent.groups[code]
                for name in ("time", "nez", "raw_nez", "pi2_nez", "data_quality"):
                    if name not in group.variables:
                        raise RuntimeError(f"/stations/{code} is missing {name}")
                times = np.asarray(group["time"][:], dtype=float)
                cadence = float(np.nanmedian(np.diff(times))) if len(times) > 1 else np.nan
                expected = float(ds["output_cadence_seconds"][i])
                if len(times) > 1 and not np.isclose(cadence, expected, atol=1.0e-6):
                    raise RuntimeError(
                        f"{code}: output cadence {cadence:g} s; expected {expected:g} s"
                    )
                print(
                    f"  {code:5s}: nested time={len(times)}, cadence={expected:g} s, "
                    f"coverage={float(ds['coverage_fraction'][i]):.1%}"
                )
        else:
            for name in ("time", "nez", "raw_nez", "pi2_nez", "data_quality"):
                if name not in ds.variables:
                    raise RuntimeError(f"Common-grid output is missing {name}")
            n_station = len(ds.dimensions["station"])
            n_time = len(ds.dimensions["time"])
            if ds["nez"].shape != (n_station, n_time, 3):
                raise RuntimeError(f"Unexpected nez shape: {ds['nez'].shape}")
            time_values = np.asarray(ds["time"][:], dtype=float)
            cadence = float(np.nanmedian(np.diff(time_values))) if n_time > 1 else np.nan
            expected = float(ds["output_cadence_seconds"][0])
            if n_time > 1 and not np.isclose(cadence, expected, atol=1.0e-6):
                raise RuntimeError(
                    f"Unexpected output time cadence {cadence:g} s; expected {expected:g} s"
                )
            print(f"Dimensions: station={n_station}, time={n_time}, component=3")
            print(f"Output grid: {expected:g} s ({1.0 / expected:g} Hz)")
            for i, code in enumerate(names):
                print(
                    f"  {code:5s}: native={float(ds['native_cadence_seconds'][i]):g} s; "
                    f"coverage={float(ds['coverage_fraction'][i]):.1%}; "
                    f"baseline={safe_string(ds['baseline_method'][i])}"
                )


# =============================================================================
# Command-line workflow
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download GMAG stations, choose native/1-minute/10-second/1-second/2-Hz output, "
            "optionally derive magnetic perturbations, and write NetCDF4."
        )
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--quiet-start", default=DEFAULT_QUIET_START,
        help=(
            "Start of the optional quiet-reference interval. Supply both quiet "
            "times for quiet-baseline perturbations; omit both to subtract the "
            "first complete active data point."
        ),
    )
    parser.add_argument(
        "--quiet-end", default=DEFAULT_QUIET_END,
        help="End of the optional quiet-reference interval.",
    )
    parser.add_argument(
        "--cadence-mode", choices=("original", "1min", "10s", "1s", "2hz"), default=None,
        help=(
            "Output cadence: original=nested per-station time dimensions, "
            "1min=common exact-UTC-minute grid, 10s=common 10-second grid, "
            "1s=common 1-second grid, "
            "2hz=common 0.5-second grid. When omitted "
            "in an interactive terminal, the program asks."
        ),
    )
    parser.add_argument(
        "--output-cadence-seconds", type=float, default=None,
        help=(
            "Deprecated compatibility option. Use 60, 10, 1, or 0.5 for the "
            "corresponding common cadence mode."
        ),
    )
    parser.add_argument(
        "--data-mode", choices=("perturbation", "actual"), default=None,
        help=(
            "perturbation=subtract a quiet or first-sample baseline; "
            "actual=retain cleaned absolute measurements. When omitted in an "
            "interactive terminal, the program asks."
        ),
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Accept the first-data-point baseline warning without an interactive pause.",
    )
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stations", nargs="*", default=None,
        help="Optional explicit station codes, still constrained to the geographic box.",
    )
    parser.add_argument(
        "--max-stations", type=int, default=0,
        help="Limit catalogue stations for a test run; 0 means no limit.",
    )
    parser.add_argument("--no-magstar-supplement", action="store_true")
    parser.add_argument("--magstar-only", action="store_true")

    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument(
        "--skip-existing", dest="download_mode", action="store_const", const="missing",
        help="Reuse existing local files and download only missing files (default).",
    )
    download_group.add_argument(
        "--force-download", dest="download_mode", action="store_const", const="force",
    )
    download_group.add_argument(
        "--existing-only", dest="download_mode", action="store_const", const="existing-only",
    )
    parser.set_defaults(download_mode="missing")
    parser.add_argument(
        "--carisma-download-workers", type=int, default=DEFAULT_CARISMA_DOWNLOAD_WORKERS
    )
    parser.add_argument(
        "--carisma-request-timeout-seconds", type=float,
        default=DEFAULT_CARISMA_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--carisma-fallback-mode", choices=("fast", "exhaustive", "none"),
        default=DEFAULT_CARISMA_FALLBACK_MODE,
    )
    parser.add_argument(
        "--max-native-cadence-seconds", type=float,
        default=DEFAULT_MAX_NATIVE_CADENCE_SECONDS,
    )
    parser.add_argument(
        "--max-short-gap-seconds", type=float, default=DEFAULT_SHORT_GAP_SECONDS
    )
    parser.add_argument("--station-map", type=Path, default=None)
    parser.add_argument("--no-station-map", action="store_true")
    parser.add_argument("--map-dpi", type=int, default=180)
    parser.add_argument(
        "--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE
    )
    parser.add_argument("--min-quiet-samples", type=int, default=60)
    parser.add_argument(
        "--quiet-baseline-rolling-window",
        default=DEFAULT_QUIET_BASELINE_ROLLING_WINDOW,
    )
    parser.add_argument(
        "--pi2-min-period-seconds", type=float,
        default=DEFAULT_PI2_MIN_PERIOD_SECONDS,
    )
    parser.add_argument(
        "--pi2-max-period-seconds", type=float,
        default=DEFAULT_PI2_MAX_PERIOD_SECONDS,
    )
    parser.add_argument("--filter-order", type=int, default=DEFAULT_FILTER_ORDER)
    parser.add_argument(
        "--filter-padding-seconds", type=int, default=DEFAULT_FILTER_PADDING_SECONDS
    )
    cleaning_group = parser.add_mutually_exclusive_group()
    cleaning_group.add_argument("--despike", dest="despike", action="store_true")
    cleaning_group.add_argument("--no-despike", dest="despike", action="store_false")
    parser.set_defaults(despike=True)
    parser.add_argument(
        "--despike-rolling-window", default=DEFAULT_DESPIKE_ROLLING_WINDOW
    )
    parser.add_argument(
        "--despike-mad-threshold", type=float, default=DEFAULT_DESPIKE_MAD_THRESHOLD
    )
    parser.add_argument(
        "--despike-min-abs-deviation-nt", type=float,
        default=DEFAULT_DESPIKE_MIN_ABS_DEVIATION_NT,
    )
    parser.add_argument(
        "--max-despike-duration-seconds", type=float,
        default=DEFAULT_MAX_DESPIKE_DURATION_SECONDS,
    )
    parser.add_argument(
        "--zero-dropout-threshold-nt", type=float,
        default=DEFAULT_ZERO_DROPOUT_THRESHOLD_NT,
    )
    parser.add_argument("--no-zero-dropout-check", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _interactive_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    print(prompt)
    for key, encoded in choices.items():
        _, description = encoded.split("|", 1)
        print(f"  {key}) {description}")
    answer = input(f"Choose [{default}]: ").strip().lower() or default
    if answer not in choices:
        raise ValueError(f"Unknown choice {answer!r}")
    return choices[answer].split("|", 1)[0]


def resolve_modes(args) -> tuple[str, str, str]:
    interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())

    cadence_mode = args.cadence_mode
    if args.output_cadence_seconds is not None:
        legacy = float(args.output_cadence_seconds)
        if np.isclose(legacy, 0.5, atol=1.0e-12):
            legacy_mode = "2hz"
        elif np.isclose(legacy, 1.0, atol=1.0e-12):
            legacy_mode = "1s"
        elif np.isclose(legacy, 10.0, atol=1.0e-12):
            legacy_mode = "10s"
        elif np.isclose(legacy, 60.0, atol=1.0e-12):
            legacy_mode = "1min"
        else:
            raise ValueError(
                "--output-cadence-seconds now accepts only 0.5, 1.0, 10.0, or 60.0; use "
                "--cadence-mode original for native station cadence"
            )
        if cadence_mode is not None and cadence_mode != legacy_mode:
            raise ValueError("Conflicting cadence options were supplied")
        cadence_mode = legacy_mode

    if cadence_mode is None:
        if interactive:
            value = _interactive_choice(
                "Select output cadence:",
                {
                    "a": "original|original cadence for each station (nested NetCDF4 groups)",
                    "b": "10s|common IMAGE-compatible 10-second cadence",
                    "c": "1min|common exact-UTC-minute cadence",
                    "d": "1s|common 1-second cadence",
                    "e": "2hz|common THEMIS GMAG cadence, 0.5 seconds (2 Hz)",
                },
                "e",
            )
            cadence_mode = value
        else:
            cadence_mode = DEFAULT_CADENCE_MODE

    data_mode = args.data_mode
    if data_mode is None:
        if interactive:
            data_mode = _interactive_choice(
                "Select magnetic-data output:",
                {
                    "a": "perturbation|magnetic perturbations",
                    "b": "actual|cleaned actual magnetic measurements",
                },
                "a",
            )
        else:
            data_mode = DEFAULT_DATA_MODE

    has_quiet_start = bool(args.quiet_start)
    has_quiet_end = bool(args.quiet_end)
    if has_quiet_start != has_quiet_end:
        raise ValueError("Supply both --quiet-start and --quiet-end, or omit both")

    if data_mode == "actual":
        baseline_method = "none"
        if has_quiet_start:
            print("NOTE: quiet interval supplied but ignored because --data-mode actual was selected")
    elif has_quiet_start:
        baseline_method = "quiet"
    else:
        baseline_method = "first-sample"
        warning = (
            "No quiet-time interval was supplied. The first complete N/E/Z data "
            "point in each station sample will be subtracted to obtain perturbations; "
            "cleaning, gap filling, and despiking will then be applied."
        )
        print(f"WARNING: {warning}")
        if interactive and not args.yes:
            input("Press Enter to continue, or Ctrl-C to stop: ")

    return cadence_mode, data_mode, baseline_method


def validate_arguments(
    args,
    start: pd.Timestamp,
    end: pd.Timestamp,
    quiet_start: Optional[pd.Timestamp],
    quiet_end: Optional[pd.Timestamp],
) -> None:
    if end < start:
        raise ValueError("--end precedes --start")
    if quiet_start is not None and quiet_end is not None and quiet_end < quiet_start:
        raise ValueError("--quiet-end precedes --quiet-start")
    if args.lat_min > args.lat_max:
        raise ValueError("--lat-min exceeds --lat-max")
    if args.lon_min > args.lon_max:
        raise ValueError("--lon-min exceeds --lon-max; use normalized -180..180 longitudes")
    if args.max_native_cadence_seconds <= 0:
        raise ValueError("Maximum native cadence must be positive")
    if args.max_short_gap_seconds < 0:
        raise ValueError("Maximum short gap cannot be negative")
    if args.carisma_download_workers <= 0:
        raise ValueError("--carisma-download-workers must be positive")
    if args.carisma_request_timeout_seconds <= 0:
        raise ValueError("--carisma-request-timeout-seconds must be positive")
    if args.map_dpi <= 0:
        raise ValueError("Map DPI must be positive")
    if not (0.0 <= args.min_coverage <= 1.0):
        raise ValueError("--min-coverage must be between 0 and 1")
    if args.filter_order <= 0:
        raise ValueError("--filter-order must be positive")
    if args.filter_padding_seconds < 0:
        raise ValueError("--filter-padding-seconds cannot be negative")
    if args.min_quiet_samples <= 0:
        raise ValueError("--min-quiet-samples must be positive")
    try:
        quiet_window = pd.to_timedelta(args.quiet_baseline_rolling_window)
    except Exception as exc:
        raise ValueError("Invalid --quiet-baseline-rolling-window") from exc
    if quiet_window <= pd.Timedelta(0):
        raise ValueError("--quiet-baseline-rolling-window must be positive")
    try:
        despike_window = pd.to_timedelta(args.despike_rolling_window)
    except Exception as exc:
        raise ValueError("Invalid --despike-rolling-window") from exc
    if despike_window <= pd.Timedelta(0):
        raise ValueError("--despike-rolling-window must be positive")
    if args.despike_mad_threshold <= 0:
        raise ValueError("--despike-mad-threshold must be positive")
    if args.despike_min_abs_deviation_nt < 0:
        raise ValueError("--despike-min-abs-deviation-nt cannot be negative")
    if args.max_despike_duration_seconds <= 0:
        raise ValueError("--max-despike-duration-seconds must be positive")
    if args.zero_dropout_threshold_nt < 0:
        raise ValueError("--zero-dropout-threshold-nt cannot be negative")
    if args.magstar_only and args.no_magstar_supplement:
        raise ValueError("--magstar-only cannot be combined with --no-magstar-supplement")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cadence_mode, data_mode, baseline_method = resolve_modes(args)

    start = parse_utc(args.start)
    end = parse_utc(args.end)
    quiet_start = parse_utc(args.quiet_start) if args.quiet_start else None
    quiet_end = parse_utc(args.quiet_end) if args.quiet_end else None
    validate_arguments(args, start, end, quiet_start, quiet_end)

    force_download = args.download_mode == "force"
    download_missing = args.download_mode != "existing-only"
    filter_start = start - pd.Timedelta(seconds=args.filter_padding_seconds)
    filter_end = end + pd.Timedelta(seconds=args.filter_padding_seconds)

    gap_padding = pd.Timedelta(seconds=args.max_short_gap_seconds)
    despike_padding = (
        pd.to_timedelta(args.despike_rolling_window)
        if args.despike else pd.Timedelta(0)
    )
    active_window = DataWindow("active", filter_start, filter_end)
    download_windows: list[DataWindow] = [active_window]
    quiet_window: Optional[DataWindow] = None
    if baseline_method == "quiet":
        assert quiet_start is not None and quiet_end is not None
        quiet_baseline_padding = pd.to_timedelta(args.quiet_baseline_rolling_window)
        quiet_load_padding = max(gap_padding, despike_padding, quiet_baseline_padding)
        quiet_window = DataWindow(
            "quiet-reference",
            quiet_start - quiet_load_padding,
            quiet_end + quiet_load_padding,
        )
        download_windows.append(quiet_window)
    daily_spans = merged_daily_spans(download_windows)

    utils, themis, carisma = import_gmag()
    catalogue_raw = load_station_catalogue(utils)
    catalogue_all = standardize_catalogue(catalogue_raw)
    catalogue_all = canonicalize_carisma_alias_rows(catalogue_all)
    before_carisma_codes = {station.code.upper() for station in catalogue_all}
    catalogue_all = merge_missing_station_catalogues(catalogue_all, built_in_carisma_catalogue())
    catalogue_all = merge_missing_station_catalogues(catalogue_all, built_in_general_supplements())
    carisma_supplement_count = sum(
        station.code.upper() not in before_carisma_codes
        for station in built_in_carisma_catalogue()
    )
    magstar_supplement_count = 0
    if not args.no_magstar_supplement:
        before_codes = {station.code.upper() for station in catalogue_all}
        magstar_rows = built_in_magstar_catalogue()
        magstar_supplement_count = sum(
            station.code.upper() not in before_codes for station in magstar_rows
        )
        catalogue_all = merge_station_catalogues(catalogue_all, magstar_rows)
    if args.magstar_only:
        catalogue_all = [s for s in catalogue_all if is_magstar_station(s)]

    selected = filter_catalogue(
        catalogue_all,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        requested_codes=args.stations,
    )
    if args.max_stations > 0:
        selected = selected[:args.max_stations]
    if not selected:
        raise RuntimeError("No GMAG catalogue stations matched the requested region/codes")

    selected_carisma_codes = sorted({
        canonical_carisma_code(station.code)
        for station in selected if is_carisma_station(station)
    })
    carisma_prefetch_counts = {"cached": 0, "downloaded": 0, "missing": 0, "error": 0}
    if selected_carisma_codes:
        print(
            f"Prefetching {len(selected_carisma_codes)} CARISMA station file set(s) "
            f"with {args.carisma_download_workers} worker(s)..."
        )
        for span_day, span_ndays in daily_spans:
            counts = prefetch_carisma_files(
                carisma,
                codes=selected_carisma_codes,
                day_start=span_day,
                ndays=span_ndays,
                force_download=force_download,
                download_missing=download_missing,
                workers=args.carisma_download_workers,
                timeout_seconds=args.carisma_request_timeout_seconds,
            )
            for key, value in counts.items():
                carisma_prefetch_counts[key] += value

    print("GMAG variable-cadence NetCDF builder")
    print("------------------------------------")
    print(f"Output interval: {start} to {end}")
    if baseline_method == "quiet":
        print(f"Quiet interval:  {quiet_start} to {quiet_end}")
    else:
        print("Quiet interval:  not used")
    print(f"Cadence mode:    {cadence_mode}")
    print(f"Data mode:       {data_mode}")
    print(f"Baseline method: {baseline_method}")
    if cadence_mode == "original":
        print("NetCDF layout:   nested /stations/<code> groups with station-specific time")
    elif cadence_mode == "1min":
        print("Output grid:     60 s; all timestamps are exact UTC minute marks")
    elif cadence_mode == "1s":
        print("Output grid:     1 s (1 Hz); native 2-Hz data are aligned to 1-s timestamps")
    elif cadence_mode == "10s":
        print("Output grid:     10 s (0.1 Hz); all stations use IMAGE-compatible timestamps")
    else:
        print("Output grid:     0.5 s (2 Hz); native 1-Hz data are interpolated")
    print(
        f"Region:          lat {args.lat_min:g}..{args.lat_max:g}, "
        f"lon {args.lon_min:g}..{args.lon_max:g}"
    )
    print(f"Catalogue sites: {len(selected)}")
    print(
        f"CARISMA:         {sum(is_carisma_station(s) for s in selected)} selected; "
        f"{carisma_supplement_count} canonical row(s) added"
    )
    if selected_carisma_codes:
        print(
            "CARISMA files:   "
            f"cached={carisma_prefetch_counts['cached']}, "
            f"downloaded={carisma_prefetch_counts['downloaded']}, "
            f"missing={carisma_prefetch_counts['missing']}, "
            f"errors={carisma_prefetch_counts['error']}"
        )
    print(
        f"MagStar:         {sum(is_magstar_station(s) for s in selected)} selected; "
        f"{magstar_supplement_count} station row(s) added"
    )
    print(f"Download mode:   {args.download_mode}")
    print(
        f"Gap filling:     interior gaps <= {args.max_short_gap_seconds:g} s"
    )

    accepted: list[LoadedStation] = []
    attempts: list[StationAttempt] = []
    selected_codes = {station.code.upper() for station in selected}

    for number, station_info in enumerate(selected, start=1):
        prefix = (
            f"[{number:3d}/{len(selected):3d}] {station_info.code:5s} "
            f"{station_info.glat:7.2f} {station_info.glon:8.2f}"
        )
        loaded: Optional[LoadedStation] = None
        loaded_from: Optional[StationLoadCandidate] = None
        candidate_errors: list[str] = []

        fallbacks = STATION_CODE_FALLBACKS.get(station_info.code.upper(), ())
        fallback_codes = {code for code, _, _ in fallbacks}
        allow_fallback = not bool(fallback_codes & selected_codes)
        candidates = station_load_candidates(
            station_info,
            allow_fallback=allow_fallback,
            carisma_fallback_mode=args.carisma_fallback_mode,
        )

        for candidate_number, candidate in enumerate(candidates):
            load_info = candidate.load_station
            if candidate_number > 0:
                print(f"{prefix}  retrying {candidate.description}")
            try:
                active_data, active_meta = load_candidate_window(
                    candidate,
                    active_window,
                    themis_module=themis,
                    carisma_module=carisma,
                    force_download=force_download,
                    download_missing=download_missing,
                )
                quiet_data = None
                quiet_meta = None
                if quiet_window is not None:
                    quiet_data, quiet_meta = load_candidate_window(
                        candidate,
                        quiet_window,
                        themis_module=themis,
                        carisma_module=carisma,
                        force_download=force_download,
                        download_missing=download_missing,
                    )

                loaded = prepare_station(
                    catalogue_station=load_info,
                    active_data=active_data,
                    active_meta=active_meta,
                    quiet_data=quiet_data,
                    quiet_meta=quiet_meta,
                    start=start,
                    end=end,
                    filter_start=filter_start,
                    filter_end=filter_end,
                    cadence_mode=cadence_mode,
                    data_mode=data_mode,
                    baseline_method=baseline_method,
                    quiet_start=quiet_start,
                    quiet_end=quiet_end,
                    max_native_cadence_seconds=args.max_native_cadence_seconds,
                    max_gap_seconds=args.max_short_gap_seconds,
                    min_coverage=args.min_coverage,
                    min_quiet_samples=args.min_quiet_samples,
                    quiet_baseline_rolling_window=args.quiet_baseline_rolling_window,
                    pi2_min_period_seconds=args.pi2_min_period_seconds,
                    pi2_max_period_seconds=args.pi2_max_period_seconds,
                    filter_order=args.filter_order,
                    despike=args.despike,
                    despike_rolling_window=args.despike_rolling_window,
                    despike_mad_threshold=args.despike_mad_threshold,
                    despike_min_abs_deviation_nt=args.despike_min_abs_deviation_nt,
                    max_despike_duration_seconds=args.max_despike_duration_seconds,
                    zero_dropout_threshold_nt=args.zero_dropout_threshold_nt,
                    zero_dropout_check=not args.no_zero_dropout_check,
                )
                output_info = candidate.output_station
                loaded.code = output_info.code
                loaded.station_name = output_info.station_name
                loaded.array_name = output_info.array_name
                provenance = (
                    f"downloader_loader={candidate.loader} | archive_code={load_info.code} | "
                    f"output_code={output_info.code} | candidate={candidate.description} | "
                    f"active_window={start.isoformat()}..{end.isoformat()} | "
                    f"baseline_method={loaded.baseline_method}"
                )
                loaded.metadata_text = (
                    f"{loaded.metadata_text} | {provenance}"
                    if loaded.metadata_text else provenance
                )
                loaded_from = candidate
                break
            except Exception as exc:
                candidate_errors.append(f"{candidate.loader}:{load_info.code}: {exc}")
                if args.debug:
                    traceback.print_exc()

        if loaded is None or loaded_from is None:
            reason = " | ".join(candidate_errors)
            attempts.append(
                StationAttempt(
                    code=station_info.code,
                    array_name=station_info.array_name,
                    glat=station_info.glat,
                    glon=station_info.glon,
                    status="skipped",
                    reason=reason,
                    baseline_method=baseline_method,
                )
            )
            print(f"{prefix}  skipped: {reason}")
            continue

        accepted.append(loaded)
        attempts.append(
            StationAttempt(
                code=loaded.code,
                array_name=loaded.array_name,
                glat=loaded.glat,
                glon=loaded.glon,
                status="accepted",
                reason=(
                    f"loader={loaded_from.loader}; archive_code={loaded_from.load_station.code}; "
                    f"{loaded_from.description}; {loaded.source_component_mapping}"
                ),
                native_cadence_seconds=loaded.native_cadence_seconds,
                quiet_native_cadence_seconds=loaded.quiet_native_cadence_seconds,
                output_cadence_seconds=loaded.output_cadence_seconds,
                coverage_fraction=loaded.coverage_fraction,
                quiet_coverage_fraction=loaded.quiet_coverage_fraction,
                baseline_method=loaded.baseline_method,
                baseline_timestamp=loaded.baseline_timestamp,
                subtracted_n_nt=float(loaded.subtracted_baseline[0]),
                subtracted_e_nt=float(loaded.subtracted_baseline[1]),
                subtracted_z_nt=float(loaded.subtracted_baseline[2]),
                output_resampling=loaded.output_resampling,
            )
        )
        print(
            f"{prefix}  accepted as {loaded.code}: native="
            f"{loaded.native_cadence_seconds:g} s, output="
            f"{loaded.output_cadence_seconds:g} s, coverage="
            f"{loaded.coverage_fraction:.1%}, baseline={loaded.baseline_method}, "
            f"subtracted N/E/Z={loaded.subtracted_baseline.tolist()}"
        )

    if not accepted:
        report_path = write_report(args.output, attempts)
        raise RuntimeError(
            "No station met the cadence, coverage, component, and baseline "
            f"requirements. See report: {report_path}"
        )

    accepted.sort(key=lambda station: (-station.glat, station.glon, station.code))
    command_line = " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])
    write_netcdf(
        output_path=args.output,
        stations=accepted,
        start=start,
        end=end,
        quiet_start=quiet_start if baseline_method == "quiet" else None,
        quiet_end=quiet_end if baseline_method == "quiet" else None,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        pi2_min_period_seconds=args.pi2_min_period_seconds,
        pi2_max_period_seconds=args.pi2_max_period_seconds,
        filter_order=args.filter_order,
        max_gap_seconds=args.max_short_gap_seconds,
        cadence_mode=cadence_mode,
        data_mode=data_mode,
        command_line=command_line,
        download_mode=args.download_mode,
        despike=args.despike,
        despike_rolling_window=args.despike_rolling_window,
        despike_mad_threshold=args.despike_mad_threshold,
        despike_min_abs_deviation_nt=args.despike_min_abs_deviation_nt,
        max_despike_duration_seconds=args.max_despike_duration_seconds,
        zero_dropout_threshold_nt=args.zero_dropout_threshold_nt,
        zero_dropout_check=not args.no_zero_dropout_check,
        quiet_baseline_rolling_window=args.quiet_baseline_rolling_window,
    )
    report_path = write_report(args.output, attempts)
    map_path: Optional[Path] = None
    if not args.no_station_map:
        map_path = args.station_map or default_station_map_path(args.output)
        try:
            write_station_map(
                output_path=map_path,
                stations=accepted,
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
                start=start,
                end=end,
                dpi=args.map_dpi,
            )
        except Exception as exc:
            print(f"WARNING: station map was not created: {exc}")
            map_path = None
    validate_netcdf(args.output)

    print("\nCompleted")
    print("---------")
    print(f"Accepted stations: {len(accepted)} of {len(selected)}")
    print(f"NetCDF:            {args.output}")
    print(f"Station report:    {report_path}")
    if map_path is not None:
        print(f"Station map:       {map_path}")
    if cadence_mode == "original":
        print(
            "Native-cadence data are under /stations/<code>; viewers requiring "
            "one root-level time dimension must use --cadence-mode 1min, 10s, 1s, or 2hz."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
