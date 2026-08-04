#!/usr/bin/env python3
"""ExploreMag magnetic-data viewer and SuperMAG data-access core.

This module can run as the standalone ``mag_viewer.py`` application and also
provides the shared loading, conversion, cleaning, plotting, and SuperMAG
download services used by ``ExploreMag.py``.

Main capabilities
-----------------
1. Query SuperMAG by UTC interval and geographic region, download available
   ground-magnetometer stations, and save NEZ/GEO vectors in NetCDF format.
2. Open ExploreMag/SuperMAG NetCDF, native-cadence grouped GMAG NetCDF, custom
   SuperMAG block/vector NetCDF, and legacy ``magnetic_viewerT.py`` CSV files.
3. Convert custom SuperMAG products to ExploreMag's rectangular NetCDF schema,
   detect their cadence, align nominal timestamps, preserve genuine outages,
   and interpolate interior gaps shorter than five seconds.
4. Display loaded stations on an interactive geographic map with synchronized
   latitude-ordered and longitude-ordered station selection.
5. Plot one station's three magnetic components and horizontal magnitude, or
   create multi-station component and dBh stacks in NEZ or GEO coordinates.
6. Calculate stacked dB/dt using cadence-appropriate display units: nT/sec for
   1-second and 2-Hz products, and nT/min for 1-minute, 10-second, and native-
   cadence products. Custom SuperMAG units are chosen from detected cadence.
7. Filter plot stations by the region selected for plots and order stacks by
   latitude, longitude, or station code.
8. Produce interpolated component, magnitude, and derivative maps for one time
   or a sequence, with automatic or user-entered color limits.
9. Build editable manual multi-panel stacks, synchronize their UTC interval,
   mark and list selected samples, and export selected values as UTF-8 text.
10. Export station maps, parameter maps, single-station figures, and automatic
    or manual stack figures as PNG files.
11. Reuse dBh supplied by a data file when available or calculate it from the
    horizontal magnetic components.
12. Retry transient SuperMAG service failures while leaving download
    cancellation responsive.

Typical use
-----------
    python mag_viewer.py
    python mag_viewer.py existing_magnetic_file.nc

For the complete ground-magnetic-field and pulsation-analysis interface, run
``python ExploreMag.py``.

Required Python packages
------------------------
    numpy, pandas, matplotlib, netCDF4

Optional but recommended
------------------------
    cartopy  (station and magnetic-parameter maps)
    scipy    (magnetic-parameter map interpolation)
    mplcursors (hover labels on stacked traces)

Tkinter must be installed through the operating system's Python distribution.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import io
import json
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.dates import (
    AutoDateLocator,
    ConciseDateFormatter,
    DateFormatter,
    date2num,
)
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator
from netCDF4 import Dataset, num2date

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# -----------------------------------------------------------------------------
# SuperMAG "Rules of the Road" acknowledgement, shown at program startup.
# -----------------------------------------------------------------------------

SUPERMAG_ACKNOWLEDGEMENT_TEXT = """

==========================RULES OF THE ROAD======================

ExploreMag is built around the SuperMAG API and the GMAG Python package, which provide its core data-download capabilities. If you use ExploreMag to download, explore, or analyze ground magnetic field data, please cite the relevant references for SuperMAG, GMAG, and ExploreMag. A list of recommended references is provided below.

Please remember that SuperMAG is made possible by the generous contribution of data by numerous collaborators. To ensure their continued operation the user must follow the below rules-of-the-road. Data, plots or derived data products are provided under the limitations of "fair use" and cannot be redistributed. Contact the individual instrument PI and the SuperMAG PI for requests that are in conflict with these restrictions.

The user is requested to acknowledge individual collaborators and SuperMAG when original data, derived data, movies, or data products are used in publications and/or presentations.

=========================When Using Data========================

In all cases:

*** Include acknowledgement as listed on the SuperMAG website.
*** Include references to a technical papers for stations used (see list below).
*** Include SuperMAG reference: Gjerloev, J. W. (2012), The SuperMAG data processing technique, J. Geophys. Res., 117, A09213, doi:10.1029/2012JA017683

In cases that only a few stations play a key role and their data are central to the scientific conclusion of the paper:

*** Offer of co-authorship to the PI (or PIs) of those stations and reference the appropriate paper (see list below)

====================When Using Substorm Lists====================

*** If the substorm onset list is central to your study please offer co-authorship to the authors of the technique you use.
*** When using substorm lists please include acknowledgements found here.
*** Include appropriate reference (see list below)
*** For details please see https://supermag.jhuapl.edu/products/?tab=description.

### Using CARISMA data

- Include the acknowledgement required by the
  [CARISMA rules of the road](https://carisma.ca/carisma-data/data-use-requirements).
  
### Using IMAGE data

- Include the acknowledgement required by the
  [IMAGE rules of the road](https://space.fmi.fi/image/www/index.php?page=rules_of_road).

### Using INTERMAGNET data

- Include the acknowledgement required by the
  [INTERMAGNET conditions of use](https://www.intermagnet.org/data-donnee/data-eng.php#conditions).

### Using THEMIS data

- Include the acknowledgement required by the
  [THEMIS rules of the road](https://themis.igpp.ucla.edu/roadrules.shtml).
 
===========================References============================

gmag Python package
Murphy, K. R., Rae, I. J., Halford, A. J., Engebretson, M., Russell, C. T., Matzka, J., ... & Tanskanen, E. (2022). GMAG: An open-source python package for ground-based magnetometers. Frontiers in Astronomy and Space Sciences, 9, 1005061.

SuperMAG
Gjerloev, J. W. (2012), The SuperMAG data processing technique, J. Geophys. Res., 117 , A09213, doi:10.1029/2012JA017683.

Collaborator EMMA
Lichtenberger J., M. Clilverd, B. Heilig, M. Vellante, J. Manninen, C. Rodger, A. Collier, A. Jørgensen, J. Reda, R. Holzworth, and R. Friedel (2013), The plasmasphere during a space weather event: first results from the PLASMON project, J. Space Weather Space Clim., 3, A23 (www.swsc-journal.org/articles/swsc/pdf/2013/01/swsc120062.pdf).

Collaborator IMAGE Chain
Tanskanen, E.I. (2009), A comprehensive high-throughput analysis of substorms observed by IMAGE magnetometer network: Years 1993-2003 examined, 114, A05204, doi:10.1029/2008JA013682.

Collaborator MACCS
Engebretson, M. J., W. J. Hughes, J. L. Alford, E. Zesta, L. J. Cahill, Jr., R. L. Arnoldy, and G. D. Reeves (1995), Magnetometer array for cusp and cleft studies observations of the spatial extent of broadband ULF magnetic pulsations at cusp/cleft latitudes , J. Geophys. Res., 100, 19371-19386, doi:10.1029/95JA00768.

Collaborator McMAC Chain
Chi, P. J., M. J. Engebretson, M. B. Moldwin, C. T. Russell, I. R. Mann, M. R. Hairston, M. Reno, J. Goldstein, L. I. Winkler, J. L. Cruz-Abeyro, D.-H. Lee, K.Yumoto, R. Dalrymple, B. Chen, and J. P. Gibson (2013), Sounding of the plasmasphere by Mid-continent MAgnetoseismic Chain magnetometers, J. Geophys. Res. Space Physics, 118, doi:10.1002/jgra.50274.

Collaborator MAGDAS / 210 Chain
Yumoto, K,. and the CPMN Group (2001), Characteristics of Pi 2 magnetic pulsations observed at the CPMN stations: A review of the STEP results, Earth Planets Space, 53, 981-992.

Collaborator CARISMA
Mann, I. R., et al. (2008), The upgraded CARISMA magnetometer array in the THEMIS era, Space Sci. Rev., 141, 413-451, doi:10.1007/s11214-008-9457-6.

Collaborator AALPIP
Clauer, C. R., et al. (2014), An autonomous adaptive low-power instrument platform (AAL-PIP) for remote high-latitude geospace data collection, Geosci. Instrum. Methods Data Syst., 3, 211-227, doi:10.5194/gi-3-211-2014

Collaborator INTERMAGNET
Love, J. J., Chulliat, A., (2013), An international network of magnetic observatories, Eos, 94(42), 373-374, doi:10.1002/2013EO420001.

===================Indices SML, SMU, SME===================

Newell, P. T., and J. W. Gjerloev (2011), Evaluation of SuperMAG auroral electrojet indices as indicators of substorms and auroral power, J. Geophys. Res., 116, A12211, doi:10.1029/2011JA016779.

==============Indices SMLs, SMLd, SMUs, SMUd===============

Gjerloev, J. W., R. A. Hoffman, S. Ohtani, J. Weygand, and R. Barnes, Response of the Auroral Electrojet Indices to Abrupt Southward IMF Turnings (2010), Annales Geophysicae, 28, 1167-1182.
Indices SME-LT, SMU-LT, SML-LT

Newell, P. T., and J. W. Gjerloev (2014), Local geomagnetic indices and the prediction of auroral power, J. Geophys. Res. Space Physics, 119, doi:10.1002/2014JA020524.
Indices SMR, SMR-LT

Newell, P. T. and J. W. Gjerloev (2012), SuperMAG-Based Partial Ring Current Indices, J. Geophys. Res., 117, doi:10.1029/2012JA017586.

======================Substorm Lists=======================

Forsyth, C., Rae, I. J., Coxon, J. C., Freeman, M. P., Jackman, C. M., Gjerloev, J., and Fazakerley, A. N. ( 2015), A new technique for determining Substorm Onsets and Phases from Indices of the Electrojet (SOPHIE), J. Geophys. Res. Space Physics, 120, 10,592-10,606, doi:doi:10.1002/2015JA021343.

Frey, H. U., Mende, S. B., Angelopoulos, V., and Donovan, E. F. (2004), Substorm onset observations by IMAGE-FUV, J. Geophys. Res., 109, A10304, doi:10.1029/2004JA010607.

Gjerloev, J. W. (2012), The SuperMAG data processing technique, J. Geophys. Res., 117, A09213,  doi:doi:10.1029/2012JA017683.

Liou, K. (2010),  Polar Ultraviolet Imager observation of auroral breakup, J. Geophys. Res.,  115, A12219, doi:doi:10.1029/2010JA015578.

Newell, P. T., and J. W. Gjerloev (2011), Evaluation of SuperMAG auroral electrojet indices as indicators of substorms and auroral power, J. Geophys. Res., 116, A12211, doi:10.1029/2011JA016779.

Newell, P. T., and J. W. Gjerloev (2011), Substorm and magnetosphere characteristic scales inferred from the SuperMAG auroral electrojet indices, J. Geophys. Res., 116, A12232, doi:10.1029/2011JA016936.

Ohtani, S., and J. Gjerloev, Is the Substorm Current Wedge an Ensemble of Wedgelets?: Revisit to Midlatitude Positive Bays, accepted, J. Geophys. Res, 2020.
"""

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import mplcursors

    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False

try:
    from scipy.interpolate import griddata

    HAS_SCIPY = True
except ImportError:
    griddata = None
    HAS_SCIPY = False


BASE_URL = "https://supermag.jhuapl.edu/services/"
TIME_UNITS = "seconds since 1970-01-01 00:00:00 UTC"
TIME_CALENDAR = "standard"
COMPONENT_LABELS = ("N", "E", "Z")
NEZ_COMPONENT_PLOT_LABELS = ("dBn (North)", "dBe (East)", "dBz (Vertical)")
GEO_COMPONENT_PLOT_LABELS = (
    "dBx (geographic north)",
    "dBy (geographic east)",
    "dBz (vertical)",
)


MANUAL_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Named substorm onset lists distributed by the SuperMAG products service.
# These identifiers match the list= values used by the website download
# backend at https://supermag.jhuapl.edu/lib/services/?service=substorms.
SUBSTORM_SERVICE_URL = "https://supermag.jhuapl.edu/lib/services/"
SUBSTORM_TECHNIQUES = (
    ("SuperMAG (Newell & Gjerloev, 2011)", "newell"),
    ("SOPHIE (Forsyth et al., 2015)", "forsyth"),
    ("IMAGE-FUV (Frey et al., 2004/2006)", "frey"),
    ("Polar UVI (Liou, 2010)", "liou"),
    ("Midlatitude positive bays (Ohtani & Gjerloev, 2020)", "ohtani"),
)

# Prefixes used for source-aware onset labels in tables and plots.
SUBSTORM_SOURCE_PREFIXES = {
    "newell": "N",
    "forsyth": "F",
    "frey": "I",
    "liou": "L",
    "ohtani": "O",
}

MANUAL_POINT_COLOR_PALETTE = (
    "#7f3c8d",
    "#11a579",
    "#3969ac",
    "#f2b701",
    "#e73f74",
    "#80ba5a",
    "#e68310",
    "#008695",
    "#cf1c90",
    "#f97b72",
    "#4b4b8f",
    "#a5aa99",
)
SUPER_MAG_INVENTORY_ATTEMPTS = 4
SUPER_MAG_STATION_ATTEMPTS = 5
SUPER_MAG_RETRY_INITIAL_SECONDS = 2.0
SUPER_MAG_RETRY_MAX_SECONDS = 12.0


class DownloadCancelled(Exception):
    """Raised when the user requests cancellation of an active download."""


class SuperMAGServiceUnavailable(RuntimeError):
    """Raised for a transient SuperMAG PHP/logon backend response."""


class UnexpectedCadenceError(RuntimeError):
    """Raised when a high-resolution request returns minute-cadence data."""


@dataclass(frozen=True)
class StationMetadata:
    code: str
    glat: float
    glon: float
    name: str = ""


@dataclass(frozen=True)
class SubstormEvent:
    """One record from a SuperMAG substorm onset list."""

    onset: datetime
    glat: float = np.nan
    glon: float = np.nan
    mlat: float = np.nan
    mlt: float = np.nan
    source: str = ""

    @property
    def has_geographic_location(self) -> bool:
        return bool(np.isfinite(self.glat) and np.isfinite(self.glon))


@dataclass
class LoadedMagneticData:
    station_codes: np.ndarray
    times: np.ndarray
    nez: np.ndarray
    glat: np.ndarray
    glon: np.ndarray
    geo: Optional[np.ndarray] = None
    dbh: Optional[np.ndarray] = None
    source_path: Optional[str] = None
    cadence_mode: str = ""
    cadence_seconds: float = np.nan

    def validate(self) -> None:
        nstation = len(self.station_codes)
        ntime = len(self.times)
        if self.nez.shape != (nstation, ntime, 3):
            raise ValueError(
                "NEZ data must have shape "
                f"(station, time, 3); received {self.nez.shape}."
            )
        if self.geo is not None and self.geo.shape != (nstation, ntime, 3):
            raise ValueError(
                "GEO data must have shape "
                f"(station, time, 3); received {self.geo.shape}."
            )
        if self.dbh is not None and self.dbh.shape != (nstation, ntime):
            raise ValueError(
                "dBh data must have shape "
                f"(station, time); received {self.dbh.shape}."
            )
        if len(self.glat) != nstation or len(self.glon) != nstation:
            raise ValueError("Station coordinate arrays do not match station count.")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def normalize_longitude(lon: float | np.ndarray) -> float | np.ndarray:
    """Return longitude(s) in the [-180, 180) convention."""
    arr = np.asarray(lon, dtype=float)
    wrapped = ((arr + 180.0) % 360.0) - 180.0
    if np.ndim(lon) == 0:
        return float(wrapped)
    return wrapped


def mask_bad_magnetometer_values(values: np.ndarray, abs_limit: float = 50000.0) -> np.ndarray:
    """Replace known SuperMAG fill/sentinel values and implausible values by NaN."""
    out = np.asarray(values, dtype=float).copy()
    for bad in (99999.0, 1000000.0):
        out[np.isclose(out, bad, rtol=0.0, atol=1.0e-6)] = np.nan
        out[np.isclose(out, -bad, rtol=0.0, atol=1.0e-6)] = np.nan
    out[np.abs(out) > abs_limit] = np.nan
    return out


def datetime_to_unix_seconds(value: datetime) -> float:
    """Convert a naive UTC or timezone-aware datetime to Unix seconds."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def unix_seconds_to_datetime(value: float) -> datetime:
    """Convert Unix seconds to a naive UTC Python datetime."""
    return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)


def netcdf_time_to_datetimes(time_var) -> np.ndarray:
    """Read a NetCDF time variable as naive UTC Python datetime objects."""
    if hasattr(time_var, "units"):
        calendar = getattr(time_var, "calendar", TIME_CALENDAR)
        converted = num2date(
            time_var[:],
            units=time_var.units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        out = []
        for value in converted:
            if isinstance(value, datetime):
                if value.tzinfo is not None:
                    value = value.astimezone(timezone.utc).replace(tzinfo=None)
                out.append(value)
            else:
                out.append(
                    datetime(
                        value.year,
                        value.month,
                        value.day,
                        value.hour,
                        value.minute,
                        value.second,
                        getattr(value, "microsecond", 0),
                    )
                )
        return np.asarray(out, dtype=object)

    return np.asarray([unix_seconds_to_datetime(v) for v in time_var[:]], dtype=object)


def decode_station_names(values: np.ndarray) -> np.ndarray:
    """Decode NetCDF station names to plain strings."""
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="replace").strip())
        else:
            decoded.append(str(value).strip())
    return np.asarray(decoded, dtype=str)


def finite_time_derivative(
    values: np.ndarray, times: np.ndarray, seconds_per_unit: float = 60.0
) -> np.ndarray:
    """
    Calculate dB/dt while preserving missing-data gaps.

    ``seconds_per_unit`` is 60 for nT/min and 1 for nT/sec.

    Gradients are evaluated independently on contiguous finite runs. This avoids
    interpolating across a station outage and avoids contaminating an entire
    trace because one fill value is present.
    """
    y = mask_bad_magnetometer_values(values)
    result = np.full(y.shape, np.nan, dtype=float)
    if len(y) < 2:
        return result

    seconds = np.asarray([datetime_to_unix_seconds(t) for t in times], dtype=float)
    finite = np.isfinite(y) & np.isfinite(seconds)
    indices = np.flatnonzero(finite)
    if len(indices) < 2:
        return result

    split_points = np.where(np.diff(indices) > 1)[0] + 1
    runs = np.split(indices, split_points)
    for run in runs:
        if len(run) < 2:
            continue
        run_times = seconds[run]
        if np.any(np.diff(run_times) <= 0):
            continue
        result[run] = np.gradient(y[run], run_times) * float(seconds_per_unit)
    return result


def derivative_display(data: LoadedMagneticData) -> tuple[float, str]:
    """Return derivative scaling and unit for the loaded product cadence."""
    mode = str(getattr(data, "cadence_mode", "") or "").strip().lower()
    cadence = float(getattr(data, "cadence_seconds", np.nan))
    if not np.isfinite(cadence) or cadence <= 0.0:
        seconds = np.asarray(
            [datetime_to_unix_seconds(value) for value in data.times], dtype=float
        )
        steps = np.diff(seconds)
        steps = steps[np.isfinite(steps) & (steps > 0.0)]
        cadence = float(np.median(steps)) if len(steps) else np.nan

    if mode in {"1s", "gmag-1s", "2hz", "gmag-2hz"}:
        return 1.0, "nT/sec"
    if mode in {"10s", "gmag-10s", "original", "native", "gmag-native"}:
        return 60.0, "nT/min"
    # Custom SuperMAG files are classified from their measured cadence.
    if mode == "custom" and np.isfinite(cadence) and cadence <= 1.1:
        return 1.0, "nT/sec"
    if np.isfinite(cadence) and cadence <= 1.1:
        return 1.0, "nT/sec"
    return 60.0, "nT/min"


def calculate_dbh(components: np.ndarray) -> np.ndarray:
    """Calculate dBh = sqrt(dBn^2 + dBe^2) in nT.

    The input must have shape ``(..., 3)``. Known SuperMAG fill values are
    masked before the horizontal magnitude is calculated from the northward
    and eastward components only.
    """
    values = np.asarray(components, dtype=float)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError(
            "Magnetic components must have a final dimension of length 3; "
            f"received {values.shape}."
        )
    clean = mask_bad_magnetometer_values(values)
    dbn = clean[..., 0]
    dbe = clean[..., 1]
    return np.sqrt(dbn * dbn + dbe * dbe)


def calculate_dbh_dt(dbh: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculate dBh/dt in nT/min for one or more station time series."""
    values = np.asarray(dbh, dtype=float)
    if values.ndim == 1:
        return finite_time_derivative(values, times)
    if values.ndim != 2:
        raise ValueError(
            "dBh must have shape (time,) or (station, time); "
            f"received {values.shape}."
        )
    return np.asarray(
        [finite_time_derivative(station_values, times) for station_values in values],
        dtype=float,
    )


def clean_magnetic_vectors(
    values: np.ndarray,
    cadence_seconds: float,
    max_gap_seconds: float = 5.0,
    window_samples: int = 31,
    mad_threshold: float = 8.0,
    minimum_spike_nt: float = 5.0,
) -> np.ndarray:
    """Mask bad values, despike, fill short interior gaps, and despike again."""
    output = mask_bad_magnetometer_values(values)
    if output.ndim != 2 or output.shape[1] != 3:
        raise ValueError("Magnetic vectors must have shape (time, 3).")
    cadence = float(cadence_seconds)
    if not np.isfinite(cadence) or cadence <= 0:
        cadence = 1.0
    half_window = max(2, int(window_samples) // 2)

    def despike(array: np.ndarray) -> np.ndarray:
        cleaned = array.copy()
        candidate = np.zeros(len(cleaned), dtype=bool)
        for component in range(3):
            series = pd.Series(cleaned[:, component])
            median = series.rolling(2 * half_window + 1, center=True, min_periods=5).median()
            deviation = (series - median).abs()
            mad = deviation.rolling(
                2 * half_window + 1, center=True, min_periods=5
            ).median()
            scale = 1.4826 * mad
            candidate |= (
                (deviation.to_numpy() >= minimum_spike_nt)
                & (deviation.to_numpy() > mad_threshold * scale.to_numpy())
                & np.isfinite(median.to_numpy())
            )
        cleaned[candidate, :] = np.nan
        return cleaned

    output = despike(output)
    limit = max(0, int(np.floor(max_gap_seconds / cadence + 1.0e-9)))
    x = np.arange(len(output), dtype=float)
    for component in range(3):
        y = output[:, component]
        missing = ~np.isfinite(y)
        starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
        ends = np.flatnonzero(missing & np.r_[~missing[1:], True])
        for start, end in zip(starts, ends):
            if end - start + 1 > limit or start == 0 or end == len(y) - 1:
                continue
            if np.isfinite(y[start - 1]) and np.isfinite(y[end + 1]):
                y[start : end + 1] = np.interp(
                    x[start : end + 1],
                    [x[start - 1], x[end + 1]],
                    [y[start - 1], y[end + 1]],
                )
    return despike(output)


def _coerce_station_time_array(
    values: np.ndarray,
    nstation: int,
    ntime: int,
    variable_name: str,
) -> np.ndarray:
    """Return a NetCDF variable as a ``(station, time)`` floating array."""
    array = np.asarray(values, dtype=float)
    if array.shape == (nstation, ntime):
        return array
    if array.shape == (ntime, nstation):
        return array.T

    squeezed = np.squeeze(array)
    if squeezed.shape == (nstation, ntime):
        return squeezed
    if squeezed.shape == (ntime, nstation):
        return squeezed.T
    if squeezed.ndim == 1 and nstation == 1 and squeezed.size == ntime:
        return squeezed.reshape(1, ntime)
    if squeezed.ndim == 1 and ntime == 1 and squeezed.size == nstation:
        return squeezed.reshape(nstation, 1)
    raise ValueError(
        f"NetCDF variable {variable_name!r} must have shape "
        f"({nstation}, {ntime}) or ({ntime}, {nstation}); received {array.shape}."
    )


def choose_longitude_window(longitudes: np.ndarray) -> tuple[float, np.ndarray, float]:
    """
    Return a central longitude, station longitudes relative to it, and span.

    The largest empty arc on the globe is excluded. This produces a compact map
    for networks that cross the dateline or combine 0..360 and -180..180 input.
    """
    lons = normalize_longitude(np.asarray(longitudes, dtype=float))
    finite = np.isfinite(lons)
    if not np.any(finite):
        return 0.0, lons, 360.0

    valid = np.mod(lons[finite], 360.0)
    if len(valid) == 1:
        center = normalize_longitude(valid[0])
        relative = normalize_longitude(lons - center)
        return float(center), relative, 0.0

    sorted_lons = np.sort(valid)
    gaps = np.diff(np.r_[sorted_lons, sorted_lons[0] + 360.0])
    gap_index = int(np.argmax(gaps))
    start = sorted_lons[(gap_index + 1) % len(sorted_lons)]
    end = sorted_lons[gap_index]
    if end < start:
        end += 360.0
    span = float(end - start)
    center_360 = (start + end) / 2.0
    center = float(normalize_longitude(center_360))
    relative = normalize_longitude(lons - center)
    return center, relative, span


def geographic_filter(
    station_codes: Iterable[str],
    metadata: dict[str, StationMetadata],
    lat_min: float,
    lat_max: float,
    lon_west: float,
    lon_east: float,
) -> list[str]:
    """Filter inventory stations by latitude and longitude, including wrap boxes."""
    lat_lo, lat_hi = sorted((lat_min, lat_max))
    lon_west = float(normalize_longitude(lon_west))
    lon_east = float(normalize_longitude(lon_east))

    selected: list[str] = []
    for code in station_codes:
        meta = metadata.get(code.upper())
        if meta is None or not np.isfinite(meta.glat) or not np.isfinite(meta.glon):
            continue
        if not (lat_lo <= meta.glat <= lat_hi):
            continue

        lon = float(normalize_longitude(meta.glon))
        if lon_west <= lon_east:
            in_lon = lon_west <= lon <= lon_east
        else:
            # Geographic box crosses ±180 degrees.
            in_lon = lon >= lon_west or lon <= lon_east
        if in_lon:
            selected.append(code.upper())

    return sorted(set(selected))


# -----------------------------------------------------------------------------
# SuperMAG API and NetCDF functions
# -----------------------------------------------------------------------------

def _open_supermag_url(url: str, timeout: float) -> bytes:
    """Open one SuperMAG URL using the same simple request style as v5."""
    # The v5 downloader is known to work intermittently with this endpoint.
    # Keep the request as close to that version as possible instead of adding
    # custom headers that might change how the server handles the call.
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def _is_supermag_backend_failure(text: str) -> bool:
    """Return True for the intermittent SuperMAG PHP/logon-validation failure."""
    lower = text.lower()
    return (
        ("shell_exec()" in lower and "logon" in lower)
        or "simplexmlelement::__construct" in lower
        or ("fatal error" in lower and "data-api.php" in lower)
        or ("fatal error" in lower and "inventory.php" in lower)
    )


def _supermag_backend_message() -> str:
    """Concise description of the transient SuperMAG backend response."""
    return (
        "SuperMAG returned an intermittent PHP/logon backend error. "
        "The request will be retried automatically before the station is "
        "reported as failed."
    )


def _retry_delay(attempt_number: int) -> float:
    """Return a bounded exponential retry delay in seconds."""
    return min(
        SUPER_MAG_RETRY_INITIAL_SECONDS * (2 ** max(attempt_number - 1, 0)),
        SUPER_MAG_RETRY_MAX_SECONDS,
    )


def _wait_for_retry(
    delay_seconds: float,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Wait before retrying, while honouring cooperative cancellation."""
    if stop_event is not None:
        if stop_event.wait(delay_seconds):
            raise DownloadCancelled("Download stopped by the user.")
    else:
        time.sleep(delay_seconds)


def summarize_supermag_api_error(exc: Exception, max_length: int = 1200) -> str:
    """Convert verbose network/API failures into a readable GUI message."""
    raw = str(exc).strip()
    lower = raw.lower()

    if isinstance(exc, UnexpectedCadenceError):
        return raw
    if isinstance(exc, SuperMAGServiceUnavailable) or _is_supermag_backend_failure(raw):
        return (
            "SuperMAG repeatedly returned its PHP/logon backend error after "
            "automatic retries. Try the download again; the service often "
            "responds successfully on a later attempt."
        )
    if "timed out" in lower:
        return "The SuperMAG request timed out after automatic retries."
    if "temporary failure in name resolution" in lower:
        return "The SuperMAG server could not be reached because name resolution failed."

    first_line = raw.splitlines()[0] if raw else repr(exc)
    if len(first_line) > max_length:
        first_line = first_line[:max_length] + "…"
    return f"{type(exc).__name__}: {first_line}"


def _fetch_inventory_once(start: datetime, end: datetime, logon: str) -> list[str]:
    """Perform one SuperMAG inventory request."""
    extent = int((end - start).total_seconds())
    if extent <= 0:
        raise ValueError("The download end time must be later than the start time.")

    query = urllib.parse.urlencode(
        {
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "extent": f"{extent:012d}",
            "logon": logon,
        }
    )
    url = f"{BASE_URL}inventory.php?{query}"
    raw = _open_supermag_url(url, timeout=90)
    text = raw.decode("utf-8", errors="replace").strip()

    if _is_supermag_backend_failure(text):
        raise SuperMAGServiceUnavailable(_supermag_backend_message())
    if not text or "<html" in text.lower() or "fatal error" in text.lower():
        raise ValueError(f"Unexpected SuperMAG inventory response:\n{text[:1200]}")

    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError(f"Unexpected SuperMAG inventory response:\n{text}")
    if lines[0].strip() != "OK":
        raise ValueError(f"SuperMAG inventory request failed:\n{text}")

    try:
        expected = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"Unexpected SuperMAG inventory response:\n{text}") from exc

    stations = [line.strip().upper() for line in lines[2:] if line.strip()]
    if len(stations) != expected:
        print(f"Warning: inventory reported {expected} stations; parsed {len(stations)}.")
    return stations


def fetch_inventory(
    start: datetime,
    end: datetime,
    logon: str,
    attempts: int = SUPER_MAG_INVENTORY_ATTEMPTS,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> list[str]:
    """Fetch inventory with retries for intermittent SuperMAG failures."""
    last_error: Optional[Exception] = None
    attempts = max(int(attempts), 1)

    for attempt in range(1, attempts + 1):
        if stop_event is not None and stop_event.is_set():
            raise DownloadCancelled("Download stopped by the user.")
        if progress_callback is not None:
            progress_callback(attempt, attempts, "Fetching station inventory")
        try:
            return _fetch_inventory_once(start, end, logon)
        except DownloadCancelled:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            _wait_for_retry(_retry_delay(attempt), stop_event)

    assert last_error is not None
    raise last_error

def load_station_metadata(path: str) -> dict[str, StationMetadata]:
    """
    Read the SuperMAG station-information text file.

    The original script expects a tab-delimited table headed by
    ``IAGA GEOLON GEOLAT``. This parser also accepts general whitespace.
    """
    metadata: dict[str, StationMetadata] = {}
    in_table = False

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("IAGA") and "GEOLON" in upper and "GEOLAT" in upper:
                in_table = True
                continue
            if not in_table or line.startswith("="):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue

            code = parts[0].strip().upper()
            try:
                glon = float(parts[1])
                glat = float(parts[2])
            except ValueError:
                continue

            name = parts[5].strip('"') if len(parts) > 5 else code
            metadata[code] = StationMetadata(
                code=code,
                glat=glat,
                glon=float(normalize_longitude(glon)),
                name=name,
            )

    if not metadata:
        raise ValueError(
            "No station records were found. Select the SuperMAG station metadata "
            "text file containing the IAGA, GEOLON, and GEOLAT table."
        )
    return metadata


def _api_component_value(record: dict, component: str, coordinate: str) -> float:
    """Extract one API component safely, returning NaN if absent."""
    try:
        return float(record[component][coordinate])
    except (KeyError, TypeError, ValueError):
        return np.nan


def _decode_supermag_json(raw: bytes, station: str) -> list[dict]:
    """Decode one station response and identify retryable server failures."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"No response was returned for station {station}.")
    if _is_supermag_backend_failure(text):
        raise SuperMAGServiceUnavailable(_supermag_backend_message())
    if "<html" in text.lower() or "fatal error" in text.lower():
        preview = text[:1200]
        raise ValueError(f"SuperMAG server error for station {station}:\n{preview}")
    try:
        records = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:1200]
        raise ValueError(
            f"SuperMAG returned a non-JSON response for station {station}:\n{preview}"
        ) from exc
    if isinstance(records, dict):
        error_text = records.get("error") or records.get("message")
        if error_text:
            raise ValueError(f"SuperMAG request failed for {station}: {error_text}")
        records = records.get("data", [])
    if not isinstance(records, list):
        raise ValueError(f"Unexpected SuperMAG data response for station {station}.")
    return records


def fetch_magnetometer_station(
    start: datetime,
    end: datetime,
    logon: str,
    station: str,
    high_resolution: bool = False,
) -> dict[str, np.ndarray]:
    """Perform one NEZ/GEO station request using the v5 URL ordering."""
    extent = int((end - start).total_seconds())
    if extent <= 0:
        raise ValueError("The download end time must be later than the start time.")

    mode_flag = "highres" if high_resolution else "all"

    # Keep the exact query layout used by the working v5 downloader: station is
    # encoded with the standard fields and the flag-only resolution parameter
    # is appended last. SuperMAG's service is sensitive enough that preserving
    # this known-working form is preferable to rearranging the parameters.
    query = urllib.parse.urlencode(
        {
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "extent": f"{extent:012d}",
            "logon": logon,
            "station": station.upper(),
        }
    )
    url = f"{BASE_URL}data-api.php?fmt=json&python&nohead&{query}&{mode_flag}"

    raw = _open_supermag_url(url, timeout=180 if high_resolution else 120)
    records = _decode_supermag_json(raw, station)
    if not records:
        resolution_text = "high-resolution " if high_resolution else ""
        raise ValueError(f"No {resolution_text}data returned for station {station}.")

    npoints = len(records)
    tval = np.full(npoints, np.nan, dtype=float)
    nez = np.full((npoints, 3), np.nan, dtype=float)
    geo = np.full((npoints, 3), np.nan, dtype=float)

    for i, record in enumerate(records):
        try:
            tval[i] = float(record.get("tval", np.nan))
        except (AttributeError, TypeError, ValueError):
            tval[i] = np.nan
        for j, component in enumerate(COMPONENT_LABELS):
            nez[i, j] = _api_component_value(record, component, "nez")
            geo[i, j] = _api_component_value(record, component, "geo")

    finite_time = np.isfinite(tval)
    if not np.any(finite_time):
        raise ValueError(f"Station {station} returned no valid timestamps.")

    tval = tval[finite_time]
    nez = nez[finite_time]
    geo = geo[finite_time]
    order = np.argsort(tval)
    tval = tval[order]
    nez = nez[order]
    geo = geo[order]

    unique_times = np.unique(tval)
    median_cadence = np.nan
    if len(unique_times) >= 2:
        positive_steps = np.diff(unique_times)
        positive_steps = positive_steps[positive_steps > 0]
        if len(positive_steps):
            median_cadence = float(np.median(positive_steps))

    # Standard requests must accept the normal 60-second product. For a
    # high-resolution request, a minute-cadence response is treated as a
    # retryable downgrade rather than immediately terminating the batch.
    if high_resolution and np.isfinite(median_cadence) and median_cadence >= 30.0:
        raise UnexpectedCadenceError(
            f"High-resolution data were requested for {station}, but this attempt "
            f"returned a median cadence of {median_cadence:g} s."
        )

    return {
        "tval": tval,
        "nez": nez,
        "geo": geo,
        "median_cadence_seconds": median_cadence,
    }


def _coerce_optional_float(value: object) -> float:
    """Convert a possibly empty CSV field to float, otherwise return NaN."""
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"nan", "na", "n/a", "none", "null", "-"}:
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _coerce_onset_datetime(value: object) -> Optional[datetime]:
    """Convert an epoch value or date string to a naive UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, (int, float)):
        try:
            return unix_seconds_to_datetime(float(value))
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None

    value_text = value.strip()
    if not value_text:
        return None
    try:
        numeric_value = float(value_text)
    except ValueError:
        numeric_value = np.nan
    if np.isfinite(numeric_value):
        try:
            return unix_seconds_to_datetime(numeric_value)
        except (OverflowError, OSError, ValueError):
            pass

    iso_text = value_text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ):
        try:
            return datetime.strptime(value_text, fmt)
        except ValueError:
            continue
    return None


def _substorm_event_key(event: SubstormEvent) -> tuple[object, ...]:
    """Stable key used to de-duplicate records and preserve GUI selections."""
    return (
        event.onset,
        round(float(event.glat), 6) if np.isfinite(event.glat) else None,
        round(float(event.glon), 6) if np.isfinite(event.glon) else None,
        event.source.strip().lower(),
    )


def _parse_substorm_csv(text: str, source: str = "") -> list[SubstormEvent]:
    """Parse CSV returned by the SuperMAG substorm-list download service."""
    cleaned = text.lstrip("\ufeff").strip()
    if not cleaned:
        return []
    if _is_supermag_backend_failure(cleaned):
        raise SuperMAGServiceUnavailable(_supermag_backend_message())
    if "<html" in cleaned.lower() or "<!doctype" in cleaned.lower():
        raise ValueError(f"Unexpected HTML response from SuperMAG:\n{cleaned[:1200]}")

    reader = csv.DictReader(io.StringIO(cleaned))
    if not reader.fieldnames:
        raise ValueError(f"Unexpected SuperMAG substorm-list response:\n{cleaned[:1200]}")

    normalized_fields = {str(name).strip().lower() for name in reader.fieldnames if name}
    if not normalized_fields.intersection({"date_utc", "datetime", "time", "onset"}):
        raise ValueError(f"Unexpected SuperMAG substorm-list CSV header:\n{cleaned[:1200]}")

    events: list[SubstormEvent] = []
    for row in reader:
        normalized = {
            str(key).strip().lower(): value
            for key, value in row.items()
            if key is not None
        }
        onset_value = next(
            (
                normalized.get(key)
                for key in ("date_utc", "onset", "datetime", "date_time", "time")
                if normalized.get(key) not in (None, "")
            ),
            None,
        )
        onset = _coerce_onset_datetime(onset_value)
        if onset is None:
            continue
        event_source = str(normalized.get("source") or source).strip()
        glon = _coerce_optional_float(normalized.get("glon"))
        if np.isfinite(glon):
            glon = float(normalize_longitude(glon))
        events.append(
            SubstormEvent(
                onset=onset,
                glat=_coerce_optional_float(normalized.get("glat")),
                glon=glon,
                mlat=_coerce_optional_float(normalized.get("mlat")),
                mlt=_coerce_optional_float(normalized.get("mlt")),
                source=event_source,
            )
        )

    unique: dict[tuple[object, ...], SubstormEvent] = {}
    for event in events:
        unique[_substorm_event_key(event)] = event
    return sorted(unique.values(), key=lambda event: (event.onset, event.source))


def fetch_substorm_events(
    start: datetime,
    end: datetime,
    logon: str,
    technique_code: str,
    attempts: int = SUPER_MAG_INVENTORY_ATTEMPTS,
) -> list[SubstormEvent]:
    """Download one SuperMAG substorm list for an exact UTC interval.

    This uses the same CSV service as the SuperMAG products download page. The
    returned records retain MLT/MLat and GLon/GLat when the selected list
    provides them.
    """
    if end <= start:
        raise ValueError("The end time must be later than the start time.")
    valid_codes = {code for _name, code in SUBSTORM_TECHNIQUES}
    if technique_code not in valid_codes:
        raise ValueError(
            f"Unknown substorm list {technique_code!r}; choose one of {sorted(valid_codes)}."
        )
    if not logon.strip():
        raise ValueError("Enter a SuperMAG logon id first.")

    query = urllib.parse.urlencode(
        {
            "service": "substorms",
            "downloadtype": "substorm_list",
            "user": logon.strip(),
            "fmt": "csv",
            "start": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "list": technique_code,
        }
    )
    url = f"{SUBSTORM_SERVICE_URL}?{query}"
    attempts = max(int(attempts), 1)
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            raw = _open_supermag_url(url, timeout=120)
            response_text = raw.decode("utf-8", errors="replace")
            return _parse_substorm_csv(response_text, source=technique_code)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            _wait_for_retry(_retry_delay(attempt))
    assert last_error is not None
    raise last_error


def fetch_substorm_onsets(
    start: datetime,
    end: datetime,
    logon: str,
    technique_code: str,
) -> list[datetime]:
    """Backward-compatible time-only wrapper around fetch_substorm_events."""
    return [
        event.onset
        for event in fetch_substorm_events(start, end, logon, technique_code)
    ]


def load_substorm_events_from_file(path: str) -> list[SubstormEvent]:
    """Read a SuperMAG-downloaded CSV or a simple time-per-line onset file."""
    file_text = Path(path).read_text(encoding="utf-8", errors="replace")
    source = Path(path).stem
    try:
        return _parse_substorm_csv(file_text, source=source)
    except ValueError:
        pass

    events: list[SubstormEvent] = []
    for raw_line in file_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidates = [line]
        fields = next(csv.reader([line]))
        if fields:
            candidates.append(fields[0].strip())
        if len(fields) >= 2:
            candidates.append(f"{fields[0].strip()} {fields[1].strip()}")
        onset = next(
            (parsed for parsed in (_coerce_onset_datetime(item) for item in candidates) if parsed),
            None,
        )
        if onset is not None:
            events.append(SubstormEvent(onset=onset, source=source))

    unique = {_substorm_event_key(event): event for event in events}
    return sorted(unique.values(), key=lambda event: event.onset)


def load_substorm_onsets_from_file(path: str) -> list[datetime]:
    """Backward-compatible time-only wrapper for local substorm-list files."""
    return [event.onset for event in load_substorm_events_from_file(path)]


def save_substorm_events_to_csv(events: Iterable[SubstormEvent], path: str) -> None:
    """Save displayed/selected substorm records in a reusable CSV format."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date_UTC", "MLT", "MLAT", "GLON", "GLAT", "Source"])
        for event in sorted(events, key=lambda item: (item.onset, item.source)):
            writer.writerow(
                [
                    event.onset.strftime(MANUAL_TIME_FORMAT),
                    "" if not np.isfinite(event.mlt) else f"{event.mlt:.6g}",
                    "" if not np.isfinite(event.mlat) else f"{event.mlat:.6g}",
                    "" if not np.isfinite(event.glon) else f"{normalize_longitude(event.glon):.6g}",
                    "" if not np.isfinite(event.glat) else f"{event.glat:.6g}",
                    event.source,
                ]
            )


def fetch_many_stations(
    start: datetime,
    end: datetime,
    logon: str,
    station_list: Iterable[str],
    high_resolution: bool = False,
    pause_seconds: float = 0.15,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    stop_event: Optional[threading.Event] = None,
    attempts_per_station: int = SUPER_MAG_STATION_ATTEMPTS,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    """Download stations sequentially with per-station automatic retries."""
    stations = list(station_list)
    all_data: dict[str, dict[str, np.ndarray]] = {}
    failures: dict[str, str] = {}
    attempts_per_station = max(int(attempts_per_station), 1)

    def cancellation_requested() -> bool:
        return stop_event is not None and stop_event.is_set()

    for number, station in enumerate(stations, start=1):
        if cancellation_requested():
            raise DownloadCancelled("Download stopped by the user.")

        last_error: Optional[Exception] = None
        for attempt in range(1, attempts_per_station + 1):
            if cancellation_requested():
                raise DownloadCancelled("Download stopped by the user.")

            if progress_callback is not None:
                suffix = "" if attempt == 1 else f"; retry {attempt}/{attempts_per_station}"
                progress_callback(
                    number - 1,
                    len(stations),
                    f"Downloading {station}{suffix}",
                )

            try:
                all_data[station] = fetch_magnetometer_station(
                    start,
                    end,
                    logon,
                    station,
                    high_resolution=high_resolution,
                )
                last_error = None
                break
            except DownloadCancelled:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= attempts_per_station:
                    break
                _wait_for_retry(_retry_delay(attempt), stop_event)

        if last_error is not None:
            failures[station] = (
                f"Failed after {attempts_per_station} attempts: "
                f"{summarize_supermag_api_error(last_error)}"
            )

        if cancellation_requested():
            raise DownloadCancelled("Download stopped by the user.")

        if pause_seconds > 0 and number < len(stations):
            _wait_for_retry(pause_seconds, stop_event)

    if progress_callback is not None:
        progress_callback(len(stations), len(stations), "Download complete")
    return all_data, failures

def save_to_netcdf(
    all_data: dict[str, dict[str, np.ndarray]],
    filename: str,
    station_metadata: dict[str, StationMetadata],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    geographic_bounds: Optional[tuple[float, float, float, float]] = None,
    requested_high_resolution: bool = False,
    max_gap_seconds: float = 5.0,
) -> None:
    """
    Save downloaded station vectors in one NetCDF file.

    Unlike the original routine, this version aligns stations to the union of
    returned timestamps. A station with a missing sample is filled with NaN
    instead of causing the entire save operation to fail because shapes differ.
    """
    if not all_data:
        raise ValueError("No successfully downloaded station data are available to save.")

    stations = sorted(all_data)
    all_times = []
    for station in stations:
        tvals = np.asarray(all_data[station]["tval"], dtype=float)
        all_times.append(np.round(tvals[np.isfinite(tvals)], 6))
    common_time = np.unique(np.concatenate(all_times))
    if len(common_time) == 0:
        raise ValueError("Downloaded station data contain no finite timestamps.")

    nstation = len(stations)
    ntime = len(common_time)
    time_lookup = {value: i for i, value in enumerate(common_time)}

    output_path = Path(filename).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(output_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", ntime)
        ds.createDimension("station", nstation)
        ds.createDimension("component", 3)

        time_var = ds.createVariable("time", "f8", ("time",))
        station_var = ds.createVariable("station", str, ("station",))
        component_var = ds.createVariable("component", "i4", ("component",))
        component_name_var = ds.createVariable("component_name", str, ("component",))
        glat_var = ds.createVariable("glat", "f4", ("station",), fill_value=np.nan)
        glon_var = ds.createVariable("glon", "f4", ("station",), fill_value=np.nan)
        nez_var = ds.createVariable(
            "nez",
            "f4",
            ("station", "time", "component"),
            zlib=True,
            complevel=4,
            fill_value=np.nan,
        )
        geo_var = ds.createVariable(
            "geo",
            "f4",
            ("station", "time", "component"),
            zlib=True,
            complevel=4,
            fill_value=np.nan,
        )
        dbh_var = ds.createVariable(
            "dBh",
            "f4",
            ("station", "time"),
            zlib=True,
            complevel=4,
            fill_value=np.nan,
        )

        time_var[:] = common_time
        time_var.units = TIME_UNITS
        time_var.calendar = TIME_CALENDAR
        time_var.standard_name = "time"
        time_var.axis = "T"

        station_var[:] = np.asarray(stations, dtype=object)
        component_var[:] = np.arange(3, dtype=np.int32)
        component_name_var[:] = np.asarray(COMPONENT_LABELS, dtype=object)
        component_var.description = "0=N, 1=E, 2=Z"
        glat_var.units = "degrees_north"
        glon_var.units = "degrees_east"
        nez_var.units = "nT"
        geo_var.units = "nT"
        nez_var.long_name = "SuperMAG local magnetic perturbation components"
        geo_var.long_name = "SuperMAG geographic magnetic perturbation components"
        dbh_var.units = "nT"
        dbh_var.long_name = "dBh = sqrt(dBn^2 + dBe^2)"

        glats = np.full(nstation, np.nan, dtype=np.float32)
        glons = np.full(nstation, np.nan, dtype=np.float32)
        nez_full = np.full((nstation, ntime, 3), np.nan, dtype=np.float32)
        geo_full = np.full((nstation, ntime, 3), np.nan, dtype=np.float32)

        for station_index, station in enumerate(stations):
            meta = station_metadata.get(station)
            if meta is not None:
                glats[station_index] = meta.glat
                glons[station_index] = normalize_longitude(meta.glon)

            station_times = np.round(
                np.asarray(all_data[station]["tval"], dtype=float), 6
            )
            station_nez = np.asarray(all_data[station]["nez"], dtype=float)
            station_geo = np.asarray(all_data[station]["geo"], dtype=float)
            finite_times = station_times[np.isfinite(station_times)]
            time_steps = np.diff(np.unique(finite_times))
            time_steps = time_steps[time_steps > 0]
            cadence = float(np.median(time_steps)) if len(time_steps) else 1.0
            station_nez = clean_magnetic_vectors(
                station_nez, cadence, max_gap_seconds=max_gap_seconds
            )
            station_geo = clean_magnetic_vectors(
                station_geo, cadence, max_gap_seconds=max_gap_seconds
            )
            for source_index, timestamp in enumerate(station_times):
                target_index = time_lookup.get(timestamp)
                if target_index is None:
                    continue
                nez_full[station_index, target_index, :] = station_nez[source_index]
                geo_full[station_index, target_index, :] = station_geo[source_index]

        glat_var[:] = glats
        glon_var[:] = glons
        nez_var[:] = nez_full
        geo_var[:] = geo_full
        dbh_var[:] = calculate_dbh(nez_full).astype(np.float32)

        ds.title = "SuperMAG magnetometer data"
        ds.source = "SuperMAG web services API"
        ds.requested_resolution = (
            "high resolution" if requested_high_resolution else "standard"
        )
        cadences = [
            float(item.get("median_cadence_seconds", np.nan))
            for item in all_data.values()
        ]
        finite_cadences = np.asarray(cadences, dtype=float)
        finite_cadences = finite_cadences[np.isfinite(finite_cadences)]
        if len(finite_cadences):
            ds.median_station_cadence_seconds = float(np.median(finite_cadences))
        ds.history = f"Created {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC"
        ds.cleaning = (
            "known fill values masked; rolling-MAD vector despike; interior gaps "
            "up to 5 seconds linearly filled; rolling-MAD vector despike repeated"
        )
        if start is not None:
            ds.download_start_utc = start.strftime("%Y-%m-%d %H:%M:%S")
        if end is not None:
            ds.download_end_utc = end.strftime("%Y-%m-%d %H:%M:%S")
        if geographic_bounds is not None:
            lat_min, lat_max, lon_west, lon_east = geographic_bounds
            ds.request_latitude_min = float(lat_min)
            ds.request_latitude_max = float(lat_max)
            ds.request_longitude_west = float(lon_west)
            ds.request_longitude_east = float(lon_east)


def _read_custom_supermag_blocks(path: str) -> tuple[
    dict[str, dict[str, np.ndarray]], dict[str, StationMetadata]
]:
    """Read a SuperMAG custom block/vector NetCDF product into station series."""
    required = {
        "id", "npnt", "time_yr", "time_mo", "time_dy", "time_hr", "time_mt",
        "time_sc", "glat", "glon", "dbn_nez", "dbe_nez", "dbz_nez",
    }
    with Dataset(path, "r") as ds:
        missing = required.difference(ds.variables)
        if missing:
            raise ValueError(f"Custom SuperMAG file is missing variables: {sorted(missing)}")
        ids = np.asarray(ds["id"][:], dtype=str)
        npnt = np.asarray(ds["npnt"][:], dtype=int)
        valid_vector = np.arange(ids.shape[1])[None, :] < npnt[:, None]
        ids = np.char.strip(ids)
        ids[~valid_vector] = ""
        frame = pd.DataFrame(
            {
                "year": ds["time_yr"][:], "month": ds["time_mo"][:],
                "day": ds["time_dy"][:], "hour": ds["time_hr"][:],
                "minute": ds["time_mt"][:],
            }
        )
        seconds = np.asarray(ds["time_sc"][:], dtype=float)
        whole_seconds = np.floor(seconds).astype(int)
        frame["second"] = whole_seconds
        timestamps = pd.to_datetime(frame, utc=True, errors="coerce")
        timestamps += pd.to_timedelta(seconds - whole_seconds, unit="s")
        unix = timestamps.astype("int64").to_numpy(dtype=float) / 1.0e9

        arrays = {
            name: mask_bad_magnetometer_values(ds[name][:])
            for name in (
                "glat", "glon", "dbn_nez", "dbe_nez", "dbz_nez",
                "dbn_geo", "dbe_geo", "dbz_geo",
            )
            if name in ds.variables
        }

    station_codes = sorted(code for code in np.unique(ids) if code)
    all_data: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, StationMetadata] = {}
    for code in station_codes:
        matches = ids == code
        rows = np.flatnonzero(matches.any(axis=1))
        columns = matches[rows].argmax(axis=1)
        finite_time = np.isfinite(unix[rows])
        rows, columns = rows[finite_time], columns[finite_time]
        nez = np.column_stack(
            [arrays[name][rows, columns] for name in ("dbn_nez", "dbe_nez", "dbz_nez")]
        )
        if all(name in arrays for name in ("dbn_geo", "dbe_geo", "dbz_geo")):
            geo = np.column_stack(
                [arrays[name][rows, columns] for name in ("dbn_geo", "dbe_geo", "dbz_geo")]
            )
        else:
            geo = nez.copy()
        order = np.argsort(unix[rows])
        station_time = unix[rows][order]
        steps = np.diff(station_time)
        cadence = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 1.0
        station_nez = nez[order]
        station_geo = geo[order]
        if len(station_time) > 1 and np.isfinite(cadence) and cadence > 0.0:
            # A station omitted from one SuperMAG block row otherwise vanishes
            # from its own time array, preventing the gap cleaner from seeing
            # the missing sample. Materialize the station cadence grid first.
            origin = float(station_time[0])
            positions = np.rint((station_time - origin) / cadence).astype(int)
            grid_length = int(np.max(positions)) + 1
            regular_time = origin + np.arange(grid_length, dtype=float) * cadence
            regular_nez = np.full((grid_length, 3), np.nan, dtype=float)
            regular_geo = np.full((grid_length, 3), np.nan, dtype=float)
            regular_nez[positions] = station_nez
            regular_geo[positions] = station_geo
            station_time = regular_time
            station_nez = regular_nez
            station_geo = regular_geo
        custom_gap_limit = 5.0 - 1.0e-6
        all_data[code] = {
            "tval": station_time,
            "nez": clean_magnetic_vectors(
                station_nez, cadence, max_gap_seconds=custom_gap_limit
            ),
            "geo": clean_magnetic_vectors(
                station_geo, cadence, max_gap_seconds=custom_gap_limit
            ),
            "median_cadence_seconds": cadence,
        }
        lat = arrays["glat"][rows, columns]
        lon = arrays["glon"][rows, columns]
        metadata[code] = StationMetadata(
            code=code,
            glat=float(np.nanmedian(lat)),
            glon=float(normalize_longitude(np.nanmedian(lon))),
        )
    if not all_data:
        raise ValueError("Custom SuperMAG file contains no station vectors.")
    return all_data, metadata


def convert_custom_supermag_netcdf(source_path: str, output_path: str) -> str:
    """Convert a SuperMAG custom block/vector file to the suite NetCDF schema."""
    all_data, metadata = _read_custom_supermag_blocks(source_path)
    cadences = np.asarray(
        [item.get("median_cadence_seconds", np.nan) for item in all_data.values()],
        dtype=float,
    )
    cadences = cadences[np.isfinite(cadences) & (cadences > 0.0)]
    cadence = float(np.median(cadences)) if len(cadences) else np.nan
    if np.isfinite(cadence):
        finite_times = np.concatenate(
            [item["tval"][np.isfinite(item["tval"])] for item in all_data.values()]
        )
        origin = float(np.min(finite_times))
        # SuperMAG block files can encode nominally common samples with jitter
        # or a station-specific phase. Snap each observation to its nearest
        # cadence slot; skipped slot numbers still preserve genuine outages.
        for item in all_data.values():
            times = np.asarray(item["tval"], dtype=float)
            positions = np.rint((times - origin) / cadence)
            snapped = origin + positions * cadence
            tolerance = max(1.0e-3, 0.51 * cadence)
            item["tval"] = np.where(np.abs(times - snapped) <= tolerance, snapped, times)
    save_to_netcdf(
        all_data,
        output_path,
        metadata,
        requested_high_resolution=True,
        max_gap_seconds=5.0 - 1.0e-6,
    )
    with Dataset(output_path, "a") as ds:
        ds.source = "Converted SuperMAG custom NetCDF product"
        ds.converted_from = str(Path(source_path).expanduser().resolve())
        ds.cadence_mode = "custom"
        if np.isfinite(cadence):
            ds.output_cadence_seconds = cadence
    return str(Path(output_path).expanduser().resolve())


def load_netcdf_file(path: str) -> LoadedMagneticData:
    """Load the merged application's NetCDF structure fully into memory."""
    with Dataset(path, "r") as ds:
        cadence_mode = str(getattr(ds, "cadence_mode", "") or "").strip()
        cadence_seconds = float(getattr(ds, "output_cadence_seconds", np.nan))
        if not cadence_mode and "Converted SuperMAG custom" in str(getattr(ds, "source", "")):
            cadence_mode = "custom"
        required = {"station", "time", "nez", "glat", "glon"}
        missing = required.difference(ds.variables)
        if missing:
            if "stations" in ds.groups and {"station", "glat", "glon"}.issubset(ds.variables):
                station_codes = decode_station_names(ds["station"][:])
                glat = np.asarray(ds["glat"][:], dtype=float)
                glon = normalize_longitude(np.asarray(ds["glon"][:], dtype=float))
                series = []
                for code in station_codes:
                    group = ds.groups["stations"].groups.get(str(code))
                    if group is None or "time" not in group.variables or "nez" not in group.variables:
                        raise ValueError(f"Native-cadence station group is missing for {code}.")
                    seconds = np.asarray(group["time"][:], dtype=float)
                    if hasattr(group["time"], "units"):
                        station_times = np.asarray(
                            [datetime_to_unix_seconds(t) for t in netcdf_time_to_datetimes(group["time"])],
                            dtype=float,
                        )
                    else:
                        station_times = seconds
                    series.append((station_times, np.asarray(group["nez"][:], dtype=float)))
                common_time = np.unique(np.concatenate([item[0] for item in series]))
                nez = np.full((len(series), len(common_time), 3), np.nan)
                lookup = {value: index for index, value in enumerate(common_time)}
                for station_index, (station_times, vectors) in enumerate(series):
                    positions = [lookup[value] for value in station_times]
                    nez[station_index, positions] = vectors
                data = LoadedMagneticData(
                    station_codes=station_codes,
                    times=np.asarray([unix_seconds_to_datetime(v) for v in common_time]),
                    nez=mask_bad_magnetometer_values(nez), glat=glat, glon=glon,
                    geo=None, dbh=calculate_dbh(nez), source_path=str(Path(path).resolve()),
                    cadence_mode="original", cadence_seconds=cadence_seconds,
                )
                data.validate()
                return data
            custom_markers = {"id", "npnt", "dbn_nez", "dbe_nez", "dbz_nez"}
            if custom_markers.issubset(ds.variables):
                all_data, metadata = _read_custom_supermag_blocks(path)
                stations = sorted(all_data)
                common_time = np.unique(np.concatenate([all_data[s]["tval"] for s in stations]))
                nez = np.full((len(stations), len(common_time), 3), np.nan)
                geo = np.full_like(nez, np.nan)
                lookup = {value: index for index, value in enumerate(common_time)}
                for station_index, station in enumerate(stations):
                    positions = [lookup[value] for value in all_data[station]["tval"]]
                    nez[station_index, positions] = all_data[station]["nez"]
                    geo[station_index, positions] = all_data[station]["geo"]
                data = LoadedMagneticData(
                    station_codes=np.asarray(stations),
                    times=np.asarray([unix_seconds_to_datetime(v) for v in common_time]),
                    nez=nez,
                    geo=geo,
                    glat=np.asarray([metadata[s].glat for s in stations]),
                    glon=np.asarray([metadata[s].glon for s in stations]),
                    dbh=calculate_dbh(nez), source_path=str(Path(path).resolve()),
                    cadence_mode="custom", cadence_seconds=float(np.median([
                        all_data[s]["median_cadence_seconds"] for s in stations
                    ])),
                )
                data.validate()
                return data
            raise ValueError(f"NetCDF file is missing variables: {sorted(missing)}")

        station_codes = decode_station_names(ds["station"][:])
        times = netcdf_time_to_datetimes(ds["time"])
        nez = np.asarray(ds["nez"][:], dtype=float)
        glat = np.asarray(ds["glat"][:], dtype=float)
        glon = normalize_longitude(np.asarray(ds["glon"][:], dtype=float))
        geo = np.asarray(ds["geo"][:], dtype=float) if "geo" in ds.variables else None

        dbh_name = next(
            (name for name in ds.variables if name.lower() == "dbh"),
            None,
        )
        if dbh_name is not None:
            dbh = _coerce_station_time_array(
                ds[dbh_name][:],
                len(station_codes),
                len(times),
                dbh_name,
            )
            dbh = mask_bad_magnetometer_values(dbh)
        else:
            dbh = calculate_dbh(nez)

    data = LoadedMagneticData(
        station_codes=station_codes,
        times=times,
        nez=nez,
        glat=glat,
        glon=glon,
        geo=geo,
        dbh=dbh,
        source_path=str(Path(path).resolve()),
        cadence_mode=cadence_mode,
        cadence_seconds=cadence_seconds,
    )
    data.validate()
    return data


def load_legacy_csv_file(path: str) -> LoadedMagneticData:
    """Load the CSV format used by magnetic_viewerT.py into the common data model."""
    frame = pd.read_csv(path)
    required = {
        "Date_UTC",
        "IAGA",
        "GEOLAT",
        "GEOLON",
        "dbn_nez",
        "dbe_nez",
        "dbz_nez",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV file is missing columns: {sorted(missing)}")

    frame["Date_UTC"] = pd.to_datetime(frame["Date_UTC"], utc=True).dt.tz_convert(None)
    station_codes = np.asarray(sorted(frame["IAGA"].astype(str).unique()), dtype=str)
    times = np.asarray(sorted(frame["Date_UTC"].unique()), dtype="datetime64[ns]")
    py_times = np.asarray(
        [pd.Timestamp(value).to_pydatetime() for value in times], dtype=object
    )
    time_lookup = {pd.Timestamp(value): i for i, value in enumerate(times)}

    nstation = len(station_codes)
    ntime = len(times)
    nez = np.full((nstation, ntime, 3), np.nan, dtype=float)
    glat = np.full(nstation, np.nan, dtype=float)
    glon = np.full(nstation, np.nan, dtype=float)

    for station_index, station in enumerate(station_codes):
        subset = frame[frame["IAGA"].astype(str) == station].sort_values("Date_UTC")
        glat[station_index] = float(subset["GEOLAT"].iloc[0])
        glon[station_index] = float(normalize_longitude(subset["GEOLON"].iloc[0]))
        for row in subset.itertuples(index=False):
            idx = time_lookup.get(pd.Timestamp(row.Date_UTC))
            if idx is None:
                continue
            nez[station_index, idx, :] = (row.dbn_nez, row.dbe_nez, row.dbz_nez)

    data = LoadedMagneticData(
        station_codes=station_codes,
        times=py_times,
        nez=nez,
        glat=glat,
        glon=glon,
        geo=None,
        dbh=calculate_dbh(nez),
        source_path=str(Path(path).resolve()),
    )
    data.validate()
    return data


def load_data_file(path: str) -> LoadedMagneticData:
    """Load a supported NetCDF or legacy CSV file."""
    suffix = Path(path).suffix.lower()
    if suffix in {".nc", ".nc4", ".cdf", ".netcdf"}:
        return load_netcdf_file(path)
    if suffix == ".csv":
        return load_legacy_csv_file(path)
    raise ValueError("Supported input formats are NetCDF (.nc/.nc4) and CSV (.csv).")


# -----------------------------------------------------------------------------
# Tkinter application
# -----------------------------------------------------------------------------

class SuperMAGDownloadViewer:
    """One-window interface for downloading, mapping, and plotting SuperMAG data."""

    def __init__(self, initial_file: Optional[str] = None) -> None:
        self.root = tk.Tk()
        self.root.title("SuperMAG Download and Magnetic Data Viewer")
        self.root.geometry("1500x940")
        self.root.minsize(1180, 760)
        self.root.report_callback_exception = self._show_callback_error
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.data: Optional[LoadedMagneticData] = None
        self.current_station_index: Optional[int] = None
        self._syncing_selection = False
        self._download_thread: Optional[threading.Thread] = None
        self._download_stop_event = threading.Event()
        self._plot_windows: list[tk.Toplevel] = []
        # Automatic stack windows keep lightweight references so active
        # substorm markers can be added or removed without rebuilding data.
        self._automatic_stack_plots: list[dict[str, object]] = []

        self.map_figure: Optional[Figure] = None
        self.map_axes = None
        self.map_canvas: Optional[FigureCanvasTkAgg] = None
        self.map_toolbar: Optional[NavigationToolbar2Tk] = None
        self.map_scatter = None
        self.map_station_indices = np.array([], dtype=int)
        self.map_highlight = None

        self.single_figure: Optional[Figure] = None
        self.single_canvas: Optional[FigureCanvasTkAgg] = None
        self.single_toolbar: Optional[NavigationToolbar2Tk] = None
        self.single_plot_ax = None

        # Shared time range used by the single-site plot and newly-created
        # automatic stack plots.
        self.data_min_time: Optional[datetime] = None
        self.data_max_time: Optional[datetime] = None
        self.time_start: Optional[datetime] = None
        self.time_end: Optional[datetime] = None
        self._start_frac = 0.0
        self._end_frac = 1.0
        self._dragging_time_handle: Optional[str] = None

        # Independent time range for the manual stack. It can be changed after
        # panels are added using explicit UTC date/time entry fields.
        self.manual_time_start: Optional[datetime] = None
        self.manual_time_end: Optional[datetime] = None

        # Manually assembled stack plot, modelled after magnetic_stackplot.py.
        self.manual_stack_panels: list[dict[str, object]] = []
        self.manual_stack_window: Optional[tk.Toplevel] = None
        self.manual_stack_figure: Optional[Figure] = None
        self.manual_stack_canvas: Optional[FigureCanvasTkAgg] = None
        self.manual_stack_plot_frame: Optional[ttk.Frame] = None
        self.manual_stack_scroll_canvas: Optional[tk.Canvas] = None
        self.manual_stack_tree: Optional[ttk.Treeview] = None
        self._manual_stack_frame_window = None
        self.manual_equal_y_scale = False
        self._manual_stack_axes: list[object] = []
        self._manual_stack_plot_data: list[dict[str, object]] = []
        self._manual_point_artists: dict[int, list[object]] = {}
        self._manual_stack_click_cid: Optional[int] = None
        self._manual_last_clicked_panel_index: Optional[int] = None

        # Separate, scrollable window listing all manual-stack point selections.
        self.manual_selection_window: Optional[tk.Toplevel] = None
        self.manual_selection_scroll_canvas: Optional[tk.Canvas] = None
        self.manual_selection_content_frame: Optional[ttk.Frame] = None
        self._manual_selection_frame_window = None

        self.MANUAL_PANEL_HEIGHT_IN = 2.35
        self.MANUAL_PANEL_WIDTH_IN = 11.5

        self._show_rules_of_the_road()

        self._make_variables()
        self._build_interface()
        self._draw_empty_map()
        self._draw_empty_single_plot()

        if initial_file:
            self.root.after(50, lambda: self.open_data_path(initial_file))

    # ----- startup acknowledgement -------------------------------------------

    def _show_rules_of_the_road(self) -> None:
        """Show the SuperMAG Rules-of-the-Road acknowledgement and block until
        the user clicks OK. Must be dismissed before the main window is usable."""
        self.root.update_idletasks()

        dialog = tk.Toplevel(self.root)
        dialog.title("ATTENTION! RULES OF THE ROAD")
        dialog.geometry("780x620")
        dialog.minsize(560, 400)
        dialog.transient(self.root)
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # force use of OK button

        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=1)

        heading = tk.Label(
            dialog,
            text="ATTENTION! RULES OF THE ROAD",
            fg="red",
            font=("TkDefaultFont", 24, "bold"),
            pady=8,
        )
        heading.grid(row=0, column=0, sticky="ew")

        text_frame = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        text_widget = tk.Text(text_frame, wrap="word", font=("TkDefaultFont", 10))
        text_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.insert("1.0", SUPERMAG_ACKNOWLEDGEMENT_TEXT)
        text_widget.configure(state="disabled")

        button_frame = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        button_frame.grid(row=2, column=0, sticky="ew")
        button_frame.columnconfigure(0, weight=1)

        ok_button = ttk.Button(
            button_frame, text="OK", command=dialog.destroy, width=12
        )
        ok_button.grid(row=0, column=0)

        dialog.update_idletasks()
        dialog.deiconify()
        dialog.attributes("-topmost", True)
        dialog.lift()
        dialog.after(300, lambda: dialog.attributes("-topmost", False))
        try:
            dialog.wait_visibility(dialog)
            dialog.grab_set()
        except tk.TclError:
            pass
        ok_button.focus_set()
        self.root.wait_window(dialog)

    # ----- UI construction -------------------------------------------------

    def _make_variables(self) -> None:
        self.logon_var = tk.StringVar(value="")
        self.metadata_file_var = tk.StringVar(value="")
        self.start_date_var = tk.StringVar(value="")
        self.start_time_var = tk.StringVar(value="")
        self.end_date_var = tk.StringVar(value="")
        self.end_time_var = tk.StringVar(value="")
        self.lat_min_var = tk.StringVar(value="")
        self.lat_max_var = tk.StringVar(value="")
        self.lon_west_var = tk.StringVar(value="")
        self.lon_east_var = tk.StringVar(value="")
        self.output_file_var = tk.StringVar(value=self._default_output_filename())
        self.high_resolution_var = tk.BooleanVar(value=False)

        self.plot_mode_var = tk.StringVar(value="single")
        self.plot_coordinate_var = tk.StringVar(value="NEZ")
        self.stack_sort_var = tk.StringVar(value="Latitude (north to south)")
        self.raw_offset_var = tk.StringVar(value="200")
        self.derivative_offset_var = tk.StringVar(value="50")
        self.map_parameter_var = tk.StringVar(value="dBz")
        self.map_color_scale_mode_var = tk.StringVar(value="automatic")
        self.map_cmin_var = tk.StringVar(value="")
        self.map_cmax_var = tk.StringVar(value="")
        self.map_start_entry_var = tk.StringVar(value="")
        self.map_end_entry_var = tk.StringVar(value="")
        self.map_status_var = tk.StringVar(value="Load data to select map times")

        self.stack_filter_lat_var = tk.BooleanVar(value=False)
        self.stack_filter_lon_var = tk.BooleanVar(value=False)
        self.stack_lat_min_var = tk.StringVar(value="25")
        self.stack_lat_max_var = tk.StringVar(value="85")
        self.stack_lon_west_var = tk.StringVar(value="-170")
        self.stack_lon_east_var = tk.StringVar(value="40")
        self.stack_filter_status_var = tk.StringVar(value="Automatic stacks: all loaded stations")
        self.manual_stack_status_var = tk.StringVar(value="Manual stack: 0 panels")
        self.manual_stack_order_var = tk.StringVar(value="Latitude (north to south)")
        self.single_time_range_var = tk.StringVar(value="Load data to select a time range")
        self.manual_start_entry_var = tk.StringVar(value="")
        self.manual_end_entry_var = tk.StringVar(value="")
        self.manual_time_range_var = tk.StringVar(value="Load data to enter a time range")
        self.manual_point_selection_var = tk.BooleanVar(value=False)
        self.manual_y_scale_button_var = tk.StringVar(value="Use same y-axis scale")
        self.manual_stack_view_mode_var = tk.StringVar(value="components")
        self.manual_stack_view_button_var = tk.StringVar(value="Show dBh")
        self.manual_component_vars = [
            tk.BooleanVar(value=True),
            tk.BooleanVar(value=True),
            tk.BooleanVar(value=True),
        ]

        self.status_var = tk.StringVar(value="Enter query settings or open an existing data file.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.loaded_file_var = tk.StringVar(value="No data file loaded")
        self.station_status_var = tk.StringVar(value="No station selected")

        # Substorm onset explorer.
        self.substorm_window: Optional[tk.Toplevel] = None
        self.substorm_technique_listbox: Optional[tk.Listbox] = None
        self.substorm_results_tree: Optional[ttk.Treeview] = None
        self.substorm_start_var = tk.StringVar(value="")
        self.substorm_end_var = tk.StringVar(value="")
        self.substorm_status_var = tk.StringVar(value="")
        self.substorm_toggle_button_var = tk.StringVar(value="Mark substorm onset")
        self._substorm_search_results: list[SubstormEvent] = []
        self.selected_substorm_events: list[SubstormEvent] = []
        # Retained for compatibility with any external code using the old field.
        self.selected_substorm_times: list[datetime] = []
        self._substorm_invocation_context = "main"

    def _build_interface(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        download_frame = ttk.LabelFrame(self.root, text="SuperMAG download query (UTC)", padding=8)
        download_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        for column in range(12):
            download_frame.columnconfigure(column, weight=0)
        download_frame.columnconfigure(3, weight=1)
        download_frame.columnconfigure(10, weight=1)

        ttk.Label(download_frame, text="Logon").grid(row=0, column=0, sticky="w")
        ttk.Entry(download_frame, textvariable=self.logon_var, width=13).grid(
            row=0, column=1, sticky="w", padx=(4, 12)
        )
        ttk.Label(download_frame, text="Station metadata file").grid(row=0, column=2, sticky="w")
        ttk.Entry(download_frame, textvariable=self.metadata_file_var).grid(
            row=0, column=3, columnspan=3, sticky="ew", padx=4
        )
        ttk.Button(download_frame, text="Browse…", command=self.choose_metadata_file).grid(
            row=0, column=6, sticky="w", padx=(0, 12)
        )
        ttk.Checkbutton(
            download_frame,
            text="High resolution",
            variable=self.high_resolution_var,
        ).grid(row=0, column=7, sticky="w", padx=(0, 12))
        ttk.Button(download_frame, text="Open data file…", command=self.choose_data_file).grid(
            row=0, column=8, sticky="w", padx=4
        )
        ttk.Button(
            download_frame,
            textvariable=self.substorm_toggle_button_var,
            command=lambda: self.toggle_substorm_marking("main"),
        ).grid(row=0, column=9, sticky="w", padx=4)
        ttk.Label(download_frame, textvariable=self.loaded_file_var).grid(
            row=0, column=10, columnspan=3, sticky="ew", padx=4
        )

        ttk.Label(download_frame, text="Start date").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(download_frame, textvariable=self.start_date_var, width=12).grid(
            row=1, column=1, sticky="w", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(download_frame, text="Start time").grid(row=1, column=2, sticky="w", pady=(7, 0))
        ttk.Entry(download_frame, textvariable=self.start_time_var, width=8).grid(
            row=1, column=3, sticky="w", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(download_frame, text="End date").grid(row=1, column=4, sticky="w", pady=(7, 0))
        ttk.Entry(download_frame, textvariable=self.end_date_var, width=12).grid(
            row=1, column=5, sticky="w", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(download_frame, text="End time").grid(row=1, column=6, sticky="w", pady=(7, 0))
        ttk.Entry(download_frame, textvariable=self.end_time_var, width=8).grid(
            row=1, column=7, sticky="w", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(download_frame, text="Format: YYYY-MM-DD, HH:MM").grid(
            row=1, column=8, columnspan=4, sticky="w", pady=(7, 0)
        )

        ttk.Label(download_frame, text="Latitude min").grid(row=2, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(download_frame, textvariable=self.lat_min_var, width=9).grid(
            row=2, column=1, sticky="w", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(download_frame, text="Latitude max").grid(row=2, column=2, sticky="w", pady=(7, 0))
        ttk.Entry(download_frame, textvariable=self.lat_max_var, width=9).grid(
            row=2, column=3, sticky="w", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(download_frame, text="Longitude west").grid(row=2, column=4, sticky="w", pady=(7, 0))
        ttk.Entry(download_frame, textvariable=self.lon_west_var, width=9).grid(
            row=2, column=5, sticky="w", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(download_frame, text="Longitude east").grid(row=2, column=6, sticky="w", pady=(7, 0))
        ttk.Entry(download_frame, textvariable=self.lon_east_var, width=9).grid(
            row=2, column=7, sticky="w", padx=(4, 12), pady=(7, 0)
        )
        ttk.Label(download_frame, text="West > east selects a dateline-crossing box").grid(
            row=2, column=8, columnspan=4, sticky="w", pady=(7, 0)
        )

        ttk.Label(download_frame, text="Output NetCDF").grid(row=3, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(download_frame, textvariable=self.output_file_var).grid(
            row=3, column=1, columnspan=7, sticky="ew", padx=4, pady=(7, 0)
        )
        ttk.Button(download_frame, text="Save as…", command=self.choose_output_file).grid(
            row=3, column=8, sticky="w", padx=4, pady=(7, 0)
        )
        self.download_button = ttk.Button(
            download_frame,
            text="Download and save NetCDF",
            command=self.start_download,
        )
        self.download_button.grid(
            row=3, column=9, columnspan=2, sticky="ew", padx=4, pady=(7, 0)
        )
        self.stop_download_button = ttk.Button(
            download_frame,
            text="Stop download",
            command=self.stop_download,
            state=tk.DISABLED,
        )
        self.stop_download_button.grid(
            row=3, column=11, sticky="ew", padx=4, pady=(7, 0)
        )

        self.progress = ttk.Progressbar(
            download_frame,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.progress.grid(row=4, column=0, columnspan=8, sticky="ew", pady=(8, 0))
        ttk.Label(download_frame, textvariable=self.status_var).grid(
            row=4, column=8, columnspan=4, sticky="w", padx=6, pady=(8, 0)
        )

        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))

        left_frame = ttk.Frame(main_pane, padding=4)
        right_frame = ttk.Frame(main_pane, padding=4)
        main_pane.add(left_frame, weight=0)
        main_pane.add(right_frame, weight=1)

        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(parent, text="Plot mode", padding=7)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        controls.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            controls,
            text="Single selected station: three components",
            value="single",
            variable=self.plot_mode_var,
            command=self.on_plot_mode_change,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(
            controls,
            text="Stacked components (three windows)",
            value="stack_raw",
            variable=self.plot_mode_var,
            command=self.on_plot_mode_change,
        ).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(
            controls,
            text="Stacked component dB/dt (cadence-aware units; three windows)",
            value="stack_derivative",
            variable=self.plot_mode_var,
            command=self.on_plot_mode_change,
        ).grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(
            controls,
            text="Stacked dBh and dBh/dt (two windows)",
            value="stack_dbh",
            variable=self.plot_mode_var,
            command=self.on_plot_mode_change,
        ).grid(row=3, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(
            controls,
            text="Magnetic parameter map",
            value="map",
            variable=self.plot_mode_var,
            command=self.on_plot_mode_change,
        ).grid(row=4, column=0, columnspan=3, sticky="w")

        ttk.Label(controls, text="Plot coordinates").grid(
            row=5, column=0, sticky="w", pady=(7, 0)
        )
        coordinate_box = ttk.Combobox(
            controls,
            textvariable=self.plot_coordinate_var,
            values=("NEZ", "GEO"),
            state="readonly",
            width=12,
        )
        coordinate_box.grid(row=5, column=1, sticky="w", pady=(7, 0))
        coordinate_box.bind("<<ComboboxSelected>>", self.on_coordinate_system_change)

        ttk.Label(controls, text="Plot start time").grid(
            row=6, column=0, sticky="w", pady=(4, 0)
        )
        plot_start_entry = ttk.Entry(
            controls, textvariable=self.map_start_entry_var, width=20
        )
        plot_start_entry.grid(
            row=6, column=1, columnspan=2, sticky="ew", pady=(4, 0)
        )
        ttk.Label(controls, text="Plot end time").grid(
            row=7, column=0, sticky="w", pady=(4, 0)
        )
        plot_end_entry = ttk.Entry(
            controls, textvariable=self.map_end_entry_var, width=20
        )
        plot_end_entry.grid(
            row=7, column=1, columnspan=2, sticky="ew", pady=(4, 0)
        )
        ttk.Label(
            controls,
            text="Plot time format: YYYY-MM-DD HH:mm:SS",
            foreground="#555555",
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(2, 0))
        plot_start_entry.bind("<Return>", self.apply_map_time_entries)
        plot_end_entry.bind("<Return>", self.apply_map_time_entries)

        ttk.Label(controls, text="Stack order").grid(row=9, column=0, sticky="w", pady=(7, 0))
        ttk.Combobox(
            controls,
            textvariable=self.stack_sort_var,
            values=("Latitude (north to south)", "Longitude (west to east)", "Station code"),
            state="readonly",
            width=25,
        ).grid(row=9, column=1, columnspan=2, sticky="ew", pady=(7, 0))

        ttk.Label(controls, text="Raw vertical offset (nT)").grid(
            row=10, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Entry(controls, textvariable=self.raw_offset_var, width=10).grid(
            row=10, column=1, sticky="w", pady=(4, 0)
        )
        ttk.Label(controls, text="Derivative offset (nT/min)").grid(
            row=11, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Entry(controls, textvariable=self.derivative_offset_var, width=10).grid(
            row=11, column=1, sticky="w", pady=(4, 0)
        )

        ttk.Separator(controls, orient=tk.HORIZONTAL).grid(
            row=12, column=0, columnspan=3, sticky="ew", pady=(7, 4)
        )
        ttk.Label(
            controls,
            text="Region selected for plots",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=13, column=0, columnspan=3, sticky="w")

        ttk.Checkbutton(
            controls,
            text="Latitude",
            variable=self.stack_filter_lat_var,
        ).grid(row=14, column=0, sticky="w", pady=(3, 0))
        lat_range_frame = ttk.Frame(controls)
        lat_range_frame.grid(row=14, column=1, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(lat_range_frame, text="min").pack(side=tk.LEFT)
        ttk.Entry(lat_range_frame, textvariable=self.stack_lat_min_var, width=7).pack(
            side=tk.LEFT, padx=(3, 7)
        )
        ttk.Label(lat_range_frame, text="max").pack(side=tk.LEFT)
        ttk.Entry(lat_range_frame, textvariable=self.stack_lat_max_var, width=7).pack(
            side=tk.LEFT, padx=(3, 0)
        )

        ttk.Checkbutton(
            controls,
            text="Longitude",
            variable=self.stack_filter_lon_var,
        ).grid(row=15, column=0, sticky="w", pady=(3, 0))
        lon_range_frame = ttk.Frame(controls)
        lon_range_frame.grid(row=15, column=1, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(lon_range_frame, text="west").pack(side=tk.LEFT)
        ttk.Entry(lon_range_frame, textvariable=self.stack_lon_west_var, width=7).pack(
            side=tk.LEFT, padx=(3, 7)
        )
        ttk.Label(lon_range_frame, text="east").pack(side=tk.LEFT)
        ttk.Entry(lon_range_frame, textvariable=self.stack_lon_east_var, width=7).pack(
            side=tk.LEFT, padx=(3, 0)
        )

        ttk.Label(
            controls,
            text="West > east includes longitudes across ±180°.",
            foreground="#555555",
        ).grid(row=16, column=0, columnspan=3, sticky="w", pady=(2, 0))
        ttk.Label(controls, textvariable=self.stack_filter_status_var).grid(
            row=17, column=0, columnspan=3, sticky="w", pady=(3, 0)
        )

        ttk.Separator(controls, orient=tk.HORIZONTAL).grid(
            row=18, column=0, columnspan=3, sticky="ew", pady=(7, 4)
        )
        ttk.Label(controls, text="Map parameter").grid(row=19, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.map_parameter_var,
            values=("dBn", "dBe", "dBz", "dBh", "dBz/dt", "dBh/dt"),
            state="readonly",
            width=13,
        ).grid(row=19, column=1, columnspan=2, sticky="w")

        ttk.Label(controls, text="Map color scale").grid(
            row=20, column=0, sticky="w", pady=(4, 0)
        )
        color_scale_mode_frame = ttk.Frame(controls)
        color_scale_mode_frame.grid(
            row=20, column=1, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Radiobutton(
            color_scale_mode_frame,
            text="Automatic",
            value="automatic",
            variable=self.map_color_scale_mode_var,
            command=self._update_map_color_scale_controls,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            color_scale_mode_frame,
            text="Enter cmin/cmax",
            value="manual",
            variable=self.map_color_scale_mode_var,
            command=self._update_map_color_scale_controls,
        ).pack(side=tk.LEFT, padx=(8, 0))

        color_limit_frame = ttk.Frame(controls)
        color_limit_frame.grid(
            row=21, column=1, columnspan=2, sticky="w", pady=(3, 0)
        )
        ttk.Label(color_limit_frame, text="cmin").pack(side=tk.LEFT)
        self.map_cmin_entry = ttk.Entry(
            color_limit_frame,
            textvariable=self.map_cmin_var,
            width=9,
        )
        self.map_cmin_entry.pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(color_limit_frame, text="cmax").pack(side=tk.LEFT)
        self.map_cmax_entry = ttk.Entry(
            color_limit_frame,
            textvariable=self.map_cmax_var,
            width=9,
        )
        self.map_cmax_entry.pack(side=tk.LEFT, padx=(3, 0))
        ttk.Label(
            controls,
            text="Color bars use open ends at both limits.",
            foreground="#555555",
        ).grid(row=21, column=0, sticky="w", pady=(3, 0))
        ttk.Label(controls, textvariable=self.map_status_var, wraplength=330).grid(
            row=22, column=0, columnspan=3, sticky="w", pady=(3, 0)
        )

        ttk.Button(controls, text="Create selected plot(s)", command=self.create_selected_plots).grid(
            row=23, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )
        ttk.Button(
            controls,
            text="Show manual station stack",
            command=self.show_manual_stackplot,
        ).grid(row=24, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Label(controls, textvariable=self.manual_stack_status_var).grid(
            row=25, column=0, columnspan=3, sticky="w", pady=(3, 0)
        )
        self._update_map_color_scale_controls()

        station_frame = ttk.LabelFrame(parent, text="Available stations", padding=5)
        station_frame.grid(row=1, column=0, sticky="nsew")
        station_frame.columnconfigure(0, weight=1)
        station_frame.columnconfigure(1, weight=1)
        station_frame.rowconfigure(1, weight=1)

        ttk.Label(station_frame, text="By latitude").grid(row=0, column=0, sticky="w")
        ttk.Label(station_frame, text="By longitude").grid(row=0, column=1, sticky="w")

        lat_frame = ttk.Frame(station_frame)
        lon_frame = ttk.Frame(station_frame)
        lat_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 3))
        lon_frame.grid(row=1, column=1, sticky="nsew", padx=(3, 0))
        lat_frame.columnconfigure(0, weight=1)
        lat_frame.rowconfigure(0, weight=1)
        lon_frame.columnconfigure(0, weight=1)
        lon_frame.rowconfigure(0, weight=1)

        self.lat_listbox = tk.Listbox(lat_frame, exportselection=False, width=23)
        self.lon_listbox = tk.Listbox(lon_frame, exportselection=False, width=23)
        lat_scroll = ttk.Scrollbar(lat_frame, orient=tk.VERTICAL, command=self.lat_listbox.yview)
        lon_scroll = ttk.Scrollbar(lon_frame, orient=tk.VERTICAL, command=self.lon_listbox.yview)
        self.lat_listbox.configure(yscrollcommand=lat_scroll.set)
        self.lon_listbox.configure(yscrollcommand=lon_scroll.set)
        self.lat_listbox.grid(row=0, column=0, sticky="nsew")
        lat_scroll.grid(row=0, column=1, sticky="ns")
        self.lon_listbox.grid(row=0, column=0, sticky="nsew")
        lon_scroll.grid(row=0, column=1, sticky="ns")
        self.lat_listbox.bind("<<ListboxSelect>>", self.on_latitude_list_select)
        self.lon_listbox.bind("<<ListboxSelect>>", self.on_longitude_list_select)

        self.lat_order = np.array([], dtype=int)
        self.lon_order = np.array([], dtype=int)
        ttk.Label(parent, textvariable=self.station_status_var).grid(
            row=2, column=0, sticky="ew", pady=(5, 0)
        )

    def _update_map_color_scale_controls(self) -> None:
        """Enable manual color-limit entries only when that mode is selected."""
        if not hasattr(self, "map_cmin_entry") or not hasattr(self, "map_cmax_entry"):
            return
        state = "normal" if self.map_color_scale_mode_var.get() == "manual" else "disabled"
        self.map_cmin_entry.configure(state=state)
        self.map_cmax_entry.configure(state=state)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=2)

        map_group = ttk.LabelFrame(parent, text="Station map", padding=4)
        plot_group = ttk.LabelFrame(parent, text="Single-site plot", padding=4)
        map_group.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        plot_group.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        map_group.columnconfigure(0, weight=1)
        map_group.rowconfigure(1, weight=1)
        plot_group.columnconfigure(0, weight=1)
        plot_group.rowconfigure(1, weight=1)

        map_buttons = ttk.Frame(map_group)
        map_buttons.grid(row=0, column=0, sticky="ew")
        ttk.Button(map_buttons, text="Fit map to stations", command=self.draw_station_map).pack(
            side=tk.LEFT
        )
        ttk.Button(map_buttons, text="Save map PNG…", command=self.save_map_png).pack(
            side=tk.RIGHT
        )
        self.map_container = ttk.Frame(map_group)
        self.map_container.grid(row=1, column=0, sticky="nsew")

        plot_buttons = ttk.Frame(plot_group)
        plot_buttons.grid(row=0, column=0, sticky="ew")
        plot_buttons.columnconfigure(2, weight=1)

        ttk.Button(
            plot_buttons,
            text="Refresh selected station",
            command=self.plot_selected_station,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            plot_buttons,
            text="Add selected station to stack →",
            command=self.add_selected_station_to_manual_stack,
        ).grid(row=0, column=1, sticky="w", padx=(5, 8))

        slider_frame = ttk.Frame(plot_buttons)
        slider_frame.grid(row=0, column=2, sticky="ew")
        self._build_single_time_slider(slider_frame)

        ttk.Button(
            plot_buttons,
            text="Save single-site PNG…",
            command=self.save_single_png,
        ).grid(row=0, column=3, sticky="e", padx=(8, 0))

        self.single_plot_container = ttk.Frame(plot_group)
        self.single_plot_container.grid(row=1, column=0, sticky="nsew")

    def _build_single_time_slider(self, parent: ttk.Frame) -> None:
        """Create the dual-handle time-range slider used by the single-site plot."""
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Time range:").grid(row=0, column=0, sticky="w", padx=(0, 4))

        self.time_slider_canvas = tk.Canvas(
            parent,
            width=330,
            height=34,
            background="white",
            highlightthickness=1,
            highlightbackground="gray",
        )
        self.time_slider_canvas.grid(row=0, column=1, sticky="ew")

        self._slider_x0 = 14.0
        self._slider_x1 = 316.0
        self._slider_y = 17.0
        self._slider_handle_radius = 7.0

        self.time_slider_canvas.create_line(
            self._slider_x0,
            self._slider_y,
            self._slider_x1,
            self._slider_y,
            fill="gray",
            width=3,
        )
        self._selected_time_range_id = self.time_slider_canvas.create_line(
            self._slider_x0,
            self._slider_y,
            self._slider_x1,
            self._slider_y,
            fill="#4a90d9",
            width=4,
        )
        self._start_time_handle_id = self.time_slider_canvas.create_oval(
            0, 0, 0, 0, fill="#2c5f8a", outline="black"
        )
        self._end_time_handle_id = self.time_slider_canvas.create_oval(
            0, 0, 0, 0, fill="#d94a4a", outline="black"
        )

        ttk.Label(parent, textvariable=self.single_time_range_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(1, 0)
        )

        self.time_slider_canvas.bind("<Button-1>", self._on_time_slider_press)
        self.time_slider_canvas.bind("<B1-Motion>", self._on_time_slider_drag)
        self.time_slider_canvas.bind("<ButtonRelease-1>", self._on_time_slider_release)
        self._redraw_time_slider_handles()

    def _build_manual_time_entries(self, parent: ttk.Frame) -> None:
        """Create UTC start/end entry fields for the manual-stack time range."""
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)

        ttk.Label(parent, text="Start (UTC)").grid(
            row=0, column=0, sticky="w", padx=(0, 4)
        )
        start_entry = ttk.Entry(
            parent,
            textvariable=self.manual_start_entry_var,
            width=21,
        )
        start_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        ttk.Label(parent, text="End (UTC)").grid(
            row=0, column=2, sticky="w", padx=(0, 4)
        )
        end_entry = ttk.Entry(
            parent,
            textvariable=self.manual_end_entry_var,
            width=21,
        )
        end_entry.grid(row=0, column=3, sticky="ew", padx=(0, 10))

        ttk.Button(
            parent,
            text="Apply time range",
            command=self.apply_manual_time_entries,
        ).grid(row=0, column=4, sticky="ew", padx=(0, 4))
        ttk.Button(
            parent,
            text="Full loaded range",
            command=self.reset_manual_time_entries,
        ).grid(row=0, column=5, sticky="ew")

        ttk.Label(
            parent,
            text="Format: YYYY-MM-DD HH:mm:SS",
            foreground="#555555",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(parent, textvariable=self.manual_time_range_var).grid(
            row=1, column=2, columnspan=4, sticky="w", pady=(3, 0)
        )

        start_entry.bind("<Return>", self.apply_manual_time_entries)
        end_entry.bind("<Return>", self.apply_manual_time_entries)

    # ----- dialogs and validation ------------------------------------------

    def _default_output_filename(self) -> str:
        start_date = self.start_date_var.get() 
        start_time = self.start_time_var.get().replace(":", "") 
        end_date = self.end_date_var.get() 
        end_time = self.end_time_var.get().replace(":", "")
        filename = f"{start_date}_{start_time}_{end_date}_{end_time}.nc"
        return str(Path.cwd() / filename)

    def choose_metadata_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select SuperMAG station metadata file",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
        )
        if path:
            self.metadata_file_var.set(path)

    def choose_output_file(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save downloaded data",
            defaultextension=".nc",
            filetypes=(("NetCDF file", "*.nc"), ("All files", "*.*")),
            initialfile=Path(self.output_file_var.get()).name,
        )
        if path:
            self.output_file_var.set(path)

    def choose_data_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open data file",
            filetypes=(
                ("Supported data", "*.nc *.nc4 *.cdf *.netcdf *.csv"),
                ("NetCDF", "*.nc *.nc4 *.cdf *.netcdf"),
                ("CSV", "*.csv"),
                ("All files", "*.*"),
            ),
        )
        if path:
            self.open_data_path(path)

    # ----- substorm onset list explorer --------------------------------------

    def _substorm_context_interval(self) -> Optional[tuple[datetime, datetime]]:
        """Return the currently displayed interval for the invoking window."""
        if self._substorm_invocation_context == "manual":
            if self.manual_time_start is not None and self.manual_time_end is not None:
                return self.manual_time_start, self.manual_time_end
        if self.time_start is not None and self.time_end is not None:
            return self.time_start, self.time_end
        try:
            start = datetime.strptime(
                f"{self.start_date_var.get().strip()} {self.start_time_var.get().strip()}",
                "%Y-%m-%d %H:%M",
            )
            end = datetime.strptime(
                f"{self.end_date_var.get().strip()} {self.end_time_var.get().strip()}",
                "%Y-%m-%d %H:%M",
            )
        except ValueError:
            return None
        return (start, end) if end > start else None

    def use_current_interval_for_substorms(self) -> None:
        interval = self._substorm_context_interval()
        if interval is None:
            messagebox.showinfo(
                "Substorm interval",
                "Load data or enter a valid start/end interval in the main window first.",
            )
            return
        start, end = interval
        self.substorm_start_var.set(start.replace(microsecond=0).strftime(MANUAL_TIME_FORMAT))
        self.substorm_end_var.set(end.replace(microsecond=0).strftime(MANUAL_TIME_FORMAT))

    def open_substorm_explorer(self, context: str = "main") -> None:
        """Open the shared downloader/selector used by the main and stack windows."""
        self._substorm_invocation_context = "manual" if context == "manual" else "main"
        if not self._substorm_search_results:
            interval = self._substorm_context_interval()
            if interval is not None:
                start, end = interval
                self.substorm_start_var.set(start.replace(microsecond=0).strftime(MANUAL_TIME_FORMAT))
                self.substorm_end_var.set(end.replace(microsecond=0).strftime(MANUAL_TIME_FORMAT))

        if self.substorm_window is not None and self.substorm_window.winfo_exists():
            self.substorm_window.deiconify()
            self.substorm_window.lift()
            return

        window = tk.Toplevel(self.root)
        window.title("Download and mark substorm onsets")
        window.geometry("1080x590")
        window.minsize(820, 450)
        window.protocol("WM_DELETE_WINDOW", window.withdraw)
        self.substorm_window = window

        window.columnconfigure(0, weight=0)
        window.columnconfigure(1, weight=1)
        window.rowconfigure(1, weight=1)

        ttk.Label(
            window,
            text="SuperMAG substorm onset lists",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 4))

        left = ttk.LabelFrame(window, text="Substorm list", padding=6)
        left.grid(row=1, column=0, sticky="ns", padx=(8, 4), pady=(0, 8))

        self.substorm_technique_listbox = tk.Listbox(
            left,
            height=9,
            exportselection=False,
            width=39,
            selectmode=tk.EXTENDED,
        )
        for name, _code in SUBSTORM_TECHNIQUES:
            self.substorm_technique_listbox.insert(tk.END, name)
        self.substorm_technique_listbox.selection_set(0)
        self.substorm_technique_listbox.pack(fill=tk.BOTH, expand=True)

        ttk.Button(
            left,
            text="Load list from file…",
            command=self.load_substorm_list_from_file,
        ).pack(fill=tk.X, pady=(8, 0))
        ttk.Button(
            left,
            text="Save displayed list as CSV…",
            command=self.save_displayed_substorm_list,
        ).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(
            left,
            text="Clear displayed records",
            command=self.clear_displayed_substorm_records,
        ).pack(fill=tk.X, pady=(4, 0))
        ttk.Label(
            left,
            text=(
                "Select one or more techniques. New downloads and loaded files "
                "are merged with the displayed records without duplicates. "
                "GLat/GLon are retained when supplied by the list."
            ),
            foreground="#555555",
            wraplength=285,
            justify="left",
        ).pack(fill=tk.X, pady=(8, 0))

        right = ttk.Frame(window, padding=(4, 0, 8, 8))
        right.grid(row=1, column=1, sticky="nsew", pady=(0, 8))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        search_frame = ttk.LabelFrame(right, text="Download time range (UTC)", padding=6)
        search_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        search_frame.columnconfigure(1, weight=1)
        search_frame.columnconfigure(3, weight=1)

        ttk.Label(search_frame, text="Start (UTC)").grid(row=0, column=0, sticky="w")
        ttk.Entry(search_frame, textvariable=self.substorm_start_var, width=21).grid(
            row=0, column=1, sticky="ew", padx=(4, 10)
        )
        ttk.Label(search_frame, text="End (UTC)").grid(row=0, column=2, sticky="w")
        ttk.Entry(search_frame, textvariable=self.substorm_end_var, width=21).grid(
            row=0, column=3, sticky="ew", padx=(4, 10)
        )
        ttk.Button(
            search_frame,
            text="Download selected list(s)",
            command=self.search_substorm_list,
        ).grid(row=0, column=4, sticky="ew", padx=(0, 4))
        ttk.Button(
            search_frame,
            text="Use displayed interval",
            command=self.use_current_interval_for_substorms,
        ).grid(row=0, column=5, sticky="ew")
        ttk.Label(
            search_frame, text="Format: YYYY-MM-DD HH:mm:SS", foreground="#555555"
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(3, 0))

        results_frame = ttk.LabelFrame(right, text="Downloaded substorm records", padding=6)
        results_frame.grid(row=1, column=0, sticky="nsew")
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        columns = ("number", "onset", "glat", "glon", "mlat", "mlt", "source")
        self.substorm_results_tree = ttk.Treeview(
            results_frame,
            columns=columns,
            show="headings",
            selectmode="extended",
        )
        headings = {
            "number": "No.",
            "onset": "Onset time (UTC)",
            "glat": "GLat",
            "glon": "GLon",
            "mlat": "MLat",
            "mlt": "MLT",
            "source": "List",
        }
        widths = {
            "number": 48,
            "onset": 180,
            "glat": 72,
            "glon": 78,
            "mlat": 72,
            "mlt": 65,
            "source": 85,
        }
        for column in columns:
            self.substorm_results_tree.heading(column, text=headings[column])
            self.substorm_results_tree.column(
                column,
                width=widths[column],
                anchor=(
                    "center"
                    if column == "number"
                    else "w"
                    if column in {"onset", "source"}
                    else "e"
                ),
                stretch=column == "onset",
            )
        results_scroll = ttk.Scrollbar(
            results_frame, orient=tk.VERTICAL, command=self.substorm_results_tree.yview
        )
        self.substorm_results_tree.configure(yscrollcommand=results_scroll.set)
        self.substorm_results_tree.grid(row=0, column=0, sticky="nsew")
        results_scroll.grid(row=0, column=1, sticky="ns")

        ttk.Label(right, textvariable=self.substorm_status_var, foreground="#555555").grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )

        bottom = ttk.Frame(window, padding=(8, 0, 8, 8))
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(
            bottom,
            text="Apply selected onsets",
            command=self.apply_selected_substorms,
        ).pack(side=tk.LEFT)
        ttk.Label(
            bottom,
            text=(
                "Selected records (or all displayed records when none are highlighted) "
                "are applied to single-site, manual-stack, automatic-stack, and map views."
            ),
            foreground="#555555",
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(bottom, text="Close", command=window.withdraw).pack(side=tk.RIGHT)
        self._populate_substorm_results(self._substorm_search_results)

    def _selected_substorm_techniques(self) -> list[tuple[str, str]]:
        """Return all highlighted substorm techniques, defaulting to Newell."""
        if self.substorm_technique_listbox is None:
            return [SUBSTORM_TECHNIQUES[0]]
        selection = self.substorm_technique_listbox.curselection()
        indices = list(selection) if selection else [0]
        return [SUBSTORM_TECHNIQUES[index] for index in indices]

    def _selected_substorm_technique(self) -> tuple[str, str]:
        """Backward-compatible single-technique accessor."""
        return self._selected_substorm_techniques()[0]

    def _parse_substorm_search_range(self) -> Optional[tuple[datetime, datetime]]:
        start_text = self.substorm_start_var.get().strip()
        end_text = self.substorm_end_var.get().strip()
        try:
            start = datetime.strptime(start_text, MANUAL_TIME_FORMAT)
            end = datetime.strptime(end_text, MANUAL_TIME_FORMAT)
        except ValueError:
            messagebox.showerror(
                "Substorm download",
                "Enter both times in the format YYYY-MM-DD HH:mm:SS.",
            )
            return None
        if end <= start:
            messagebox.showerror(
                "Substorm download", "The end time must be later than the start time."
            )
            return None
        return start, end

    @staticmethod
    def _substorm_value_text(value: float) -> str:
        return "" if not np.isfinite(value) else f"{value:.2f}"

    @staticmethod
    def _substorm_source_category(source: str) -> str:
        """Normalize source names/codes to one of the five supported lists."""
        text = str(source).strip().lower()
        aliases = (
            ("newell", ("newell",)),
            ("forsyth", ("forsyth", "sophie")),
            ("frey", ("frey", "image-fuv", "image_fuv", "image fuv")),
            ("liou", ("liou", "polar uvi", "polar_uvi")),
            ("ohtani", ("ohtani", "midlatitude", "positive bay")),
        )
        for category, tokens in aliases:
            if text == category or any(token in text for token in tokens):
                return category
        return text or "unknown"

    def _substorm_event_label_map(
        self, events: Iterable[SubstormEvent]
    ) -> dict[tuple[object, ...], str]:
        """Assign N/F/I/L/O labels independently within each source list."""
        counters: dict[str, int] = {}
        labels: dict[tuple[object, ...], str] = {}
        for event in sorted(events, key=lambda item: (item.onset, item.source)):
            category = self._substorm_source_category(event.source)
            counters[category] = counters.get(category, 0) + 1
            prefix = SUBSTORM_SOURCE_PREFIXES.get(category, "U")
            labels[_substorm_event_key(event)] = f"{prefix}{counters[category]}"
        return labels

    def _merge_substorm_results(self, events: Iterable[SubstormEvent]) -> int:
        """Merge records into the displayed collection and return added count."""
        existing = {
            _substorm_event_key(event): event
            for event in self._substorm_search_results
        }
        before = len(existing)
        for event in events:
            existing[_substorm_event_key(event)] = event
        self._populate_substorm_results(list(existing.values()))
        return len(existing) - before

    def clear_displayed_substorm_records(self) -> None:
        """Clear downloaded records and remove all currently drawn onsets."""
        self.unmark_substorm_onsets()
        self._populate_substorm_results([])
        self.substorm_status_var.set("Cleared all displayed substorm records.")

    def _populate_substorm_results(self, events: list[SubstormEvent]) -> None:
        tree = self.substorm_results_tree
        selected_keys = {
            _substorm_event_key(event) for event in self.selected_substorm_events
        }
        # Preserve highlighted-but-not-yet-applied rows while appending lists.
        if tree is not None:
            for item_id in tree.selection():
                try:
                    index = int(str(item_id).rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                if 0 <= index < len(self._substorm_search_results):
                    selected_keys.add(
                        _substorm_event_key(self._substorm_search_results[index])
                    )

        unique = {_substorm_event_key(event): event for event in events}
        self._substorm_search_results = sorted(
            unique.values(), key=lambda event: (event.onset, event.source)
        )
        if tree is None:
            return

        label_by_key = self._substorm_event_label_map(self._substorm_search_results)
        tree.delete(*tree.get_children())
        for index, event in enumerate(self._substorm_search_results):
            item_id = f"event_{index}"
            event_key = _substorm_event_key(event)
            tree.insert(
                "",
                tk.END,
                iid=item_id,
                values=(
                    label_by_key.get(event_key, str(index + 1)),
                    event.onset.strftime(MANUAL_TIME_FORMAT),
                    self._substorm_value_text(event.glat),
                    self._substorm_value_text(normalize_longitude(event.glon))
                    if np.isfinite(event.glon)
                    else "",
                    self._substorm_value_text(event.mlat),
                    self._substorm_value_text(event.mlt),
                    event.source,
                ),
            )
            if event_key in selected_keys:
                tree.selection_add(item_id)

    def search_substorm_list(self) -> None:
        time_range = self._parse_substorm_search_range()
        if time_range is None:
            return
        start, end = time_range
        techniques = self._selected_substorm_techniques()
        logon = self.logon_var.get().strip()
        if not logon:
            messagebox.showerror("Substorm download", "Enter a SuperMAG logon id first.")
            return

        downloaded: list[SubstormEvent] = []
        failures: list[str] = []
        for list_number, (name, code) in enumerate(techniques, start=1):
            self.substorm_status_var.set(
                f"Downloading {name} ({list_number}/{len(techniques)})…"
            )
            self.root.update_idletasks()
            try:
                events = fetch_substorm_events(start, end, logon, code)
            except Exception as exc:
                failures.append(
                    f"{name}: {summarize_supermag_api_error(exc)}"
                )
                continue
            downloaded.extend(
                event for event in events if start <= event.onset <= end
            )

        added = self._merge_substorm_results(downloaded)
        located = sum(event.has_geographic_location for event in downloaded)
        source_count = len(
            {self._substorm_source_category(event.source) for event in self._substorm_search_results}
        )
        self.substorm_status_var.set(
            f"Added {added} new record(s); {len(self._substorm_search_results)} total "
            f"from {source_count} list(s). {located} downloaded record(s) include GLat/GLon."
        )
        if failures:
            messagebox.showwarning(
                "Substorm download",
                "Some selected lists could not be downloaded:\n\n"
                + "\n\n".join(failures),
            )

    def load_substorm_list_from_file(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Open one or more substorm list files",
            filetypes=(("Text/CSV", "*.txt *.csv *.dat"), ("All files", "*.*")),
        )
        if not paths:
            return

        loaded_events: list[SubstormEvent] = []
        failures: list[str] = []
        time_range = self._parse_substorm_search_range()
        for path in paths:
            try:
                events = load_substorm_events_from_file(path)
            except OSError as exc:
                failures.append(f"{Path(path).name}: {exc}")
                continue
            if time_range is not None:
                start, end = time_range
                events = [event for event in events if start <= event.onset <= end]
            loaded_events.extend(events)

        added = self._merge_substorm_results(loaded_events)
        located = sum(event.has_geographic_location for event in loaded_events)
        self.substorm_status_var.set(
            f"Added {added} new record(s) from {len(paths) - len(failures)} file(s); "
            f"{len(self._substorm_search_results)} total. "
            f"{located} loaded record(s) include GLat/GLon."
        )
        if failures:
            messagebox.showwarning(
                "Substorm list files",
                "Some files could not be read:\n\n"
                + "\n".join(failures),
            )

    def save_displayed_substorm_list(self) -> None:
        if not self._substorm_search_results:
            messagebox.showinfo("Save substorm list", "Download or load a list first.")
            return
        categories = {
            self._substorm_source_category(event.source)
            for event in self._substorm_search_results
        }
        code = next(iter(categories)) if len(categories) == 1 else "combined"
        path = filedialog.asksaveasfilename(
            title="Save displayed substorm list",
            defaultextension=".csv",
            filetypes=(("CSV file", "*.csv"), ("All files", "*.*")),
            initialfile=f"SuperMAG_{code}_substorms.csv",
        )
        if not path:
            return
        try:
            save_substorm_events_to_csv(self._substorm_search_results, path)
        except OSError as exc:
            messagebox.showerror("Save substorm list", f"Could not write the file:\n\n{exc}")
            return
        self.substorm_status_var.set(f"Saved {len(self._substorm_search_results)} records to {path}.")

    def _selected_substorm_events(self) -> list[SubstormEvent]:
        tree = self.substorm_results_tree
        if tree is None:
            return []
        selected_items = tree.selection()
        items = selected_items if selected_items else tree.get_children()
        events: list[SubstormEvent] = []
        for item_id in items:
            try:
                index = int(str(item_id).rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if 0 <= index < len(self._substorm_search_results):
                events.append(self._substorm_search_results[index])
        unique = {_substorm_event_key(event): event for event in events}
        return sorted(unique.values(), key=lambda event: (event.onset, event.source))

    def _selected_substorm_onset_times(self) -> list[datetime]:
        """Compatibility helper returning times from the current tree selection."""
        return [event.onset for event in self._selected_substorm_events()]

    def _update_substorm_toggle_text(self) -> None:
        self.substorm_toggle_button_var.set(
            "Unmark substorm onset"
            if self.selected_substorm_events
            else "Mark substorm onset"
        )

    def toggle_substorm_marking(self, context: str = "main") -> None:
        """Open the selector when unmarked, or remove every marker when marked."""
        if self.selected_substorm_events:
            self.unmark_substorm_onsets()
        else:
            self.open_substorm_explorer(context)

    def unmark_substorm_onsets(self) -> None:
        """Remove all active onset markers from every plot type."""
        self.selected_substorm_events = []
        self.selected_substorm_times = []
        self._update_substorm_toggle_text()

        if self.data is not None and self.current_station_index is not None:
            self.plot_selected_station()
        if self.manual_stack_window is not None and self.manual_stack_window.winfo_exists():
            self._rebuild_manual_stackplot()
        self._refresh_automatic_stack_substorm_markers()
        if self.data is not None:
            self.draw_station_map()
        self.substorm_status_var.set("Removed all marked substorm onsets from the plots.")

    def apply_selected_substorms(self) -> None:
        """Apply the current event selection consistently to all relevant plots."""
        events = self._selected_substorm_events()
        if not events:
            messagebox.showinfo(
                "Mark substorm onset",
                "Download or load a substorm list first, then select one or more records.",
            )
            return
        self.selected_substorm_events = events
        self.selected_substorm_times = [event.onset for event in events]
        self._update_substorm_toggle_text()

        if self.data is not None and self.current_station_index is not None:
            self.plot_selected_station()
        if self.manual_stack_window is not None and self.manual_stack_window.winfo_exists():
            self._rebuild_manual_stackplot()
        self._refresh_automatic_stack_substorm_markers()
        if self.data is not None:
            self.draw_station_map()

        located = sum(event.has_geographic_location for event in events)
        self.substorm_status_var.set(
            f"Applied {len(events)} onset(s) to all plot types; "
            f"{located} location(s) marked on the station map."
        )

    def mark_substorm_onset_on_single_plot(self) -> None:
        """Backward-compatible callback for older saved UI bindings."""
        self.toggle_substorm_marking("main")

    def mark_substorm_onset_on_manual_stack(self) -> None:
        """Toggle markers or open the shared selector from the manual stack."""
        self.toggle_substorm_marking("manual")

    def _numbered_selected_substorm_events(self) -> list[tuple[str, SubstormEvent]]:
        """Return selected events with stable source-specific labels."""
        label_by_key = self._substorm_event_label_map(self._substorm_search_results)
        selected_events = sorted(
            self.selected_substorm_events,
            key=lambda event: (event.onset, event.source),
        )

        # Selected records loaded outside the current table still receive labels
        # continuing from the corresponding source category.
        category_counts: dict[str, int] = {}
        for event in self._substorm_search_results:
            category = self._substorm_source_category(event.source)
            category_counts[category] = category_counts.get(category, 0) + 1

        numbered_events: list[tuple[str, SubstormEvent]] = []
        for event in selected_events:
            event_key = _substorm_event_key(event)
            label = label_by_key.get(event_key)
            if label is None:
                category = self._substorm_source_category(event.source)
                category_counts[category] = category_counts.get(category, 0) + 1
                prefix = SUBSTORM_SOURCE_PREFIXES.get(category, "U")
                label = f"{prefix}{category_counts[category]}"
            numbered_events.append((label, event))
        return numbered_events

    def _draw_selected_substorm_lines(
        self,
        ax,
        *,
        show_numbers: bool = True,
        number_y: float = 1.005,
        line_color: str = "black",
        line_alpha: float = 0.85,
        line_zorder: float = 0.4,
        line_ax=None,
    ) -> list[object]:
        """Draw active onsets behind data and return all created artists."""
        artists: list[object] = []
        x_min, x_max = sorted(ax.get_xlim())
        target_ax = line_ax if line_ax is not None else ax
        for event_label, event in self._numbered_selected_substorm_events():
            onset_value = float(date2num(event.onset))
            if onset_value < x_min or onset_value > x_max:
                continue
            line = target_ax.axvline(
                event.onset,
                color=line_color,
                linestyle="--",
                linewidth=1.15,
                alpha=line_alpha,
                zorder=line_zorder,
            )
            artists.append(line)
            if show_numbers:
                annotation = ax.annotate(
                    str(event_label),
                    xy=(event.onset, number_y),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 1),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=9,
                    fontweight="bold",
                    clip_on=False,
                    zorder=8,
                    bbox=dict(
                        boxstyle="round,pad=0.10",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.78,
                    ),
                )
                artists.append(annotation)
        return artists

    def _refresh_automatic_stack_substorm_markers(self) -> None:
        """Update markers in every still-open automatic stack window."""
        active_records: list[dict[str, object]] = []
        for record in self._automatic_stack_plots:
            window = record.get("window")
            if window is None or not window.winfo_exists():
                continue
            for artist in record.get("onset_artists", []):
                try:
                    artist.remove()
                except (ValueError, AttributeError):
                    pass
            ax = record.get("axis")
            canvas = record.get("canvas")
            artists: list[object] = []
            if ax is not None and self.selected_substorm_events:
                artists = self._draw_selected_substorm_lines(
                    ax,
                    show_numbers=True,
                    line_color="black",
                    line_alpha=0.85,
                    line_zorder=0.25,
                )
            record["onset_artists"] = artists
            if canvas is not None:
                canvas.draw_idle()
            active_records.append(record)
        self._automatic_stack_plots = active_records

    def _draw_selected_substorm_locations(self, ax, center_lon: float) -> None:
        """Draw lime-green dashed onset-location circles on the station map."""
        numbered_located_events = [
            (event_number, event)
            for event_number, event in self._numbered_selected_substorm_events()
            if event.has_geographic_location
        ]
        if not numbered_located_events:
            return
        lons = np.asarray(
            [normalize_longitude(event.glon) for _number, event in numbered_located_events],
            dtype=float,
        )
        lats = np.asarray(
            [event.glat for _number, event in numbered_located_events],
            dtype=float,
        )
        if HAS_CARTOPY and hasattr(ax, "projection"):
            x_values = lons
            transform = ccrs.PlateCarree()
        else:
            x_values = normalize_longitude(lons - center_lon)
            transform = None

        scatter_kwargs = dict(
            s=620,
            facecolors="none",
            edgecolors="#32CD32",
            linewidths=2.2,
            linestyles="--",
            marker="o",
            zorder=10,
        )
        if transform is not None:
            scatter_kwargs["transform"] = transform
        ax.scatter(x_values, lats, **scatter_kwargs)

        if len(numbered_located_events) > 1:
            for (number, _event), x_value, lat in zip(
                numbered_located_events, x_values, lats
            ):
                annotation_kwargs = dict(
                    xy=(x_value, lat),
                    xytext=(9, 9),
                    textcoords="offset points",
                    color="#32CD32",
                    fontsize=10,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    zorder=11,
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.55, pad=1.2),
                )
                if transform is not None:
                    annotation_kwargs["transform"] = transform
                ax.annotate(str(number), **annotation_kwargs)

    def _parse_download_settings(
        self,
    ) -> tuple[datetime, datetime, float, float, float, float, str, str, str, bool]:
        try:
            start = datetime.strptime(
                f"{self.start_date_var.get().strip()} {self.start_time_var.get().strip()}",
                "%Y-%m-%d %H:%M",
            )
            end = datetime.strptime(
                f"{self.end_date_var.get().strip()} {self.end_time_var.get().strip()}",
                "%Y-%m-%d %H:%M",
            )
        except ValueError as exc:
            raise ValueError("Dates/times must use YYYY-MM-DD and HH:MM.") from exc

        if end <= start:
            raise ValueError("End date/time must be later than start date/time.")

        try:
            lat_min = float(self.lat_min_var.get())
            lat_max = float(self.lat_max_var.get())
            lon_west = float(self.lon_west_var.get())
            lon_east = float(self.lon_east_var.get())
        except ValueError as exc:
            raise ValueError("Latitude and longitude bounds must be numeric.") from exc

        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError("Latitudes must be between -90 and 90 degrees.")
        if not (-360 <= lon_west <= 360 and -360 <= lon_east <= 360):
            raise ValueError("Longitudes must be between -360 and 360 degrees.")

        logon = self.logon_var.get().strip()
        metadata_path = self.metadata_file_var.get().strip()
        output_path = self.output_file_var.get().strip()
        if not logon:
            raise ValueError("Enter the SuperMAG logon name used for API access.")
        if not metadata_path or not Path(metadata_path).is_file():
            raise ValueError("Select a valid SuperMAG station metadata text file.")
        if not output_path:
            raise ValueError("Select an output NetCDF filename.")

        return (
            start,
            end,
            lat_min,
            lat_max,
            lon_west,
            lon_east,
            logon,
            metadata_path,
            output_path,
            bool(self.high_resolution_var.get()),
        )

    # ----- download workflow ------------------------------------------------

    def start_download(self) -> None:
        if self._download_thread is not None and self._download_thread.is_alive():
            messagebox.showinfo("Download", "A download is already running.")
            return

        try:
            settings = self._parse_download_settings()
        except Exception as exc:
            messagebox.showerror("Download settings", str(exc))
            return

        output_path = Path(settings[8]).expanduser()
        if output_path.exists() and not messagebox.askyesno(
            "Replace NetCDF file",
            f"The output file already exists:\n\n{output_path}\n\nReplace it?",
        ):
            return

        self._download_stop_event.clear()
        self.download_button.configure(state=tk.DISABLED)
        self.stop_download_button.configure(state=tk.NORMAL)
        self.progress_var.set(0.0)
        self.status_var.set("Reading station metadata…")
        self._download_thread = threading.Thread(
            target=self._download_worker,
            args=settings,
            daemon=True,
        )
        self._download_thread.start()

    def stop_download(self) -> None:
        """Request a cooperative stop of the active download."""
        if self._download_thread is None or not self._download_thread.is_alive():
            self.stop_download_button.configure(state=tk.DISABLED)
            return

        self._download_stop_event.set()
        self.stop_download_button.configure(state=tk.DISABLED)
        self.status_var.set(
            "Stopping download after the current SuperMAG request finishes…"
        )

    def _download_worker(
        self,
        start: datetime,
        end: datetime,
        lat_min: float,
        lat_max: float,
        lon_west: float,
        lon_east: float,
        logon: str,
        metadata_path: str,
        output_path: str,
        high_resolution: bool,
    ) -> None:
        try:
            if self._download_stop_event.is_set():
                raise DownloadCancelled("Download stopped by the user.")
            metadata = load_station_metadata(metadata_path)
            inventory_warning = None
            if self._download_stop_event.is_set():
                raise DownloadCancelled("Download stopped by the user.")

            try:
                self._thread_status("Fetching station inventory…", 1.0)

                def inventory_progress(attempt: int, attempts: int, text: str) -> None:
                    suffix = "" if attempt == 1 else f"; retry {attempt}/{attempts}"
                    self._thread_status(f"{text}{suffix}…", 1.0)

                inventory = fetch_inventory(
                    start,
                    end,
                    logon,
                    progress_callback=inventory_progress,
                    stop_event=self._download_stop_event,
                )
                if self._download_stop_event.is_set():
                    raise DownloadCancelled("Download stopped by the user.")
                selected = geographic_filter(
                    inventory,
                    metadata,
                    lat_min,
                    lat_max,
                    lon_west,
                    lon_east,
                )
                selection_message = (
                    f"Selected {len(selected)} of {len(inventory)} available stations."
                )
            except DownloadCancelled:
                raise
            except Exception as inventory_exc:
                if self._download_stop_event.is_set():
                    raise DownloadCancelled("Download stopped by the user.")
                inventory_warning = summarize_supermag_api_error(inventory_exc)
                print(
                    "SuperMAG inventory request failed; using metadata fallback:\n"
                    f"{inventory_warning}",
                    file=sys.stderr,
                )
                self._thread_status(
                    "Inventory service unavailable; selecting stations from metadata…",
                    1.5,
                )
                selected = geographic_filter(
                    metadata.keys(),
                    metadata,
                    lat_min,
                    lat_max,
                    lon_west,
                    lon_east,
                )
                selection_message = (
                    f"Inventory fallback selected {len(selected)} metadata stations "
                    "inside the requested region."
                )

            if not selected:
                raise ValueError(
                    "No stations in the metadata catalogue fall inside the requested "
                    "latitude/longitude range."
                )
            self._thread_status(selection_message, 2.0)

            def progress(done: int, total: int, text: str) -> None:
                percent = 2.0 + (done / max(total, 1)) * 88.0
                resolution = "high-resolution" if high_resolution else "standard"
                self._thread_status(
                    f"{text} [{resolution}] ({done}/{total})", percent
                )

            all_data, failures = fetch_many_stations(
                start,
                end,
                logon,
                selected,
                high_resolution=high_resolution,
                progress_callback=progress,
                stop_event=self._download_stop_event,
            )
            if self._download_stop_event.is_set():
                raise DownloadCancelled("Download stopped by the user.")
            if not all_data:
                example_failures = list(failures.items())[:5]
                details = "\n".join(
                    f"{station}: {reason}" for station, reason in example_failures
                )
                if len(failures) > len(example_failures):
                    details += f"\n…and {len(failures) - len(example_failures)} more."
                raise RuntimeError(
                    "All station downloads failed after automatic retries.\n" + details
                )

            self._thread_status("Writing NetCDF file…", 92.0)
            save_to_netcdf(
                all_data,
                output_path,
                metadata,
                start=start,
                end=end,
                geographic_bounds=(lat_min, lat_max, lon_west, lon_east),
                requested_high_resolution=high_resolution,
            )
            if self._download_stop_event.is_set():
                try:
                    Path(output_path).unlink(missing_ok=True)
                except OSError:
                    pass
                raise DownloadCancelled("Download stopped by the user.")

            self._thread_status("Loading saved NetCDF file…", 97.0)
            loaded = load_netcdf_file(output_path)
            if self._download_stop_event.is_set():
                try:
                    Path(output_path).unlink(missing_ok=True)
                except OSError:
                    pass
                raise DownloadCancelled("Download stopped by the user.")

            self.root.after(
                0,
                lambda: self._download_finished(
                    loaded,
                    output_path,
                    failures,
                    len(selected),
                    inventory_warning,
                    high_resolution,
                ),
            )
        except DownloadCancelled:
            self.root.after(0, self._download_cancelled)
        except Exception as exc:
            # Keep the full traceback in the terminal for debugging, but never
            # place multi-page PHP/HTML output in a Tk message box.
            print(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                file=sys.stderr,
            )
            user_message = summarize_supermag_api_error(exc)
            self.root.after(
                0,
                lambda text=user_message: self._download_failed(text),
            )

    def _thread_status(self, text: str, percent: float) -> None:
        self.root.after(0, lambda: self.status_var.set(text))
        self.root.after(0, lambda: self.progress_var.set(percent))

    def _download_finished(
        self,
        loaded: LoadedMagneticData,
        output_path: str,
        failures: dict[str, str],
        selected_count: int,
        inventory_warning: Optional[str] = None,
        high_resolution: bool = False,
    ) -> None:
        if self._download_stop_event.is_set():
            try:
                Path(output_path).unlink(missing_ok=True)
            except OSError:
                pass
            self._download_cancelled()
            return

        self.download_button.configure(state=tk.NORMAL)
        self.stop_download_button.configure(state=tk.DISABLED)
        self._download_stop_event.clear()
        self.progress_var.set(100.0)
        self.set_loaded_data(loaded)
        successful = len(loaded.station_codes)
        status = f"Saved {successful} stations to {output_path}"
        if failures:
            status += f"; {len(failures)} station(s) failed"
        self.status_var.set(status)

        resolution_text = (
            "high resolution" if high_resolution else "standard resolution"
        )
        details = (
            f"Requested resolution: {resolution_text}\n"
            f"Requested stations: {selected_count}\n"
            f"Downloaded and saved: {successful}\n"
            f"Failed: {len(failures)}\n\n"
            f"NetCDF file:\n{output_path}"
        )
        if inventory_warning:
            details += (
                "\n\nInventory fallback used:\n"
                f"{inventory_warning}"
            )
        if failures:
            short_failures = list(failures.items())[:12]
            details += "\n\nFailures:\n" + "\n".join(
                f"{station}: {reason}" for station, reason in short_failures
            )
            if len(failures) > len(short_failures):
                details += f"\n…and {len(failures) - len(short_failures)} more."
        messagebox.showinfo("Download complete", details)

    def _download_cancelled(self) -> None:
        self.download_button.configure(state=tk.NORMAL)
        self.stop_download_button.configure(state=tk.DISABLED)
        self._download_stop_event.clear()
        self.status_var.set("Download stopped. No new NetCDF file was saved.")

    def _download_failed(self, error_text: str) -> None:
        self.download_button.configure(state=tk.NORMAL)
        self.stop_download_button.configure(state=tk.DISABLED)
        self._download_stop_event.clear()
        self.status_var.set("Download failed after automatic retries; no new NetCDF file was created.")
        messagebox.showerror("Download failed", error_text)

    # ----- data loading and station selection -------------------------------

    def open_data_path(self, path: str) -> None:
        try:
            loaded = load_data_file(path)
            self.set_loaded_data(loaded)
            self.status_var.set(
                f"Loaded {len(loaded.station_codes)} stations and {len(loaded.times)} samples."
            )
        except Exception as exc:
            messagebox.showerror("Open data file", f"Could not load {path}:\n\n{exc}")

    def set_loaded_data(self, data: LoadedMagneticData) -> None:
        data.validate()
        # Clear any station index belonging to the previously loaded file before
        # redrawing the map or station lists.
        self.current_station_index = None
        self.data = data
        if data.geo is None and self.plot_coordinate_var.get().strip().upper() == "GEO":
            self.plot_coordinate_var.set("NEZ")
        self.loaded_file_var.set(
            Path(data.source_path).name if data.source_path else "Downloaded data"
        )

        # A manual stack belongs to the currently loaded file. Clear panels
        # when another file is opened or a new download is loaded.
        self.manual_stack_panels = []
        self._update_manual_stack_status()
        self._refresh_manual_point_selection_window()
        if self.manual_stack_window is not None and self.manual_stack_window.winfo_exists():
            self._rebuild_manual_stackplot()

        self._update_query_fields_from_loaded_data()
        self._initialize_time_range()
        self._initialize_map_time_entries()
        self.populate_station_lists()
        self.draw_station_map()

        if len(data.station_codes):
            self.select_station_index(0)
        else:
            self.current_station_index = None
            self._draw_empty_single_plot()

    def _update_query_fields_from_loaded_data(self) -> None:
        if self.data is None or len(self.data.times) == 0:
            return
        start = min(self.data.times)
        end = max(self.data.times)
        self.start_date_var.set(start.strftime("%Y-%m-%d"))
        self.start_time_var.set(start.strftime("%H:%M"))
        self.end_date_var.set(end.strftime("%Y-%m-%d"))
        self.end_time_var.set(end.strftime("%H:%M"))

        finite = np.isfinite(self.data.glat) & np.isfinite(self.data.glon)
        if np.any(finite):
            lat_min = float(np.nanmin(self.data.glat[finite]))
            lat_max = float(np.nanmax(self.data.glat[finite]))
            self.lat_min_var.set(f"{lat_min:.2f}")
            self.lat_max_var.set(f"{lat_max:.2f}")
            self.stack_lat_min_var.set(f"{lat_min:.2f}")
            self.stack_lat_max_var.set(f"{lat_max:.2f}")

            center, relative, _ = choose_longitude_window(self.data.glon[finite])
            rel_finite = relative[np.isfinite(relative)]
            west = float(normalize_longitude(center + np.nanmin(rel_finite)))
            east = float(normalize_longitude(center + np.nanmax(rel_finite)))
            self.lon_west_var.set(f"{west:.2f}")
            self.lon_east_var.set(f"{east:.2f}")
            self.stack_lon_west_var.set(f"{west:.2f}")
            self.stack_lon_east_var.set(f"{east:.2f}")

    def populate_station_lists(self) -> None:
        self.lat_listbox.delete(0, tk.END)
        self.lon_listbox.delete(0, tk.END)
        if self.data is None:
            self.lat_order = np.array([], dtype=int)
            self.lon_order = np.array([], dtype=int)
            return

        codes = self.data.station_codes
        glat = self.data.glat
        glon = normalize_longitude(self.data.glon)
        self.lat_order = np.lexsort((codes, -np.nan_to_num(glat, nan=-999.0)))
        self.lon_order = np.lexsort((codes, np.nan_to_num(glon, nan=999.0)))

        for index in self.lat_order:
            lat_text = f"{glat[index]:7.2f}°" if np.isfinite(glat[index]) else "   n/a  "
            self.lat_listbox.insert(tk.END, f"{codes[index]:5s}  {lat_text}")
        for index in self.lon_order:
            lon_text = f"{glon[index]:8.2f}°" if np.isfinite(glon[index]) else "    n/a  "
            self.lon_listbox.insert(tk.END, f"{codes[index]:5s}  {lon_text}")

    def on_latitude_list_select(self, _event=None) -> None:
        if self._syncing_selection:
            return
        selection = self.lat_listbox.curselection()
        if selection and len(self.lat_order):
            self.select_station_index(int(self.lat_order[selection[0]]))

    def on_longitude_list_select(self, _event=None) -> None:
        if self._syncing_selection:
            return
        selection = self.lon_listbox.curselection()
        if selection and len(self.lon_order):
            self.select_station_index(int(self.lon_order[selection[0]]))

    def select_station_index(self, index: int) -> None:
        if self.data is None or index < 0 or index >= len(self.data.station_codes):
            return
        self.current_station_index = int(index)
        code = self.data.station_codes[index]
        self.station_status_var.set(
            f"Selected {code}: lat={self.data.glat[index]:.2f}°, "
            f"lon={normalize_longitude(self.data.glon[index]):.2f}°"
        )
        self._sync_station_lists(index)
        self.highlight_station_on_map()
        if self.plot_mode_var.get() == "single":
            self.plot_selected_station()

    def _sync_station_lists(self, station_index: int) -> None:
        self._syncing_selection = True
        try:
            self.lat_listbox.selection_clear(0, tk.END)
            self.lon_listbox.selection_clear(0, tk.END)

            lat_positions = np.flatnonzero(self.lat_order == station_index)
            if len(lat_positions):
                pos = int(lat_positions[0])
                self.lat_listbox.selection_set(pos)
                self.lat_listbox.see(pos)

            lon_positions = np.flatnonzero(self.lon_order == station_index)
            if len(lon_positions):
                pos = int(lon_positions[0])
                self.lon_listbox.selection_set(pos)
                self.lon_listbox.see(pos)
        finally:
            self._syncing_selection = False

    # ----- map ---------------------------------------------------------------

    def _clear_container(self, container: ttk.Frame) -> None:
        for widget in container.winfo_children():
            widget.destroy()

    def _draw_empty_map(self) -> None:
        self._clear_container(self.map_container)
        self.map_figure = Figure(figsize=(10, 5), dpi=100)
        ax = self.map_figure.add_subplot(111)
        ax.text(0.5, 0.5, "Download or open a data file to display stations.", ha="center", va="center")
        ax.set_axis_off()
        self.map_axes = ax
        self.map_canvas = FigureCanvasTkAgg(self.map_figure, master=self.map_container)
        self.map_canvas.draw()
        self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.map_toolbar = NavigationToolbar2Tk(self.map_canvas, self.map_container, pack_toolbar=False)
        self.map_toolbar.update()
        self.map_toolbar.pack(fill=tk.X)

    def draw_station_map(self) -> None:
        if self.data is None:
            self._draw_empty_map()
            return

        finite = np.isfinite(self.data.glat) & np.isfinite(self.data.glon)
        if not np.any(finite):
            self._draw_empty_map()
            self.status_var.set("Loaded data contain no finite station coordinates.")
            return

        indices = np.flatnonzero(finite)
        lats = self.data.glat[indices]
        lons = normalize_longitude(self.data.glon[indices])
        codes = self.data.station_codes[indices]
        center_lon, relative_lons, lon_span = choose_longitude_window(lons)

        self._clear_container(self.map_container)
        self.map_figure = Figure(figsize=(11, 6), dpi=100)

        if HAS_CARTOPY:
            projection = ccrs.PlateCarree(central_longitude=center_lon)
            ax = self.map_figure.add_subplot(111, projection=projection)
            ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
            ax.add_feature(cfeature.LAKES, alpha=0.45)
            grid = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.45,
                alpha=0.45,
                linestyle="--",
            )
            grid.top_labels = False
            grid.right_labels = False

            lat_min = float(np.nanmin(lats))
            lat_max = float(np.nanmax(lats))
            lat_pad = max(2.0, 0.12 * max(lat_max - lat_min, 1.0))
            rel_min = float(np.nanmin(relative_lons))
            rel_max = float(np.nanmax(relative_lons))
            lon_pad = max(3.0, 0.12 * max(rel_max - rel_min, 1.0))

            if lon_span >= 330.0:
                ax.set_global()
            else:
                ax.set_extent(
                    [
                        max(-180.0, rel_min - lon_pad),
                        min(180.0, rel_max + lon_pad),
                        max(-90.0, lat_min - lat_pad),
                        min(90.0, lat_max + lat_pad),
                    ],
                    crs=projection,
                )
            ax.set_aspect("auto")
            self.map_scatter = ax.scatter(
                lons,
                lats,
                s=75,
                c="#2866b2",
                alpha=0.78,
                edgecolors="black",
                linewidths=0.8,
                picker=6,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            for lon, lat, code in zip(lons, lats, codes):
                ax.annotate(
                    code,
                    xy=(lon, lat),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold",
                    transform=ccrs.PlateCarree(),
                    zorder=6,
                )
        else:
            ax = self.map_figure.add_subplot(111)
            plot_lons = relative_lons
            self.map_scatter = ax.scatter(
                plot_lons,
                lats,
                s=75,
                c="#2866b2",
                alpha=0.78,
                edgecolors="black",
                linewidths=0.8,
                picker=6,
            )
            for lon, lat, code in zip(plot_lons, lats, codes):
                ax.annotate(code, (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=7)
            ax.set_xlabel(f"Longitude relative to {center_lon:.1f}°")
            ax.set_ylabel("Geographic latitude")
            ax.grid(True, alpha=0.3)
            lon_pad = max(3.0, 0.12 * max(np.ptp(plot_lons), 1.0))
            lat_pad = max(2.0, 0.12 * max(np.ptp(lats), 1.0))
            ax.set_xlim(np.nanmin(plot_lons) - lon_pad, np.nanmax(plot_lons) + lon_pad)
            ax.set_ylim(max(-90, np.nanmin(lats) - lat_pad), min(90, np.nanmax(lats) + lat_pad))
            ax.set_aspect("auto")

        self._draw_selected_substorm_locations(ax, center_lon)
        ax.set_title(f"Available magnetometer stations (n={len(indices)})")
        self.map_figure.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.09)
        self.map_axes = ax
        self.map_station_indices = indices
        self.map_highlight = None

        self.map_canvas = FigureCanvasTkAgg(self.map_figure, master=self.map_container)
        self.map_canvas.draw()
        self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.map_canvas.mpl_connect("pick_event", self.on_map_pick)
        self.map_toolbar = NavigationToolbar2Tk(self.map_canvas, self.map_container, pack_toolbar=False)
        self.map_toolbar.update()
        self.map_toolbar.pack(fill=tk.X)
        self.highlight_station_on_map()

    def on_map_pick(self, event) -> None:
        if self.map_scatter is None or event.artist is not self.map_scatter:
            return
        if len(event.ind) == 0:
            return
        local_index = int(event.ind[0])
        if local_index < len(self.map_station_indices):
            self.select_station_index(int(self.map_station_indices[local_index]))

    def highlight_station_on_map(self) -> None:
        if (
            self.data is None
            or self.current_station_index is None
            or self.map_axes is None
            or self.map_canvas is None
        ):
            return
        index = self.current_station_index
        lon = float(normalize_longitude(self.data.glon[index]))
        lat = float(self.data.glat[index])
        if not np.isfinite(lon) or not np.isfinite(lat):
            return

        if self.map_highlight is not None:
            try:
                self.map_highlight.remove()
            except ValueError:
                pass
            self.map_highlight = None

        # Draw the selected station as a normal-sized green station marker.
        # This separate collection gives the selected station its own high
        # z-order without changing the drawing order of all blue stations.
        kwargs = dict(
            s=75,
            c="limegreen",
            alpha=0.98,
            edgecolors="black",
            linewidths=1.0,
            marker="o",
            picker=False,
            zorder=20,
        )
        if HAS_CARTOPY and hasattr(self.map_axes, "projection"):
            kwargs["transform"] = ccrs.PlateCarree()
            x_value = lon
        else:
            center_lon, _, _ = choose_longitude_window(
                self.data.glon[np.isfinite(self.data.glon)]
            )
            x_value = float(normalize_longitude(lon - center_lon))

        self.map_highlight = self.map_axes.scatter([x_value], [lat], **kwargs)
        self.map_canvas.draw_idle()

    # ----- time-range slider -------------------------------------------------

    def _initialize_time_range(self) -> None:
        """Reset the dual-handle slider to the full interval of the loaded file."""
        if self.data is None or len(self.data.times) == 0:
            self.data_min_time = None
            self.data_max_time = None
            self.time_start = None
            self.time_end = None
            self._start_frac = 0.0
            self._end_frac = 1.0
            self.single_time_range_var.set("Load data to select a time range")
            self.manual_time_start = None
            self.manual_time_end = None
            self.manual_start_entry_var.set("")
            self.manual_end_entry_var.set("")
            self.manual_time_range_var.set("Load data to enter a time range")
            self._redraw_time_slider_handles()
            return

        self.data_min_time = min(self.data.times)
        self.data_max_time = max(self.data.times)
        self._start_frac = 0.0
        self._end_frac = 1.0
        self._update_time_range_from_fractions()
        self._redraw_time_slider_handles()
        self.manual_time_start = self.data_min_time
        self.manual_time_end = self.data_max_time
        self._update_manual_time_entry_values()

    def _time_frac_to_x(self, fraction: float) -> float:
        return self._slider_x0 + fraction * (self._slider_x1 - self._slider_x0)

    def _time_x_to_frac(self, x_value: float) -> float:
        span = self._slider_x1 - self._slider_x0
        if span <= 0:
            return 0.0
        return max(0.0, min(1.0, (x_value - self._slider_x0) / span))

    def _redraw_time_slider_handles(self) -> None:
        if not hasattr(self, "time_slider_canvas"):
            return
        radius = self._slider_handle_radius
        y_value = self._slider_y
        start_x = self._time_frac_to_x(self._start_frac)
        end_x = self._time_frac_to_x(self._end_frac)
        self.time_slider_canvas.coords(
            self._selected_time_range_id, start_x, y_value, end_x, y_value
        )
        self.time_slider_canvas.coords(
            self._start_time_handle_id,
            start_x - radius,
            y_value - radius,
            start_x + radius,
            y_value + radius,
        )
        self.time_slider_canvas.coords(
            self._end_time_handle_id,
            end_x - radius,
            y_value - radius,
            end_x + radius,
            y_value + radius,
        )

    def _update_time_range_from_fractions(self) -> None:
        if self.data_min_time is None or self.data_max_time is None:
            self.time_start = None
            self.time_end = None
            self.single_time_range_var.set("Load data to select a time range")
            return

        interval = self.data_max_time - self.data_min_time
        self.time_start = self.data_min_time + interval * self._start_frac
        self.time_end = self.data_min_time + interval * self._end_frac
        if interval.total_seconds() < 120:
            time_format = "%Y-%m-%d %H:%M:%S"
        else:
            time_format = "%Y-%m-%d %H:%M"
        self.single_time_range_var.set(
            f"{self.time_start.strftime(time_format)}  →  "
            f"{self.time_end.strftime(time_format)} UTC"
        )

    def _on_time_slider_press(self, event) -> None:
        if self.data_min_time is None:
            return
        start_x = self._time_frac_to_x(self._start_frac)
        end_x = self._time_frac_to_x(self._end_frac)
        self._dragging_time_handle = (
            "start" if abs(event.x - start_x) <= abs(event.x - end_x) else "end"
        )
        self._on_time_slider_drag(event)

    def _on_time_slider_drag(self, event) -> None:
        if self._dragging_time_handle is None:
            return
        fraction = self._time_x_to_frac(event.x)
        if self._dragging_time_handle == "start":
            self._start_frac = min(fraction, self._end_frac)
        else:
            self._end_frac = max(fraction, self._start_frac)
        self._redraw_time_slider_handles()
        self._update_time_range_from_fractions()

    def _on_time_slider_release(self, _event) -> None:
        if self._dragging_time_handle is None:
            return
        self._dragging_time_handle = None
        self.plot_selected_station()

    def _initialize_map_time_entries(self) -> None:
        """Initialize the independent magnetic-map time fields."""
        if self.data is None or len(self.data.times) == 0:
            self.map_start_entry_var.set("")
            self.map_end_entry_var.set("")
            self.map_status_var.set("Load data to select map times")
            return
        first_time = min(self.data.times).replace(microsecond=0)
        last_time = max(self.data.times).replace(microsecond=0)
        self.map_start_entry_var.set(first_time.strftime(MANUAL_TIME_FORMAT))
        self.map_end_entry_var.set(last_time.strftime(MANUAL_TIME_FORMAT))
        self.map_status_var.set(
            "Map times affect magnetic parameter maps only; equal times display one sample"
        )

    def _update_manual_time_entry_values(self) -> None:
        """Synchronize manual-stack entry text and its status label."""
        if self.manual_time_start is None or self.manual_time_end is None:
            self.manual_start_entry_var.set("")
            self.manual_end_entry_var.set("")
            self.manual_time_range_var.set("Load data to enter a time range")
            return

        self.manual_start_entry_var.set(
            self.manual_time_start.strftime(MANUAL_TIME_FORMAT)
        )
        self.manual_end_entry_var.set(
            self.manual_time_end.strftime(MANUAL_TIME_FORMAT)
        )
        self.manual_time_range_var.set(
            f"Displayed: {self.manual_time_start.strftime(MANUAL_TIME_FORMAT)} → "
            f"{self.manual_time_end.strftime(MANUAL_TIME_FORMAT)} UTC"
        )

    def _sync_manual_time_range_to_main(self) -> None:
        """Initialize manual entries from the active single-site time interval."""
        self.manual_time_start = (self.time_start or self.data_min_time).replace(
            microsecond=0
        )
        self.manual_time_end = (self.time_end or self.data_max_time).replace(
            microsecond=0
        )
        self._update_manual_time_entry_values()

    def apply_manual_time_entries(self, _event=None) -> None:
        """Validate the manual UTC entries and redraw all existing panels."""
        if self.data_min_time is None or self.data_max_time is None:
            messagebox.showinfo("Manual stack", "Load a data file first.")
            return

        start_text = self.manual_start_entry_var.get().strip()
        end_text = self.manual_end_entry_var.get().strip()
        try:
            start = datetime.strptime(start_text, MANUAL_TIME_FORMAT)
            end = datetime.strptime(end_text, MANUAL_TIME_FORMAT)
        except ValueError:
            messagebox.showerror(
                "Manual-stack time range",
                "Enter both times in the format YYYY-MM-DD HH:mm:SS.",
            )
            return

        if end <= start:
            messagebox.showerror(
                "Manual-stack time range",
                "The end time must be later than the start time.",
            )
            return

        if start < self.data_min_time or end > self.data_max_time:
            messagebox.showerror(
                "Manual-stack time range",
                "The entered range must stay within the loaded data interval:\n\n"
                f"{self.data_min_time.strftime(MANUAL_TIME_FORMAT)} to "
                f"{self.data_max_time.strftime(MANUAL_TIME_FORMAT)} UTC",
            )
            return

        self.manual_time_start = start
        self.manual_time_end = end
        self._update_manual_time_entry_values()
        self._refresh_manual_stack_tree()
        self._rebuild_manual_stackplot()

    def reset_manual_time_entries(self) -> None:
        """Restore the manual stack to the full loaded time interval."""
        if self.data_min_time is None or self.data_max_time is None:
            return
        self.manual_time_start = self.data_min_time
        self.manual_time_end = self.data_max_time
        self._update_manual_time_entry_values()
        self._refresh_manual_stack_tree()
        self._rebuild_manual_stackplot()

    def _manual_selected_time_mask(self) -> np.ndarray:
        if self.data is None:
            return np.array([], dtype=bool)
        if self.manual_time_start is None or self.manual_time_end is None:
            return np.ones(len(self.data.times), dtype=bool)
        return np.asarray(
            [
                self.manual_time_start <= value <= self.manual_time_end
                for value in self.data.times
            ],
            dtype=bool,
        )

    def _selected_time_mask(self) -> np.ndarray:
        """Return a mask for the shared plot interval."""
        if self.data is None:
            return np.array([], dtype=bool)
        if self.time_start is None or self.time_end is None:
            return np.ones(len(self.data.times), dtype=bool)
        if self.time_start == self.time_end and len(self.data.times):
            offsets = np.asarray(
                [
                    abs((value - self.time_start).total_seconds())
                    for value in self.data.times
                ],
                dtype=float,
            )
            mask = np.zeros(len(self.data.times), dtype=bool)
            mask[int(np.nanargmin(offsets))] = True
            return mask
        return np.asarray(
            [self.time_start <= value <= self.time_end for value in self.data.times],
            dtype=bool,
        )

    # ----- coordinate-system selection --------------------------------------

    def _active_coordinate_data(
        self,
    ) -> tuple[str, np.ndarray, tuple[str, str, str], tuple[str, str, str]]:
        """Return the selected coordinate array and component labels."""
        if self.data is None:
            raise ValueError("No magnetic data are loaded.")

        coordinate = self.plot_coordinate_var.get().strip().upper()
        if coordinate == "GEO":
            if self.data.geo is None:
                raise ValueError(
                    "The loaded file does not contain a GEO variable. Choose NEZ "
                    "or open a NetCDF file containing both NEZ and GEO data."
                )
            return (
                "GEO",
                self.data.geo,
                ("X", "Y", "Z"),
                GEO_COMPONENT_PLOT_LABELS,
            )
        return (
            "NEZ",
            self.data.nez,
            ("N", "E", "Z"),
            NEZ_COMPONENT_PLOT_LABELS,
        )

    def on_coordinate_system_change(self, _event=None) -> None:
        """Refresh plots when the user changes between NEZ and GEO."""
        if self.data is None:
            return
        try:
            self._active_coordinate_data()
        except ValueError as exc:
            self.plot_coordinate_var.set("NEZ")
            messagebox.showerror("Plot coordinates", str(exc))
        self.plot_selected_station()
        if self.manual_stack_window is not None and self.manual_stack_window.winfo_exists():
            self._rebuild_manual_stackplot()
        self._refresh_manual_point_selection_window()

    # ----- single-site plot --------------------------------------------------

    def _draw_empty_single_plot(self) -> None:
        self._clear_container(self.single_plot_container)
        self.single_figure = Figure(figsize=(10, 4), dpi=100)
        ax = self.single_figure.add_subplot(111)
        ax.text(0.5, 0.5, "Select a station to display the three selected-coordinate components.", ha="center", va="center")
        ax.set_axis_off()
        self.single_plot_ax = None
        self.single_canvas = FigureCanvasTkAgg(self.single_figure, master=self.single_plot_container)
        self.single_canvas.draw()
        self.single_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.single_toolbar = NavigationToolbar2Tk(
            self.single_canvas, self.single_plot_container, pack_toolbar=False
        )
        self.single_toolbar.update()
        self.single_toolbar.pack(fill=tk.X)

    def plot_selected_station(self) -> None:
        if self.data is None or self.current_station_index is None:
            self._draw_empty_single_plot()
            return

        time_mask = self._selected_time_mask()
        if len(time_mask) != len(self.data.times):
            time_mask = np.ones(len(self.data.times), dtype=bool)

        index = self.current_station_index
        code = self.data.station_codes[index]
        coordinate, coordinate_data, component_symbols, component_long_labels = (
            self._active_coordinate_data()
        )
        plot_times = self.data.times[time_mask]
        values = coordinate_data[index, time_mask, :]

        self._clear_container(self.single_plot_container)
        self.single_figure = Figure(figsize=(11, 4.5), dpi=100)
        ax = self.single_figure.add_subplot(111)
        self.single_plot_ax = ax

        if len(plot_times) == 0:
            ax.text(
                0.5,
                0.5,
                f"{code}: no samples in the selected time range.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            line_specs = (
            ("#1f5aa6", component_long_labels[0]),
            ("#b53434", component_long_labels[1]),
            ("#2e8b57", component_long_labels[2]),)

            for component_index, (color, label) in enumerate(line_specs):
                ax.plot(
                    plot_times,
                    mask_bad_magnetometer_values(values[:, component_index]),
                    color=color,
                    label=label,
                    linewidth=1.2,
                    alpha=0.88,
                    zorder=3,
                )

            # dBh is the horizontal magnitude sqrt(dBn^2 + dBe^2).
            if self.data.dbh is not None:
                dbh_values = self.data.dbh[index, time_mask]
            else:
                dbh_values = calculate_dbh(self.data.nez[index, time_mask, :])

            dbh_values = mask_bad_magnetometer_values(dbh_values)

            dbh_ax = ax.twinx()

            # Put the entire secondary axes behind the component axes.
            dbh_ax.set_zorder(ax.get_zorder() - 1)

            # Make the primary axes background transparent so the grey line remains visible.
            ax.patch.set_visible(False)

            dbh_ax.plot(
                plot_times,
                dbh_values,
                color="lightgrey",
                label="dBh (horizontal)",
                linewidth=1.35,
                alpha=0.92,
                zorder=1,
            )
            dbh_ax.set_ylabel("dBh (nT)", color="black")
            dbh_ax.tick_params(axis="y", colors="black")
            dbh_ax.spines["right"].set_color("black")

            latitude = self.data.glat[index]
            longitude = float(normalize_longitude(self.data.glon[index]))
            ax.set_title(
                f"Raw {coordinate} data and dBh — {code}  "
                f"(lat={latitude:.2f}°, lon={longitude:.2f}°)"
            )
            ax.set_xlabel("Date/time UTC")
            ax.set_ylabel("Magnetic field components (nT)")
            ax.grid(True, alpha=0.3)
            self._draw_selected_substorm_lines(
                ax,
                line_ax=dbh_ax,
                line_color="black",
                line_alpha=0.78,
                line_zorder=0.1,
            )

            primary_handles, primary_labels = ax.get_legend_handles_labels()
            dbh_handles, dbh_labels = dbh_ax.get_legend_handles_labels()
            ax.legend(
                primary_handles + dbh_handles,
                primary_labels + dbh_labels,
                loc="upper right",
                fontsize=8,
            )

            locator = AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
            ax.xaxis.set_minor_locator(NullLocator())
            single_top = 0.82 if self.selected_substorm_events else 0.88
            self.single_figure.subplots_adjust(
                left=0.08, right=0.90, top=single_top, bottom=0.18
            )

        self.single_canvas = FigureCanvasTkAgg(
            self.single_figure, master=self.single_plot_container
        )
        self.single_canvas.draw()
        self.single_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.single_toolbar = NavigationToolbar2Tk(
            self.single_canvas, self.single_plot_container, pack_toolbar=False
        )
        self.single_toolbar.update()
        self.single_toolbar.pack(fill=tk.X)

    # ----- manually assembled station stack ---------------------------------

    def add_selected_station_to_manual_stack(self) -> None:
        """Add the selected station and active time interval as one stack panel."""
        if self.data is None or self.current_station_index is None:
            messagebox.showinfo("Manual stack", "Select a station after loading data.")
            return

        time_mask = self._selected_time_mask()
        if not np.any(time_mask):
            messagebox.showinfo(
                "Manual stack", "No samples fall inside the selected time range."
            )
            return

        station_index = self.current_station_index
        if not self.manual_stack_panels:
            self._sync_manual_time_range_to_main()
        panel = {
            "station_index": int(station_index),
            "station": str(self.data.station_codes[station_index]),
            "glat": float(self.data.glat[station_index]),
            "glon": float(normalize_longitude(self.data.glon[station_index])),
            "selected_points": [],
            "next_point_color_index": 0,
            "ymin": None,
            "ymax": None,
        }
        self.manual_stack_panels.append(panel)
        self._update_manual_stack_status()
        self.show_manual_stackplot()
        self._refresh_manual_point_selection_window()
        self._rebuild_manual_stackplot()

    def show_manual_stackplot(self) -> None:
        """Create or reveal the separate scrollable manual stack window."""
        if self.manual_stack_window is not None and self.manual_stack_window.winfo_exists():
            self.manual_stack_window.deiconify()
            self.manual_stack_window.lift()
            return

        window = tk.Toplevel(self.root)
        window.title("Manually Selected Station Stack Plot")
        window.geometry("1180x650")
        window.minsize(800, 350)
        window.protocol("WM_DELETE_WINDOW", window.withdraw)
        self.manual_stack_window = window

        top = ttk.Frame(window, padding=5)
        top.pack(fill=tk.X)
        ttk.Label(
            top,
            text="Manual station stack",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side=tk.LEFT)
        for component_index, component_name in enumerate(("X", "Y", "Z")):
            ttk.Checkbutton(
                top,
                text=component_name,
                variable=self.manual_component_vars[component_index],
                command=self._rebuild_manual_stackplot,
            ).pack(side=tk.LEFT, padx=(10 if component_index == 0 else 2, 0))
        ttk.Button(
            top,
            textvariable=self.substorm_toggle_button_var,
            command=self.mark_substorm_onset_on_manual_stack,
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(
            top,
            text="Save PNG…",
            command=self.save_manual_stackplot,
        ).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(
            top,
            text="Clear all",
            command=self.clear_manual_stackplot,
        ).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(
            top,
            text="Remove last",
            command=self.remove_last_manual_stack_panel,
        ).pack(side=tk.RIGHT, padx=(4, 0))

        manual_time_frame = ttk.LabelFrame(
            window,
            text="Manual-stack time range (UTC)",
            padding=5,
        )
        manual_time_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        self._build_manual_time_entries(manual_time_frame)

        interaction_frame = ttk.LabelFrame(
            window,
            text="Manual-stack plot controls",
            padding=5,
        )
        interaction_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Checkbutton(
            interaction_frame,
            text="Select and label data points by clicking near a trace",
            variable=self.manual_point_selection_var,
            command=self.on_manual_point_selection_toggle,
        ).pack(side=tk.LEFT)
        ttk.Button(
            interaction_frame,
            textvariable=self.manual_y_scale_button_var,
            command=self.toggle_manual_equal_y_scales,
        ).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(
            interaction_frame,
            textvariable=self.manual_stack_view_button_var,
            command=self.toggle_manual_stack_view,
        ).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(
            interaction_frame,
            text="Remove selected points",
            command=self.remove_selected_manual_data_points,
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(
            interaction_frame,
            text=(
                "Selections appear as colored circles on the plots and as an "
                "organized list in a separate window. Click the same point again "
                "to remove it."
            ),
            foreground="#555555",
        ).pack(side=tk.LEFT, padx=(12, 0))

        station_manager = ttk.LabelFrame(
            window,
            text="Stations in the manual stack",
            padding=5,
        )
        station_manager.pack(fill=tk.X, padx=5, pady=(0, 5))
        station_manager.columnconfigure(0, weight=1)
        station_manager.rowconfigure(0, weight=1)

        tree_frame = ttk.Frame(station_manager)
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.manual_stack_tree = ttk.Treeview(
            tree_frame,
            columns=("station", "latitude", "longitude", "points", "ymin", "ymax"),
            show="headings",
            height=5,
            selectmode="extended",
        )
        self.manual_stack_tree.heading("station", text="Station")
        self.manual_stack_tree.heading("latitude", text="Latitude")
        self.manual_stack_tree.heading("longitude", text="Longitude")
        self.manual_stack_tree.heading("points", text="Points")
        self.manual_stack_tree.heading("ymin", text="ymin")
        self.manual_stack_tree.heading("ymax", text="ymax")
        self.manual_stack_tree.column("station", width=90, anchor="center", stretch=False)
        self.manual_stack_tree.column("latitude", width=90, anchor="e", stretch=False)
        self.manual_stack_tree.column("longitude", width=100, anchor="e", stretch=False)
        self.manual_stack_tree.column("points", width=65, anchor="center", stretch=False)
        self.manual_stack_tree.column("ymin", width=90, anchor="center", stretch=True)
        self.manual_stack_tree.column("ymax", width=90, anchor="center", stretch=True)
        tree_scroll = ttk.Scrollbar(
            tree_frame,
            orient=tk.VERTICAL,
            command=self.manual_stack_tree.yview,
        )
        self.manual_stack_tree.configure(yscrollcommand=tree_scroll.set)
        self.manual_stack_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.manual_stack_tree.bind(
            "<Delete>", lambda _event: self.remove_selected_manual_stack_panels()
        )
        self.manual_stack_tree.bind("<Double-1>", self._edit_manual_y_axis_cell)

        manager_controls = ttk.Frame(station_manager, padding=(8, 0, 0, 0))
        manager_controls.grid(row=0, column=1, sticky="ns")
        ttk.Button(
            manager_controls,
            text="Remove selected",
            command=self.remove_selected_manual_stack_panels,
        ).pack(fill=tk.X, pady=(0, 8))
        ttk.Label(manager_controls, text="Order existing panels").pack(anchor="w")
        ttk.Combobox(
            manager_controls,
            textvariable=self.manual_stack_order_var,
            values=(
                "Latitude (north to south)",
                "Longitude (west to east)",
            ),
            state="readonly",
            width=25,
        ).pack(fill=tk.X, pady=(3, 4))
        ttk.Button(
            manager_controls,
            text="Apply order",
            command=self.order_manual_stack_panels,
        ).pack(fill=tk.X)
        ttk.Button(
            manager_controls,
            text="Apply manual y-axis",
            command=self.apply_manual_y_axes,
        ).pack(fill=tk.X, pady=(4, 0))
        ttk.Label(
            manager_controls,
            text=(
                "Use Ctrl/Shift to select multiple stations.\n"
                "The Delete key also removes the selection."
            ),
            foreground="#555555",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(8, 0))

        container = ttk.Frame(window)
        container.pack(fill=tk.BOTH, expand=True)
        self.manual_stack_scroll_canvas = tk.Canvas(
            container, highlightthickness=0, background="white"
        )
        vertical_scroll = ttk.Scrollbar(
            container,
            orient=tk.VERTICAL,
            command=self.manual_stack_scroll_canvas.yview,
        )
        self.manual_stack_scroll_canvas.configure(yscrollcommand=vertical_scroll.set)
        self.manual_stack_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vertical_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.manual_stack_plot_frame = ttk.Frame(self.manual_stack_scroll_canvas)
        self._manual_stack_frame_window = self.manual_stack_scroll_canvas.create_window(
            (0, 0), window=self.manual_stack_plot_frame, anchor="nw"
        )
        self.manual_stack_plot_frame.bind(
            "<Configure>",
            lambda _event: self.manual_stack_scroll_canvas.configure(
                scrollregion=self.manual_stack_scroll_canvas.bbox("all")
            ),
        )
        self.manual_stack_scroll_canvas.bind(
            "<Configure>",
            lambda event: self.manual_stack_scroll_canvas.itemconfigure(
                self._manual_stack_frame_window, width=event.width
            ),
        )
        self._refresh_manual_stack_tree()
        self._rebuild_manual_stackplot()

    def on_manual_point_selection_toggle(self) -> None:
        """Open or hide the point-list window when selection mode changes."""
        if self.manual_point_selection_var.get():
            self.show_manual_point_selection_window()
        elif (
            self.manual_selection_window is not None
            and self.manual_selection_window.winfo_exists()
        ):
            self.manual_selection_window.withdraw()

    def _close_manual_point_selection_window(self) -> None:
        """Hide the point-list window and turn off point-selection mode."""
        self.manual_point_selection_var.set(False)
        if (
            self.manual_selection_window is not None
            and self.manual_selection_window.winfo_exists()
        ):
            self.manual_selection_window.withdraw()

    def show_manual_point_selection_window(self) -> None:
        """Create or reveal the synchronized manual-stack point-list window."""
        if (
            self.manual_selection_window is not None
            and self.manual_selection_window.winfo_exists()
        ):
            self.manual_selection_window.deiconify()
            self.manual_selection_window.lift()
            self._refresh_manual_point_selection_window()
            return

        window = tk.Toplevel(self.root)
        window.title("Selected manual-stack data points")
        window.geometry("820x680")
        window.minsize(560, 320)
        window.protocol("WM_DELETE_WINDOW", self._close_manual_point_selection_window)
        self.manual_selection_window = window

        top = ttk.Frame(window, padding=7)
        top.pack(fill=tk.X)
        ttk.Label(
            top,
            text="Selected manual-stack data points",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Button(
            top,
            text="Export list as TXT…",
            command=self.export_manual_selected_points_txt,
        ).pack(side=tk.RIGHT)

        ttk.Label(
            window,
            text=(
                "Each subplot has its own section. Point colors match the circle "
                "markers shown on the corresponding plot."
            ),
            foreground="#555555",
            padding=(7, 0, 7, 6),
        ).pack(fill=tk.X)

        container = ttk.Frame(window)
        container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(container, highlightthickness=0, background="white")
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        content = ttk.Frame(canvas, padding=7)
        frame_window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(frame_window, width=event.width),
        )

        self.manual_selection_scroll_canvas = canvas
        self.manual_selection_content_frame = content
        self._manual_selection_frame_window = frame_window
        self._refresh_manual_point_selection_window()

    @staticmethod
    def _manual_point_color(color_index: int) -> str:
        """Return a stable, distinct color for one selected point."""
        color_index = max(int(color_index), 0)
        if color_index < len(MANUAL_POINT_COLOR_PALETTE):
            return MANUAL_POINT_COLOR_PALETTE[color_index]

        # Continue with a golden-ratio hue sequence after the fixed palette.
        hue = (0.618033988749895 * color_index) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
        return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"

    def _ensure_manual_point_colors(self, panel: dict[str, object]) -> None:
        """Assign persistent colors to old or newly loaded point records."""
        points = panel.get("selected_points", [])
        if not isinstance(points, list):
            panel["selected_points"] = []
            panel["next_point_color_index"] = 0
            return

        next_index = max(int(panel.get("next_point_color_index", 0)), 0)
        existing_indices: list[int] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            try:
                existing_indices.append(int(point.get("color_index", -1)))
            except (TypeError, ValueError):
                pass
        if existing_indices:
            next_index = max(next_index, max(existing_indices) + 1)

        for point in points:
            if not isinstance(point, dict):
                continue
            color = str(point.get("color", "")).strip()
            if color:
                continue
            point["color_index"] = next_index
            point["color"] = self._manual_point_color(next_index)
            next_index += 1

        panel["next_point_color_index"] = next_index

    @staticmethod
    def _manual_point_view_mode(point: dict[str, object]) -> str:
        """Return the stored view for a point; old records are component points."""
        mode = str(point.get("view_mode", "components")).strip().lower()
        return "dbh" if mode == "dbh" else "components"

    @classmethod
    def _sorted_manual_points(cls, panel: dict[str, object]) -> list[dict[str, object]]:
        """Return selections ordered by view, component, and then time."""
        points = panel.get("selected_points", [])
        if not isinstance(points, list):
            return []
        valid_points = [point for point in points if isinstance(point, dict)]
        return sorted(
            valid_points,
            key=lambda point: (
                0 if cls._manual_point_view_mode(point) == "components" else 1,
                int(point.get("component_index", -1)),
                int(point.get("time_index", -1)),
            ),
        )

    def _manual_selection_section_title(
        self,
        panel_number: int,
        panel: dict[str, object],
        coordinate: str,
    ) -> str:
        """Build the title used for one subplot section in the list window."""
        station = str(panel.get("station", ""))
        glat = float(panel.get("glat", np.nan))
        glon = float(normalize_longitude(panel.get("glon", np.nan)))
        lat_text = f"{glat:.2f}°" if np.isfinite(glat) else "n/a"
        lon_text = f"{glon:.2f}°" if np.isfinite(glon) else "n/a"
        return (
            f"Subplot {panel_number}: {station}  |  "
            f"lat={lat_text}, lon={lon_text}  |  {coordinate}"
        )

    def _manual_point_display_record(
        self,
        panel: dict[str, object],
        point: dict[str, object],
        coordinate_data: np.ndarray,
        component_long_labels: tuple[str, str, str],
    ) -> Optional[tuple[str, str]]:
        """Return (color, formatted text) for one stored point selection."""
        if self.data is None:
            return None
        try:
            station_index = int(panel.get("station_index", -1))
            time_index = int(point.get("time_index", -1))
            component_index = int(point.get("component_index", -1))
        except (TypeError, ValueError):
            return None
        if not (
            0 <= station_index < coordinate_data.shape[0]
            and 0 <= time_index < coordinate_data.shape[1]
            and time_index < len(self.data.times)
        ):
            return None

        point_view_mode = self._manual_point_view_mode(point)
        if point_view_mode == "dbh":
            if component_index != 0:
                return None
            if self.data.dbh is not None:
                value = float(self.data.dbh[station_index, time_index])
            else:
                value = float(calculate_dbh(self.data.nez[station_index])[time_index])
            component_text = "dBh (horizontal)"
        else:
            if not (
                0 <= component_index < coordinate_data.shape[2]
                and component_index < len(component_long_labels)
            ):
                return None
            value = float(coordinate_data[station_index, time_index, component_index])
            component_text = component_long_labels[component_index]

        value = float(mask_bad_magnetometer_values(np.asarray([value]))[0])
        if not np.isfinite(value):
            return None

        point_time = pd.Timestamp(self.data.times[time_index]).to_pydatetime()
        if point_time.tzinfo is not None:
            point_time = point_time.astimezone(timezone.utc).replace(tzinfo=None)
        color = str(point.get("color", "#000000"))
        text = (
            f"{component_text}, {point_time:%Y-%m-%d %H:%M:%S} UTC, "
            f"{value:.3f} nT"
        )
        return color, text

    def _refresh_manual_point_selection_window(self) -> None:
        """Rebuild all station sections and point rows in the list window."""
        content = self.manual_selection_content_frame
        if (
            content is None
            or not content.winfo_exists()
            or self.manual_selection_window is None
            or not self.manual_selection_window.winfo_exists()
        ):
            return

        for widget in content.winfo_children():
            widget.destroy()

        if self.data is None or not self.manual_stack_panels:
            ttk.Label(
                content,
                text="No manual-stack subplots are currently available.",
                padding=16,
            ).pack(anchor="center")
            return

        try:
            coordinate, coordinate_data, _symbols, component_long_labels = (
                self._active_coordinate_data()
            )
        except ValueError as exc:
            ttk.Label(content, text=str(exc), padding=16).pack(anchor="center")
            return

        for panel_number, panel in enumerate(self.manual_stack_panels, start=1):
            self._ensure_manual_point_colors(panel)
            section = ttk.LabelFrame(
                content,
                text=self._manual_selection_section_title(
                    panel_number, panel, coordinate
                ),
                padding=8,
            )
            section.pack(fill=tk.X, expand=True, pady=(0, 8))

            displayed_count = 0
            for point in self._sorted_manual_points(panel):
                record = self._manual_point_display_record(
                    panel, point, coordinate_data, component_long_labels
                )
                if record is None:
                    continue
                color, point_text = record
                row = ttk.Frame(section)
                row.pack(fill=tk.X, anchor="w", pady=1)
                bullet = tk.Canvas(
                    row,
                    width=18,
                    height=18,
                    highlightthickness=0,
                    background="white",
                )
                bullet.create_oval(
                    4,
                    4,
                    14,
                    14,
                    fill=color,
                    outline="black",
                    width=1,
                )
                bullet.pack(side=tk.LEFT, padx=(0, 5))
                ttk.Label(row, text=point_text).pack(side=tk.LEFT, anchor="w")
                displayed_count += 1

            if displayed_count == 0:
                ttk.Label(
                    section,
                    text="No selected points.",
                    foreground="#666666",
                ).pack(anchor="w")

    def export_manual_selected_points_txt(self) -> None:
        """Export the organized manual-stack selection list as UTF-8 text."""
        if self.data is None or not self.manual_stack_panels:
            messagebox.showinfo(
                "Export selected points",
                "There are no manual-stack subplots to export.",
            )
            return

        try:
            coordinate, coordinate_data, _symbols, component_long_labels = (
                self._active_coordinate_data()
            )
        except ValueError as exc:
            messagebox.showerror("Export selected points", str(exc))
            return

        selected_count = sum(
            len(self._sorted_manual_points(panel))
            for panel in self.manual_stack_panels
        )
        if selected_count == 0:
            messagebox.showinfo(
                "Export selected points",
                "No data points have been selected.",
            )
            return

        initial_name = self._plot_filename(
            f"{coordinate}_manual_stack_selected_points"
        )
        initial_name = str(Path(initial_name).with_suffix(".txt").name)
        path = filedialog.asksaveasfilename(
            title="Export selected manual-stack points",
            defaultextension=".txt",
            filetypes=(("Text file", "*.txt"), ("All files", "*.*")),
            initialfile=initial_name,
        )
        if not path:
            return

        lines = [
            "Manual-stack selected data points",
            f"Coordinate system: {coordinate}",
            "",
        ]
        for panel_number, panel in enumerate(self.manual_stack_panels, start=1):
            self._ensure_manual_point_colors(panel)
            lines.append(
                self._manual_selection_section_title(
                    panel_number, panel, coordinate
                )
            )
            exported_in_section = 0
            for point in self._sorted_manual_points(panel):
                record = self._manual_point_display_record(
                    panel, point, coordinate_data, component_long_labels
                )
                if record is None:
                    continue
                _color, point_text = record
                lines.append(f"  ● {point_text}")
                exported_in_section += 1
            if exported_in_section == 0:
                lines.append("  No selected points.")
            lines.append("")

        try:
            Path(path).write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror(
                "Export selected points",
                f"Could not write the text file:\n\n{exc}",
            )
            return
        messagebox.showinfo(
            "Export selected points",
            f"Selected points were exported to:\n\n{path}",
        )

    def _update_manual_stack_status(self) -> None:
        panel_count = len(self.manual_stack_panels)
        station_count = len(
            {str(panel.get("station", "")) for panel in self.manual_stack_panels}
        )
        panel_word = "panel" if panel_count == 1 else "panels"
        station_word = "station" if station_count == 1 else "stations"
        self.manual_stack_status_var.set(
            f"Manual stack: {panel_count} {panel_word}, "
            f"{station_count} {station_word}"
        )

    def _edit_manual_y_axis_cell(self, event) -> None:
        """Overlay an entry on a double-clicked ymin/ymax Treeview cell."""
        tree = self.manual_stack_tree
        if tree is None:
            return
        item_id = tree.identify_row(event.y)
        column_id = tree.identify_column(event.x)
        field_by_column = {"#5": "ymin", "#6": "ymax"}
        field = field_by_column.get(column_id)
        if not item_id or field is None:
            return
        try:
            panel_index = int(item_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return
        if not 0 <= panel_index < len(self.manual_stack_panels):
            return
        bbox = tree.bbox(item_id, column_id)
        if not bbox:
            return
        x_value, y_value, width, height = bbox
        editor = ttk.Entry(tree)
        current = self.manual_stack_panels[panel_index].get(field)
        editor.insert(0, "" if current is None else str(current))
        editor.select_range(0, tk.END)
        editor.place(x=x_value, y=y_value, width=width, height=height)
        editor.focus_set()

        def commit(_event=None) -> None:
            if not editor.winfo_exists():
                return
            text = editor.get().strip()
            self.manual_stack_panels[panel_index][field] = text or None
            editor.destroy()
            self._refresh_manual_stack_tree()

        editor.bind("<Return>", commit)
        editor.bind("<FocusOut>", commit)
        editor.bind("<Escape>", lambda _event: editor.destroy())

    def apply_manual_y_axes(self) -> None:
        """Validate and apply the per-panel limits entered in the station table."""
        for panel in self.manual_stack_panels:
            station = str(panel.get("station", "station"))
            ymin_text = panel.get("ymin")
            ymax_text = panel.get("ymax")
            if ymin_text in (None, "") and ymax_text in (None, ""):
                panel["ymin"] = None
                panel["ymax"] = None
                continue
            if ymin_text in (None, "") or ymax_text in (None, ""):
                messagebox.showerror(
                    "Manual y-axis",
                    f"Enter both ymin and ymax for {station}, or leave both blank.",
                )
                return
            try:
                ymin = float(ymin_text)
                ymax = float(ymax_text)
            except (TypeError, ValueError):
                messagebox.showerror(
                    "Manual y-axis", f"The ymin and ymax values for {station} must be numbers."
                )
                return
            if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
                messagebox.showerror(
                    "Manual y-axis", f"For {station}, ymax must be a finite number greater than ymin."
                )
                return
            panel["ymin"] = ymin
            panel["ymax"] = ymax
        self._refresh_manual_stack_tree()
        self._rebuild_manual_stackplot()

    def _refresh_manual_stack_tree(self) -> None:
        """Synchronize the selectable station table with the panel sequence."""
        tree = self.manual_stack_tree
        if tree is None or not tree.winfo_exists():
            return

        previous_selection = set(tree.selection())
        for item in tree.get_children():
            tree.delete(item)

        for panel_index, panel in enumerate(self.manual_stack_panels):
            station = str(panel.get("station", ""))
            glat = float(panel.get("glat", np.nan))
            glon = float(normalize_longitude(panel.get("glon", np.nan)))
            lat_text = f"{glat:.2f}°" if np.isfinite(glat) else "n/a"
            lon_text = f"{glon:.2f}°" if np.isfinite(glon) else "n/a"
            selected_points = panel.get("selected_points", [])
            point_count = len(selected_points) if isinstance(selected_points, list) else 0
            ymin = panel.get("ymin")
            ymax = panel.get("ymax")
            ymin_text = "" if ymin is None else str(ymin)
            ymax_text = "" if ymax is None else str(ymax)

            item_id = f"panel_{panel_index}"
            tree.insert(
                "",
                tk.END,
                iid=item_id,
                values=(station, lat_text, lon_text, point_count, ymin_text, ymax_text),
            )
            if item_id in previous_selection:
                tree.selection_add(item_id)

    def clear_manual_stackplot(self) -> None:
        self.manual_stack_panels = []
        self._manual_last_clicked_panel_index = None
        self._update_manual_stack_status()
        self._refresh_manual_stack_tree()
        self._refresh_manual_point_selection_window()
        self._rebuild_manual_stackplot()

    def remove_last_manual_stack_panel(self) -> None:
        if self.manual_stack_panels:
            self.manual_stack_panels.pop()
        self._manual_last_clicked_panel_index = None
        self._update_manual_stack_status()
        self._refresh_manual_stack_tree()
        self._refresh_manual_point_selection_window()
        self._rebuild_manual_stackplot()

    def remove_selected_manual_stack_panels(self) -> None:
        """Remove one or more panels selected in the manual-stack station table."""
        tree = self.manual_stack_tree
        if tree is None or not tree.winfo_exists():
            return

        selected_items = tree.selection()
        if not selected_items:
            messagebox.showinfo(
                "Manual stack",
                "Select one or more stations in the manual-stack table first.",
            )
            return

        selected_indices: list[int] = []
        for item_id in selected_items:
            try:
                selected_indices.append(int(item_id.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue

        for panel_index in sorted(set(selected_indices), reverse=True):
            if 0 <= panel_index < len(self.manual_stack_panels):
                del self.manual_stack_panels[panel_index]

        self._manual_last_clicked_panel_index = None
        self._update_manual_stack_status()
        self._refresh_manual_stack_tree()
        self._refresh_manual_point_selection_window()
        self._rebuild_manual_stackplot()

    def order_manual_stack_panels(self) -> None:
        """Reorder current manual panels by station latitude or longitude."""
        if len(self.manual_stack_panels) < 2:
            self._refresh_manual_stack_tree()
            return

        mode = self.manual_stack_order_var.get()
        indexed_panels = list(enumerate(self.manual_stack_panels))

        if mode.startswith("Latitude"):
            def sort_key(item):
                original_index, panel = item
                latitude = float(panel.get("glat", np.nan))
                station = str(panel.get("station", ""))
                return (
                    0 if np.isfinite(latitude) else 1,
                    -latitude if np.isfinite(latitude) else 0.0,
                    station,
                    original_index,
                )
        elif mode.startswith("Longitude"):
            def sort_key(item):
                original_index, panel = item
                longitude = float(normalize_longitude(panel.get("glon", np.nan)))
                station = str(panel.get("station", ""))
                return (
                    0 if np.isfinite(longitude) else 1,
                    longitude if np.isfinite(longitude) else 0.0,
                    station,
                    original_index,
                )
        else:
            return

        self.manual_stack_panels = [
            panel for _original_index, panel in sorted(indexed_panels, key=sort_key)
        ]
        self._manual_last_clicked_panel_index = None
        self._update_manual_stack_status()
        self._refresh_manual_stack_tree()
        self._refresh_manual_point_selection_window()
        self._rebuild_manual_stackplot()

    def toggle_manual_equal_y_scales(self) -> None:
        """Toggle between independent and common y-axis limits."""
        self.manual_equal_y_scale = not self.manual_equal_y_scale
        self.manual_y_scale_button_var.set(
            "Use individual y-axis scales"
            if self.manual_equal_y_scale
            else "Use same y-axis scale"
        )
        self._rebuild_manual_stackplot()

    def toggle_manual_stack_view(self) -> None:
        """Switch every manual-stack panel between components and dBh."""
        current_mode = self.manual_stack_view_mode_var.get().strip().lower()
        new_mode = "components" if current_mode == "dbh" else "dbh"
        self.manual_stack_view_mode_var.set(new_mode)
        self.manual_stack_view_button_var.set(
            "Show components" if new_mode == "dbh" else "Show dBh"
        )
        self._rebuild_manual_stackplot()
        self._refresh_manual_point_selection_window()

    def remove_selected_manual_data_points(self) -> None:
        """Clear stored point selections from selected manual-stack subplots."""
        selected_indices: list[int] = []
        tree = self.manual_stack_tree
        if tree is not None and tree.winfo_exists():
            for item_id in tree.selection():
                try:
                    selected_indices.append(int(item_id.rsplit("_", 1)[1]))
                except (IndexError, ValueError):
                    continue

        if not selected_indices and self._manual_last_clicked_panel_index is not None:
            selected_indices = [self._manual_last_clicked_panel_index]

        selected_indices = sorted(set(selected_indices))
        if not selected_indices:
            messagebox.showinfo(
                "Manual stack",
                "Select one or more subplot rows in the station table, or click a subplot first.",
            )
            return

        changed = False
        for panel_index in selected_indices:
            if not (0 <= panel_index < len(self.manual_stack_panels)):
                continue
            panel = self.manual_stack_panels[panel_index]
            points = panel.get("selected_points", [])
            if isinstance(points, list) and points:
                panel["selected_points"] = []
                panel["next_point_color_index"] = 0
                changed = True
                self._refresh_manual_selected_point_artists(panel_index)

        if changed:
            self._refresh_manual_stack_tree()
            self._refresh_manual_point_selection_window()
            if self.manual_stack_canvas is not None:
                self.manual_stack_canvas.draw_idle()

    def _select_manual_panel_in_tree(self, panel_index: int) -> None:
        """Select the station-table row corresponding to a clicked subplot."""
        tree = self.manual_stack_tree
        if tree is None or not tree.winfo_exists():
            return
        item_id = f"panel_{panel_index}"
        if tree.exists(item_id):
            tree.selection_set(item_id)
            tree.focus(item_id)
            tree.see(item_id)

    def _nearest_manual_data_point(
        self,
        event,
        panel_index: int,
        maximum_distance_pixels: float = 24.0,
    ) -> Optional[tuple[int, int]]:
        """Return the nearest visible point as (global time index, component)."""
        if not (0 <= panel_index < len(self._manual_stack_plot_data)):
            return None
        if event.x is None or event.y is None:
            return None

        plot_data = self._manual_stack_plot_data[panel_index]
        ax = plot_data.get("axis")
        times_num = np.asarray(plot_data.get("times_num", []), dtype=float)
        global_indices = np.asarray(plot_data.get("global_indices", []), dtype=int)
        values = np.asarray(plot_data.get("values", []), dtype=float)
        if ax is None or len(times_num) == 0 or values.ndim != 2:
            return None

        click_xy = np.asarray([float(event.x), float(event.y)], dtype=float)
        best_distance_squared = float(maximum_distance_pixels) ** 2
        best_result: Optional[tuple[int, int]] = None

        component_count = min(values.shape[1], 3)
        for component_index in range(component_count):
            component_values = values[:, component_index]
            finite = np.isfinite(times_num) & np.isfinite(component_values)
            if not np.any(finite):
                continue
            local_indices = np.flatnonzero(finite)
            data_xy = np.column_stack(
                (times_num[local_indices], component_values[local_indices])
            )
            display_xy = ax.transData.transform(data_xy)
            distances_squared = np.sum((display_xy - click_xy) ** 2, axis=1)
            nearest_position = int(np.argmin(distances_squared))
            distance_squared = float(distances_squared[nearest_position])
            if distance_squared <= best_distance_squared:
                local_index = int(local_indices[nearest_position])
                best_distance_squared = distance_squared
                best_result = (
                    int(global_indices[local_index]),
                    int(component_index),
                )

        return best_result

    def _on_manual_stack_click(self, event) -> None:
        """Add or remove a point label in the clicked manual-stack subplot."""
        if not self.manual_point_selection_var.get() or event.button != 1:
            return
        if event.inaxes is None:
            return

        panel_index = getattr(event.inaxes, "_manual_panel_index", None)
        if panel_index is None:
            return
        panel_index = int(panel_index)
        self._manual_last_clicked_panel_index = panel_index
        self._select_manual_panel_in_tree(panel_index)

        nearest = self._nearest_manual_data_point(event, panel_index)
        if nearest is None:
            return
        time_index, component_index = nearest

        panel = self.manual_stack_panels[panel_index]
        stored_points = panel.get("selected_points", [])
        if not isinstance(stored_points, list):
            stored_points = []

        current_view_mode = self.manual_stack_view_mode_var.get().strip().lower()
        if current_view_mode != "dbh":
            current_view_mode = "components"

        matching_position = None
        for point_position, point in enumerate(stored_points):
            if not isinstance(point, dict):
                continue
            if (
                self._manual_point_view_mode(point) == current_view_mode
                and int(point.get("time_index", -1)) == time_index
                and int(point.get("component_index", -1)) == component_index
            ):
                matching_position = point_position
                break

        if matching_position is None:
            color_index = max(int(panel.get("next_point_color_index", 0)), 0)
            stored_points.append(
                {
                    "time_index": int(time_index),
                    "component_index": int(component_index),
                    "view_mode": current_view_mode,
                    "color_index": color_index,
                    "color": self._manual_point_color(color_index),
                }
            )
            panel["next_point_color_index"] = color_index + 1
            stored_points.sort(
                key=lambda point: (
                    int(point.get("component_index", -1)),
                    int(point.get("time_index", -1)),
                )
            )
        else:
            del stored_points[matching_position]

        panel["selected_points"] = stored_points
        self._refresh_manual_selected_point_artists(panel_index)
        self._refresh_manual_stack_tree()
        self._refresh_manual_point_selection_window()
        if self.manual_stack_canvas is not None:
            self.manual_stack_canvas.draw_idle()

    def _refresh_manual_selected_point_artists(self, panel_index: int) -> None:
        """Redraw stored color-coded point markers for one subplot."""
        for artist in self._manual_point_artists.get(panel_index, []):
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._manual_point_artists[panel_index] = []

        if not (
            0 <= panel_index < len(self.manual_stack_panels)
            and 0 <= panel_index < len(self._manual_stack_plot_data)
        ):
            return

        panel = self.manual_stack_panels[panel_index]
        self._ensure_manual_point_colors(panel)
        stored_points = self._sorted_manual_points(panel)
        if not stored_points:
            return

        plot_data = self._manual_stack_plot_data[panel_index]
        ax = plot_data.get("axis")
        global_indices = np.asarray(plot_data.get("global_indices", []), dtype=int)
        plot_times = np.asarray(plot_data.get("plot_times", []), dtype=object)
        values = np.asarray(plot_data.get("values", []), dtype=float)
        plot_view_mode = str(plot_data.get("view_mode", "components")).strip().lower()
        if plot_view_mode != "dbh":
            plot_view_mode = "components"
        if ax is None or len(global_indices) == 0 or values.ndim != 2:
            return

        local_lookup = {
            int(global_index): int(local_index)
            for local_index, global_index in enumerate(global_indices)
        }
        artists: list[object] = []
        for point in stored_points:
            if self._manual_point_view_mode(point) != plot_view_mode:
                continue
            time_index = int(point.get("time_index", -1))
            component_index = int(point.get("component_index", -1))
            local_index = local_lookup.get(time_index)
            if local_index is None or not (0 <= component_index < values.shape[1]):
                continue
            value = float(values[local_index, component_index])
            if not np.isfinite(value):
                continue
            point_time = plot_times[local_index]
            color = str(point.get("color", "#000000"))

            marker, = ax.plot(
                [point_time],
                [value],
                marker="o",
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.9,
                linestyle="None",
                zorder=9,
            )
            artists.append(marker)

        self._manual_point_artists[panel_index] = artists

    def _rebuild_manual_stackplot(self) -> None:
        self._refresh_manual_stack_tree()
        if (
            self.manual_stack_plot_frame is None
            or not self.manual_stack_plot_frame.winfo_exists()
        ):
            return

        for widget in self.manual_stack_plot_frame.winfo_children():
            widget.destroy()

        self._manual_stack_axes = []
        self._manual_stack_plot_data = []
        self._manual_point_artists = {}
        self._manual_stack_click_cid = None

        panel_count = len(self.manual_stack_panels)
        if panel_count == 0:
            self.manual_stack_figure = None
            self.manual_stack_canvas = None
            ttk.Label(
                self.manual_stack_plot_frame,
                text=(
                    "Select a station and time range, then use "
                    "“Add selected station to stack →”."
                ),
                padding=20,
            ).pack(anchor="center")
            self._refresh_manual_point_selection_window()
            return

        figure_height = self.MANUAL_PANEL_HEIGHT_IN * panel_count
        figure = Figure(
            figsize=(self.MANUAL_PANEL_WIDTH_IN, figure_height),
            dpi=100,
        )
        axes = figure.subplots(panel_count, 1, sharex=True)
        if panel_count == 1:
            axes = [axes]

        coordinate, coordinate_data, component_symbols, component_long_labels = (
            self._active_coordinate_data()
        )
        manual_view_mode = self.manual_stack_view_mode_var.get().strip().lower()
        if manual_view_mode != "dbh":
            manual_view_mode = "components"
        dbh_view = manual_view_mode == "dbh"
        component_enabled = tuple(var.get() for var in self.manual_component_vars)
        line_specs = (
            ((0, "#333333", "dBh (horizontal)"),)
            if dbh_view
            else tuple(
                (index, color, component_long_labels[index])
                for index, color in enumerate(("#1f5aa6", "#b53434", "#2e8b57"))
                if component_enabled[index]
            )
        )
        time_mask = self._manual_selected_time_mask()
        global_indices = np.flatnonzero(time_mask)
        plot_times = np.asarray(self.data.times[time_mask], dtype=object)
        plot_times_num = (
            np.asarray(date2num(list(plot_times)), dtype=float)
            if len(plot_times)
            else np.asarray([], dtype=float)
        )
        range_start = self.manual_time_start
        range_end = self.manual_time_end
        if isinstance(range_start, datetime) and isinstance(range_end, datetime):
            range_text = (
                f"{range_start:%Y-%m-%d %H:%M:%S} to "
                f"{range_end:%Y-%m-%d %H:%M:%S} UTC"
            )
        else:
            range_text = "selected interval"

        self._manual_stack_axes = list(axes)
        all_finite_values: list[np.ndarray] = []
        component_colors = (
            ("#333333",)
            if dbh_view
            else tuple(
                color if component_enabled[index] else ""
                for index, color in enumerate(("#1f5aa6", "#b53434", "#2e8b57"))
            )
        )
        plotted_symbols = ("dBh",) if dbh_view else tuple(component_symbols)

        for panel_index, (ax, panel) in enumerate(
            zip(axes, self.manual_stack_panels)
        ):
            ax._manual_panel_index = panel_index
            station_index = int(panel.get("station_index", -1))
            station = str(panel.get("station", ""))
            glat = float(panel.get("glat", np.nan))
            glon = float(normalize_longitude(panel.get("glon", np.nan)))
            value_columns = 1 if dbh_view else 3
            values = np.full((len(plot_times), value_columns), np.nan, dtype=float)

            if (
                station_index < 0
                or station_index >= len(self.data.station_codes)
                or len(plot_times) == 0
            ):
                ax.text(
                    0.5,
                    0.5,
                    f"{station}: no samples in the selected time range.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                if dbh_view:
                    if self.data.dbh is not None:
                        dbh_values = self.data.dbh[station_index, time_mask]
                    else:
                        dbh_values = calculate_dbh(
                            self.data.nez[station_index, time_mask, :]
                        )
                    values = mask_bad_magnetometer_values(
                        np.asarray(dbh_values, dtype=float)
                    ).reshape(-1, 1)
                else:
                    values = np.asarray(
                        coordinate_data[station_index, time_mask, :], dtype=float
                    )

                if not dbh_view:
                    for component_index, enabled in enumerate(component_enabled):
                        if not enabled:
                            values[:, component_index] = np.nan
                for component_index, color, label in line_specs:
                    component_values = mask_bad_magnetometer_values(
                        values[:, component_index]
                    )
                    values[:, component_index] = component_values
                    ax.plot(
                        plot_times,
                        component_values,
                        color=color,
                        label=label,
                        linewidth=1.15 if dbh_view else 1.05,
                        alpha=0.92 if dbh_view else 0.88,
                        zorder=2,
                    )
                ax.legend(
                    loc="upper right",
                    fontsize=7,
                    ncol=1 if dbh_view else max(1, len(line_specs)),
                )

            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                all_finite_values.append(finite_values)

            self._manual_stack_plot_data.append(
                {
                    "axis": ax,
                    "global_indices": global_indices.copy(),
                    "plot_times": plot_times.copy(),
                    "times_num": plot_times_num.copy(),
                    "values": values.copy(),
                    "component_symbols": plotted_symbols,
                    "component_colors": component_colors,
                    "view_mode": manual_view_mode,
                }
            )

            lat_text = f"{glat:.2f}°" if np.isfinite(glat) else "n/a"
            lon_text = f"{glon:.2f}°" if np.isfinite(glon) else "n/a"
            quantity_label = "dBh (nT)" if dbh_view else "dB (nT)"
            ax.set_ylabel(
                f"{station}, {quantity_label}\n{lat_text} N, {lon_text} E",
                fontsize=8,
            )
            ax.grid(True, alpha=0.3)
            self._draw_selected_substorm_lines(
                ax,
                show_numbers=(panel_index % 3 == 0),
                line_color="gold",
                line_alpha=0.9,
                line_zorder=0.35,
            )
            ax.xaxis.set_minor_locator(NullLocator())

        if self.manual_equal_y_scale and all_finite_values:
            combined_values = np.concatenate(all_finite_values)
            common_min = float(np.nanmin(combined_values))
            common_max = float(np.nanmax(combined_values))
            if np.isclose(common_min, common_max):
                padding = max(abs(common_min) * 0.05, 1.0)
            else:
                padding = (common_max - common_min) * 0.05
            for ax in axes:
                ax.set_ylim(common_min - padding, common_max + padding)

        # Explicit per-station limits take precedence over automatic/equal scaling.
        for ax, panel in zip(axes, self.manual_stack_panels):
            ymin = panel.get("ymin")
            ymax = panel.get("ymax")
            if isinstance(ymin, (int, float)) and isinstance(ymax, (int, float)):
                if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                    ax.set_ylim(float(ymin), float(ymax))

        for panel_index in range(panel_count):
            self._refresh_manual_selected_point_artists(panel_index)

        axes[-1].set_xlabel("Date/time UTC")
        locator = AutoDateLocator()
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(ConciseDateFormatter(locator))
        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)

        top_margin_in = 0.28
        bottom_margin_in = 0.55
        figure.subplots_adjust(
            left=0.085,
            right=0.98,
            top=1.0 - top_margin_in / figure_height,
            bottom=bottom_margin_in / figure_height,
            hspace=0.10 if self.selected_substorm_events else 0.06,
        )

        canvas = FigureCanvasTkAgg(figure, master=self.manual_stack_plot_frame)
        self._manual_stack_click_cid = canvas.mpl_connect(
            "button_press_event", self._on_manual_stack_click
        )
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(height=max(260, int(figure_height * 100)))
        widget.pack(fill=tk.BOTH, expand=True)

        self.manual_stack_figure = figure
        self.manual_stack_canvas = canvas
        self._refresh_manual_point_selection_window()
        self._resize_manual_stack_window(panel_count)

    def _resize_manual_stack_window(self, panel_count: int) -> None:
        if self.manual_stack_window is None or not self.manual_stack_window.winfo_exists():
            return
        target_height = min(330 + panel_count * 180, 960)
        self.manual_stack_window.geometry(f"1180x{max(520, target_height)}")

    def save_manual_stackplot(self) -> None:
        if self.manual_stack_figure is None:
            messagebox.showinfo("Save manual stack", "There is no manual stack to save.")
            return
        self.save_figure_png(
            self.manual_stack_figure,
            self._plot_filename(
                (
                    "dBh_manual_station_stack"
                    if self.manual_stack_view_mode_var.get().strip().lower() == "dbh"
                    else f"{self.plot_coordinate_var.get().strip().upper()}_manual_station_stack"
                )
            ),
        )

    # ----- magnetic parameter maps ------------------------------------------

    def _parse_map_time_entries(self) -> tuple[datetime, datetime]:
        """Validate the independent magnetic-map start/end fields."""
        start_text = self.map_start_entry_var.get().strip()
        end_text = self.map_end_entry_var.get().strip()
        try:
            start = datetime.strptime(start_text, MANUAL_TIME_FORMAT)
            end = datetime.strptime(end_text, MANUAL_TIME_FORMAT)
        except ValueError as exc:
            raise ValueError(
                "Enter both map times in the format YYYY-MM-DD HH:mm:SS."
            ) from exc
        if end < start:
            raise ValueError("The map end time cannot be earlier than the start time.")
        if self.data is None or len(self.data.times) == 0:
            raise ValueError("No magnetic data are loaded.")
        data_start = min(self.data.times)
        data_end = max(self.data.times)
        if start < data_start or start > data_end or end < data_start or end > data_end:
            raise ValueError(
                "Map times must stay within the loaded data interval:\n"
                f"{data_start.strftime(MANUAL_TIME_FORMAT)} to "
                f"{data_end.strftime(MANUAL_TIME_FORMAT)} UTC"
            )
        return start, end

    def _parse_plot_time_entries(self) -> tuple[datetime, datetime]:
        """Backward-compatible alias for the magnetic-map time parser."""
        return self._parse_map_time_entries()

    def _synchronize_main_plot_time_range(
        self, start: datetime, end: datetime
    ) -> None:
        """Backward-compatible helper for setting the slider-backed interval."""
        self.time_start = start
        self.time_end = end
        if self.data_min_time is not None and self.data_max_time is not None:
            total_seconds = (self.data_max_time - self.data_min_time).total_seconds()
            if total_seconds > 0:
                self._start_frac = max(
                    0.0,
                    min(1.0, (start - self.data_min_time).total_seconds() / total_seconds),
                )
                self._end_frac = max(
                    self._start_frac,
                    min(1.0, (end - self.data_min_time).total_seconds() / total_seconds),
                )
            else:
                self._start_frac = 0.0
                self._end_frac = 1.0
        self._redraw_time_slider_handles()
        self.single_time_range_var.set(
            f"{start.strftime(MANUAL_TIME_FORMAT)}  →  "
            f"{end.strftime(MANUAL_TIME_FORMAT)} UTC"
        )

    def apply_map_time_entries(self, _event=None) -> bool:
        """Validate map times without changing single-site or stack intervals."""
        try:
            start, end = self._parse_map_time_entries()
        except ValueError as exc:
            messagebox.showerror("Map time range", str(exc))
            return False
        self.map_status_var.set(
            f"Map interval: {start.strftime(MANUAL_TIME_FORMAT)} → "
            f"{end.strftime(MANUAL_TIME_FORMAT)} UTC"
        )
        return True

    def apply_plot_time_entries(self, _event=None) -> bool:
        """Backward-compatible alias for applying the independent map times."""
        return self.apply_map_time_entries(_event)

    def _map_region(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Return station indices and geographic bounds for the requested map."""
        if self.data is None:
            raise ValueError("No magnetic data are loaded.")

        try:
            lat_min = float(self.stack_lat_min_var.get())
            lat_max = float(self.stack_lat_max_var.get())
            lon_west = float(self.stack_lon_west_var.get())
            lon_east = float(self.stack_lon_east_var.get())
        except ValueError as exc:
            raise ValueError("Map latitude and longitude bounds must be numeric.") from exc

        lat_lo, lat_hi = sorted((lat_min, lat_max))
        if lat_lo < -90.0 or lat_hi > 90.0:
            raise ValueError("Map latitudes must be between -90 and 90 degrees.")
        lon_west = float(normalize_longitude(lon_west))
        lon_east = float(normalize_longitude(lon_east))

        lats = np.asarray(self.data.glat, dtype=float)
        lons = normalize_longitude(self.data.glon)
        mask = np.isfinite(lats) & np.isfinite(lons)
        mask &= (lats >= lat_lo) & (lats <= lat_hi)
        if lon_west <= lon_east:
            mask &= (lons >= lon_west) & (lons <= lon_east)
        else:
            mask &= (lons >= lon_west) | (lons <= lon_east)
        indices = np.flatnonzero(mask)
        if len(indices) < 3:
            raise ValueError(
                "At least three stations with finite coordinates are required "
                "inside the selected map region."
            )
        return indices, (lat_lo, lat_hi, lon_west, lon_east)

    def _map_parameter_data(self) -> tuple[np.ndarray, str, str, bool]:
        """Return all-station map data, label, filename token, and signed flag."""
        if self.data is None:
            raise ValueError("No magnetic data are loaded.")
        selection = self.map_parameter_var.get().strip()
        if selection == "dBn":
            return mask_bad_magnetometer_values(self.data.nez[:, :, 0]), "dBn (nT)", "dBn", True
        if selection == "dBe":
            return mask_bad_magnetometer_values(self.data.nez[:, :, 1]), "dBe (nT)", "dBe", True
        if selection == "dBz":
            return mask_bad_magnetometer_values(self.data.nez[:, :, 2]), "dBz (nT)", "dBz", True
        if selection == "dBh":
            dbh = self.data.dbh if self.data.dbh is not None else calculate_dbh(self.data.nez)
            return mask_bad_magnetometer_values(dbh), "dBh (nT)", "dBh", False
        if selection == "dBz/dt":
            values = np.asarray(
                [finite_time_derivative(row, self.data.times) for row in self.data.nez[:, :, 2]],
                dtype=float,
            )
            return values, "dBz/dt (nT/min)", "dBz_dt", True
        if selection == "dBh/dt":
            dbh = self.data.dbh if self.data.dbh is not None else calculate_dbh(self.data.nez)
            return calculate_dbh_dt(dbh, self.data.times), "dBh/dt (nT/min)", "dBh_dt", True
        raise ValueError("Map parameter must be dBn, dBe, dBz, dBh, dBz/dt, or dBh/dt.")

    def _map_color_limits(self, value_points: np.ndarray) -> tuple[float, float]:
        """Return automatic or user-entered map color limits."""
        mode = self.map_color_scale_mode_var.get().strip().lower()
        finite_values = np.asarray(value_points, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            raise ValueError("No finite station values are available for the map color scale.")

        if mode == "manual":
            cmin_text = self.map_cmin_var.get().strip()
            cmax_text = self.map_cmax_var.get().strip()
            if not cmin_text or not cmax_text:
                raise ValueError("Enter both cmin and cmax for the manual map color scale.")
            try:
                cmin = float(cmin_text)
                cmax = float(cmax_text)
            except ValueError as exc:
                raise ValueError("Map cmin and cmax must be numeric.") from exc
            if not np.isfinite(cmin) or not np.isfinite(cmax):
                raise ValueError("Map cmin and cmax must be finite numbers.")
            if cmin >= cmax:
                raise ValueError("Map cmin must be smaller than cmax.")
            return cmin, cmax

        if mode != "automatic":
            raise ValueError("Map color scale mode must be automatic or manual.")

        cmin = float(np.nanmin(finite_values))
        cmax = float(np.nanmax(finite_values))
        if cmin == cmax:
            padding = max(abs(cmin) * 0.05, 1.0)
            cmin -= padding
            cmax += padding
        return cmin, cmax

    def _nearest_map_time_index(self, requested_time: datetime) -> int:
        if self.data is None or len(self.data.times) == 0:
            raise ValueError("No magnetic data are loaded.")
        offsets = np.asarray(
            [abs((value - requested_time).total_seconds()) for value in self.data.times],
            dtype=float,
        )
        return int(np.nanargmin(offsets))

    def _map_output_directory(self) -> Path:
        if self.data is not None and self.data.source_path:
            return Path(self.data.source_path).expanduser().resolve().parent
        output_text = self.output_file_var.get().strip()
        if output_text:
            return Path(output_text).expanduser().resolve().parent
        return Path.cwd()

    def _create_component_map_figure(
        self,
        values_all_stations: np.ndarray,
        time_index: int,
        station_indices: np.ndarray,
        bounds: tuple[float, float, float, float],
        colorbar_label: str,
        signed_values: bool,
    ) -> Figure:
        if self.data is None:
            raise ValueError("No magnetic data are loaded.")
        if not HAS_CARTOPY:
            raise RuntimeError(
                "Cartopy is required for magnetic parameter maps. Install cartopy "
                "in the Python environment used to run this program."
            )
        if not HAS_SCIPY or griddata is None:
            raise RuntimeError(
                "SciPy is required for map interpolation. Install scipy in the "
                "Python environment used to run this program."
            )

        lat_lo, lat_hi, lon_west, lon_east = bounds
        station_lats = np.asarray(self.data.glat[station_indices], dtype=float)
        station_lons = normalize_longitude(self.data.glon[station_indices])
        station_values = mask_bad_magnetometer_values(
            np.asarray(values_all_stations[station_indices, time_index], dtype=float)
        )

        crosses_dateline = lon_west > lon_east
        if crosses_dateline:
            lon_plot = np.asarray(station_lons, dtype=float).copy()
            lon_plot[lon_plot < lon_west] += 360.0
            grid_west = lon_west
            grid_east = lon_east + 360.0
        else:
            lon_plot = np.asarray(station_lons, dtype=float)
            grid_west = lon_west
            grid_east = lon_east

        center_unwrapped = 0.5 * (grid_west + grid_east)
        lon_native = lon_plot - center_unwrapped
        native_west = grid_west - center_unwrapped
        native_east = grid_east - center_unwrapped

        finite = (
            np.isfinite(station_lats)
            & np.isfinite(lon_native)
            & np.isfinite(station_values)
        )
        if np.count_nonzero(finite) < 3:
            raise ValueError(
                "At least three finite station values are required at this time "
                "for map interpolation."
            )

        lon_points = lon_native[finite]
        lat_points = station_lats[finite]
        value_points = station_values[finite]
        code_points = self.data.station_codes[station_indices][finite]

        lon_span = max(native_east - native_west, 0.1)
        lat_span = max(lat_hi - lat_lo, 0.1)
        lon_step = max(0.25, lon_span / 260.0)
        lat_step = max(0.25, lat_span / 220.0)
        lon_grid = np.arange(native_west, native_east + lon_step, lon_step)
        lat_grid = np.arange(lat_lo, lat_hi + lat_step, lat_step)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        interpolated = griddata(
            np.column_stack([lon_points, lat_points]),
            value_points,
            (lon_mesh, lat_mesh),
            method="linear",
        )

        # Use a diverging colormap for signed quantities and a sequential
        # colormap for the nonnegative dBh magnitude. Automatic limits follow
        # the finite station values at the selected time; manual limits are
        # read from the interface and validated centrally.
        vmin, vmax = self._map_color_limits(value_points)
        cmap = "RdBu_r" if signed_values else "plasma"

        central_longitude = float(normalize_longitude(center_unwrapped))
        projection = ccrs.PlateCarree(central_longitude=central_longitude)
        data_crs = ccrs.PlateCarree()
        figure = Figure(figsize=(10.5, 7.5), dpi=100)
        ax = figure.add_subplot(111, projection=projection)
        ax.set_extent([native_west, native_east, lat_lo, lat_hi], crs=projection)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="black", linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor="white")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.7)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS, alpha=0.3)
        gridliner = ax.gridlines(
            crs=data_crs,
            draw_labels=True,
            linewidth=0.5,
            alpha=0.5,
            linestyle="--",
        )
        gridliner.top_labels = False
        gridliner.right_labels = False

        mappable = ax.pcolormesh(
            lon_mesh,
            lat_mesh,
            interpolated,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            transform=projection,
            zorder=1,
        )
        ax.scatter(
            lon_points,
            lat_points,
            c=value_points,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=34,
            edgecolors="black",
            linewidths=0.45,
            transform=projection,
            zorder=3,
        )
        for lon, lat, code in zip(lon_points, lat_points, code_points):
            ax.text(
                lon,
                lat,
                str(code),
                fontsize=7,
                ha="left",
                va="bottom",
                transform=projection,
                zorder=4,
            )

        sample_time = self.data.times[time_index]
        ax.set_title(
            f"{self.map_parameter_var.get()} at "
            f"{sample_time.strftime(MANUAL_TIME_FORMAT)} UTC "
            f"({np.count_nonzero(finite)} stations)"
        )
        colorbar = figure.colorbar(
            mappable,
            ax=ax,
            pad=0.035,
            shrink=0.88,
            extend="both",
        )
        colorbar.set_label(colorbar_label)
        figure.subplots_adjust(left=0.06, right=0.9, top=0.91, bottom=0.08)
        return figure

    def _map_filename(self, token: str, time_index: int) -> str:
        if self.data is None:
            timestamp = datetime.utcnow()
        else:
            timestamp = self.data.times[time_index]
        return f"{timestamp:%Y%m%d_%H%M%S}_{token}_map.png"

    def _display_component_map(self, figure: Figure, default_name: str) -> None:
        window = tk.Toplevel(self.root)
        window.title(f"SuperMAG {self.map_parameter_var.get()} map")
        window.geometry("1120x820")
        window.minsize(800, 600)
        self._plot_windows.append(window)

        top = ttk.Frame(window, padding=4)
        top.pack(fill=tk.X)
        ttk.Label(
            top,
            text=f"{self.map_parameter_var.get()} map",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Button(
            top,
            text="Export as PNG…",
            command=lambda: self.save_figure_png(figure, default_name),
        ).pack(side=tk.RIGHT)

        canvas_frame = ttk.Frame(window)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        def close_window() -> None:
            try:
                self._plot_windows.remove(window)
            except ValueError:
                pass
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)

    def create_map_output(self) -> None:
        if self.data is None:
            messagebox.showinfo("Map", "Download or open a data file first.")
            return
        try:
            start, end = self._parse_map_time_entries()
            station_indices, bounds = self._map_region()
            map_values, colorbar_label, token, signed_values = self._map_parameter_data()

            if start == end:
                time_index = self._nearest_map_time_index(start)
                figure = self._create_component_map_figure(
                    map_values,
                    time_index,
                    station_indices,
                    bounds,
                    colorbar_label,
                    signed_values,
                )
                default_name = self._map_filename(token, time_index)
                self._display_component_map(figure, default_name)
                actual_time = self.data.times[time_index]
                self.map_status_var.set(
                    f"Displayed nearest sample: {actual_time.strftime(MANUAL_TIME_FORMAT)} UTC"
                )
                self.status_var.set(
                    f"Displayed {self.map_parameter_var.get()} map at "
                    f"{actual_time.strftime(MANUAL_TIME_FORMAT)} UTC."
                )
                return

            time_indices = [
                index
                for index, value in enumerate(self.data.times)
                if start <= value <= end
            ]
            if not time_indices:
                raise ValueError("No data samples fall inside the entered map time range.")

            output_directory = self._map_output_directory()
            output_directory.mkdir(parents=True, exist_ok=True)
            saved_paths: list[Path] = []
            self.map_status_var.set(
                f"Saving {len(time_indices)} {self.map_parameter_var.get()} maps…"
            )
            self.root.update_idletasks()
            for sequence_number, time_index in enumerate(time_indices, start=1):
                figure = self._create_component_map_figure(
                    map_values,
                    time_index,
                    station_indices,
                    bounds,
                    colorbar_label,
                    signed_values,
                )
                output_path = output_directory / self._map_filename(token, time_index)
                figure.savefig(output_path, dpi=300, bbox_inches="tight")
                figure.clear()
                saved_paths.append(output_path)
                self.map_status_var.set(
                    f"Saving map {sequence_number}/{len(time_indices)}…"
                )
                self.root.update_idletasks()

            message = (
                f"Saved {len(saved_paths)} {self.map_parameter_var.get()} maps to "
                f"{output_directory}"
            )
            self.map_status_var.set(message)
            self.status_var.set(message)
        except Exception as exc:
            self.map_status_var.set("Map creation failed")
            messagebox.showerror("Map", str(exc))

    # ----- stack plots -------------------------------------------------------

    def on_plot_mode_change(self) -> None:
        mode = self.plot_mode_var.get()
        if mode == "single":
            self.plot_selected_station()
        elif mode == "map":
            self.map_status_var.set(
                "Map times affect magnetic parameter maps only; equal times display one sample."
            )

    def create_selected_plots(self) -> None:
        if self.data is None:
            messagebox.showinfo("Plot", "Download or open a data file first.")
            return

        mode = self.plot_mode_var.get()
        if mode == "map":
            self.create_map_output()
            return
        if mode == "single":
            self.plot_selected_station()
            return

        if mode == "stack_dbh":
            try:
                raw_offset = float(self.raw_offset_var.get())
                derivative_offset = float(self.derivative_offset_var.get())
            except ValueError:
                messagebox.showerror(
                    "Plot",
                    "The raw and derivative vertical offsets must be numeric.",
                )
                return
            if raw_offset < 0 or derivative_offset < 0:
                messagebox.showerror(
                    "Plot",
                    "Vertical offsets must be zero or positive.",
                )
                return
        else:
            derivative = mode == "stack_derivative"
            try:
                offset = float(
                    self.derivative_offset_var.get()
                    if derivative
                    else self.raw_offset_var.get()
                )
            except ValueError:
                messagebox.showerror(
                    "Plot", "The selected vertical offset must be numeric."
                )
                return
            if offset < 0:
                messagebox.showerror(
                    "Plot", "Vertical offset must be zero or positive."
                )
                return

        try:
            order = self._stack_station_order()
        except ValueError as exc:
            messagebox.showerror("Stack filters", str(exc))
            return

        if len(order) == 0:
            messagebox.showinfo(
                "Stack filters",
                "No loaded stations match the selected latitude/longitude filter.",
            )
            return

        time_mask = self._selected_time_mask()
        if not np.any(time_mask):
            messagebox.showinfo(
                "Plot", "No data samples fall inside the selected time range."
            )
            return

        if mode == "stack_dbh":
            self._create_dbh_stack_plot_window(
                order,
                raw_offset,
                derivative=False,
                time_mask=time_mask,
            )
            self._create_dbh_stack_plot_window(
                order,
                derivative_offset,
                derivative=True,
                time_mask=time_mask,
            )
            return

        for component_index in range(3):
            self._create_stack_plot_window(
                component_index, order, offset, derivative, time_mask
            )

    def _filtered_stack_station_indices(self) -> np.ndarray:
        """Return station indices passing the optional latitude/longitude filters."""
        if self.data is None:
            return np.array([], dtype=int)

        mask = np.ones(len(self.data.station_codes), dtype=bool)
        filter_descriptions: list[str] = []

        if self.stack_filter_lat_var.get():
            try:
                lat_min = float(self.stack_lat_min_var.get())
                lat_max = float(self.stack_lat_max_var.get())
            except ValueError as exc:
                raise ValueError("Stack latitude bounds must be numeric.") from exc
            lat_lo, lat_hi = sorted((lat_min, lat_max))
            mask &= np.isfinite(self.data.glat)
            mask &= (self.data.glat >= lat_lo) & (self.data.glat <= lat_hi)
            filter_descriptions.append(f"lat {lat_lo:g}° to {lat_hi:g}°")

        if self.stack_filter_lon_var.get():
            try:
                lon_west = float(self.stack_lon_west_var.get())
                lon_east = float(self.stack_lon_east_var.get())
            except ValueError as exc:
                raise ValueError("Stack longitude bounds must be numeric.") from exc

            lon_west = float(normalize_longitude(lon_west))
            lon_east = float(normalize_longitude(lon_east))
            lons = normalize_longitude(self.data.glon)
            finite_lon = np.isfinite(lons)
            if lon_west <= lon_east:
                in_lon = finite_lon & (lons >= lon_west) & (lons <= lon_east)
            else:
                in_lon = finite_lon & ((lons >= lon_west) | (lons <= lon_east))
            mask &= in_lon
            filter_descriptions.append(f"lon {lon_west:g}° to {lon_east:g}°")

        indices = np.flatnonzero(mask)
        if filter_descriptions:
            description = " and ".join(filter_descriptions)
            self.stack_filter_status_var.set(
                f"Automatic stacks: {len(indices)} matching stations ({description})"
            )
        else:
            self.stack_filter_status_var.set(
                f"Automatic stacks: all {len(indices)} loaded stations"
            )
        return indices

    def _stack_station_order(self) -> np.ndarray:
        if self.data is None:
            return np.array([], dtype=int)

        indices = self._filtered_stack_station_indices()
        if len(indices) == 0:
            return indices

        choice = self.stack_sort_var.get()
        codes = self.data.station_codes[indices]
        if choice.startswith("Latitude"):
            local_order = np.lexsort(
                (codes, -np.nan_to_num(self.data.glat[indices], nan=-999.0))
            )
        elif choice.startswith("Longitude"):
            lons = normalize_longitude(self.data.glon[indices])
            local_order = np.lexsort((codes, np.nan_to_num(lons, nan=999.0)))
        else:
            local_order = np.argsort(codes)
        return indices[local_order]

    def _create_stack_plot_window(
        self,
        component_index: int,
        order: np.ndarray,
        offset: float,
        derivative: bool,
        time_mask: np.ndarray,
    ) -> None:
        if self.data is None:
            return

        coordinate, coordinate_data, component_symbols, component_long_labels = (
            self._active_coordinate_data()
        )
        component = component_symbols[component_index]
        quantity = f"dB{component}/dt" if derivative else f"dB{component}"
        derivative_scale, derivative_units = derivative_display(self.data)
        units = derivative_units if derivative else "nT"
        window = tk.Toplevel(self.root)
        window.title(f"Stacked {quantity} ({coordinate})")
        window.geometry("1250x760")
        window.minsize(850, 520)
        self._plot_windows.append(window)

        top = ttk.Frame(window, padding=4)
        top.pack(fill=tk.X)
        plot_times = self.data.times[time_mask]
        range_text = ""
        if len(plot_times):
            range_text = (
                f"; {plot_times[0]:%Y-%m-%d %H:%M:%S} to "
                f"{plot_times[-1]:%Y-%m-%d %H:%M:%S} UTC"
            )
        ttk.Label(
            top,
            text=(
                f"{quantity} ({coordinate}) — {len(order)} stations; "
                f"offset={offset:g} {units}{range_text}"
            ),
            font=("TkDefaultFont", 10, "bold"),
        ).pack(side=tk.LEFT)

        figure = Figure(figsize=(13, 7), dpi=100)
        ax = figure.add_subplot(111)
        canvas_frame = ttk.Frame(window)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        cmap = __import__("matplotlib").colormaps["tab20"]
        lines = []
        for stack_index, station_index in enumerate(order):
            raw = coordinate_data[station_index, time_mask, component_index]
            y = (
                finite_time_derivative(raw, plot_times, derivative_scale)
                if derivative
                else mask_bad_magnetometer_values(raw)
            )
            if offset != 0.0 and np.any(np.isfinite(y)):
                y = y - np.nanmedian(y) + stack_index * offset

            color = cmap((stack_index % 20) / 19 if len(order) > 1 else 0.0)
            line = ax.plot(
                plot_times,
                y,
                linewidth=0.8,
                alpha=0.86,
                color=color,
                zorder=2,
            )[0]
            line._station_code = self.data.station_codes[station_index]
            line._station_lat = self.data.glat[station_index]
            line._station_lon = normalize_longitude(self.data.glon[station_index])
            lines.append(line)

            finite_y = np.flatnonzero(np.isfinite(y))
            if len(finite_y):
                ax.text(
                    1.003,
                    y[finite_y[-1]],
                    self.data.station_codes[station_index],
                    transform=ax.get_yaxis_transform(),
                    color=color,
                    fontsize=7,
                    va="center",
                    ha="left",
                    clip_on=False,
                )

        sort_label = self.stack_sort_var.get()
        ax.set_title(
            f"Stacked {quantity} in {coordinate} coordinates "
            f"({sort_label.lower()}; active time range)"
        )
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(f"dB/dt ({derivative_units})" if derivative else "dB (nT)")
        ax.grid(True, alpha=0.28)
        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        figure.subplots_adjust(left=0.08, right=0.91, top=0.86, bottom=0.13)

        onset_artists = self._draw_selected_substorm_lines(
            ax,
            show_numbers=True,
            line_color="black",
            line_alpha=0.85,
            line_zorder=0.25,
        )

        if HAS_MPLCURSORS and lines:
            cursor = mplcursors.cursor(lines, hover=True)

            @cursor.connect("add")
            def _on_add(selection):
                line = selection.artist
                x_value, y_value = selection.target
                try:
                    time_text = DateFormatter("%Y-%m-%d %H:%M:%S")(x_value)
                except Exception:
                    time_text = str(x_value)
                selection.annotation.set_text(
                    f"{line._station_code}\n"
                    f"lat={line._station_lat:.2f}, lon={line._station_lon:.2f}\n"
                    f"{time_text} UTC\n{y_value:.2f} {units}"
                )

        canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        stack_record: dict[str, object] = {
            "window": window,
            "axis": ax,
            "canvas": canvas,
            "onset_artists": onset_artists,
        }
        self._automatic_stack_plots.append(stack_record)

        default_name = self._plot_filename(f"{coordinate}_{quantity}")
        ttk.Button(
            top,
            text="Save PNG…",
            command=lambda fig=figure, name=default_name: self.save_figure_png(fig, name),
        ).pack(side=tk.RIGHT)
        ttk.Button(
            top,
            textvariable=self.substorm_toggle_button_var,
            command=lambda: self.toggle_substorm_marking("main"),
        ).pack(side=tk.RIGHT, padx=(0, 5))

        def close_window() -> None:
            try:
                self._plot_windows.remove(window)
            except ValueError:
                pass
            try:
                self._automatic_stack_plots.remove(stack_record)
            except ValueError:
                pass
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)

    def _create_dbh_stack_plot_window(
        self,
        order: np.ndarray,
        offset: float,
        derivative: bool,
        time_mask: np.ndarray,
    ) -> None:
        """Create one stacked dBh or dBh/dt window for the filtered stations."""
        if self.data is None:
            return

        quantity = "dBh/dt" if derivative else "dBh"
        derivative_scale, derivative_units = derivative_display(self.data)
        units = derivative_units if derivative else "nT"
        window = tk.Toplevel(self.root)
        window.title(f"Stacked {quantity}")
        window.geometry("1250x760")
        window.minsize(850, 520)
        self._plot_windows.append(window)

        top = ttk.Frame(window, padding=4)
        top.pack(fill=tk.X)
        plot_times = self.data.times[time_mask]
        range_text = ""
        if len(plot_times):
            range_text = (
                f"; {plot_times[0]:%Y-%m-%d %H:%M:%S} to "
                f"{plot_times[-1]:%Y-%m-%d %H:%M:%S} UTC"
            )
        ttk.Label(
            top,
            text=(
                f"{quantity} — {len(order)} stations; "
                f"offset={offset:g} {units}{range_text}"
            ),
            font=("TkDefaultFont", 10, "bold"),
        ).pack(side=tk.LEFT)

        figure = Figure(figsize=(13, 7), dpi=100)
        ax = figure.add_subplot(111)
        canvas_frame = ttk.Frame(window)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        if self.data.dbh is not None:
            dbh_data = self.data.dbh
        else:
            dbh_data = calculate_dbh(self.data.nez)

        cmap = __import__("matplotlib").colormaps["tab20"]
        lines = []
        for stack_index, station_index in enumerate(order):
            raw = mask_bad_magnetometer_values(
                dbh_data[station_index, time_mask]
            )
            y = (
                finite_time_derivative(raw, plot_times, derivative_scale)
                if derivative
                else raw
            )
            if offset != 0.0 and np.any(np.isfinite(y)):
                y = y - np.nanmedian(y) + stack_index * offset

            color = cmap((stack_index % 20) / 19 if len(order) > 1 else 0.0)
            line = ax.plot(
                plot_times,
                y,
                linewidth=0.8,
                alpha=0.86,
                color=color,
                zorder=2,
            )[0]
            line._station_code = self.data.station_codes[station_index]
            line._station_lat = self.data.glat[station_index]
            line._station_lon = normalize_longitude(
                self.data.glon[station_index]
            )
            lines.append(line)

            finite_y = np.flatnonzero(np.isfinite(y))
            if len(finite_y):
                ax.text(
                    1.003,
                    y[finite_y[-1]],
                    self.data.station_codes[station_index],
                    transform=ax.get_yaxis_transform(),
                    color=color,
                    fontsize=7,
                    va="center",
                    ha="left",
                    clip_on=False,
                )

        sort_label = self.stack_sort_var.get()
        ax.set_title(
            f"Stacked {quantity} "
            f"({sort_label.lower()}; active time range)"
        )
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(f"{quantity} ({units})")
        ax.grid(True, alpha=0.28)
        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        figure.subplots_adjust(left=0.08, right=0.91, top=0.86, bottom=0.13)

        onset_artists = self._draw_selected_substorm_lines(
            ax,
            show_numbers=True,
            line_color="black",
            line_alpha=0.85,
            line_zorder=0.25,
        )

        if HAS_MPLCURSORS and lines:
            cursor = mplcursors.cursor(lines, hover=True)

            @cursor.connect("add")
            def _on_add(selection):
                line = selection.artist
                x_value, y_value = selection.target
                try:
                    time_text = DateFormatter("%Y-%m-%d %H:%M:%S")(x_value)
                except Exception:
                    time_text = str(x_value)
                selection.annotation.set_text(
                    f"{line._station_code}\n"
                    f"lat={line._station_lat:.2f}, "
                    f"lon={line._station_lon:.2f}\n"
                    f"{time_text} UTC\n"
                    f"{y_value:.2f} {units}"
                )

        canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        stack_record: dict[str, object] = {
            "window": window,
            "axis": ax,
            "canvas": canvas,
            "onset_artists": onset_artists,
        }
        self._automatic_stack_plots.append(stack_record)

        default_name = self._plot_filename(quantity)
        ttk.Button(
            top,
            text="Save PNG…",
            command=lambda fig=figure, name=default_name: self.save_figure_png(
                fig, name
            ),
        ).pack(side=tk.RIGHT)
        ttk.Button(
            top,
            textvariable=self.substorm_toggle_button_var,
            command=lambda: self.toggle_substorm_marking("main"),
        ).pack(side=tk.RIGHT, padx=(0, 5))

        def close_window() -> None:
            try:
                self._plot_windows.remove(window)
            except ValueError:
                pass
            try:
                self._automatic_stack_plots.remove(stack_record)
            except ValueError:
                pass
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)

    def _plot_filename(self, quantity: str) -> str:
        if self.data is None or not len(self.data.times):
            prefix = "supermag"
        else:
            prefix = min(self.data.times).strftime("%Y%m%d_%H%M")
        safe_quantity = quantity.replace("/", "_per_")
        return f"{prefix}_stacked_{safe_quantity}.png"

    # ----- save/export -------------------------------------------------------

    def save_figure_png(self, figure: Optional[Figure], default_name: str) -> None:
        if figure is None:
            messagebox.showinfo("Save PNG", "There is no figure to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save figure as PNG",
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"),),
            initialfile=default_name,
        )
        if not path:
            return
        try:
            figure.savefig(path, dpi=300, bbox_inches="tight")
            self.status_var.set(f"Saved PNG: {path}")
        except Exception as exc:
            messagebox.showerror("Save PNG", f"Could not save the figure:\n\n{exc}")

    def save_map_png(self) -> None:
        self.save_figure_png(self.map_figure, "station_map.png")

    def save_single_png(self) -> None:
        coordinate = self.plot_coordinate_var.get().strip().upper()
        if self.data is not None and self.current_station_index is not None:
            code = self.data.station_codes[self.current_station_index]
            name = f"{code}_{coordinate}.png"
        else:
            name = f"single_station_{coordinate}.png"
        self.save_figure_png(self.single_figure, name)

    # ----- errors and shutdown ----------------------------------------------

    def _show_callback_error(self, exc_type, value, tb) -> None:
        error_text = "".join(traceback.format_exception(exc_type, value, tb))
        print(error_text, file=sys.stderr)
        messagebox.showerror("Application error", error_text)

    def close(self) -> None:
        self._download_stop_event.set()
        for window in list(self._plot_windows):
            try:
                window.destroy()
            except tk.TclError:
                pass
        if self.manual_stack_window is not None:
            try:
                self.manual_stack_window.destroy()
            except tk.TclError:
                pass
        if self.manual_selection_window is not None:
            try:
                self.manual_selection_window.destroy()
            except tk.TclError:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_file",
        nargs="?",
        help="Optional existing NetCDF or legacy CSV file to load at startup.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_arguments(argv)
    app = SuperMAGDownloadViewer(initial_file=args.data_file)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
