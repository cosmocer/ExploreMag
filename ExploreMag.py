#!/usr/bin/env python3
"""ExploreMag ground-magnetic-field exploration and pulsation-analysis GUI.

ExploreMag combines the data access and plotting services in ``mag_viewer.py``,
the multi-network GMAG downloader in ``gmag_download_and_clean.py``, and the
Pi2 polarization engine in ``mag_analyzer.py`` into one desktop application.

Current capabilities
--------------------
1. Download SuperMAG 1-minute data or GMAG data at 1 second, 2 Hz, 10 seconds,
   or each station's native cadence for a selected UTC interval and region.
2. Import custom SuperMAG NetCDF products, detect and align their cadence, fill
   interior gaps shorter than five seconds, convert them to ExploreMag NetCDF,
   save them as ``<source>_converted_for_exploremag.nc``, and open the result.
3. Open existing ExploreMag, SuperMAG, and GMAG files; select stations from
   synchronized lists, the station map, or entered latitude/longitude bounds.
4. Plot single-station components, automatic or manual multi-station stacks,
   cadence-aware dB/dt, dBh, scalar maps, vector maps, and map sequences.
5. Download, save, reopen, and visualize global, sunlit/dark-sector, and
   regional SuperMAG indices, including regional time-versus-MLT views.
6. Perform wavelet, Pi-band, Pc-band, and Pi2 polarization analyses with
   editable time intervals, data export, interactive cursors, and PNG output.
7. Map total horizontal ULF power using the cadence-resolvable portion of the
   requested band, report the effective band, and create selectable stacked
   station power plots with editable time and y-axis ranges.
8. Display Pi2 polarization ellipses and diagnostics with explicit coordinate
   conventions, normalized map symbols, and safeguards for ambiguous azimuth
   or rotation estimates.

Run
---
    python ExploreMag.py
    python ExploreMag.py existing_magnetic_file.nc

Keep ``ExploreMag.py``, ``mag_viewer.py``, ``mag_analyzer.py``, and
``gmag_download_and_clean.py`` together. High-resolution THEMIS-hosted IMAGE
data are preferred; the dedicated GMAG IMAGE loader is used as a fallback.
Cadence suitability, coverage, cleaning, and baseline requirements remain
enforced by the selected download mode.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import csv
import json
import inspect
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import traceback
import urllib.parse
from tkinter import simpledialog

import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib import dates as mdates
from matplotlib.lines import Line2D

try:
    import ppigrf
except ImportError:
    ppigrf = None

try:
    from scipy import signal as scipy_signal
    from scipy.stats import chi2 as scipy_chi2
except ImportError:  # handled with a clear GUI error when an analysis is requested
    scipy_signal = None
    scipy_chi2 = None

import mag_viewer as viewer
import gmag_download_and_clean as gmag_downloader
from mag_analyzer import (
    StationPi2Result,
    create_polarization_figure,
    run_pi2_analysis,
)

def _gmag_station_directory_candidates(utils_module) -> list[Path]:
    """Return candidate directories containing GMAG's non-Python metadata."""
    candidates: list[Path] = []
    configured = os.environ.get("GMAG_STATION_PATH", "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        candidates.append(
            configured_path.parent
            if configured_path.name.lower() == "station_list.csv"
            else configured_path
        )

    utils_file = getattr(utils_module, "__file__", None)
    if utils_file:
        candidates.append(Path(utils_file).resolve().parent / "Stations")

    analyzer_path = Path(__file__).resolve()
    for parent in analyzer_path.parents:
        candidates.extend(
            (
                parent / "gmag" / "Stations",
                parent / "gmag" / "gmag" / "Stations",
                parent / "gmag" / "docs" / "_data",
            )
        )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _load_gmag_station_catalogue_compat(utils_module):
    """Load GMAG station metadata even when its wheel omitted package data.

    GMAG 2.0.2's ``setup.py`` does not declare the ``Stations`` directory as
    package data. A wheel installation therefore imports successfully while
    ``utils.load_station_geo(param="ALL")`` returns ``None``. Prefer the normal
    API, then look for the same catalogue in an explicitly configured location
    or a nearby source checkout.
    """
    errors: list[str] = []
    for all_token in ("ALL", "*", "all"):
        try:
            catalogue = utils_module.load_station_geo(param=all_token)
            if isinstance(catalogue, viewer.pd.DataFrame) and not catalogue.empty:
                return catalogue.copy()
        except Exception as exc:
            errors.append(f"default param={all_token!r}: {exc!r}")

    searched: list[str] = []
    for candidate in _gmag_station_directory_candidates(utils_module):
        station_file = candidate / "station_list.csv"
        searched.append(str(station_file.resolve(strict=False)))
        if not station_file.is_file():
            continue
        try:
            catalogue = utils_module.load_station_geo(
                param="ALL", path=str(station_file.parent)
            )
            if isinstance(catalogue, viewer.pd.DataFrame) and not catalogue.empty:
                return catalogue.copy()
        except Exception as exc:
            errors.append(f"{station_file}: {exc!r}")

    details = " ; ".join(errors) if errors else "no catalogue files were readable"
    raise RuntimeError(
        "GMAG station catalogue is unavailable. GMAG imported, but its wheel "
        "did not include Stations/station_list.csv. Set GMAG_STATION_PATH to "
        "the directory containing station_list.csv or keep a gmag source "
        f"checkout beside the project. Searched: {searched}. Details: {details}"
    )


_FIRST_SAMPLE_BASELINE_ERROR = (
    "No complete N/E/Z sample is available at the beginning of the "
    "requested interval for first-data-point baselining"
)
_IMAGE_BASELINE_SEARCH_LIMIT_SECONDS = 300.0


def _call_gmag_image_load(
    image_module,
    *,
    code: str,
    day_start,
    ndays: int,
    force_download: bool,
    download_files: bool,
):
    """Call ``gmag.arrays.image.load`` across supported GMAG signatures.

    The official IMAGE loader uses the network's native product.  This helper
    deliberately does not resample it: the downloader's normal cadence check
    decides whether the returned data are suitable for the selected 1-s/2-Hz
    output mode.
    """
    start_text = viewer.pd.Timestamp(day_start).strftime("%Y-%m-%d")
    load = image_module.load
    signature = inspect.signature(load)
    parameters = signature.parameters

    kwargs: dict[str, Any] = {}
    if "ndays" in parameters:
        kwargs["ndays"] = int(ndays)
    if "dl" in parameters:
        kwargs["dl"] = bool(download_files)
    elif "download" in parameters:
        kwargs["download"] = bool(download_files)
    if "force" in parameters:
        kwargs["force"] = bool(force_download)
    elif "force_download" in parameters:
        kwargs["force_download"] = bool(force_download)

    try:
        return load(str(code).strip().upper(), start_text, **kwargs)
    except TypeError as first_error:
        # Older source checkouts used positional ndays and did not expose every
        # cache-control keyword. Retry only for an apparent signature mismatch.
        message = str(first_error).lower()
        signature_tokens = (
            "unexpected keyword", "positional argument", "required positional",
            "multiple values", "takes ",
        )
        if not any(token in message for token in signature_tokens):
            raise
        try:
            return load(str(code).strip().upper(), start_text, int(ndays))
        except TypeError:
            raise first_error


def _install_gmag_image_loader_compatibility(downloader_module) -> None:
    """Add a dedicated IMAGE fallback to the existing multi-array downloader.

    High-resolution THEMIS-hosted files remain the first attempt.  For a
    catalogue row labelled IMAGE, the official ``gmag.arrays.image`` loader is
    appended as a network-specific fallback instead of treating a missing
    THEMIS CDF as proof that the station has no data.
    """
    if getattr(downloader_module, "_exploremag_image_loader_compat", False):
        return

    required = (
        "station_load_candidates", "load_candidate_window",
        "StationLoadCandidate", "unpack_load_result", "utc_index",
    )
    missing = [name for name in required if not hasattr(downloader_module, name)]
    if missing:
        raise RuntimeError(
            "The GMAG downloader is too old for the IMAGE compatibility patch; "
            f"missing: {', '.join(missing)}"
        )

    try:
        from gmag.arrays import image as gmag_image
        image_import_error: Optional[Exception] = None
    except Exception as exc:
        # Do not disable THEMIS/CARISMA downloads merely because an older GMAG
        # checkout lacks the IMAGE module. Raise only if the IMAGE fallback is
        # actually reached for an IMAGE station.
        gmag_image = None
        image_import_error = exc

    original_candidates = downloader_module.station_load_candidates
    original_load_window = downloader_module.load_candidate_window

    def station_load_candidates_image_compat(*args, **kwargs):
        candidates = list(original_candidates(*args, **kwargs))
        station = args[0] if args else kwargs.get("station")
        if station is None:
            return candidates
        array_token = re.sub(
            r"[^a-z0-9]+", "", str(getattr(station, "array_name", "")).lower()
        )
        if "image" not in array_token:
            return candidates

        code = str(getattr(station, "code", "")).strip().upper()
        if not code:
            return candidates
        duplicate = any(
            str(getattr(candidate, "loader", "")).upper() == "IMAGE"
            and str(getattr(getattr(candidate, "load_station", None), "code", "")).upper()
            == code
            for candidate in candidates
        )
        if duplicate:
            return candidates

        candidates.append(
            downloader_module.StationLoadCandidate(
                load_station=station,
                output_station=station,
                loader="IMAGE",
                description=(
                    f"direct IMAGE archive code {code}; native cadence retained "
                    "and validated before output"
                ),
            )
        )
        return candidates

    def load_candidate_window_image_compat(
        candidate,
        window,
        *,
        themis_module,
        carisma_module,
        force_download: bool,
        download_missing: bool,
    ):
        if str(getattr(candidate, "loader", "")).upper() != "IMAGE":
            return original_load_window(
                candidate,
                window,
                themis_module=themis_module,
                carisma_module=carisma_module,
                force_download=force_download,
                download_missing=download_missing,
            )

        load_info = candidate.load_station
        try:
            if gmag_image is None:
                raise RuntimeError(
                    "The installed GMAG package has no gmag.arrays.image module. "
                    "Install/update the official GMAG source checkout before "
                    "requesting direct IMAGE fallback data. Original import error: "
                    + repr(image_import_error)
                )
            result = _call_gmag_image_load(
                gmag_image,
                code=load_info.code,
                day_start=window.day_start,
                ndays=window.ndays,
                force_download=force_download,
                download_files=download_missing,
            )
            if result is None:
                raise RuntimeError("IMAGE loader returned no result")
            data, meta = downloader_module.unpack_load_result(result)
        except Exception as exc:
            raise RuntimeError(
                f"{window.name} window {window.start} to {window.end}: {exc}"
            ) from exc

        index = downloader_module.utc_index(data.index)
        valid = ~index.isna()
        data = data.loc[np.asarray(valid)].copy()
        data.index = index[valid]
        data = data.loc[(data.index >= window.start) & (data.index <= window.end)]
        if data.empty:
            raise RuntimeError(
                f"{window.name} window {window.start} to {window.end}: "
                "no samples returned by the direct IMAGE archive"
            )
        return data, meta

    station_load_candidates_image_compat._exploremag_original = original_candidates
    load_candidate_window_image_compat._exploremag_original = original_load_window
    downloader_module.station_load_candidates = station_load_candidates_image_compat
    downloader_module.load_candidate_window = load_candidate_window_image_compat
    downloader_module._exploremag_image_loader_compat = True


def _pad_loaded_station_front(
    downloader_module,
    loaded_station,
    missing_rows: int,
) -> None:
    """Restore the omitted beginning of a station after a baseline retry."""
    if missing_rows <= 0:
        return
    for attribute in ("raw_nez", "perturbation_nez", "pi2_nez"):
        values = getattr(loaded_station, attribute, None)
        if values is None:
            continue
        array = np.asarray(values)
        shape = (missing_rows,) + array.shape[1:]
        pad = np.full(shape, np.nan, dtype=array.dtype)
        setattr(loaded_station, attribute, np.concatenate((pad, array), axis=0))

    quality = getattr(loaded_station, "quality", None)
    if quality is not None:
        quality_array = np.asarray(quality)
        missing_flag = int(getattr(downloader_module, "QUALITY_MISSING", 2))
        quality_pad = np.full(
            (missing_rows,) + quality_array.shape[1:],
            missing_flag,
            dtype=quality_array.dtype,
        )
        loaded_station.quality = np.concatenate(
            (quality_pad, quality_array), axis=0
        )

    raw = np.asarray(getattr(loaded_station, "raw_nez", np.empty((0, 3))))
    if raw.ndim == 2 and raw.shape[1] >= 2 and len(raw):
        horizontal = np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1])
        loaded_station.coverage_fraction = float(np.count_nonzero(horizontal) / len(raw))


def _install_gmag_first_sample_baseline_compatibility(downloader_module) -> None:
    """Retry first-point baselining from the first complete vector after start.

    Some high-resolution IMAGE files begin with staggered/missing components.
    The old downloader rejects the station before reaching a complete N/E/Z
    vector.  On that one specific error, this wrapper advances the station's
    processing grid to the first complete vector found within five minutes,
    lets the downloader calculate and report the real baseline timestamp, and
    then pads the omitted leading output rows back with NaNs/missing flags.
    No magnetic samples are invented or interpolated for the baseline.
    """
    if getattr(downloader_module, "_exploremag_first_sample_baseline_compat", False):
        return
    original_prepare = getattr(downloader_module, "prepare_station", None)
    extract_frame = getattr(downloader_module, "extract_station_frame", None)
    if original_prepare is None or extract_frame is None:
        raise RuntimeError(
            "The GMAG downloader does not expose prepare_station/extract_station_frame; "
            "the first-complete-vector baseline fix cannot be installed."
        )

    signature = inspect.signature(original_prepare)

    def prepare_station_first_complete_compat(*args, **kwargs):
        try:
            return original_prepare(*args, **kwargs)
        except Exception as exc:
            if _FIRST_SAMPLE_BASELINE_ERROR.lower() not in str(exc).lower():
                raise
            # Exception target variables are cleared after the except block;
            # retain the original failure under a separate name for fallback.
            original_error = exc

        bound = signature.bind_partial(*args, **kwargs)
        arguments = bound.arguments
        target_time = arguments.get("target_time")
        data = arguments.get("data")
        meta = arguments.get("meta")
        station = arguments.get("catalogue_station")
        if target_time is None or data is None or station is None:
            raise original_error

        try:
            original_target = viewer.pd.DatetimeIndex(
                viewer.pd.to_datetime(target_time, utc=True, errors="coerce")
            )
            if len(original_target) < 2 or original_target.isna().any():
                raise ValueError("target grid contains invalid times")
            code = str(getattr(station, "code", "")).strip().upper()
            frame, _columns, _mapping, _coordinate = extract_frame(data, code, meta)
            frame = frame.loc[frame.index >= original_target[0], ["N", "E", "Z"]]
            complete = frame.notna().all(axis=1)
            if not complete.any():
                raise ValueError("no complete vector occurs after the output start")
            baseline_time = viewer.pd.Timestamp(frame.index[np.flatnonzero(complete)[0]])
            delay_seconds = float(
                (baseline_time - original_target[0]) / viewer.pd.Timedelta(seconds=1)
            )
            if delay_seconds < 0.0 or delay_seconds > _IMAGE_BASELINE_SEARCH_LIMIT_SECONDS:
                raise ValueError(
                    f"first complete vector is {delay_seconds:g} s after output start"
                )

            retry_index = int(np.searchsorted(original_target.asi8, baseline_time.value))
            if retry_index >= len(original_target) - 1:
                raise ValueError("too few output samples remain after the first complete vector")
            retry_target = original_target[retry_index:]
            arguments["target_time"] = retry_target
            loaded = original_prepare(*bound.args, **bound.kwargs)
            _pad_loaded_station_front(downloader_module, loaded, retry_index)

            retry_start = viewer.pd.Timestamp(retry_target[0])
            note = (
                "first-complete-vector baseline fallback: complete vector detected at "
                f"{baseline_time.isoformat()} ({delay_seconds:g} s after the requested "
                f"start); output processing restarted at {retry_start.isoformat()} "
                "and earlier rows were retained as NaN"
            )
            existing = str(getattr(loaded, "output_resampling", "")).strip()
            loaded.output_resampling = f"{existing} | {note}" if existing else note
            return loaded
        except Exception as retry_error:
            try:
                original_error.add_note(
                    "First-complete-vector fallback also failed: " + str(retry_error)
                )
            except AttributeError:
                pass
            raise original_error

    prepare_station_first_complete_compat._exploremag_original = original_prepare
    downloader_module.prepare_station = prepare_station_first_complete_compat
    downloader_module._exploremag_first_sample_baseline_compat = True


def _install_gmag_downloader_compatibility(downloader_module) -> None:
    """Install all analyzer-side patches required by the bundled downloader."""
    _install_gmag_image_loader_compatibility(downloader_module)
    _install_gmag_first_sample_baseline_compatibility(downloader_module)


def _install_gmag_runtime_compatibility() -> None:
    """Bridge GMAG 2.0.2 to wheel installs and current cdflib releases."""
    from gmag import utils as gmag_utils
    from gmag.arrays import themis as gmag_themis

    station_directory = next(
        (
            path
            for path in _gmag_station_directory_candidates(gmag_utils)
            if (path / "station_list.csv").is_file()
        ),
        None,
    )
    if station_directory is not None and not getattr(
        gmag_utils, "_exploremag_metadata_compat", False
    ):
        original_geo = gmag_utils.load_station_geo
        original_coor = gmag_utils.load_station_coor

        def load_station_geo_compat(*args, **kwargs):
            if not kwargs.get("path"):
                kwargs["path"] = str(station_directory)
            return original_geo(*args, **kwargs)

        def load_station_coor_compat(*args, **kwargs):
            if not kwargs.get("path"):
                kwargs["path"] = str(station_directory)
            return original_coor(*args, **kwargs)

        gmag_utils.load_station_geo = load_station_geo_compat
        gmag_utils.load_station_coor = load_station_coor_compat
        gmag_utils._exploremag_metadata_compat = True

    # cdflib <=1.3.1 returned the CDF label variable as (component, 1);
    # cdflib 1.3.12 squeezes it to (component,). GMAG 2.0.2 assumes the old
    # shape and calls c_col[0].astype(str), producing the report's
    # "'str' object has no attribute 'astype'" failure. Restore the legacy
    # label shape only while GMAG's THEMIS reader is active.
    if not getattr(gmag_themis, "_exploremag_cdflib_compat", False):
        original_load = gmag_themis.load

        def themis_load_compat(*args, **kwargs):
            cdf_class = gmag_themis.cdflib.CDF
            original_varget = cdf_class.varget

            def varget_compat(cdf_self, variable, *vargs, **vkwargs):
                result = original_varget(cdf_self, variable, *vargs, **vkwargs)
                if (
                    str(variable).lower().endswith("_labl")
                    and isinstance(result, np.ndarray)
                    and result.ndim == 1
                ):
                    return result.reshape(-1, 1)
                return result

            cdf_class.varget = varget_compat
            try:
                return original_load(*args, **kwargs)
            finally:
                cdf_class.varget = original_varget

        gmag_themis.load = themis_load_compat
        gmag_themis._exploremag_cdflib_compat = True


    _install_gmag_downloader_compatibility(gmag_downloader)


class ExploreMagAnalyzer(viewer.SuperMAGDownloadViewer):
    """v27 viewer with multi-selection and interactive pulsation analyses."""

    SUPERMAG_INDEX_KEYS = (
        "sme", "sml", "smu", "mlat", "mlt", "glat", "glon", "stid", "num",
        "smes", "smls", "smus", "mlats", "mlts", "glats", "glons", "stids", "nums",
        "smed", "smld", "smud", "mlatd", "mltd", "glatd", "glond", "stidd", "numd",
        "smer", "smlr", "smur", "mlatr", "mltr", "glatr", "glonr", "stidr", "numr",
        "smr", "ltsmr", "ltnum", "nsmr",
    )
    SUPERMAG_INDEX_VAR_NAMES = {
        "sme": "SME", "sml": "SML", "smu": "SMU",
        "mlat": "AACGM_LAT", "mlt": "AACGM_MLT",
        "glat": "GEO_LAT", "glon": "GEO_LON",
        "stid": "STATION_ID", "num": "STATION_NUM",
        "smes": "SME_S", "smls": "SML_S", "smus": "SMU_S",
        "mlats": "AACGM_LAT_S", "mlts": "AACGM_MLT_S",
        "glats": "GEO_LAT_S", "glons": "GEO_LON_S",
        "stids": "STATION_ID_S", "nums": "STATION_NUM_S",
        "smed": "SME_D", "smld": "SML_D", "smud": "SMU_D",
        "mlatd": "AACGM_LAT_D", "mltd": "AACGM_MLT_D",
        "glatd": "GEO_LAT_D", "glond": "GEO_LON_D",
        "stidd": "STATION_ID_D", "numd": "STATION_NUM_D",
        "smer": "SME_R", "smlr": "SML_R", "smur": "SMU_R",
        "mlatr": "AACGM_LAT_R", "mltr": "AACGM_MLT_R",
        "glatr": "GEO_LAT_R", "glonr": "GEO_LON_R",
        "stidr": "STATION_ID_R", "numr": "STATION_NUM_R",
        "smr": "SMR", "ltsmr": "SMR_LT", "ltnum": "LT_NUM",
        "nsmr": "SMR_STATION_NUM",
    }
    SUPERMAG_INDEX_PLOT_GROUPS = (
        (("sme", "sml", "smu"), "Global auroral electrojet indices"),
        (("smes", "smls", "smus"), "Sunlit-sector indices"),
        (("smed", "smld", "smud"), "Dark-sector indices"),
        (("smur",), "Regional upper electrojet index by MLT"),
        (("smlr",), "Regional lower electrojet index by MLT"),
        (("smr",), "Partial ring-current index"),
    )
    SUPERMAG_R_SELECTED_MLTS = (0, 6, 12, 18)
    SUPERMAG_R_MLT_TICKS = (0, 3, 6, 9, 12, 15, 18, 21)

    PC_BANDS = (
        ("Pc2", 5.0, 10.0),
        ("Pc3", 10.0, 45.0),
        ("Pc4", 45.0, 150.0),
        ("Pc5", 150.0, 600.0),
    )
    PI_BANDS = (
        ("Pi1", 1.0, 40.0),
        ("Pi2", 40.0, 150.0),
        ("Pi3/Ps6", 150.0, None),
    )
    SCALOGRAM_PERIOD_RANGE = (5.0, 600.0)
    TOTAL_ULF_PERIOD_RANGE = (5.0, 150.0)
    ULF_DETAIL_SCALOGRAM_RANGE = (5.0, 200.0)
    SUPERMAG_INDEX_ABSOLUTE_LIMIT = 1.0e5

    # For a nearly circular ellipse the major-axis azimuth is poorly
    # constrained.  Classical Pi2 work commonly suppresses azimuth when
    # |b/a| exceeds about 0.8; retain the ellipse but omit its major axis.
    POLARIZATION_AZIMUTH_MAX_RATIO = 0.80

    MAP_NONE = "None"
    VECTOR_MAP_OPTIONS = (
        "H⃗",
        "rotated H⃗",
        "dH⃗/dt",
        "rotated dH⃗/dt",
    )
    VECTOR_MAP_SPECS = {
        "H⃗": (False, False, "H_vector", "horizontal magnetic field"),
        "rotated H⃗": (False, True, "H_vector_rotated",
                       "90° clockwise rotated horizontal magnetic field"),
        "dH⃗/dt": (True, False, "dH_dt_vector",
                    "horizontal magnetic-field time derivative"),
        "rotated dH⃗/dt": (True, True, "dH_dt_vector_rotated",
                            "90° clockwise rotated horizontal magnetic-field time derivative"),
    }

    def __init__(self, initial_file: Optional[str] = None) -> None:
        # These attributes are used by dynamically dispatched methods during
        # the parent constructor.
        self.selected_station_indices: set[int] = set()
        self._pi2_results: dict[int, StationPi2Result] = {}
        self._last_supermag_indices: Optional[
            tuple[datetime, datetime, np.ndarray, dict[str, np.ndarray]]
        ] = None
        self._supermag_indices_file: Optional[Path] = None
        self._pi3_ps6_map_cache: Optional[np.ndarray] = None
        self._total_ulf_map_cache: Optional[np.ndarray] = None
        self._map_vector_derivative_cache: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._download_start_entry = None
        self._download_end_entry = None
        self._download_start_date_entry = None
        self._download_start_time_entry = None
        self._download_end_date_entry = None
        self._download_end_time_entry = None
        self._create_plots_button = None
        self._halt_map_button = None
        self._map_generation_halted = False
        self._gmag_process: Optional[subprocess.Popen[str]] = None
        self._gmag_process_lock = threading.Lock()
        self._gmag_progress_current = 0
        self._gmag_progress_total = 0
        self._gmag_download_active = False
        self._install_foreground_dialog_wrappers()
        super().__init__(initial_file=initial_file)
        self.root.title(
            "ExploreMag: A Software Suite for Exploratory Analysis of Ground-Level Magnetic Field Data"
        )

    def _dialog_parent(self):
        """Return the currently active application window for modal dialogs."""
        root = getattr(self, "root", None)
        if root is None:
            return None
        try:
            if not root.winfo_exists():
                return None
        except Exception:
            return None

        candidate = None
        try:
            candidate = root.grab_current()
        except Exception:
            candidate = None
        if candidate is None:
            try:
                focus = root.focus_get()
                if focus is not None:
                    candidate = focus.winfo_toplevel()
            except Exception:
                candidate = None
        if candidate is None or candidate is root:
            for window in reversed(getattr(self, "_plot_windows", [])):
                try:
                    if window.winfo_exists() and window.winfo_viewable():
                        candidate = window
                        break
                except Exception:
                    continue
        if candidate is None:
            candidate = root
        try:
            candidate.lift()
            candidate.attributes("-topmost", True)
            candidate.after_idle(lambda w=candidate: w.attributes("-topmost", False))
        except Exception:
            pass
        return candidate

    def _install_foreground_dialog_wrappers(self) -> None:
        """Make Tk message boxes and file choosers modal above the active window."""
        for module, names in (
            (
                viewer.messagebox,
                (
                    "showinfo", "showwarning", "showerror", "askquestion",
                    "askokcancel", "askyesno", "askyesnocancel", "askretrycancel",
                ),
            ),
            (
                viewer.filedialog,
                (
                    "askopenfilename", "askopenfilenames", "asksaveasfilename",
                    "askdirectory",
                ),
            ),
        ):
            for name in names:
                current = getattr(module, name, None)
                if current is None:
                    continue
                original = getattr(current, "_geomag_original_dialog", current)

                def wrapper(*args, _original=original, **kwargs):
                    if "parent" not in kwargs:
                        parent = self._dialog_parent()
                        if parent is not None:
                            kwargs["parent"] = parent
                    return _original(*args, **kwargs)

                wrapper._geomag_original_dialog = original
                setattr(module, name, wrapper)

    @staticmethod
    def _parse_gui_datetime(text: str, label: str) -> datetime:
        value = str(text).strip()
        formats = (
            viewer.MANUAL_TIME_FORMAT,
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
        )
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise ValueError(
            f"{label} must use YYYY-MM-DD HH:mm:SS UTC (minutes are also accepted)."
        )

    def _download_query_interval(self) -> tuple[datetime, datetime]:
        """Read and combine the separate date/time fields in the download query."""
        # mag_viewer displays four independent entries:
        # Start date, Start time, End date, and End time.  Always combine those
        # fields before parsing; a date-only value must not be treated as a full
        # timestamp.
        date_time_entries = (
            self._download_start_date_entry,
            self._download_start_time_entry,
            self._download_end_date_entry,
            self._download_end_time_entry,
        )
        if all(entry is not None for entry in date_time_entries):
            start_text = (
                f"{self._download_start_date_entry.get().strip()} "
                f"{self._download_start_time_entry.get().strip()}"
            )
            end_text = (
                f"{self._download_end_date_entry.get().strip()} "
                f"{self._download_end_time_entry.get().strip()}"
            )
            start = self._parse_gui_datetime(start_text, "Download start")
            end = self._parse_gui_datetime(end_text, "Download end")
            if end <= start:
                raise ValueError("Download end time must be later than download start time.")
            return start, end

        # Compatibility with viewer revisions that expose separate Tk variables
        # instead of discoverable Entry widgets.
        variable_quartets = (
            ("start_date_var", "start_time_var", "end_date_var", "end_time_var"),
            ("download_start_date_var", "download_start_time_var",
             "download_end_date_var", "download_end_time_var"),
            ("date_start_var", "time_start_var", "date_end_var", "time_end_var"),
        )
        for names in variable_quartets:
            variables = [getattr(self, name, None) for name in names]
            if any(variable is None for variable in variables):
                continue
            try:
                values = [str(variable.get()).strip() for variable in variables]
            except Exception:
                continue
            if not all(values):
                continue
            start = self._parse_gui_datetime(
                f"{values[0]} {values[1]}", "Download start"
            )
            end = self._parse_gui_datetime(
                f"{values[2]} {values[3]}", "Download end"
            )
            if end <= start:
                raise ValueError("Download end time must be later than download start time.")
            return start, end

        # Final compatibility fallback for a viewer that provides already
        # combined full timestamps in two entries.
        if self._download_start_entry is not None and self._download_end_entry is not None:
            start_text = str(self._download_start_entry.get()).strip()
            end_text = str(self._download_end_entry.get()).strip()
            if (" " in start_text or "T" in start_text) and (" " in end_text or "T" in end_text):
                start = self._parse_gui_datetime(start_text, "Download start")
                end = self._parse_gui_datetime(end_text, "Download end")
                if end <= start:
                    raise ValueError("Download end time must be later than download start time.")
                return start, end

        raise ValueError("Could not locate the main download date/time entries.")

    def _capture_download_time_entries(self, frame) -> None:
        labels = []
        entries = []
        for widget in frame.winfo_children():
            info = widget.grid_info()
            if not info:
                continue
            try:
                row = int(info.get("row", -1))
                column = int(info.get("column", -1))
            except (TypeError, ValueError):
                continue
            try:
                widget_class = str(widget.winfo_class())
            except Exception:
                widget_class = ""
            if "Label" in widget_class:
                try:
                    labels.append((str(widget.cget("text")).lower(), row, column))
                except Exception:
                    pass
            elif "Entry" in widget_class:
                entries.append((row, column, widget))

        def normalized_label(value: str) -> str:
            return " ".join(value.replace(":", " ").split())

        def entry_after_exact(label_text: str):
            target = normalized_label(label_text)
            matching = [
                item for item in labels
                if normalized_label(item[0]) == target
            ]
            for _, row, column in matching:
                choices = [
                    item for item in entries
                    if item[0] == row and item[1] > column
                ]
                if choices:
                    return min(choices, key=lambda item: item[1])[2]
            return None

        self._download_start_date_entry = entry_after_exact("start date")
        self._download_start_time_entry = entry_after_exact("start time")
        self._download_end_date_entry = entry_after_exact("end date")
        self._download_end_time_entry = entry_after_exact("end time")

        # Keep the old two-entry attributes only for compatibility with a
        # future/older viewer whose fields already contain complete timestamps.
        self._download_start_entry = self._download_start_date_entry
        self._download_end_entry = self._download_end_date_entry

    def _find_label_frame(self, parent, title: str):
        target = title.strip().lower()
        stack = [parent]
        while stack:
            widget = stack.pop()
            try:
                if target in str(widget.cget("text")).strip().lower():
                    return widget
            except Exception:
                pass
            try:
                stack.extend(widget.winfo_children())
            except Exception:
                pass
        return None

    def _make_variables(self) -> None:
        super()._make_variables()
        tk = viewer.tk
        self.pi2_start_var = tk.StringVar(value="")
        self.pi2_end_var = tk.StringVar(value="")
        self.quiet_start_var = tk.StringVar(value="")
        self.quiet_end_var = tk.StringVar(value="")
        self.pi2_onset_var = tk.StringVar(value="")
        self.pi2_coordinate_var = tk.StringVar(value="GEO")
        self.pi2_confirm_signs_var = tk.BooleanVar(value=True)
        self.indices_output_file_var = tk.StringVar(value="supermag_indices.nc")
        self.pi2_status_var = tk.StringVar(
            value="Select stations and choose a pulsation analysis."
        )
        self.map_cadence_var = tk.StringVar(value="")
        self.map_vector_parameter_var = tk.StringVar(value=self.MAP_NONE)
        self.map_vector_scale_var = tk.StringVar(value="")
        self.map_vector_arrow_size_var = tk.StringVar(value="1")
        self.map_colormap_var = tk.StringVar(value="RdBu_r")
        self.download_source_var = tk.StringVar(value="supermag-1min")
        self.pi2_band_min_var = tk.StringVar(value="40")
        self.pi2_band_max_var = tk.StringVar(value="150")

    def _build_interface(self) -> None:
        super()._build_interface()
        frame = self._find_label_frame(self.root, "SuperMAG download query")
        if frame is None:
            return
        frame.configure(text="Ground-level magnetic field data download query (UTC)")
        self._capture_download_time_entries(frame)

        # The station catalogue ships with exploremag and is always resolved
        # relative to this source file, independent of the launch directory.
        metadata_path = Path(__file__).resolve().with_name(
            "20250712-supermag-stations.txt"
        )
        self.metadata_file_var.set(str(metadata_path))

        # Remove the inherited metadata chooser and obsolete high-resolution
        # checkbox; cadence/source is selected explicitly beside Logon.
        for widget in list(frame.winfo_children()):
            try:
                text = str(widget.cget("text")).strip()
            except Exception:
                text = ""
            variable = ""
            try:
                variable = str(widget.cget("textvariable"))
            except Exception:
                pass
            if (
                text in {"Station metadata file", "Browse…", "High resolution"}
                or variable == str(self.metadata_file_var)
                or variable == str(self.high_resolution_var)
            ):
                widget.destroy()

        source_frame = viewer.ttk.Frame(frame)
        source_frame.grid(row=0, column=2, columnspan=6, sticky="w", padx=(0, 8))
        for text, value in (
            ("SuperMAG 1-minute", "supermag-1min"),
            ("gmag 1-second", "gmag-1s"),
            ("gmag 2 Hz", "gmag-2hz"),
            ("gmag 10-second", "gmag-10s"),
            ("native cadence", "gmag-native"),
        ):
            viewer.ttk.Radiobutton(
                source_frame,
                text=text,
                value=value,
                variable=self.download_source_var,
            ).pack(side=viewer.tk.LEFT, padx=(0, 10))
        viewer.ttk.Button(
            source_frame,
            text="Custom SuperMAG",
            command=self._start_custom_supermag_conversion,
        ).pack(side=viewer.tk.LEFT, padx=(0, 10))

        # Move the parent's progress/status row down to make room for the
        # independent SuperMAG-index output row.
        row_widgets = []
        for widget in frame.winfo_children():
            info = widget.grid_info()
            if not info or "row" not in info:
                continue
            try:
                row = int(info["row"])
            except (TypeError, ValueError):
                continue
            if row >= 4:
                row_widgets.append((row, widget))
        for row, widget in sorted(row_widgets, key=lambda item: item[0], reverse=True):
            widget.grid_configure(row=row + 1)

        viewer.ttk.Label(frame, text="SuperMAG indices NetCDF").grid(
            row=4, column=0, sticky="w", pady=(5, 0)
        )
        viewer.ttk.Entry(
            frame,
            textvariable=self.indices_output_file_var,
            width=52,
        ).grid(
            row=4, column=1, columnspan=3, sticky="ew", padx=(4, 5), pady=(5, 0)
        )
        viewer.ttk.Button(
            frame,
            text="Save as…",
            command=self.choose_supermag_indices_output_file,
        ).grid(row=4, column=4, sticky="ew", padx=(0, 5), pady=(5, 0))
        viewer.ttk.Button(
            frame,
            text="Download and save SuperMAG indices",
            command=self.download_and_save_supermag_indices,
        ).grid(row=4, column=5, sticky="ew", padx=(0, 5), pady=(5, 0))
        viewer.ttk.Button(
            frame,
            text="Open SuperMAG indices",
            command=self.open_supermag_indices_file,
        ).grid(row=4, column=6, sticky="ew", pady=(5, 0))

    def start_download(self) -> None:
        """Dispatch the selected ground-magnetometer download source."""
        source = self.download_source_var.get()
        if source == "supermag-1min":
            self.high_resolution_var.set(False)
            super().start_download()
            return
        if source == "custom-supermag":
            self._start_custom_supermag_conversion()
            return
        self._start_gmag_download(source)

    def _start_custom_supermag_conversion(self) -> None:
        source = viewer.filedialog.askopenfilename(
            parent=self._dialog_parent(),
            title="Select custom SuperMAG NetCDF",
            filetypes=[("NetCDF", "*.nc *.nc4 *.netcdf"), ("All files", "*")],
        )
        if not source:
            return
        source_path = Path(source).expanduser().resolve()
        output_path = source_path.with_name(
            f"{source_path.stem}_converted_for_exploremag.nc"
        )
        self.output_file_var.set(str(output_path))
        if output_path.exists() and not viewer.messagebox.askyesno(
            "Replace NetCDF file", f"The output file already exists:\n\n{output_path}\n\nReplace it?"
        ):
            return
        self.download_button.configure(state=viewer.tk.DISABLED)
        self.stop_download_button.configure(state=viewer.tk.DISABLED)
        self.progress_var.set(10.0)
        self.status_var.set("Converting and cleaning custom SuperMAG data…")

        def worker() -> None:
            try:
                viewer.convert_custom_supermag_netcdf(source, str(output_path))
                loaded = viewer.load_netcdf_file(str(output_path))
                self.root.after(0, lambda: self._gmag_download_finished(loaded, output_path, "custom"))
            except Exception as exc:
                self.root.after(0, lambda text=str(exc): self._gmag_download_failed(text))

        self._download_thread = threading.Thread(target=worker, daemon=True)
        self._download_thread.start()

    def _ask_gmag_baseline(self) -> Optional[tuple[str, str]]:
        """Ask for a quiet interval, or explicitly select first-sample baseline."""
        use_quiet = viewer.messagebox.askyesnocancel(
            "GMAG perturbation baseline",
            "Use a user-defined quiet-time interval for the GMAG baseline?\n\n"
            "Yes: enter quiet start/end times.\n"
            "No: subtract the first complete N/E/Z sample at each station.",
        )
        if use_quiet is None:
            return None
        if not use_quiet:
            return "", ""
        start = simpledialog.askstring(
            "Quiet-time baseline",
            "Quiet interval start (YYYY-MM-DD HH:MM:SS UTC):",
            parent=self._dialog_parent(),
        )
        if start is None:
            return None
        end = simpledialog.askstring(
            "Quiet-time baseline",
            "Quiet interval end (YYYY-MM-DD HH:MM:SS UTC):",
            parent=self._dialog_parent(),
        )
        if end is None:
            return None
        quiet_start = self._parse_gui_datetime(start, "Quiet start")
        quiet_end = self._parse_gui_datetime(end, "Quiet end")
        if quiet_end <= quiet_start:
            raise ValueError("Quiet interval end must be later than its start.")
        return quiet_start.isoformat(), quiet_end.isoformat()

    def _start_gmag_download(self, source: str) -> None:
        if self._download_thread is not None and self._download_thread.is_alive():
            viewer.messagebox.showinfo("Download", "A download is already running.")
            return
        try:
            start, end = self._download_query_interval()
            lat_min = float(self.lat_min_var.get())
            lat_max = float(self.lat_max_var.get())
            lon_min = float(self.lon_west_var.get())
            lon_max = float(self.lon_east_var.get())
            if lat_min > lat_max:
                raise ValueError("Latitude minimum exceeds latitude maximum.")
            if lon_min > lon_max:
                raise ValueError(
                    "GMAG downloads require a non-dateline-crossing longitude range "
                    "with west ≤ east."
                )
            output_text = self.output_file_var.get().strip()
            if not output_text:
                raise ValueError("Select an output NetCDF filename.")
            output_path = Path(output_text).expanduser().resolve()
            baseline = self._ask_gmag_baseline()
            if baseline is None:
                return
        except Exception as exc:
            viewer.messagebox.showerror("GMAG download settings", str(exc))
            return
        if output_path.exists() and not viewer.messagebox.askyesno(
            "Replace NetCDF file",
            f"The output file already exists:\n\n{output_path}\n\nReplace it?",
        ):
            return

        cadence_mode = {
            "gmag-1s": "1s",
            "gmag-2hz": "2hz",
            "gmag-10s": "10s",
            "gmag-native": "original",
        }.get(source)
        if cadence_mode is None:
            viewer.messagebox.showerror("GMAG download settings", f"Unknown source mode: {source}")
            return
        argv = [
            "--start", start.isoformat(),
            "--end", end.isoformat(),
            "--cadence-mode", cadence_mode,
            "--data-mode", "perturbation",
            "--lat-min", str(lat_min),
            "--lat-max", str(lat_max),
            "--lon-min", str(lon_min),
            "--lon-max", str(lon_max),
            "--output", str(output_path),
            "--yes",
        ]
        if cadence_mode in {"10s", "original"}:
            # IMAGE's native product is commonly 10-second data. Faster common
            # modes retain the stricter cadence guard, while these two modes
            # must accept IMAGE without fabricating higher-rate measurements.
            argv.extend(["--max-native-cadence-seconds", "10.1"])
        if baseline[0]:
            argv.extend(["--quiet-start", baseline[0], "--quiet-end", baseline[1]])
        else:
            argv.extend(["--quiet-start", "", "--quiet-end", ""])

        self._download_stop_event.clear()
        self._gmag_download_active = True
        self._gmag_progress_current = 0
        self._gmag_progress_total = 0
        self.download_button.configure(state=viewer.tk.DISABLED)
        self.stop_download_button.configure(state=viewer.tk.NORMAL)
        self.progress_var.set(2.0)
        self.status_var.set(
            f"Starting GMAG {cadence_mode} download; station progress will appear below…"
        )
        self._download_thread = threading.Thread(
            target=self._gmag_download_worker,
            args=(argv, output_path, cadence_mode),
            daemon=True,
        )
        self._download_thread.start()

    @staticmethod
    def _gmag_worker_bootstrap() -> str:
        """Return child-process code that installs the analyzer compatibility hooks."""
        return r"""
import importlib.util
from pathlib import Path
import sys

analyzer_path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(analyzer_path.parent))
spec = importlib.util.spec_from_file_location("_geomag_analyzer_worker", analyzer_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load analyzer module from {analyzer_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._install_gmag_runtime_compatibility()
module.gmag_downloader.load_station_catalogue = module._load_gmag_station_catalogue_compat
raise SystemExit(module.gmag_downloader.main(sys.argv[2:]))
"""

    def _set_gmag_process(
        self, process: Optional[subprocess.Popen[str]]
    ) -> None:
        with self._gmag_process_lock:
            self._gmag_process = process

    def _terminate_gmag_process(self, force: bool = False) -> None:
        with self._gmag_process_lock:
            process = self._gmag_process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(
                    process.pid,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
            elif force:
                process.kill()
            else:
                process.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill() if force else process.terminate()
            except Exception:
                pass

    def stop_download(self) -> None:
        """Stop either the inherited SuperMAG download or a GMAG subprocess."""
        with self._gmag_process_lock:
            gmag_process = self._gmag_process
        gmag_thread_running = (
            self._download_thread is not None and self._download_thread.is_alive()
        )
        if gmag_process is not None or self._gmag_download_active or (
            gmag_thread_running
            and self.download_source_var.get().strip().startswith("gmag-")
        ):
            self._download_stop_event.set()
            self.stop_download_button.configure(state=viewer.tk.DISABLED)
            self.status_var.set("Stopping GMAG download…")
            self._terminate_gmag_process(force=False)
            self.root.after(1500, lambda: self._terminate_gmag_process(force=True))
            return
        super().stop_download()

    def _update_gmag_progress(
        self,
        current: int,
        total: int,
        detail: str,
    ) -> None:
        if self._download_stop_event.is_set():
            return
        total = max(int(total), 1)
        current = int(np.clip(current, 0, total))
        self._gmag_progress_current = current
        self._gmag_progress_total = total
        progress = 5.0 + 90.0 * current / total
        self.progress_var.set(min(progress, 95.0))
        detail = detail.strip()
        suffix = f": {detail}" if detail else ""
        self.status_var.set(
            f"GMAG station {current}/{total} ({progress:.0f}%){suffix}"
        )

    def _update_gmag_phase(self, message: str, progress: float) -> None:
        if self._download_stop_event.is_set():
            return
        self.progress_var.set(float(np.clip(progress, 0.0, 99.0)))
        self.status_var.set(message)

    def _gmag_download_worker(
        self, argv: list[str], output_path: Path, cadence_mode: str
    ) -> None:
        process: Optional[subprocess.Popen[str]] = None
        output_tail: list[str] = []
        try:
            command = [
                sys.executable,
                "-u",
                "-c",
                self._gmag_worker_bootstrap(),
                str(Path(__file__).resolve()),
                *argv,
            ]
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(Path(__file__).resolve().parent),
                env=environment,
                start_new_session=(os.name != "nt"),
            )
            self._set_gmag_process(process)
            if self._download_stop_event.is_set():
                self._terminate_gmag_process(force=False)

            station_pattern = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]\s*(.*)")
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.rstrip()
                    if not line:
                        continue
                    print(line)
                    output_tail.append(line)
                    del output_tail[:-30]
                    match = station_pattern.search(line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        detail = match.group(3)
                        self.root.after(
                            0,
                            lambda c=current, t=total, d=detail: self._update_gmag_progress(
                                c, t, d
                            ),
                        )
                    elif line.startswith("Prefetching"):
                        self.root.after(
                            0,
                            lambda text=line: self._update_gmag_phase(text, 4.0),
                        )
                    elif line.startswith("Catalogue sites:"):
                        self.root.after(
                            0,
                            lambda text=line: self._update_gmag_phase(text, 5.0),
                        )
                    elif any(
                        token in line.lower()
                        for token in ("writing netcdf", "saved netcdf", "station report")
                    ):
                        self.root.after(
                            0,
                            lambda text=line: self._update_gmag_phase(text, 97.0),
                        )
                    if self._download_stop_event.is_set():
                        self._terminate_gmag_process(force=False)

            return_code = process.wait()
            if self._download_stop_event.is_set():
                self.root.after(0, self._gmag_download_cancelled)
                return
            if return_code != 0:
                details = "\n".join(output_tail[-12:])
                raise RuntimeError(
                    f"GMAG downloader exited with status {return_code}."
                    + (f"\n\nLast output:\n{details}" if details else "")
                )

            self.root.after(
                0,
                lambda: self._update_gmag_phase(
                    "GMAG download finished; opening the NetCDF file…", 98.0
                ),
            )
            loaded = viewer.load_netcdf_file(str(output_path))
            if self._download_stop_event.is_set():
                self.root.after(0, self._gmag_download_cancelled)
                return

            generated_report = output_path.with_name(
                output_path.stem + "_station_report.csv"
            )
            software_report = Path(__file__).resolve().with_name(
                generated_report.name
            )
            if (
                generated_report.is_file()
                and generated_report.resolve() != software_report.resolve()
            ):
                shutil.copy2(generated_report, software_report)
            self.root.after(
                0,
                lambda: self._gmag_download_finished(
                    loaded, output_path, cadence_mode
                ),
            )
        except Exception as exc:
            print(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                file=viewer.sys.stderr,
            )
            if self._download_stop_event.is_set():
                self.root.after(0, self._gmag_download_cancelled)
            else:
                self.root.after(
                    0,
                    lambda text=str(exc): self._gmag_download_failed(text),
                )
        finally:
            if process is not None and process.poll() is None:
                self._terminate_gmag_process(force=True)
            self._set_gmag_process(None)

    def _gmag_download_cancelled(self) -> None:
        self._gmag_download_active = False
        self.download_button.configure(state=viewer.tk.NORMAL)
        self.stop_download_button.configure(state=viewer.tk.DISABLED)
        self.progress_var.set(0.0)
        self.status_var.set("GMAG download stopped by the user.")

    def _gmag_download_failed(self, text: str) -> None:
        self._gmag_download_active = False
        self.download_button.configure(state=viewer.tk.NORMAL)
        self.stop_download_button.configure(state=viewer.tk.DISABLED)
        self.progress_var.set(0.0)
        self.status_var.set("GMAG download failed.")
        viewer.messagebox.showerror("GMAG download", text)

    def _gmag_download_finished(
        self, loaded: viewer.LoadedMagneticData, output_path: Path, cadence_mode: str
    ) -> None:
        self._gmag_download_active = False
        self.download_button.configure(state=viewer.tk.NORMAL)
        self.stop_download_button.configure(state=viewer.tk.DISABLED)
        self.progress_var.set(100.0)
        self.set_loaded_data(loaded)
        report_path = Path(__file__).resolve().with_name(
            output_path.stem + "_station_report.csv"
        )
        if cadence_mode == "custom":
            self.status_var.set(
                f"Converted, cleaned, saved, and opened {len(loaded.station_codes)} stations."
            )
            viewer.messagebox.showinfo(
                "Custom SuperMAG conversion complete",
                f"Suite-compatible NetCDF was saved and opened automatically:\n{output_path}",
            )
            return
        self.status_var.set(
            f"Saved and opened {len(loaded.station_codes)} GMAG stations at "
            f"{cadence_mode} cadence."
        )
        viewer.messagebox.showinfo(
            "GMAG download complete",
            f"NetCDF was saved and opened automatically:\n{output_path}\n\n"
            f"Station download report:\n{report_path}",
        )

    def _build_left_panel(self, parent) -> None:
        super()._build_left_panel(parent)
        for listbox in (self.lat_listbox, self.lon_listbox):
            listbox.configure(
                selectmode=viewer.tk.EXTENDED,
                selectbackground="#c8c8c8",
                selectforeground="black",
                activestyle="dotbox",
            )

        controls = self._find_label_frame(parent, "Plot mode")
        if controls is None:
            return

        # Locate the inherited scalar-map selector before rearranging its row.
        map_parameter_box = None
        stack = [parent]
        while stack:
            widget = stack.pop()
            try:
                if str(widget.winfo_class()) == "TCombobox" and str(
                    widget.cget("textvariable")
                ) == str(self.map_parameter_var):
                    map_parameter_box = widget
                    break
            except Exception:
                pass
            try:
                stack.extend(widget.winfo_children())
            except Exception:
                pass

        # Rename the inherited map plot mode.
        for widget in controls.winfo_children():
            try:
                if str(widget.cget("text")).strip() == "Magnetic parameter map":
                    widget.configure(text="Plot maps")
                    break
            except Exception:
                pass

        # Insert the independent SuperMAG-index plot mode.
        row_widgets = []
        for widget in controls.winfo_children():
            info = widget.grid_info()
            if not info or "row" not in info:
                continue
            try:
                row = int(info["row"])
            except (TypeError, ValueError):
                continue
            if row >= 4:
                row_widgets.append((row, widget))
        for row, widget in sorted(row_widgets, key=lambda item: item[0], reverse=True):
            widget.grid_configure(row=row + 1)

        viewer.ttk.Radiobutton(
            controls,
            text="Plot SuperMAG indices",
            value="sm_indices",
            variable=self.plot_mode_var,
            command=self._on_sm_indices_mode_change,
        ).grid(row=4, column=0, columnspan=3, sticky="w")

        # Insert cadence directly below Plot end time.
        row_widgets = []
        for widget in controls.winfo_children():
            info = widget.grid_info()
            if info and int(info.get("row", -1)) >= 9:
                row_widgets.append((int(info["row"]), widget))
        for row, widget in sorted(row_widgets, key=lambda item: item[0], reverse=True):
            widget.grid_configure(row=row + 1)
        viewer.ttk.Label(controls, text="Plot cadence (s)").grid(
            row=9, column=0, sticky="w", pady=(4, 0)
        )
        cadence_entry = viewer.ttk.Entry(
            controls, textvariable=self.map_cadence_var, width=20
        )
        cadence_entry.grid(row=9, column=1, columnspan=2, sticky="ew", pady=(4, 0))
        cadence_entry.bind("<Return>", self.apply_map_time_entries)

        # Configure independent scalar and vector map selectors on one row.
        if map_parameter_box is not None:
            values = [str(value) for value in map_parameter_box.cget("values")]
            for parameter in ("Pi3/Ps6", "Total ULF"):
                if parameter not in values:
                    values.append(parameter)
            values = [value for value in values if value != self.MAP_NONE]
            map_parameter_box.configure(
                values=(self.MAP_NONE, *values),
                width=15,
            )
            map_row = int(map_parameter_box.grid_info().get("row", 0))
            map_parameter_box.grid_configure(
                row=map_row, column=1, columnspan=1, sticky="ew", padx=(0, 4)
            )
            viewer.ttk.Combobox(
                controls,
                textvariable=self.map_vector_parameter_var,
                values=(self.MAP_NONE, *self.VECTOR_MAP_OPTIONS),
                state="readonly",
                width=19,
            ).grid(row=map_row, column=2, sticky="ew")
            for widget in controls.winfo_children():
                try:
                    if (
                        str(widget.winfo_class()) == "TLabel"
                        and str(widget.cget("text")).strip() == "Map parameter"
                    ):
                        widget.configure(text="Map parameters")
                        break
                except Exception:
                    pass

            # Keep arrow lengths comparable between consecutive maps when the
            # user supplies a fixed reference magnitude.
            row_widgets = []
            for widget in controls.winfo_children():
                info = widget.grid_info()
                if info and int(info.get("row", -1)) > map_row:
                    row_widgets.append((int(info["row"]), widget))
            for row, widget in sorted(
                row_widgets, key=lambda item: item[0], reverse=True
            ):
                widget.grid_configure(row=row + 1)
            vector_scale_row = map_row + 1
            viewer.ttk.Label(
                controls, text="Unit vector scale"
            ).grid(row=vector_scale_row, column=0, sticky="w", pady=(4, 0))
            viewer.ttk.Entry(
                controls,
                textvariable=self.map_vector_scale_var,
                width=10,
            ).grid(
                row=vector_scale_row,
                column=1,
                sticky="ew",
                pady=(4, 0),
            )
            viewer.ttk.Label(
                controls, text="Arrow size"
            ).grid(
                row=vector_scale_row,
                column=2,
                sticky="e",
                padx=(8, 4),
                pady=(4, 0),
            )
            viewer.ttk.Entry(
                controls,
                textvariable=self.map_vector_arrow_size_var,
                width=7,
            ).grid(
                row=vector_scale_row,
                column=3,
                sticky="ew",
                pady=(4, 0),
            )

        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        controls.columnconfigure(3, weight=1)

        # Add a Matplotlib colormap selector immediately below Map color scale.
        color_scale_row = None
        for widget in controls.winfo_children():
            try:
                if str(widget.cget("text")).strip() == "Map color scale":
                    color_scale_row = int(widget.grid_info().get("row", -1))
                    break
            except Exception:
                pass
        if color_scale_row is not None and color_scale_row >= 0:
            row_widgets = []
            for widget in controls.winfo_children():
                info = widget.grid_info()
                if not info or "row" not in info:
                    continue
                row = int(info["row"])
                if row > color_scale_row:
                    row_widgets.append((row, widget))
            for row, widget in sorted(
                row_widgets, key=lambda item: item[0], reverse=True
            ):
                widget.grid_configure(row=row + 1)

            cmap_row = color_scale_row + 1
            viewer.ttk.Label(controls, text="Colormap").grid(
                row=cmap_row, column=0, sticky="w", pady=(4, 0)
            )
            viewer.ttk.Combobox(
                controls,
                textvariable=self.map_colormap_var,
                values=tuple(sorted(mpl.colormaps)),
                state="readonly",
                width=25,
            ).grid(
                row=cmap_row, column=1, columnspan=2, sticky="ew", pady=(4, 0)
            )

        # Split the inherited full-width action row into Create and Halt
        # controls.  Halt is enabled only while a consecutive map sequence is
        # being generated.
        for widget in controls.winfo_children():
            try:
                if str(widget.cget("text")).strip() != "Create selected plot(s)":
                    continue
                self._create_plots_button = widget
                button_row = int(widget.grid_info().get("row", 0))
                widget.grid_configure(
                    row=button_row,
                    column=0,
                    columnspan=2,
                    sticky="ew",
                    padx=(0, 3),
                )
                self._halt_map_button = viewer.ttk.Button(
                    controls,
                    text="Halt process",
                    command=self.halt_map_generation,
                    state="disabled",
                )
                self._halt_map_button.grid(
                    row=button_row,
                    column=2,
                    columnspan=2,
                    sticky="ew",
                    padx=(3, 0),
                    pady=(8, 0),
                )
                break
            except Exception:
                continue

    def _build_right_panel(self, parent) -> None:
        """Build the inherited plots and label the stack button for multi-selection."""
        super()._build_right_panel(parent)
        stack = [parent]
        while stack:
            widget = stack.pop()
            try:
                if str(widget.cget("text")).strip() == "Add selected station to stack →":
                    widget.configure(text="Add selected station(s) to the stack →")
                    break
            except Exception:
                pass
            try:
                stack.extend(widget.winfo_children())
            except Exception:
                pass

    def _on_sm_indices_mode_change(self) -> None:
        # Do not dispatch to the parent mode handler: v27 does not know this
        # added value and can otherwise fall back to a stacked-component mode.
        self.plot_mode_var.set("sm_indices")
        try:
            self.status_var.set(
                "SuperMAG-index plot mode selected. Click Create selected plots."
            )
        except Exception:
            pass

    def _dispatch_create_selected_plots(self) -> None:
        """Compatibility entry point for any callbacks retained by older GUIs."""
        self.create_selected_plots()

    def create_selected_plots(self) -> None:
        """Dispatch index plots before the parent checks magnetometer data.

        ``SuperMAGDownloadViewer._build_left_panel`` binds its button to this
        method through normal dynamic dispatch.  Keeping the dispatch here
        makes the two data sources independent: an indices plot needs only the
        indices cache, while all original modes continue through the parent.
        """
        if self.plot_mode_var.get().strip() == "sm_indices":
            self.plot_supermag_indices()
            return
        super().create_selected_plots()

    @classmethod
    def _find_supermag_index_value(cls, value: Any, target: str) -> Any:
        if not isinstance(value, dict):
            return None
        for key, candidate in value.items():
            if str(key).lower() != target.lower():
                continue
            if isinstance(candidate, dict):
                for nested_key in ("value", "index", "all", target, target.lower(), target.upper()):
                    if nested_key in candidate:
                        return candidate[nested_key]
            return candidate
        for candidate in value.values():
            if isinstance(candidate, dict):
                found = cls._find_supermag_index_value(candidate, target)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _normalize_supermag_index_value(value: Any) -> Any:
        if value is None:
            return np.nan
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            return [ExploreMagAnalyzer._normalize_supermag_index_value(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    @staticmethod
    def _index_value_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, tuple, np.ndarray)):
            return len(value) == 0 or all(
                ExploreMagAnalyzer._index_value_missing(item) for item in value
            )
        try:
            return bool(np.isnan(float(value)))
        except (TypeError, ValueError):
            return str(value).strip() == ""

    @staticmethod
    def _pack_supermag_index_column(values: list[Any]) -> np.ndarray:
        scalar = all(not isinstance(value, (list, tuple, np.ndarray)) for value in values)
        if scalar:
            converted = []
            numeric = True
            for value in values:
                try:
                    converted.append(float(value))
                except (TypeError, ValueError):
                    numeric = False
                    break
            if numeric:
                return np.asarray(converted, dtype=float)
        return np.asarray(values, dtype=object)

    @staticmethod
    def _coerce_supermag_index_time(record: dict[str, Any]) -> Optional[datetime]:
        for key in ("tval", "time", "timestamp", "datetime", "date"):
            value = None
            for actual_key, candidate in record.items():
                if str(actual_key).lower() == key:
                    value = candidate
                    break
            if value is None:
                continue
            if isinstance(value, dict):
                value = value.get("value", value.get("tval"))
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None and np.isfinite(numeric):
                # SuperMAG tval is Unix time in seconds.
                return datetime.fromtimestamp(numeric, tz=timezone.utc).replace(tzinfo=None)
            if isinstance(value, str):
                candidate = value.strip().replace("Z", "+00:00")
                try:
                    parsed = datetime.fromisoformat(candidate)
                except ValueError:
                    continue
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                return parsed
        return None

    @classmethod
    def _decode_supermag_indices_response(
        cls, raw: bytes
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        response_text = raw.decode("utf-8", errors="replace").strip()
        if hasattr(viewer, "_is_supermag_backend_failure") and viewer._is_supermag_backend_failure(
            response_text
        ):
            raise RuntimeError("SuperMAG returned its temporary PHP/logon backend error.")
        if not response_text or "<html" in response_text.lower() or "fatal error" in response_text.lower():
            raise ValueError("Unexpected SuperMAG indices response:\n" + response_text[:1200])

        starts = [
            position
            for position in (response_text.find("["), response_text.find("{"))
            if position >= 0
        ]
        if starts:
            response_text = response_text[min(starts):]
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            closing = max(response_text.rfind("]"), response_text.rfind("}"))
            if closing < 0:
                raise ValueError("SuperMAG returned malformed indices JSON.") from exc
            try:
                payload = json.loads(response_text[: closing + 1])
            except json.JSONDecodeError as nested_exc:
                raise ValueError("SuperMAG returned malformed indices JSON.") from nested_exc

        if isinstance(payload, dict):
            for key in ("data", "indices", "records"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("SuperMAG indices response did not contain a record list.")

        records: list[tuple[datetime, dict[str, Any]]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            sample_time = cls._coerce_supermag_index_time(item)
            if sample_time is None:
                continue
            values = {
                key: cls._normalize_supermag_index_value(
                    cls._find_supermag_index_value(item, key)
                )
                for key in cls.SUPERMAG_INDEX_KEYS
            }
            if all(cls._index_value_missing(value) for value in values.values()):
                continue
            records.append((sample_time, values))

        if not records:
            raise ValueError("No SuperMAG index samples were returned.")
        records.sort(key=lambda item: item[0])
        times = np.asarray([item[0] for item in records], dtype=object)
        index_data = {
            key: cls._pack_supermag_index_column([item[1][key] for item in records])
            for key in cls.SUPERMAG_INDEX_KEYS
        }
        return times, index_data

    def _fetch_supermag_indices(
        self,
        start: datetime,
        end: datetime,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        extent = int((end - start).total_seconds())
        if extent <= 0:
            raise ValueError("End time must be later than start time.")
        logon = self.logon_var.get().strip()
        if not logon:
            raise ValueError("Enter your SuperMAG logon before downloading indices.")

        query = urllib.parse.urlencode(
            {
                "start": start.strftime("%Y-%m-%dT%H:%M"),
                "extent": f"{extent:012d}",
                "logon": logon,
            }
        )
        requested = urllib.parse.quote(",".join(self.SUPERMAG_INDEX_KEYS), safe=",")
        url = (
            f"{viewer.BASE_URL}indices.php?fmt=json&python&nohead&{query}"
            f"&indices={requested}"
        )
        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            self.status_var.set(
                f"Downloading all SuperMAG indices (attempt {attempt}/3)…"
            )
            self.root.update_idletasks()
            try:
                raw = viewer._open_supermag_url(url, timeout=180)
                times, data = self._decode_supermag_indices_response(raw)
                self._last_supermag_indices = (start, end, times, data)
                return times, data
            except Exception as exc:
                last_error = exc
                if attempt < 3 and hasattr(viewer, "_wait_for_retry"):
                    viewer._wait_for_retry(min(2.0 * attempt, 4.0))
        assert last_error is not None
        raise last_error

    def choose_supermag_indices_output_file(self) -> None:
        try:
            start, end = self._download_query_interval()
            initial = f"SuperMAG_indices_{start:%Y%m%d_%H%M%S}_{end:%Y%m%d_%H%M%S}.nc"
        except Exception:
            initial = "{start:%Y%m%d}_supermag_indices.nc"
        path = viewer.filedialog.asksaveasfilename(
            title="Save SuperMAG indices NetCDF",
            defaultextension=".nc",
            filetypes=(("NetCDF files", "*.nc"), ("All files", "*.*")),
            initialfile=initial,
        )
        if path:
            self.indices_output_file_var.set(path)

    @staticmethod
    def _netcdf_value_array(variable) -> np.ndarray:
        """Read a NetCDF variable while replacing masked values safely."""
        values = variable[:]
        if np.ma.isMaskedArray(values):
            if getattr(values.dtype, "kind", "") in "SUO":
                values = values.filled("")
            else:
                values = values.filled(np.nan)
        return np.asarray(values)

    @staticmethod
    def _decode_netcdf_time_variable(time_variable) -> np.ndarray:
        """Decode the UNIX_TIME variable written by this application."""
        raw = ExploreMagAnalyzer._netcdf_value_array(time_variable)
        raw = np.asarray(raw, dtype=float).reshape(-1)
        units = str(getattr(time_variable, "units", "")).lower()

        # Files written by this application store Unix seconds.  Use the same
        # interpretation when units are absent, because the variable is named
        # UNIX_TIME and this avoids a dependency on netCDF4.num2date exports.
        if not units or "1970-01-01" in units or "unix" in units:
            output = []
            for value in raw:
                if not np.isfinite(value):
                    output.append(None)
                else:
                    output.append(
                        datetime.fromtimestamp(
                            float(value), tz=timezone.utc
                        ).replace(tzinfo=None)
                    )
            return np.asarray(output, dtype=object)

        num2date = getattr(viewer, "num2date", None)
        if num2date is None:
            raise ValueError(
                f"Unsupported NetCDF time units: {getattr(time_variable, 'units', '')}"
            )
        calendar = getattr(time_variable, "calendar", "standard")
        decoded = num2date(raw, units=time_variable.units, calendar=calendar)
        output = []
        for value in np.asarray(decoded, dtype=object).reshape(-1):
            if hasattr(value, "to_pydatetime"):
                value = value.to_pydatetime()
            if not isinstance(value, datetime):
                value = datetime(
                    value.year, value.month, value.day,
                    value.hour, value.minute, value.second,
                    value.microsecond,
                )
            if value.tzinfo is not None:
                value = value.astimezone(timezone.utc).replace(tzinfo=None)
            output.append(value)
        return np.asarray(output, dtype=object)

    @classmethod
    def _load_supermag_indices_netcdf(
        cls, path: str | Path
    ) -> tuple[datetime, datetime, np.ndarray, dict[str, np.ndarray]]:
        """Read a saved SuperMAG-index NetCDF file into the plot cache."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"SuperMAG indices file not found:\n{source}")

        with viewer.Dataset(str(source), "r") as dataset:
            time_variable = None
            for candidate in ("UNIX_TIME", "unix_time", "TIME", "time"):
                if candidate in dataset.variables:
                    time_variable = dataset.variables[candidate]
                    break
            if time_variable is None:
                raise ValueError(
                    "The selected NetCDF file has no UNIX_TIME/time variable."
                )

            times = cls._decode_netcdf_time_variable(time_variable)
            valid_time_mask = np.asarray(
                [isinstance(value, datetime) for value in times], dtype=bool
            )
            if not np.any(valid_time_mask):
                raise ValueError("The indices file contains no valid timestamps.")

            variable_by_api_key: dict[str, Any] = {}
            for variable in dataset.variables.values():
                api_key = str(getattr(variable, "api_key", "")).strip().lower()
                if api_key:
                    variable_by_api_key[api_key] = variable

            index_data: dict[str, np.ndarray] = {}
            for key in cls.SUPERMAG_INDEX_KEYS:
                variable = variable_by_api_key.get(key)
                if variable is None:
                    candidates = (
                        cls.SUPERMAG_INDEX_VAR_NAMES[key],
                        key,
                        key.upper(),
                    )
                    for candidate in candidates:
                        if candidate in dataset.variables:
                            variable = dataset.variables[candidate]
                            break
                if variable is None:
                    index_data[key] = np.full(len(times), np.nan, dtype=float)
                    continue

                values = cls._netcdf_value_array(variable)
                if values.ndim == 0:
                    values = np.repeat(values.reshape(1), len(times), axis=0)
                if values.shape[0] != len(times):
                    raise ValueError(
                        f"Variable {variable.name} has {values.shape[0]} samples, "
                        f"but the time coordinate has {len(times)}."
                    )
                index_data[key] = values

            finite_plot_series = 0
            for keys, _title in cls.SUPERMAG_INDEX_PLOT_GROUPS:
                for key in keys:
                    values = cls._numeric_index_series(index_data, key)
                    if len(values) == len(times) and np.any(np.isfinite(values)):
                        finite_plot_series += 1
            if finite_plot_series == 0:
                raise ValueError(
                    "The selected file contains none of the plottable SuperMAG "
                    "index variables (SME/SML/SMU and related indices)."
                )

            valid_times = times[valid_time_mask]
            start = valid_times[0]
            end = valid_times[-1]
            start_attr = str(getattr(dataset, "request_start_utc", "")).strip()
            end_attr = str(getattr(dataset, "request_end_utc", "")).strip()
            for text_value, target in ((start_attr, "start"), (end_attr, "end")):
                if not text_value:
                    continue
                try:
                    parsed = cls._parse_gui_datetime(
                        text_value,
                        f"Indices file {target}",
                    )
                except ValueError:
                    continue
                if target == "start":
                    start = parsed
                else:
                    end = parsed

        return start, end, times, index_data

    def open_supermag_indices_file(self) -> None:
        """Choose and read a previously saved SuperMAG-index NetCDF file."""
        current = Path(self.indices_output_file_var.get().strip()).expanduser()
        initial_directory = str(current.parent) if current.parent.is_dir() else None
        initial_file = current.name if current.suffix.lower() == ".nc" else None
        options = dict(
            title="Open SuperMAG indices NetCDF",
            filetypes=(("NetCDF files", "*.nc"), ("All files", "*.*")),
        )
        if initial_directory:
            options["initialdir"] = initial_directory
        if initial_file:
            options["initialfile"] = initial_file
        path = viewer.filedialog.askopenfilename(**options)
        if not path:
            return

        try:
            start, end, times, index_data = self._load_supermag_indices_netcdf(path)
        except Exception as exc:
            self.status_var.set("Could not open the SuperMAG-index file.")
            viewer.messagebox.showerror("Open SuperMAG indices", str(exc))
            return

        self._last_supermag_indices = (start, end, times, index_data)
        self._supermag_indices_file = Path(path).expanduser()
        self.indices_output_file_var.set(str(self._supermag_indices_file))
        self.status_var.set(
            f"Opened {len(times)} SuperMAG-index samples: "
            f"{self._supermag_indices_file}"
        )
        viewer.messagebox.showinfo(
            "SuperMAG indices opened",
            f"Read {len(times)} samples from:\n\n{self._supermag_indices_file}",
        )

    @staticmethod
    def _index_sequence(value: Any) -> list[Any]:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _save_supermag_indices_netcdf(
        self,
        path: str | Path,
        times: np.ndarray,
        index_data: dict[str, np.ndarray],
        start: datetime,
        end: datetime,
    ) -> Path:
        output = Path(path).expanduser()
        if output.suffix.lower() != ".nc":
            output = output.with_suffix(".nc")
        output.parent.mkdir(parents=True, exist_ok=True)

        with viewer.Dataset(str(output), "w", format="NETCDF4") as dataset:
            dataset.title = "SuperMAG indices"
            dataset.source = "SuperMAG indices API"
            dataset.history = (
                f"Created {datetime.now(timezone.utc).isoformat()} by "
                "ExploreMag.py"
            )
            dataset.request_start_utc = start.strftime("%Y-%m-%d %H:%M:%S")
            dataset.request_end_utc = end.strftime("%Y-%m-%d %H:%M:%S")
            dataset.requested_api_keys = ",".join(self.SUPERMAG_INDEX_KEYS)
            dataset.createDimension("UNIX_TIME", len(times))
            time_var = dataset.createVariable("UNIX_TIME", "f8", ("UNIX_TIME",))
            time_var.units = getattr(
                viewer, "TIME_UNITS", "seconds since 1970-01-01 00:00:00 UTC"
            )
            time_var.calendar = getattr(viewer, "TIME_CALENDAR", "standard")
            time_var[:] = np.asarray(
                [
                    value.replace(tzinfo=timezone.utc).timestamp()
                    if value.tzinfo is None
                    else value.astimezone(timezone.utc).timestamp()
                    for value in times
                ],
                dtype=float,
            )

            for key in self.SUPERMAG_INDEX_KEYS:
                values = list(index_data.get(key, np.full(len(times), np.nan)))
                sequences = [self._index_sequence(value) for value in values]
                width = max((len(value) for value in sequences), default=1)
                name = self.SUPERMAG_INDEX_VAR_NAMES[key]
                string_hint = "stid" in key
                numeric = not string_hint
                if numeric:
                    for sequence in sequences:
                        for value in sequence:
                            if self._index_value_missing(value):
                                continue
                            try:
                                float(value)
                            except (TypeError, ValueError):
                                numeric = False
                                break
                        if not numeric:
                            break

                dimensions = ("UNIX_TIME",)
                if width > 1:
                    dimension_name = f"{name}_ELEMENT"
                    dataset.createDimension(dimension_name, width)
                    dimensions = ("UNIX_TIME", dimension_name)

                if numeric:
                    array = np.full((len(times), width), np.nan, dtype=float)
                    for row, sequence in enumerate(sequences):
                        for column, value in enumerate(sequence[:width]):
                            try:
                                array[row, column] = float(value)
                            except (TypeError, ValueError):
                                pass
                    variable = dataset.createVariable(name, "f8", dimensions)
                    variable[:] = array[:, 0] if width == 1 else array
                else:
                    array = np.full((len(times), width), "", dtype=object)
                    for row, sequence in enumerate(sequences):
                        for column, value in enumerate(sequence[:width]):
                            if not self._index_value_missing(value):
                                array[row, column] = str(value)
                    variable = dataset.createVariable(name, str, dimensions)
                    variable[:] = array[:, 0] if width == 1 else array
                variable.api_key = key
                if key.startswith(("sme", "sml", "smu", "smr")):
                    variable.units = "nT"

        return output

    def download_and_save_supermag_indices(self) -> None:
        try:
            start, end = self._download_query_interval()
            output_text = self.indices_output_file_var.get().strip()
            if not output_text:
                raise ValueError("Choose an output NetCDF filename for the indices.")
            times, index_data = self._fetch_supermag_indices(start, end)
            output = self._save_supermag_indices_netcdf(
                output_text, times, index_data, start, end
            )
        except Exception as exc:
            message = (
                viewer.summarize_supermag_api_error(exc)
                if hasattr(viewer, "summarize_supermag_api_error")
                else str(exc)
            )
            self.status_var.set("SuperMAG-index download/save failed.")
            viewer.messagebox.showerror("SuperMAG indices", message)
            return

        self.indices_output_file_var.set(str(output))
        self._supermag_indices_file = output
        self.status_var.set(f"Saved {len(times)} SuperMAG-index samples: {output}")
        viewer.messagebox.showinfo(
            "SuperMAG indices saved",
            f"Saved {len(times)} samples and all requested index variables to:\n\n{output}",
        )

    @classmethod
    def _numeric_index_series(
        cls, index_data: dict[str, np.ndarray], key: str
    ) -> np.ndarray:
        source = index_data.get(key)
        if source is None:
            return np.array([], dtype=float)
        source = np.asarray(source, dtype=object)
        output = np.full(len(source), np.nan, dtype=float)
        for index, value in enumerate(source):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                value = value[0] if value else np.nan
            try:
                output[index] = float(value)
            except (TypeError, ValueError):
                pass
        output[np.abs(output) > cls.SUPERMAG_INDEX_ABSOLUTE_LIMIT] = np.nan
        return output

    @classmethod
    def _numeric_index_matrix(
        cls,
        index_data: dict[str, np.ndarray],
        key: str,
        expected_time_count: Optional[int] = None,
    ) -> np.ndarray:
        """Return a numeric time-by-element matrix for a SuperMAG index.

        SuperMAG regional indices such as SMU_R and SML_R are normally stored
        as ``(time, MLT)`` arrays.  API responses may instead arrive as an
        object array containing one sequence per time sample, so both layouts
        are normalized here.
        """
        source = index_data.get(key)
        if source is None:
            return np.empty((0, 0), dtype=float)

        raw = np.asarray(source)
        if np.ma.isMaskedArray(raw):
            raw = np.ma.filled(raw, np.nan)

        if raw.ndim == 0:
            raw = raw.reshape(1, 1)
        elif raw.ndim == 1:
            rows: list[list[Any]] = []
            maximum_width = 1
            for value in raw:
                if isinstance(value, np.ndarray):
                    sequence = value.reshape(-1).tolist()
                elif isinstance(value, (list, tuple)):
                    sequence = list(value)
                else:
                    sequence = [value]
                rows.append(sequence)
                maximum_width = max(maximum_width, len(sequence))
            expanded = np.full((len(rows), maximum_width), np.nan, dtype=object)
            for row_index, sequence in enumerate(rows):
                expanded[row_index, : len(sequence)] = sequence
            raw = expanded
        else:
            raw = raw.reshape(raw.shape[0], -1)

        if (
            expected_time_count is not None
            and raw.shape[0] != expected_time_count
            and raw.ndim == 2
            and raw.shape[1] == expected_time_count
        ):
            raw = raw.T

        output = np.full(raw.shape, np.nan, dtype=float)
        for index in np.ndindex(raw.shape):
            try:
                output[index] = float(raw[index])
            except (TypeError, ValueError):
                pass
        output[np.abs(output) > cls.SUPERMAG_INDEX_ABSOLUTE_LIMIT] = np.nan
        return output

    @classmethod
    def _regional_mlt_matrix(
        cls,
        index_data: dict[str, np.ndarray],
        key: str,
        time_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and return a regional index matrix with integer MLT bins."""
        matrix = cls._numeric_index_matrix(
            index_data, key, expected_time_count=time_count
        )
        variable_name = cls.SUPERMAG_INDEX_VAR_NAMES[key]
        if matrix.ndim != 2 or matrix.shape[0] != time_count:
            raise ValueError(
                f"{variable_name} must have shape (time, MLT); got {matrix.shape}."
            )
        if matrix.shape[1] <= max(cls.SUPERMAG_R_SELECTED_MLTS):
            raise ValueError(
                f"{variable_name} has only {matrix.shape[1]} MLT columns; "
                "columns 00, 06, 12 and 18 are required."
            )
        if not np.any(np.isfinite(matrix)):
            raise ValueError(f"{variable_name} contains no finite values.")
        return matrix, np.arange(matrix.shape[1], dtype=int)

    @classmethod
    def _plot_regional_mlt_lines(
        cls,
        axis,
        times: np.ndarray,
        matrix: np.ndarray,
        variable_name: str,
    ) -> int:
        """Plot the 00, 06, 12 and 18 MLT columns of a regional index."""
        plotted = 0
        for mlt_value in cls.SUPERMAG_R_SELECTED_MLTS:
            values = matrix[:, int(mlt_value)]
            if not np.any(np.isfinite(values)):
                continue
            axis.plot(
                times,
                values,
                linewidth=1.0,
                label=f"{variable_name} {mlt_value:02d} MLT",
            )
            plotted += 1
        return plotted

    @classmethod
    def _supermag_r_image_figure(
        cls,
        times: np.ndarray,
        smu_r: np.ndarray,
        sml_r: np.ndarray,
        source_name: str,
        start: datetime,
        end: datetime,
    ) -> Figure:
        """Create the second-window time-versus-MLT SMU_R/SML_R figure."""
        time_numbers = mdates.date2num(np.asarray(times, dtype=object))
        if len(time_numbers) > 1:
            differences = np.diff(time_numbers)
            differences = differences[np.isfinite(differences) & (differences > 0.0)]
            half_step = float(np.nanmedian(differences) / 2.0) if len(differences) else 0.0
        else:
            half_step = 0.5 / (24.0 * 60.0)
        if not np.isfinite(half_step) or half_step <= 0.0:
            half_step = 0.5 / (24.0 * 60.0)

        x_extent = [time_numbers[0] - half_step, time_numbers[-1] + half_step]
        maximum_mlt_count = max(smu_r.shape[1], sml_r.shape[1])
        ticks = [
            value for value in cls.SUPERMAG_R_MLT_TICKS
            if value < maximum_mlt_count
        ]

        figure = Figure(figsize=(14, 8.5), dpi=100)
        axes = [figure.add_subplot(2, 1, 1)]
        axes.append(figure.add_subplot(2, 1, 2, sharex=axes[0]))

        image0 = axes[0].imshow(
            smu_r.T,
            aspect="auto",
            origin="lower",
            extent=[x_extent[0], x_extent[1], -0.5, smu_r.shape[1] - 0.5],
            cmap="RdYlBu_r",
        )
        axes[0].set_title("SMU: time versus magnetic local time")
        axes[0].set_ylabel("MLT")
        axes[0].set_yticks([value for value in ticks if value < smu_r.shape[1]])
        axes[0].set_yticklabels(
            [f"{value:02d}" for value in ticks if value < smu_r.shape[1]]
        )
        axes[0].set_ylim(-0.5, smu_r.shape[1] - 0.5)
        axes[0].tick_params(labelbottom=False)
        figure.colorbar(image0, ax=axes[0], pad=0.012, label="SMU (nT)")

        image1 = axes[1].imshow(
            sml_r.T,
            aspect="auto",
            origin="lower",
            extent=[x_extent[0], x_extent[1], -0.5, sml_r.shape[1] - 0.5],
            cmap="PuOr",
        )
        axes[1].set_title("SML: time versus magnetic local time")
        axes[1].set_ylabel("MLT")
        axes[1].set_xlabel("UTC (HH:MM)")
        axes[1].set_yticks([value for value in ticks if value < sml_r.shape[1]])
        axes[1].set_yticklabels(
            [f"{value:02d}" for value in ticks if value < sml_r.shape[1]]
        )
        axes[1].set_ylim(-0.5, sml_r.shape[1] - 0.5)
        figure.colorbar(image1, ax=axes[1], pad=0.012, label="SML (nT)")

        axes[1].xaxis_date()
        cls._configure_time_axis(axes[1])
        axes[1].set_xlim(times[0], times[-1])
        figure.suptitle(
            f"SuperMAG regional indices by MLT — {source_name}\n"
            f"{cls._format_gui_time(start)} to {cls._format_gui_time(end)} UTC",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        return figure

    def _supermag_indices_for_plot(
        self,
    ) -> tuple[datetime, datetime, np.ndarray, dict[str, np.ndarray]]:
        """Return opened/downloaded local indices, loading the path field if needed."""
        path_text = self.indices_output_file_var.get().strip()
        path = Path(path_text).expanduser() if path_text else None

        # If the field points to a different existing file, read it immediately.
        if path is not None and path.is_file() and path != self._supermag_indices_file:
            loaded = self._load_supermag_indices_netcdf(path)
            self._last_supermag_indices = loaded
            self._supermag_indices_file = path

        if self._last_supermag_indices is None:
            if path is not None and path.is_file():
                loaded = self._load_supermag_indices_netcdf(path)
                self._last_supermag_indices = loaded
                self._supermag_indices_file = path
            else:
                raise ValueError(
                    "Open a SuperMAG indices NetCDF file, or use Download and save "
                    "SuperMAG indices first."
                )
        return self._last_supermag_indices

    @staticmethod
    def _slice_supermag_index_data(
        index_data: dict[str, np.ndarray], mask: np.ndarray
    ) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for key, values in index_data.items():
            array = np.asarray(values)
            if array.ndim == 0:
                output[key] = array
            elif array.shape[0] == len(mask):
                output[key] = array[mask]
            else:
                output[key] = array
        return output

    def plot_supermag_indices(self) -> None:
        """Plot line indices and open a second SMU_R/SML_R time-MLT window."""
        try:
            file_start, file_end, all_times, all_index_data = (
                self._supermag_indices_for_plot()
            )
            all_times = np.asarray(all_times, dtype=object)
            valid = np.asarray(
                [isinstance(value, datetime) for value in all_times], dtype=bool
            )
            if np.count_nonzero(valid) < 2:
                raise ValueError(
                    "The SuperMAG indices file contains fewer than two valid times."
                )
            all_times = all_times[valid]
            all_index_data = self._slice_supermag_index_data(all_index_data, valid)
            sort_order = np.argsort(
                np.asarray([value.timestamp() for value in all_times], dtype=float)
            )
            all_times = all_times[sort_order]
            all_index_data = self._slice_supermag_index_data(
                all_index_data, sort_order
            )

            # Respect the main Plot start/end entries when they are available.
            requested_start = file_start
            requested_end = file_end
            interval_was_read = False
            try:
                if hasattr(self, "_parse_map_time_entries"):
                    requested_start, requested_end = self._parse_map_time_entries()
                    interval_was_read = True
                elif self.data is not None:
                    requested_start, requested_end = self._plot_interval()
                    interval_was_read = True
            except Exception:
                requested_start, requested_end = file_start, file_end

            mask = np.asarray(
                [
                    requested_start <= value <= requested_end
                    for value in all_times
                ],
                dtype=bool,
            )
            if np.count_nonzero(mask) < 2:
                if interval_was_read:
                    raise ValueError(
                        "The selected Plot start/end interval does not overlap the "
                        "opened SuperMAG indices file."
                    )
                mask = np.ones(len(all_times), dtype=bool)

            times = all_times[mask]
            index_data = self._slice_supermag_index_data(all_index_data, mask)
            start = times[0]
            end = times[-1]
            smu_r, _ = self._regional_mlt_matrix(
                index_data, "smur", len(times)
            )
            sml_r, _ = self._regional_mlt_matrix(
                index_data, "smlr", len(times)
            )
        except Exception as exc:
            self.status_var.set("SuperMAG-index plot failed.")
            viewer.messagebox.showerror("SuperMAG indices", str(exc))
            return

        figure = Figure(figsize=(14, 11.5), dpi=100)
        axes = [figure.add_subplot(6, 1, 1)]
        axes.extend(
            figure.add_subplot(6, 1, row, sharex=axes[0])
            for row in range(2, 7)
        )

        # Panels 1-3: scalar global, sunlit-sector, and dark-sector indices.
        for axis, (keys, title) in zip(
            axes[:3], self.SUPERMAG_INDEX_PLOT_GROUPS[:3]
        ):
            plotted = 0
            for key in keys:
                values = self._numeric_index_series(index_data, key)
                if len(values) != len(times) or not np.any(np.isfinite(values)):
                    continue
                axis.plot(
                    times,
                    values,
                    linewidth=1.0,
                    label=self.SUPERMAG_INDEX_VAR_NAMES[key],
                )
                plotted += 1
            axis.axhline(0.0, color="0.55", linewidth=0.7, zorder=0)
            axis.set_ylabel("nT")
            axis.set_title(title, fontsize=10)
            axis.grid(True, alpha=0.28)
            if plotted:
                axis.legend(loc="upper right", ncol=min(3, plotted), fontsize=8)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No finite values in selected interval",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )

        # Panel 4: SMU_R columns corresponding to 00, 06, 12 and 18 MLT.
        regional_specs = (
            (axes[3], smu_r, "SMU", "SMU_R at selected MLT sectors"),
            (axes[4], sml_r, "SML", "SML_R at selected MLT sectors"),
        )
        for axis, matrix, variable_name, title in regional_specs:
            plotted = self._plot_regional_mlt_lines(
                axis, times, matrix, variable_name
            )
            axis.axhline(0.0, color="0.55", linewidth=0.7, zorder=0)
            axis.set_ylabel("nT")
            axis.set_title(title, fontsize=10)
            axis.grid(True, alpha=0.28)
            if plotted:
                axis.legend(
                    loc="upper right",
                    ncol=min(4, plotted),
                    fontsize=8,
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No finite values at 00, 06, 12 or 18 MLT",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )

        # Panel 6: scalar SMR index.
        smr_values = self._numeric_index_series(index_data, "smr")
        if len(smr_values) == len(times) and np.any(np.isfinite(smr_values)):
            axes[5].plot(times, smr_values, linewidth=1.0, label="SMR")
            axes[5].legend(loc="upper right", fontsize=8)
        else:
            axes[5].text(
                0.5,
                0.5,
                "No finite values in selected interval",
                transform=axes[5].transAxes,
                ha="center",
                va="center",
            )
        axes[5].axhline(0.0, color="0.55", linewidth=0.7, zorder=0)
        axes[5].set_ylabel("nT")
        axes[5].set_title("Partial ring-current index", fontsize=10)
        axes[5].grid(True, alpha=0.28)

        for axis in axes[:-1]:
            axis.tick_params(labelbottom=False)
        axes[-1].set_xlabel("UTC (HH:MM)")
        axes[-1].set_xlim(times[0], times[-1])
        self._configure_time_axis(axes[-1])
        source_name = (
            self._supermag_indices_file.name
            if self._supermag_indices_file is not None
            else "downloaded indices"
        )
        figure.suptitle(
            f"SuperMAG indices — {source_name}\n"
            f"{self._format_gui_time(start)} to {self._format_gui_time(end)} UTC",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.965))
        self._new_figure_window(
            "SuperMAG indices",
            figure,
            "supermag_indices.png",
            geometry="1450x950",
            cursor_axes=axes,
        )

        regional_figure = self._supermag_r_image_figure(
            times, smu_r, sml_r, source_name, start, end
        )
        self._new_figure_window(
            "SuperMAG SMU_R and SML_R by MLT",
            regional_figure,
            "supermag_SMU_R_SML_R_time_MLT.png",
            geometry="1450x900",
        )
        self.status_var.set(
            f"Plotted {len(times)} SuperMAG-index samples from {source_name}; "
            "opened line and time-MLT image windows."
        )

    # Backward-compatible name retained for older callbacks or external code.
    def download_and_plot_supermag_indices(self) -> None:
        self.plot_supermag_indices()

    def _build_right_panel(self, parent) -> None:
        super()._build_right_panel(parent)
        parent.rowconfigure(2, weight=0)

        frame = viewer.ttk.LabelFrame(parent, text="Pulsation analysis", padding=6)
        frame.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        frame.columnconfigure(0, weight=0)
        for column in range(1, 4):
            frame.columnconfigure(column, weight=1)

        viewer.ttk.Label(
            frame,
            text="Single/multiple-site analysis:",
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        viewer.ttk.Button(
            frame,
            text="Wavelet scalogram",
            command=self.show_wavelet_scalograms,
        ).grid(row=0, column=1, sticky="ew", padx=(0, 4))
        viewer.ttk.Button(
            frame,
            text="Pi pulsations",
            command=self.show_pi_pulsation_analysis,
        ).grid(row=0, column=2, sticky="ew", padx=4)
        viewer.ttk.Button(
            frame,
            text="Pc pulsations",
            command=self.show_pc_pulsation_analysis,
        ).grid(row=0, column=3, sticky="ew", padx=(4, 0))

        viewer.ttk.Label(
            frame,
            text="Map analysis for selected stations:",
        ).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        viewer.ttk.Button(
            frame,
            text="Pi2 ellipse/Polarization",
            command=self.open_pi2_analysis_window,
        ).grid(row=1, column=1, sticky="ew", padx=(0, 4), pady=(6, 0))
        viewer.ttk.Button(
            frame,
            text="Total ULF wave power",
            command=self.show_total_ulf_wave_power_map,
        ).grid(row=1, column=2, sticky="ew", padx=4, pady=(6, 0))

        selection_row = viewer.ttk.Frame(frame)
        selection_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        viewer.ttk.Button(
            selection_row,
            text="Select all stations",
            command=self.select_all_stations,
        ).pack(side=viewer.tk.LEFT)
        viewer.ttk.Button(
            selection_row,
            text="Select station according to lat/lon",
            command=self.select_stations_by_plot_region,
        ).pack(side=viewer.tk.LEFT, padx=(5, 0))
        viewer.ttk.Button(
            selection_row,
            text="Clear station selection",
            command=self.clear_station_selection,
        ).pack(side=viewer.tk.LEFT, padx=(5, 0))

        viewer.ttk.Label(
            frame,
            textvariable=self.pi2_status_var,
            wraplength=900,
        ).grid(row=2, column=2, columnspan=2, sticky="w", padx=(8, 0), pady=(6, 0))

    # ------------------------------------------------------------------
    # Multi-station selection
    # ------------------------------------------------------------------

    def set_loaded_data(self, data: viewer.LoadedMagneticData) -> None:
        self.selected_station_indices.clear()
        self._pi2_results.clear()
        self._pi3_ps6_map_cache = None
        self._total_ulf_map_cache = None
        self._map_vector_derivative_cache = None
        super().set_loaded_data(data)
        self.pi2_coordinate_var.set("GEO" if data.geo is not None else "NEZ")
        self.pi2_start_var.set("")
        self.pi2_end_var.set("")
        self.quiet_start_var.set("")
        self.quiet_end_var.set("")
        self.pi2_onset_var.set("")
        self.pi2_status_var.set(
            f"Loaded {len(data.station_codes)} stations. Choose a pulsation analysis."
        )

    def _pi3_ps6_map_data(self) -> np.ndarray:
        """Return >150 s horizontal pulsation amplitude for every station/time."""
        if self.data is None:
            raise ValueError("No magnetic data are loaded.")
        if self._pi3_ps6_map_cache is not None:
            return self._pi3_ps6_map_cache
        self._require_scipy()

        times = self._time_array_ns(self.data.times)
        if len(times) < 16:
            raise ValueError("At least 16 samples are required for a Pi3/Ps6 map.")
        seconds = (
            (times - times[0]).astype("timedelta64[ns]").astype(np.int64) / 1.0e9
        )
        differences = np.diff(seconds)
        differences = differences[np.isfinite(differences) & (differences > 0.0)]
        if not len(differences):
            raise ValueError("Could not determine the magnetic-data cadence.")
        cadence = float(np.median(differences))

        source = viewer.mask_bad_magnetometer_values(self.data.nez[:, :, :2])
        output = np.full(source.shape[:2], np.nan, dtype=float)
        sample_axis = np.arange(len(times), dtype=float)
        for station_index in range(source.shape[0]):
            regular = np.empty((len(times), 2), dtype=float)
            valid_station = True
            for component in range(2):
                values = source[station_index, :, component]
                finite = np.isfinite(values)
                if np.count_nonzero(finite) < 8:
                    valid_station = False
                    break
                regular[:, component] = np.interp(
                    sample_axis, sample_axis[finite], values[finite]
                )
            if not valid_station:
                continue
            try:
                filtered, _ = self._pi_band_components(
                    regular, cadence, 150.0, None
                )
            except ValueError:
                continue
            output[station_index] = np.hypot(filtered[:, 0], filtered[:, 1])

        if not np.any(np.isfinite(output)):
            raise ValueError("No station produced a finite Pi3/Ps6 map value.")
        self._pi3_ps6_map_cache = output
        return output

    @classmethod
    def _map_selection_is_none(cls, value: str) -> bool:
        return str(value).strip().lower() in {"", "none"}

    def _map_layer_description(self) -> str:
        scalar = self.map_parameter_var.get().strip()
        vector = self.map_vector_parameter_var.get().strip()
        layers = []
        if not self._map_selection_is_none(scalar):
            layers.append(scalar)
        if not self._map_selection_is_none(vector):
            layers.append(vector)
        return " + ".join(layers) if layers else "map"

    def _map_output_token_for_layers(self, scalar_token: str = "") -> str:
        vector = self.map_vector_parameter_var.get().strip()
        tokens = [scalar_token] if scalar_token else []
        if not self._map_selection_is_none(vector):
            try:
                tokens.append(self.VECTOR_MAP_SPECS[vector][2])
            except KeyError as exc:
                raise ValueError(f"Unknown vector map parameter: {vector}") from exc
        return "_with_".join(tokens) if tokens else "map"

    def _selected_map_colormap(self) -> str:
        name = self.map_colormap_var.get().strip()
        if not name:
            raise ValueError("Choose a Matplotlib colormap for the scalar map layer.")
        try:
            mpl.colormaps[name]
        except KeyError as exc:
            raise ValueError(f"Unknown Matplotlib colormap: {name}") from exc
        return name

    def _selected_vector_scale(self) -> Optional[float]:
        """Return the fixed reference-vector magnitude, or None for autoscaling."""
        text = self.map_vector_scale_var.get().strip()
        if not text:
            return None
        try:
            scale = float(text)
        except ValueError as exc:
            raise ValueError(
                "Vector scale must be a positive number, or blank for automatic scaling."
            ) from exc
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "Vector scale must be a positive number, or blank for automatic scaling."
            )
        return scale

    def _selected_vector_arrow_size(self) -> float:
        """Return the positive display-size multiplier for map arrows."""
        text = self.map_vector_arrow_size_var.get().strip()
        if not text:
            return 1.0
        try:
            multiplier = float(text)
        except ValueError as exc:
            raise ValueError("Arrow size must be a positive number.") from exc
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("Arrow size must be a positive number.")
        return multiplier

    def _horizontal_vector_derivatives(self) -> tuple[np.ndarray, np.ndarray]:
        if self.data is None:
            raise ValueError("No magnetic data are loaded.")
        if self._map_vector_derivative_cache is None:
            clean = viewer.mask_bad_magnetometer_values(self.data.nez[:, :, :2])
            north_dt = np.asarray(
                [viewer.finite_time_derivative(row, self.data.times) for row in clean[:, :, 0]],
                dtype=float,
            )
            east_dt = np.asarray(
                [viewer.finite_time_derivative(row, self.data.times) for row in clean[:, :, 1]],
                dtype=float,
            )
            self._map_vector_derivative_cache = (north_dt, east_dt)
        return self._map_vector_derivative_cache

    def _vector_components_for_map(
        self, time_index: int
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
        if self.data is None:
            raise ValueError("No magnetic data are loaded.")
        selection = self.map_vector_parameter_var.get().strip()
        if self._map_selection_is_none(selection):
            return (
                np.full(len(self.data.station_codes), np.nan),
                np.full(len(self.data.station_codes), np.nan),
                "",
                "",
            )
        try:
            derivative, rotate_clockwise, _token, long_label = (
                self.VECTOR_MAP_SPECS[selection]
            )
        except KeyError as exc:
            raise ValueError(f"Unknown vector map parameter: {selection}") from exc

        if derivative:
            north_all, east_all = self._horizontal_vector_derivatives()
            north = np.asarray(north_all[:, time_index], dtype=float)
            east = np.asarray(east_all[:, time_index], dtype=float)
            units = "nT/min"
        else:
            clean = viewer.mask_bad_magnetometer_values(self.data.nez[:, :, :2])
            north = np.asarray(clean[:, time_index, 0], dtype=float)
            east = np.asarray(clean[:, time_index, 1], dtype=float)
            units = "nT"

        # In geographic east/north coordinates, a 90° clockwise rotation is
        # (east, north) -> (north, -east).
        if rotate_clockwise:
            east, north = north.copy(), -east.copy()
        return east, north, long_label, units

    @staticmethod
    def _quiver_reference_magnitude(east: np.ndarray, north: np.ndarray) -> float:
        magnitudes = np.hypot(east, north)
        positive = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]
        if not len(positive):
            return 1.0
        return float(np.nanpercentile(positive, 75.0))

    def _create_component_map_figure(
        self,
        values_all_stations: np.ndarray,
        time_index: int,
        station_indices: np.ndarray,
        bounds: tuple[float, float, float, float],
        colorbar_label: str,
        signed_values: bool,
    ) -> Figure:
        """Create a scalar map, a vector map, or a scalar/vector overlay."""
        if self.data is None:
            raise ValueError("No magnetic data are loaded.")

        scalar_selected = not self._map_selection_is_none(
            self.map_parameter_var.get()
        )
        vector_selected = not self._map_selection_is_none(
            self.map_vector_parameter_var.get()
        )
        if not scalar_selected and not vector_selected:
            raise ValueError("Select a scalar map parameter, a vector map parameter, or both.")

        if scalar_selected:
            figure = viewer.SuperMAGDownloadViewer._create_component_map_figure(
                self,
                values_all_stations,
                time_index,
                station_indices,
                bounds,
                colorbar_label,
                signed_values,
            )
            axis = figure.axes[0]

            # Apply the user's Matplotlib colormap to both the interpolated
            # field and the station-value markers.
            from matplotlib.collections import PathCollection, QuadMesh

            cmap_name = self._selected_map_colormap()
            for artist in axis.collections:
                if isinstance(artist, (QuadMesh, PathCollection)):
                    values = artist.get_array()
                    if values is not None and np.size(values):
                        artist.set_cmap(cmap_name)
                        artist.changed()
        else:
            # A vector-only map must not depend on having three stations for
            # scalar interpolation, so build only the geographic base map.
            if not viewer.HAS_CARTOPY:
                raise RuntimeError(
                    "Cartopy is required for vector maps. Install cartopy in "
                    "the Python environment used to run this program."
                )
            selected = np.asarray(station_indices, dtype=int)
            lats = np.asarray(self.data.glat[selected], dtype=float)
            lons = np.asarray(
                viewer.normalize_longitude(self.data.glon[selected]), dtype=float
            )
            finite_locations = np.isfinite(lats) & np.isfinite(lons)
            if not np.any(finite_locations):
                raise ValueError("No finite station locations are available for the vector map.")
            lats = lats[finite_locations]
            lons = lons[finite_locations]
            codes = np.asarray(self.data.station_codes[selected], dtype=str)[finite_locations]
            center_lon, relative_lons, _span = viewer.choose_longitude_window(lons)
            projection = viewer.ccrs.PlateCarree(central_longitude=center_lon)
            figure = Figure(figsize=(11, 8), dpi=100)
            axis = figure.add_subplot(111, projection=projection)
            axis.add_feature(
                viewer.cfeature.LAND,
                facecolor="#f0f0f0",
                edgecolor="black",
                linewidth=0.5,
                zorder=0,
            )
            axis.add_feature(viewer.cfeature.OCEAN, facecolor="white", zorder=0)
            axis.add_feature(viewer.cfeature.COASTLINE, linewidth=0.8)
            axis.add_feature(viewer.cfeature.BORDERS, linestyle=":", linewidth=0.7)
            axis.add_feature(viewer.cfeature.LAKES, alpha=0.5)
            gridlines = axis.gridlines(
                crs=viewer.ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.5,
                alpha=0.5,
                linestyle="--",
            )
            gridlines.top_labels = False
            gridlines.right_labels = False

            rel_min = float(np.nanmin(relative_lons))
            rel_max = float(np.nanmax(relative_lons))
            lat_min = float(np.nanmin(lats))
            lat_max = float(np.nanmax(lats))
            lon_pad = max(3.0, 0.08 * max(rel_max - rel_min, 1.0) + 2.0)
            lat_pad = max(2.5, 0.08 * max(lat_max - lat_min, 1.0) + 1.5)
            axis.set_extent(
                [
                    max(-180.0, rel_min - lon_pad),
                    min(180.0, rel_max + lon_pad),
                    max(-90.0, lat_min - lat_pad),
                    min(90.0, lat_max + lat_pad),
                ],
                crs=projection,
            )
            axis.scatter(
                lons,
                lats,
                s=34,
                facecolors="white",
                edgecolors="black",
                linewidths=0.65,
                transform=viewer.ccrs.PlateCarree(),
                zorder=3,
            )
            for lon, lat, code in zip(lons, lats, codes):
                axis.text(
                    lon,
                    lat,
                    str(code),
                    fontsize=7,
                    ha="left",
                    va="bottom",
                    transform=viewer.ccrs.PlateCarree(),
                    zorder=4,
                )
            figure.subplots_adjust(left=0.06, right=0.96, top=0.91, bottom=0.08)

        vector_count = 0
        if vector_selected:
            selected = np.asarray(station_indices, dtype=int)
            east_all, north_all, vector_label, vector_units = (
                self._vector_components_for_map(time_index)
            )
            east = np.asarray(east_all[selected], dtype=float)
            north = np.asarray(north_all[selected], dtype=float)
            lats = np.asarray(self.data.glat[selected], dtype=float)
            lons = np.asarray(viewer.normalize_longitude(self.data.glon[selected]), dtype=float)
            finite_vectors = (
                np.isfinite(lats)
                & np.isfinite(lons)
                & np.isfinite(east)
                & np.isfinite(north)
                & ((np.abs(east) > 0.0) | (np.abs(north) > 0.0))
            )
            vector_count = int(np.count_nonzero(finite_vectors))
            if vector_count == 0:
                raise ValueError(
                    f"No finite, non-zero vectors are available for {self.map_vector_parameter_var.get()}."
                )

            vector_scale = self._selected_vector_scale()
            arrow_size = self._selected_vector_arrow_size()
            east_to_plot = east[finite_vectors]
            north_to_plot = north[finite_vectors]
            reference = (
                vector_scale
                if vector_scale is not None
                else self._quiver_reference_magnitude(east_to_plot, north_to_plot)
            )
            # Normalize explicitly instead of asking Quiver/GeoAxes to
            # interpret magnetic data units as a physical scale.  A
            # normalized magnitude of 1 is always the displayed reference
            # magnitude, while Arrow size controls its physical length.
            reference_arrow_inches = 0.8 * arrow_size
            east_to_plot = east_to_plot / reference
            north_to_plot = north_to_plot / reference
            quiver_options: dict[str, Any] = {
                "scale": 1.0 / reference_arrow_inches,
                "scale_units": "inches",
                "units": "inches",
                "width": 0.025 * arrow_size,
                "minlength": 0,
            }
            quiver = axis.quiver(
                lons[finite_vectors],
                lats[finite_vectors],
                east_to_plot,
                north_to_plot,
                transform=viewer.ccrs.PlateCarree(),
                color="black",
                alpha=0.92,
                pivot="tail",
                zorder=7,
                **quiver_options,
            )
            axis.quiverkey(
                quiver,
                X=0.78,
                Y=0.035,
                U=1.0,
                label=f"{reference:.3g} {vector_units}",
                labelpos="E",
                coordinates="axes",
                color="black",
            )

        sample_time = self.data.times[time_index]
        scalar_count = int(
            np.count_nonzero(
                np.isfinite(
                    np.asarray(values_all_stations, dtype=float)[
                        np.asarray(station_indices, dtype=int), time_index
                    ]
                )
            )
        ) if scalar_selected else 0
        count_text = []
        if scalar_selected:
            count_text.append(f"{scalar_count} scalar stations")
        if vector_selected:
            count_text.append(f"{vector_count} vectors")
        axis.set_title(
            f"{self._map_layer_description()} at "
            f"{sample_time.strftime(viewer.MANUAL_TIME_FORMAT)} UTC "
            f"({', '.join(count_text)})"
        )
        return figure

    def _display_component_map(self, figure: Figure, default_name: str) -> None:
        """Use the combined layer description in the inherited map window."""
        original = self.map_parameter_var.get()
        try:
            self.map_parameter_var.set(self._map_layer_description())
            viewer.SuperMAGDownloadViewer._display_component_map(
                self, figure, default_name
            )
        finally:
            self.map_parameter_var.set(original)

    def _map_parameter_data(self) -> tuple[np.ndarray, str, str, bool]:
        selection = self.map_parameter_var.get().strip()
        if selection == "Pi3/Ps6":
            return (
                self._pi3_ps6_map_data(),
                "Pi3/Ps6 horizontal amplitude (nT; periods >150 s)",
                "Pi3_Ps6",
                False,
            )
        if selection == "Total ULF":
            return (
                self._total_ulf_map_data(),
                "Running horizontal ULF wave power (nT²; 5–150 s)",
                "Total_ULF",
                False,
            )
        return viewer.SuperMAGDownloadViewer._map_parameter_data(self)

    def _total_ulf_map_data(self) -> np.ndarray:
        """Return running 5–150 s horizontal power for every station and sample."""
        if self.data is None:
            raise ValueError("No magnetic data are loaded.")
        if self._total_ulf_map_cache is not None:
            return self._total_ulf_map_cache
        self._require_scipy()
        times = self._time_array_ns(self.data.times)
        seconds = (
            (times - times[0]).astype("timedelta64[ns]").astype(np.int64) / 1.0e9
        )
        differences = np.diff(seconds)
        differences = differences[np.isfinite(differences) & (differences > 0)]
        if not len(differences):
            raise ValueError("Could not determine the magnetic-data cadence.")
        cadence = float(np.median(differences))
        source = viewer.mask_bad_magnetometer_values(self.data.nez[:, :, :2])
        output = np.full(source.shape[:2], np.nan, dtype=float)
        x = np.arange(len(times), dtype=float)
        for station in range(source.shape[0]):
            regular = np.empty((len(times), 2), dtype=float)
            if any(np.count_nonzero(np.isfinite(source[station, :, c])) < 8 for c in range(2)):
                continue
            for component in range(2):
                finite = np.isfinite(source[station, :, component])
                regular[:, component] = np.interp(
                    x, x[finite], source[station, finite, component]
                )
            try:
                filtered = self._bandpass_components(
                    regular, cadence, *self.TOTAL_ULF_PERIOD_RANGE
                )
            except ValueError:
                continue
            instantaneous = np.sum(filtered ** 2, axis=1)
            output[station] = self._rolling_mean(
                instantaneous, max(3, int(round(10.0 / cadence)))
            )
        if not np.any(np.isfinite(output)):
            raise ValueError("No station produced finite Total ULF power.")
        self._total_ulf_map_cache = output
        return output

    def _map_plot_cadence_seconds(self) -> Optional[float]:
        text = self.map_cadence_var.get().strip()
        if not text:
            return None
        try:
            cadence = float(text)
        except ValueError as exc:
            raise ValueError("Plot cadence must be a positive number of seconds.") from exc
        if not np.isfinite(cadence) or cadence <= 0:
            raise ValueError("Plot cadence must be a positive number of seconds.")
        return cadence

    def halt_map_generation(self) -> None:
        """Request a safe stop before the next consecutive map is created."""
        self._map_generation_halted = True
        try:
            self.map_status_var.set(
                "Halting map generation after the current map operation…"
            )
        except Exception:
            pass

    def _set_map_generation_active(self, active: bool) -> None:
        """Update the Create/Halt button states for a map sequence."""
        if self._create_plots_button is not None:
            self._create_plots_button.configure(
                state="disabled" if active else "normal"
            )
        if self._halt_map_button is not None:
            self._halt_map_button.configure(
                state="normal" if active else "disabled"
            )

    def apply_map_time_entries(self, _event=None) -> bool:
        """Validate both the map interval and optional output cadence."""
        try:
            start, end = self._parse_map_time_entries()
            cadence = self._map_plot_cadence_seconds()
        except ValueError as exc:
            viewer.messagebox.showerror("Map time range", str(exc))
            return False
        cadence_text = (
            "every loaded sample" if cadence is None else f"every {cadence:g} s"
        )
        self.map_status_var.set(
            f"Map interval: {start.strftime(viewer.MANUAL_TIME_FORMAT)} → "
            f"{end.strftime(viewer.MANUAL_TIME_FORMAT)} UTC; {cadence_text}"
        )
        return True

    def create_map_output(self) -> None:
        """Create scalar maps, vector maps, or scalar/vector overlays."""
        if self.data is None:
            viewer.messagebox.showinfo("Map", "Download or open a data file first.")
            return
        try:
            scalar_selected = not self._map_selection_is_none(
                self.map_parameter_var.get()
            )
            vector_selected = not self._map_selection_is_none(
                self.map_vector_parameter_var.get()
            )
            if not scalar_selected and not vector_selected:
                raise ValueError(
                    "Select a scalar map parameter, a vector map parameter, or both."
                )

            start, end = self._parse_map_time_entries()
            station_indices, bounds = self._map_region()
            if scalar_selected:
                map_values, colorbar_label, scalar_token, signed_values = (
                    self._map_parameter_data()
                )
                # Validate the selected colormap before a long map sequence begins.
                self._selected_map_colormap()
            else:
                map_values = np.zeros(
                    (len(self.data.station_codes), len(self.data.times)),
                    dtype=float,
                )
                colorbar_label = ""
                scalar_token = ""
                signed_values = False
            token = self._map_output_token_for_layers(scalar_token)
            cadence = self._map_plot_cadence_seconds()

            if start == end:
                time_indices = [self._nearest_map_time_index(start)]
            elif cadence is None:
                time_indices = [
                    i for i, value in enumerate(self.data.times) if start <= value <= end
                ]
            else:
                requested = []
                value = start
                from datetime import timedelta
                while value <= end:
                    requested.append(value)
                    value += timedelta(seconds=cadence)
                time_indices = []
                for value in requested:
                    index = self._nearest_map_time_index(value)
                    if not time_indices or index != time_indices[-1]:
                        time_indices.append(index)

            if not time_indices:
                raise ValueError("No data samples fall inside the entered map time range.")

            if start == end:
                index = time_indices[0]
                figure = self._create_component_map_figure(
                    map_values,
                    index,
                    station_indices,
                    bounds,
                    colorbar_label,
                    signed_values,
                )
                self._display_component_map(
                    figure, self._map_filename(token, index)
                )
                actual = self.data.times[index]
                self.map_status_var.set(
                    f"Displayed {self._map_layer_description()} at the nearest sample: "
                    f"{actual.strftime(viewer.MANUAL_TIME_FORMAT)} UTC"
                )
                return

            output_directory = self._map_output_directory()
            output_directory.mkdir(parents=True, exist_ok=True)
            self._map_generation_halted = False
            saved_count = 0
            self._set_map_generation_active(True)
            try:
                for number, index in enumerate(time_indices, 1):
                    # Process Tk button events so Halt remains responsive while
                    # this otherwise synchronous sequence is running.
                    self.root.update()
                    if self._map_generation_halted:
                        break
                    self.map_status_var.set(
                        f"Saving map {number}/{len(time_indices)}…"
                    )
                    self.root.update()
                    if self._map_generation_halted:
                        break
                    figure = self._create_component_map_figure(
                        map_values,
                        index,
                        station_indices,
                        bounds,
                        colorbar_label,
                        signed_values,
                    )
                    self.root.update()
                    if self._map_generation_halted:
                        figure.clear()
                        break
                    figure.savefig(
                        output_directory / self._map_filename(token, index),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    figure.clear()
                    saved_count += 1
            finally:
                self._set_map_generation_active(False)

            if self._map_generation_halted:
                message = (
                    f"Map generation halted; saved {saved_count} of "
                    f"{len(time_indices)} {self._map_layer_description()} maps "
                    f"to {output_directory}"
                )
            else:
                message = (
                    f"Saved {saved_count} {self._map_layer_description()} maps "
                    f"to {output_directory}"
                )
            self.map_status_var.set(message)
            self.status_var.set(message)
        except Exception as exc:
            self.map_status_var.set("Map creation failed")
            viewer.messagebox.showerror("Map", str(exc))

    def _apply_station_selection(
        self, indices: set[int], active_index: Optional[int] = None
    ) -> None:
        if self.data is None:
            self.selected_station_indices.clear()
            self.current_station_index = None
            return
        valid = {
            int(index)
            for index in indices
            if 0 <= int(index) < len(self.data.station_codes)
        }
        self.selected_station_indices = valid

        if active_index is not None and int(active_index) in valid:
            self.current_station_index = int(active_index)
        elif self.current_station_index not in valid:
            self.current_station_index = min(valid) if valid else None

        if not valid:
            self.station_status_var.set("No stations selected")
        else:
            codes = [str(self.data.station_codes[index]) for index in sorted(valid)]
            preview = ", ".join(codes[:5])
            if len(codes) > 5:
                preview += f", … (+{len(codes) - 5})"
            active_code = (
                str(self.data.station_codes[self.current_station_index])
                if self.current_station_index is not None
                else "none"
            )
            self.station_status_var.set(
                f"Selected {len(valid)} station(s): {preview}; active plot: {active_code}"
            )

        self._sync_station_lists(self.current_station_index)
        self.highlight_station_on_map()
        if self.plot_mode_var.get() == "single":
            if self.current_station_index is None:
                self._draw_empty_single_plot()
            else:
                self.plot_selected_station()

    def select_station_index(self, index: int) -> None:
        """Compatibility selection: replace the set with one active station."""
        self._apply_station_selection({int(index)}, active_index=int(index))

    def _sync_station_lists(self, station_index: Optional[int] = None) -> None:
        self._syncing_selection = True
        try:
            self.lat_listbox.selection_clear(0, viewer.tk.END)
            self.lon_listbox.selection_clear(0, viewer.tk.END)
            for position, data_index in enumerate(self.lat_order):
                if int(data_index) in self.selected_station_indices:
                    self.lat_listbox.selection_set(position)
            for position, data_index in enumerate(self.lon_order):
                if int(data_index) in self.selected_station_indices:
                    self.lon_listbox.selection_set(position)

            if station_index is not None:
                lat_positions = np.flatnonzero(self.lat_order == station_index)
                lon_positions = np.flatnonzero(self.lon_order == station_index)
                if len(lat_positions):
                    self.lat_listbox.see(int(lat_positions[0]))
                if len(lon_positions):
                    self.lon_listbox.see(int(lon_positions[0]))
        finally:
            self._syncing_selection = False

    def _selection_from_listbox(self, listbox, order: np.ndarray) -> None:
        positions = [int(value) for value in listbox.curselection()]
        indices = {int(order[position]) for position in positions if position < len(order)}
        active_index: Optional[int] = None
        if positions:
            active_position = int(listbox.index(viewer.tk.ACTIVE))
            if active_position in positions and active_position < len(order):
                active_index = int(order[active_position])
            else:
                active_index = int(order[positions[-1]])
        self._apply_station_selection(indices, active_index=active_index)

    def on_latitude_list_select(self, _event=None) -> None:
        if not self._syncing_selection:
            self._selection_from_listbox(self.lat_listbox, self.lat_order)

    def on_longitude_list_select(self, _event=None) -> None:
        if not self._syncing_selection:
            self._selection_from_listbox(self.lon_listbox, self.lon_order)

    def on_map_pick(self, event) -> None:
        if self.map_scatter is None or event.artist is not self.map_scatter or not len(event.ind):
            return
        local_index = int(event.ind[0])
        if local_index >= len(self.map_station_indices):
            return
        station_index = int(self.map_station_indices[local_index])
        updated = set(self.selected_station_indices)
        if station_index in updated:
            updated.remove(station_index)
            active = self.current_station_index
            if active == station_index:
                active = min(updated) if updated else None
        else:
            updated.add(station_index)
            active = station_index
        self._apply_station_selection(updated, active_index=active)

    def add_selected_station_to_manual_stack(self) -> None:
        """Add every selected station to the manual stack as a separate panel."""
        if self.data is None or not self.selected_station_indices:
            viewer.messagebox.showinfo(
                "Manual stack", "Select one or more stations after loading data."
            )
            return

        time_mask = self._selected_time_mask()
        if not np.any(time_mask):
            viewer.messagebox.showinfo(
                "Manual stack", "No samples fall inside the selected time range."
            )
            return

        selected = sorted(
            int(index)
            for index in self.selected_station_indices
            if 0 <= int(index) < len(self.data.station_codes)
        )
        if not selected:
            viewer.messagebox.showinfo(
                "Manual stack", "Select one or more valid stations."
            )
            return

        if not self.manual_stack_panels:
            self._sync_manual_time_range_to_main()
        for station_index in selected:
            self.manual_stack_panels.append(
                {
                    "station_index": station_index,
                    "station": str(self.data.station_codes[station_index]),
                    "glat": float(self.data.glat[station_index]),
                    "glon": float(
                        viewer.normalize_longitude(self.data.glon[station_index])
                    ),
                    "selected_points": [],
                    "next_point_color_index": 0,
                    "ymin": None,
                    "ymax": None,
                }
            )

        self._update_manual_stack_status()
        self.show_manual_stackplot()
        self._refresh_manual_point_selection_window()
        self._rebuild_manual_stackplot()

    def highlight_station_on_map(self) -> None:
        if self.map_axes is None or self.map_canvas is None or self.data is None:
            return
        if self.map_highlight is not None:
            try:
                self.map_highlight.remove()
            except (ValueError, AttributeError):
                pass
            self.map_highlight = None

        indices = [
            index
            for index in sorted(self.selected_station_indices)
            if np.isfinite(self.data.glat[index]) and np.isfinite(self.data.glon[index])
        ]
        if not indices:
            self.map_canvas.draw_idle()
            return

        lons = viewer.normalize_longitude(self.data.glon[indices])
        lats = self.data.glat[indices]
        kwargs = dict(
            s=80,
            c="limegreen",
            alpha=0.98,
            edgecolors="black",
            linewidths=1.0,
            marker="o",
            picker=False,
            zorder=20,
        )
        if viewer.HAS_CARTOPY and hasattr(self.map_axes, "projection"):
            kwargs["transform"] = viewer.ccrs.PlateCarree()
            x_values = lons
        else:
            finite_lons = self.data.glon[np.isfinite(self.data.glon)]
            center_lon, _, _ = viewer.choose_longitude_window(finite_lons)
            x_values = viewer.normalize_longitude(lons - center_lon)
        self.map_highlight = self.map_axes.scatter(x_values, lats, **kwargs)
        self.map_canvas.draw_idle()

    def select_all_stations(self) -> None:
        if self.data is None:
            return
        indices = set(range(len(self.data.station_codes)))
        active = self.current_station_index if self.current_station_index in indices else 0
        self._apply_station_selection(indices, active_index=active)

    def select_stations_by_plot_region(self) -> None:
        """Select stations inside the latitude/longitude plot-region entries."""
        if self.data is None:
            viewer.messagebox.showinfo(
                "Station selection", "Open or download a magnetic data file first."
            )
            return
        try:
            lat_min, lat_max = sorted(
                (float(self.stack_lat_min_var.get()), float(self.stack_lat_max_var.get()))
            )
            lon_west = float(viewer.normalize_longitude(float(self.stack_lon_west_var.get())))
            lon_east = float(viewer.normalize_longitude(float(self.stack_lon_east_var.get())))
        except ValueError:
            viewer.messagebox.showerror(
                "Station selection", "Latitude and longitude ranges must be numeric."
            )
            return

        lats = np.asarray(self.data.glat, dtype=float)
        lons = np.asarray(viewer.normalize_longitude(self.data.glon), dtype=float)
        mask = np.isfinite(lats) & np.isfinite(lons)
        mask &= (lats >= lat_min) & (lats <= lat_max)
        if lon_west <= lon_east:
            mask &= (lons >= lon_west) & (lons <= lon_east)
        else:
            mask &= (lons >= lon_west) | (lons <= lon_east)
        indices = set(int(index) for index in np.flatnonzero(mask))
        self._apply_station_selection(
            indices, active_index=min(indices) if indices else None
        )
        self.pi2_status_var.set(
            f"Selected {len(indices)} station(s) in latitude {lat_min:g}–{lat_max:g}°, "
            f"longitude {lon_west:g}–{lon_east:g}°."
        )

    def clear_station_selection(self) -> None:
        self._apply_station_selection(set(), active_index=None)

    def _confirm_small_region_map_analysis(
        self,
        analysis_name: str,
        station_indices: list[int],
    ) -> bool:
        """Confirm map creation for fewer than five tightly grouped stations."""
        if self.data is None or not (1 <= len(station_indices) < 5):
            return True

        finite_indices = [
            int(index)
            for index in station_indices
            if np.isfinite(self.data.glat[int(index)])
            and np.isfinite(self.data.glon[int(index)])
        ]
        if not finite_indices:
            return True

        latitudes = np.asarray(self.data.glat[finite_indices], dtype=float)
        longitudes = np.asarray(
            viewer.normalize_longitude(self.data.glon[finite_indices]), dtype=float
        )
        _, relative_longitudes, _ = viewer.choose_longitude_window(longitudes)
        latitude_span = float(np.ptp(latitudes)) if len(latitudes) > 1 else 0.0
        longitude_span = (
            float(np.ptp(relative_longitudes)) if len(relative_longitudes) > 1 else 0.0
        )

        # A tightly grouped selection can make Cartopy spend extra time
        # resolving a very small map extent.  Warn only for 1--4 stations.
        small_region = latitude_span <= 12.0 and longitude_span <= 20.0
        if not small_region:
            return True

        codes = [str(self.data.station_codes[index]) for index in finite_indices]
        result = {"continue": False}
        dialog = viewer.tk.Toplevel(self.root)
        dialog.title(f"{analysis_name}: small map region")
        dialog_parent = self._dialog_parent() or self.root
        dialog.transient(dialog_parent)
        dialog.resizable(False, False)
        dialog.lift()
        try:
            dialog.attributes("-topmost", True)
            dialog.after_idle(lambda: dialog.attributes("-topmost", False))
        except viewer.tk.TclError:
            pass

        body = viewer.ttk.Frame(dialog, padding=14)
        body.pack(fill=viewer.tk.BOTH, expand=True)
        viewer.ttk.Label(
            body,
            text=(
                f"Only {len(finite_indices)} station(s) are selected in a small "
                f"geographic region ({', '.join(codes)}).\n\n"
                "Maps for a few closely spaced stations may take longer to load. "
                "Continue with the analysis?"
            ),
            wraplength=520,
            justify=viewer.tk.LEFT,
        ).pack(fill=viewer.tk.X)

        buttons = viewer.ttk.Frame(body)
        buttons.pack(fill=viewer.tk.X, pady=(14, 0))

        def finish(continue_analysis: bool) -> None:
            result["continue"] = bool(continue_analysis)
            try:
                dialog.grab_release()
            except viewer.tk.TclError:
                pass
            dialog.destroy()

        viewer.ttk.Button(
            buttons, text="Cancel", command=lambda: finish(False)
        ).pack(side=viewer.tk.RIGHT)
        continue_button = viewer.ttk.Button(
            buttons, text="Continue", command=lambda: finish(True)
        )
        continue_button.pack(side=viewer.tk.RIGHT, padx=(0, 6))
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
        dialog.bind("<Escape>", lambda _event: finish(False))
        dialog.bind("<Return>", lambda _event: finish(True))
        dialog.grab_set()
        continue_button.focus_set()
        self.root.wait_window(dialog)
        return bool(result["continue"])

    # ------------------------------------------------------------------
    # Pi2 time controls and onset in the existing single-site plot
    # ------------------------------------------------------------------

    @staticmethod
    def _format_gui_time(value: datetime) -> str:
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.replace(microsecond=0).strftime(viewer.MANUAL_TIME_FORMAT)

    def use_plot_range_as_pi2(self) -> None:
        if self.time_start is None or self.time_end is None:
            viewer.messagebox.showinfo("Pi2 interval", "Load data and choose a plot range first.")
            return
        self.pi2_start_var.set(self._format_gui_time(self.time_start))
        self.pi2_end_var.set(self._format_gui_time(self.time_end))

    def use_plot_range_as_quiet(self) -> None:
        if self.time_start is None or self.time_end is None:
            viewer.messagebox.showinfo("Quiet interval", "Load data and choose a plot range first.")
            return
        self.quiet_start_var.set(self._format_gui_time(self.time_start))
        self.quiet_end_var.set(self._format_gui_time(self.time_end))

    @staticmethod
    def _parse_required_time(text: str, label: str) -> datetime:
        try:
            return datetime.strptime(text.strip(), viewer.MANUAL_TIME_FORMAT)
        except ValueError as exc:
            raise ValueError(
                f"{label} must use YYYY-MM-DD HH:mm:SS UTC."
            ) from exc

    def _parse_optional_onset(self, strict: bool = True) -> Optional[datetime]:
        text = self.pi2_onset_var.get().strip()
        if not text:
            return None
        try:
            return datetime.strptime(text, viewer.MANUAL_TIME_FORMAT)
        except ValueError:
            if strict:
                raise ValueError("Optional onset must use YYYY-MM-DD HH:mm:SS UTC.")
            return None

    def plot_selected_station(self) -> None:
        super().plot_selected_station()
        if self.single_plot_ax is None or self.single_canvas is None:
            return
        onset = self._parse_optional_onset(strict=False)
        if onset is None:
            return
        self.single_plot_ax.axvline(
            onset,
            color="red",
            linestyle="--",
            linewidth=1.25,
            label="Pi2 onset",
            zorder=12,
        )
        handles = []
        labels = []
        if self.single_figure is not None:
            for axis in self.single_figure.axes:
                axis_handles, axis_labels = axis.get_legend_handles_labels()
                handles.extend(axis_handles)
                labels.extend(axis_labels)
        if handles:
            unique: dict[str, object] = {}
            for handle, label in zip(handles, labels):
                unique[label] = handle
            self.single_plot_ax.legend(
                list(unique.values()), list(unique.keys()), loc="upper right", fontsize=8
            )
        self.single_canvas.draw_idle()


    # ------------------------------------------------------------------
    # Shared pulsation-analysis utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _require_scipy() -> None:
        if scipy_signal is None:
            raise RuntimeError(
                "SciPy is required for pulsation filtering and wavelet analysis. "
                "Install it with: python -m pip install scipy"
            )

    def _plot_interval(self) -> tuple[datetime, datetime]:
        if self.data is None:
            raise ValueError("Open or download a ground-magnetometer data file first.")
        if self.time_start is not None and self.time_end is not None:
            return self.time_start, self.time_end
        times = self._time_array_ns(self.data.times)
        if len(times) < 2:
            raise ValueError("The loaded file has too few time samples.")
        return self._datetime64_to_datetime(times[0]), self._datetime64_to_datetime(times[-1])

    @staticmethod
    def _time_array_ns(values) -> np.ndarray:
        raw = np.asarray(values)
        if np.issubdtype(raw.dtype, np.datetime64):
            return raw.astype("datetime64[ns]")
        converted = []
        for value in raw:
            if isinstance(value, datetime) and value.tzinfo is not None:
                value = value.astimezone(timezone.utc).replace(tzinfo=None)
            converted.append(np.datetime64(value, "ns"))
        return np.asarray(converted, dtype="datetime64[ns]")

    @staticmethod
    def _datetime64_to_datetime(value: np.datetime64) -> datetime:
        nanoseconds = value.astype("datetime64[ns]").astype(np.int64)
        return datetime.utcfromtimestamp(float(nanoseconds) / 1.0e9)

    def _pulsation_coordinate_data(self) -> tuple[np.ndarray, tuple[str, str], str]:
        if self.data is None:
            raise ValueError("Open or download a ground-magnetometer data file first.")
        if self.data.nez is not None:
            return self.data.nez, ("N", "E"), "NEZ"
        if self.data.geo is not None:
            return self.data.geo, ("X", "Y"), "GEO"
        raise ValueError("The loaded file contains neither NEZ nor GEO magnetic components.")

    def _prepare_station_timeseries(
        self,
        station_index: int,
        start: datetime,
        end: datetime,
        include_horizontal_magnitude: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, tuple[str, ...], str]:
        self._require_scipy()
        coordinate_data, horizontal_labels, coordinate_name = (
            self._pulsation_coordinate_data()
        )
        all_times = self._time_array_ns(self.data.times)
        start64 = np.datetime64(start.replace(tzinfo=None), "ns")
        end64 = np.datetime64(end.replace(tzinfo=None), "ns")
        mask = (all_times >= start64) & (all_times <= end64)
        if np.count_nonzero(mask) < 16:
            raise ValueError("The selected plot interval contains fewer than 16 samples.")

        station_values = viewer.mask_bad_magnetometer_values(
            coordinate_data[int(station_index), :, :]
        )
        values = np.asarray(station_values[mask, :2], dtype=float)
        times = all_times[mask]

        time_seconds = (
            (times - times[0]).astype("timedelta64[ns]").astype(np.int64) / 1.0e9
        )
        differences = np.diff(time_seconds)
        differences = differences[np.isfinite(differences) & (differences > 0.0)]
        if not len(differences):
            raise ValueError("Could not determine a positive data cadence.")
        cadence = float(np.median(differences))
        if cadence <= 0.0:
            raise ValueError("The data cadence must be positive.")
        if float(np.max(differences)) > 1.5 * cadence:
            raise ValueError(
                "The selected interval contains a timestamp gap larger than "
                "1.5 sampling intervals. Analyze continuous data segments separately."
            )

        regular_seconds = np.arange(0.0, time_seconds[-1] + 0.5 * cadence, cadence)
        regular_horizontal = np.empty((len(regular_seconds), 2), dtype=float)
        for component in range(2):
            series = values[:, component]
            finite = np.isfinite(series)
            if np.count_nonzero(finite) < 8:
                raise ValueError(
                    f"Horizontal component {horizontal_labels[component]} has fewer "
                    "than eight finite samples."
                )
            if not np.all(finite):
                raise ValueError(
                    f"Horizontal component {horizontal_labels[component]} contains "
                    "missing samples. Interpolating across gaps can manufacture "
                    "pulsation power and phase; select a continuous interval."
                )
            regular_horizontal[:, component] = np.interp(
                regular_seconds, time_seconds[finite], series[finite]
            )

        if include_horizontal_magnitude:
            # H is calculated from the interpolated physical horizontal
            # perturbations before filtering.
            horizontal_magnitude = np.hypot(
                regular_horizontal[:, 0], regular_horizontal[:, 1]
            )
            analysis_values = np.column_stack(
                (regular_horizontal, horizontal_magnitude)
            )
            labels: tuple[str, ...] = (
                horizontal_labels[0],
                horizontal_labels[1],
                "H",
            )
        else:
            analysis_values = regular_horizontal
            labels = horizontal_labels

        regular_times = (
            times[0]
            + np.rint(regular_seconds * 1.0e9).astype("timedelta64[ns]")
        )
        return regular_times, analysis_values, cadence, labels, coordinate_name

    def _prepare_station_scalogram_timeseries(
        self,
        station_index: int,
        start: datetime,
        end: datetime,
    ) -> tuple[np.ndarray, np.ndarray, float, tuple[str, str, str, str], str]:
        """Prepare already-baselined H, N/E, E/Y, and Z component series."""
        self._require_scipy()
        coordinate_data, horizontal_labels, coordinate_name = self._pulsation_coordinate_data()
        all_times = self._time_array_ns(self.data.times)
        start64 = np.datetime64(start.replace(tzinfo=None), "ns")
        end64 = np.datetime64(end.replace(tzinfo=None), "ns")
        mask = (all_times >= start64) & (all_times <= end64)
        if np.count_nonzero(mask) < 16:
            raise ValueError("The selected plot interval contains fewer than 16 samples.")

        station_values = viewer.mask_bad_magnetometer_values(
            coordinate_data[int(station_index), :, :]
        )
        values = np.asarray(station_values[mask, :3], dtype=float)
        times = all_times[mask]
        time_seconds = (
            (times - times[0]).astype("timedelta64[ns]").astype(np.int64) / 1.0e9
        )
        differences = np.diff(time_seconds)
        differences = differences[np.isfinite(differences) & (differences > 0.0)]
        if not len(differences):
            raise ValueError("Could not determine a positive data cadence.")
        cadence = float(np.median(differences))
        if float(np.max(differences)) > 1.5 * cadence:
            raise ValueError(
                "The selected interval contains a timestamp gap larger than "
                "1.5 sampling intervals. Analyze continuous data segments separately."
            )
        regular_seconds = np.arange(0.0, time_seconds[-1] + 0.5 * cadence, cadence)
        regular_components = np.empty((len(regular_seconds), 3), dtype=float)
        component_labels = (horizontal_labels[0], horizontal_labels[1], "Z")
        for component, label in enumerate(component_labels):
            series = values[:, component]
            finite = np.isfinite(series)
            if np.count_nonzero(finite) < 8:
                raise ValueError(f"Component {label} has fewer than eight finite samples.")
            if not np.all(finite):
                raise ValueError(
                    f"Component {label} contains missing samples. Interpolating "
                    "across gaps can manufacture pulsation power and phase; "
                    "select a continuous interval."
                )
            regular_components[:, component] = np.interp(
                regular_seconds, time_seconds[finite], series[finite]
            )

        # Retained for non-scalogram time-series consumers.  H scalograms are
        # formed from the two complex component wavelet coefficients below.
        horizontal_magnitude = np.hypot(
            regular_components[:, 0], regular_components[:, 1]
        )
        analysis_values = np.column_stack(
            (horizontal_magnitude, regular_components)
        )
        regular_times = times[0] + np.rint(
            regular_seconds * 1.0e9
        ).astype("timedelta64[ns]")
        labels = ("H", horizontal_labels[0], horizontal_labels[1], "Z")
        return regular_times, analysis_values, cadence, labels, coordinate_name

    @staticmethod
    def _period_grid(
        cadence_s: float,
        duration_s: float,
        requested_min_s: float = 5.0,
        requested_max_s: float = 600.0,
        count: int = 72,
    ) -> np.ndarray:
        minimum = max(float(requested_min_s), 2.2 * float(cadence_s))
        maximum = min(float(requested_max_s), max(minimum, duration_s / 3.0))
        if maximum <= minimum * 1.02:
            raise ValueError(
                f"The {cadence_s:g} s cadence and selected duration cannot resolve "
                f"the requested {requested_min_s:g}–{requested_max_s:g} s periods."
            )
        return np.geomspace(minimum, maximum, int(max(count, 16)))

    @staticmethod
    def _morlet_power(
        values: np.ndarray,
        cadence_s: float,
        periods_s: np.ndarray,
        omega0: float = 6.0,
    ) -> np.ndarray:
        """Return summed component Morlet power for regular data.

        Scale-to-period conversion follows Torrence and Compo (1998).
        """
        values = np.asarray(values, dtype=float)
        periods_s = np.asarray(periods_s, dtype=float)
        number_samples = values.shape[0]
        total_power = np.zeros((len(periods_s), number_samples), dtype=float)
        for row, period in enumerate(periods_s):
            fourier_factor = 4.0 * np.pi / (
                omega0 + np.sqrt(2.0 + omega0**2)
            )
            scale_samples = max(period / (fourier_factor * cadence_s), 1.0)
            half_width = int(max(8, np.ceil(5.0 * scale_samples)))
            sample_offset = np.arange(-half_width, half_width + 1, dtype=float)
            wavelet = (
                np.pi ** (-0.25)
                * np.exp(1j * omega0 * sample_offset / scale_samples)
                * np.exp(-0.5 * (sample_offset / scale_samples) ** 2)
                / np.sqrt(scale_samples)
            )
            for component in range(values.shape[1]):
                coefficients = scipy_signal.fftconvolve(
                    values[:, component], np.conjugate(wavelet[::-1]), mode="same"
                )
                total_power[row] += np.abs(coefficients) ** 2
        return total_power

    @staticmethod
    def _morlet_coefficients(
        values: np.ndarray,
        cadence_s: float,
        periods_s: np.ndarray,
        omega0: float = 6.0,
    ) -> np.ndarray:
        """Return Morlet coefficients using the Torrence–Compo Fourier factor."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        coefficients = np.empty(
            (len(periods_s), values.shape[0], values.shape[1]), dtype=complex
        )
        for row, period in enumerate(np.asarray(periods_s, dtype=float)):
            fourier_factor = 4.0 * np.pi / (
                omega0 + np.sqrt(2.0 + omega0**2)
            )
            scale = max(period / (fourier_factor * cadence_s), 1.0)
            half_width = int(max(8, np.ceil(5.0 * scale)))
            offset = np.arange(-half_width, half_width + 1, dtype=float)
            wavelet = (
                np.pi ** (-0.25)
                * np.exp(1j * omega0 * offset / scale)
                * np.exp(-0.5 * (offset / scale) ** 2)
                / np.sqrt(scale)
            )
            for component in range(values.shape[1]):
                coefficients[row, :, component] = scipy_signal.fftconvolve(
                    values[:, component], np.conjugate(wavelet[::-1]), mode="same"
                )
        return coefficients

    @classmethod
    def _horizontal_wavelet_power(
        cls, horizontal_values: np.ndarray, cadence_s: float, periods_s: np.ndarray
    ) -> np.ndarray:
        """Return total horizontal power |W_N|² + |W_E|²."""
        coefficients = cls._morlet_coefficients(
            np.asarray(horizontal_values)[:, :2], cadence_s, periods_s
        )
        return (
            np.abs(coefficients[:, :, 0]) ** 2
            + np.abs(coefficients[:, :, 1]) ** 2
        )

    @staticmethod
    def _red_noise_significance(
        source_values: np.ndarray,
        cadence_s: float,
        periods_s: np.ndarray,
    ) -> np.ndarray:
        """Approximate Torrence–Compo 95% AR(1) red-noise power threshold."""
        values = np.asarray(source_values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        threshold = np.zeros(len(periods_s), dtype=float)
        quantile = float(scipy_chi2.ppf(0.95, 2) / 2.0) if scipy_chi2 else 2.9957
        for component in range(values.shape[1]):
            series = values[:, component] - np.mean(values[:, component])
            variance = float(np.var(series))
            if len(series) > 2 and variance > 0:
                alpha = float(np.corrcoef(series[:-1], series[1:])[0, 1])
                alpha = float(np.clip(np.nan_to_num(alpha), -0.99, 0.99))
            else:
                alpha = 0.0
            frequency = cadence_s / np.asarray(periods_s, dtype=float)
            background = variance * (1.0 - alpha ** 2) / (
                1.0 + alpha ** 2 - 2.0 * alpha * np.cos(2.0 * np.pi * frequency)
            )
            threshold += background * quantile
        return threshold

    def _draw_scalogram(
        self,
        axis,
        times: np.ndarray,
        periods: np.ndarray,
        power: np.ndarray,
        source_values: np.ndarray,
        cadence: float,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Draw power, 95% red-noise contour, and cone of influence."""
        reference = np.maximum(
            np.nanmedian(power, axis=1, keepdims=True), np.finfo(float).tiny
        )
        display_power = 10.0 * np.log10(
            np.maximum(power, np.finfo(float).tiny) / reference
        )
        finite = display_power[np.isfinite(display_power)]
        automatic_min = float(np.nanpercentile(finite, 5.0))
        automatic_max = float(np.nanpercentile(finite, 99.0))
        mesh = axis.pcolormesh(
            times,
            periods,
            display_power,
            shading="auto",
            cmap="turbo",
            vmin=automatic_min if vmin is None else vmin,
            vmax=automatic_max if vmax is None else vmax,
            rasterized=True,
        )
        threshold = self._red_noise_significance(source_values, cadence, periods)
        ratio = power / np.maximum(threshold[:, None], np.finfo(float).tiny)
        if np.nanmin(ratio) <= 1.0 <= np.nanmax(ratio):
            axis.contour(
                times, periods, ratio, levels=[1.0], colors="black",
                linewidths=0.8, linestyles="solid"
            )
        distance = np.minimum(
            np.arange(len(times)), np.arange(len(times))[::-1]
        ) * float(cadence)
        omega0 = 6.0
        fourier_factor = 4.0 * np.pi / (
            omega0 + np.sqrt(2.0 + omega0**2)
        )
        coi = np.maximum(
            float(cadence),
            (fourier_factor / np.sqrt(2.0)) * distance,
        )
        coi = np.clip(coi, float(periods[0]), float(periods[-1]))
        axis.plot(times, coi, color="white", linewidth=1.2)
        axis.fill_between(
            times, coi, float(np.max(periods)), facecolor="none", alpha=0.35,
            hatch="///", edgecolor="white", linewidth=0.0
        )
        for boundary in (40.0, 150.0):
            if periods[0] <= boundary <= periods[-1]:
                axis.axhline(
                    boundary, color="white", linestyle="--", linewidth=0.8
                )
                axis.text(
                    0.995, boundary, f"{boundary:g} s",
                    transform=axis.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8, color="white",
                )
        return mesh

    def _ask_scalogram_color_limits(
        self, labels: tuple[str, ...], powers: list[np.ndarray]
    ) -> Optional[list[tuple[float, float]]]:
        """Allow per-panel limits, initialized to one common power range."""
        relative_powers = []
        for power in powers:
            reference = np.maximum(
                np.nanmedian(power, axis=1, keepdims=True), np.finfo(float).tiny
            )
            relative_powers.append(
                10.0 * np.log10(
                    np.maximum(power, np.finfo(float).tiny) / reference
                )
            )
        finite = np.concatenate([p[np.isfinite(p)] for p in relative_powers])
        common_min, common_max = float(np.min(finite)), float(np.max(finite))
        if common_min == common_max:
            common_max = common_min + 1.0
        dialog = viewer.tk.Toplevel(self.root)
        dialog.title("Wavelet scalogram color limits")
        dialog.transient(self.root)
        variables = []
        viewer.ttk.Label(
            dialog,
            text="Enter cmin/cmax for each subplot (all begin with identical limits).",
            padding=8,
        ).grid(row=0, column=0, columnspan=5, sticky="w")
        for row, label in enumerate(labels, 1):
            low = viewer.tk.StringVar(value=f"{common_min:.6g}")
            high = viewer.tk.StringVar(value=f"{common_max:.6g}")
            variables.append((low, high))
            viewer.ttk.Label(dialog, text=label, padding=4).grid(row=row, column=0)
            viewer.ttk.Label(dialog, text="cmin").grid(row=row, column=1)
            viewer.ttk.Entry(dialog, textvariable=low, width=14).grid(row=row, column=2)
            viewer.ttk.Label(dialog, text="cmax").grid(row=row, column=3)
            viewer.ttk.Entry(dialog, textvariable=high, width=14).grid(row=row, column=4)
        result: list[tuple[float, float]] = []
        accepted = viewer.tk.BooleanVar(value=False)
        def accept() -> None:
            try:
                parsed = [(float(lo.get()), float(hi.get())) for lo, hi in variables]
                if any(not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi for lo, hi in parsed):
                    raise ValueError
            except ValueError:
                viewer.messagebox.showerror(
                    "Color limits", "Each cmin/cmax pair must be finite and cmin < cmax.",
                    parent=dialog
                )
                return
            result.extend(parsed)
            accepted.set(True)
            dialog.destroy()
        buttons = viewer.ttk.Frame(dialog, padding=8)
        buttons.grid(row=len(labels) + 1, column=0, columnspan=5, sticky="e")
        viewer.ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
        viewer.ttk.Button(buttons, text="Plot", command=accept).pack(side="right", padx=5)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.grab_set()
        self.root.wait_window(dialog)
        return result if accepted.get() else None

    @staticmethod
    def _bandpass_components(
        values: np.ndarray,
        cadence_s: float,
        minimum_period_s: float,
        maximum_period_s: float,
    ) -> np.ndarray:
        sampling_frequency = 1.0 / float(cadence_s)
        nyquist = 0.5 * sampling_frequency
        low_frequency = 1.0 / float(maximum_period_s)
        high_frequency = 1.0 / float(minimum_period_s)
        if high_frequency >= 0.98 * nyquist:
            raise ValueError(
                f"A {cadence_s:g} s cadence cannot resolve the {minimum_period_s:g} s "
                "short-period edge without violating Nyquist."
            )
        if low_frequency <= 0.0 or low_frequency >= high_frequency:
            raise ValueError("Invalid pulsation period band.")
        sos = scipy_signal.butter(
            4,
            [low_frequency, high_frequency],
            btype="bandpass",
            fs=sampling_frequency,
            output="sos",
        )
        try:
            return scipy_signal.sosfiltfilt(sos, values, axis=0)
        except ValueError as exc:
            raise ValueError(
                "The selected interval is too short for stable zero-phase band-pass filtering."
            ) from exc

    @classmethod
    def _ulf_bandpass_components(
        cls,
        values: np.ndarray,
        cadence_s: float,
        minimum_period_s: float,
        maximum_period_s: float,
    ) -> tuple[np.ndarray, float, str]:
        """Filter the resolvable part of a ULF band and describe that band."""
        nyquist_limited_minimum = 2.0 * float(cadence_s) / 0.95
        effective_minimum = max(float(minimum_period_s), nyquist_limited_minimum)
        if effective_minimum >= float(maximum_period_s):
            raise ValueError(
                f"A {cadence_s:g} s cadence cannot resolve any part of the "
                f"{minimum_period_s:g}–{maximum_period_s:g} s ULF band."
            )
        filtered = cls._bandpass_components(
            values, cadence_s, effective_minimum, maximum_period_s
        )
        if effective_minimum > minimum_period_s * 1.01:
            note = (
                f"{effective_minimum:.1f}–{maximum_period_s:g} s "
                f"(cadence-limited from {minimum_period_s:g}–{maximum_period_s:g} s)"
            )
        else:
            note = f"{minimum_period_s:g}–{maximum_period_s:g} s"
        return filtered, effective_minimum, note

    @staticmethod
    def _rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
        samples = int(max(1, samples))
        if samples == 1:
            return np.asarray(values, dtype=float)
        kernel = np.ones(samples, dtype=float) / float(samples)
        return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")

    @staticmethod
    def _configure_time_axis(axis) -> None:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    @classmethod
    def _configure_scalogram_axis(
        cls,
        axis,
        visible_max_s: float = 600.0,
        multiline_labels: bool = False,
    ) -> None:
        """Match the GOES scalogram: logarithmic period, short periods on top."""
        lower = max(2.5, cls.SCALOGRAM_PERIOD_RANGE[0])
        upper = float(visible_max_s)
        axis.set_yscale("log")
        axis.set_ylim(upper, lower)
        preferred = np.asarray([2.5, 5, 10, 20, 40, 60, 100, 150, 300, 600, 750])
        ticks = preferred[(preferred >= lower) & (preferred <= upper)]
        axis.set_yticks(ticks)
        axis.set_yticklabels([f"{tick:g}" for tick in ticks])
        axis.set_ylabel("Period\n(s)")
        axis.grid(False)

    @staticmethod
    def _apply_zero_phase_filter(sos: np.ndarray, values: np.ndarray) -> np.ndarray:
        try:
            return scipy_signal.sosfiltfilt(sos, values, axis=0)
        except ValueError as exc:
            raise ValueError(
                "The selected interval is too short for stable zero-phase filtering."
            ) from exc

    @classmethod
    def _pi_band_components(
        cls,
        values: np.ndarray,
        cadence_s: float,
        minimum_period_s: float,
        maximum_period_s: Optional[float],
    ) -> tuple[np.ndarray, str]:
        """Filter one Pi category and report any cadence-limited period edge."""
        sampling_frequency = 1.0 / float(cadence_s)
        nyquist = 0.5 * sampling_frequency
        if maximum_period_s is None:
            cutoff_frequency = 1.0 / float(minimum_period_s)
            if cutoff_frequency >= 0.98 * nyquist:
                raise ValueError(
                    f"A {cadence_s:g} s cadence cannot resolve periods longer than "
                    f"{minimum_period_s:g} s with a stable low-pass filter."
                )
            sos = scipy_signal.butter(
                4,
                cutoff_frequency,
                btype="lowpass",
                fs=sampling_frequency,
                output="sos",
            )
            return cls._apply_zero_phase_filter(sos, values), f">{minimum_period_s:g} s"

        low_frequency = 1.0 / float(maximum_period_s)
        requested_high = 1.0 / float(minimum_period_s)
        high_frequency = min(requested_high, 0.95 * nyquist)
        if low_frequency <= 0.0 or high_frequency <= low_frequency:
            raise ValueError(
                f"A {cadence_s:g} s cadence cannot resolve the requested "
                f"{minimum_period_s:g}–{maximum_period_s:g} s Pi band."
            )
        effective_minimum = 1.0 / high_frequency
        sos = scipy_signal.butter(
            4,
            [low_frequency, high_frequency],
            btype="bandpass",
            fs=sampling_frequency,
            output="sos",
        )
        note = f"{minimum_period_s:g}–{maximum_period_s:g} s"
        if effective_minimum > minimum_period_s * 1.01:
            note += f"; cadence-resolved edge ≥{effective_minimum:.2f} s"
        return cls._apply_zero_phase_filter(sos, values), note

    def _attach_time_value_cursor(
        self,
        canvas,
        axes,
        scalogram_data: Optional[dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    ) -> None:
        """Add synchronized crosshairs and nearest time/value readout."""
        axes = [axis for axis in axes if axis is not None]
        scalogram_data = scalogram_data or {}
        # Preserve the established datetime limits.  Creating a cursor line at
        # x=0 (1970-01-01 in Matplotlib date coordinates) can trigger autoscaling
        # and make a modern scalogram appear to vanish after the first redraw.
        original_x_limits = {axis: axis.get_xlim() for axis in axes}
        vertical_lines = {}
        for axis in axes:
            left, right = original_x_limits[axis]
            cursor_x = 0.5 * (left + right)
            vertical_lines[axis] = axis.axvline(
                cursor_x,
                color="black",
                linewidth=0.8,
                linestyle="--",
                alpha=0.65,
                visible=False,
                label="_cursor",
                zorder=100,
            )
            axis.set_xlim(left, right)
        annotations = {
            axis: axis.annotate(
                "",
                xy=(0.0, 0.0),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
                visible=False,
                zorder=101,
            )
            for axis in axes
        }

        def hide_cursor(_event=None) -> None:
            changed = False
            for line in vertical_lines.values():
                if line.get_visible():
                    line.set_visible(False)
                    changed = True
            for annotation in annotations.values():
                if annotation.get_visible():
                    annotation.set_visible(False)
                    changed = True
            if changed:
                canvas.draw_idle()

        def on_motion(event) -> None:
            if event.inaxes not in axes or event.xdata is None:
                hide_cursor()
                return
            active_axis = event.inaxes
            for axis, line in vertical_lines.items():
                line.set_xdata([event.xdata, event.xdata])
                line.set_visible(True)
            for axis, annotation in annotations.items():
                annotation.set_visible(axis is active_axis)

            time_text = mdates.num2date(event.xdata).strftime("%H:%M:%S")
            annotation = annotations[active_axis]
            if active_axis in scalogram_data and event.ydata is not None:
                times, periods, power = scalogram_data[active_axis]
                time_numbers = mdates.date2num(np.asarray(times))
                time_index = int(np.argmin(np.abs(time_numbers - event.xdata)))
                period_index = int(np.argmin(np.abs(np.asarray(periods) - event.ydata)))
                value = float(power[period_index, time_index])
                period = float(periods[period_index])
                annotation.xy = (event.xdata, period)
                annotation.set_text(
                    f"{time_text} UTC\nPeriod: {period:.1f} s\nPower: {value:.3g} nT²"
                )
            else:
                readouts = []
                nearest_y = event.ydata if event.ydata is not None else 0.0
                for line in active_axis.get_lines():
                    if line is vertical_lines[active_axis] or line.get_label() == "_cursor":
                        continue
                    x_values = np.asarray(line.get_xdata())
                    y_values = np.asarray(line.get_ydata(), dtype=float)
                    if not len(x_values) or len(x_values) != len(y_values):
                        continue
                    try:
                        x_numbers = mdates.date2num(x_values)
                    except Exception:
                        continue
                    index = int(np.argmin(np.abs(x_numbers - event.xdata)))
                    value = float(y_values[index])
                    if not np.isfinite(value):
                        continue
                    label = line.get_label()
                    if not label or label.startswith("_"):
                        label = "Value"
                    readouts.append(f"{label}: {value:.3g}")
                    nearest_y = value
                annotation.xy = (event.xdata, nearest_y)
                annotation.set_text(time_text + " UTC" + ("\n" + "\n".join(readouts) if readouts else ""))
            canvas.draw_idle()

        motion_id = canvas.mpl_connect("motion_notify_event", on_motion)
        leave_id = canvas.mpl_connect("figure_leave_event", hide_cursor)
        canvas._pulsation_cursor_state = {
            "motion_id": motion_id,
            "leave_id": leave_id,
            "lines": vertical_lines,
            "annotations": annotations,
            "callbacks": (on_motion, hide_cursor),
        }

    @staticmethod
    def _export_time_text(value: np.datetime64 | datetime) -> str:
        if isinstance(value, np.datetime64):
            nanoseconds = value.astype("datetime64[ns]").astype(np.int64)
            value = datetime.utcfromtimestamp(float(nanoseconds) / 1.0e9)
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")

    def _export_pulsation_band_data(
        self,
        analysis_kind: str,
        station_indices: list[int],
        start: datetime,
        end: datetime,
    ) -> None:
        if self.data is None or not station_indices:
            return
        analysis_kind = analysis_kind.strip().lower()
        if analysis_kind not in {"pi", "pc"}:
            raise ValueError("analysis_kind must be 'pi' or 'pc'.")

        token = "Pi" if analysis_kind == "pi" else "Pc"
        default_name = (
            f"{token}_pulsations_"
            f"{start:%Y%m%d_%H%M%S}_{end:%Y%m%d_%H%M%S}.csv"
        )
        filename = viewer.filedialog.asksaveasfilename(
            parent=self.root,
            title=f"Export {token} pulsation data",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filename:
            return

        failures = []
        row_count = 0
        with open(filename, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "station",
                    "time_utc",
                    "coordinate_system",
                    "analysis",
                    "band",
                    "period_range_s",
                    "component",
                    "filtered_value_nT",
                ]
            )
            for station_index in station_indices:
                code = str(self.data.station_codes[int(station_index)])
                try:
                    times, values, cadence, labels, coordinate_name = (
                        self._prepare_station_timeseries(
                            station_index,
                            start,
                            end,
                            include_horizontal_magnitude=True,
                        )
                    )
                    bands = self.PI_BANDS if analysis_kind == "pi" else self.PC_BANDS
                    for band_name, minimum_period, maximum_period in bands:
                        if analysis_kind == "pi":
                            filtered, period_text = self._pi_band_components(
                                values,
                                cadence,
                                minimum_period,
                                maximum_period,
                            )
                        else:
                            filtered = self._bandpass_components(
                                values,
                                cadence,
                                minimum_period,
                                maximum_period,
                            )
                            period_text = (
                                f"{minimum_period:g}-{maximum_period:g}"
                            )
                        for sample_index, sample_time in enumerate(times):
                            time_text = self._export_time_text(sample_time)
                            for component_index, label in enumerate(labels):
                                writer.writerow(
                                    [
                                        code,
                                        time_text,
                                        coordinate_name,
                                        token,
                                        band_name,
                                        period_text,
                                        label,
                                        f"{float(filtered[sample_index, component_index]):.10g}",
                                    ]
                                )
                                row_count += 1
                except Exception as exc:
                    failures.append(f"{code}: {exc}")

        message = f"Exported {row_count:,} rows to:\n{filename}"
        if failures:
            message += "\n\nSkipped/partial stations:\n" + "\n".join(failures[:10])
            viewer.messagebox.showwarning(f"{token} export", message)
        else:
            viewer.messagebox.showinfo(f"{token} export", message)

    def _export_multi_station_ulf_data(
        self,
        station_indices: list[int],
        start: datetime,
        end: datetime,
    ) -> None:
        if self.data is None or not station_indices:
            return
        default_name = (
            "supermag_multi_station_ULF_power_"
            f"{start:%Y%m%d_%H%M%S}_{end:%Y%m%d_%H%M%S}.csv"
        )
        filename = viewer.filedialog.asksaveasfilename(
            parent=self.root,
            title="Export multi-station ULF wave power",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filename:
            return

        minimum_period, maximum_period = self.TOTAL_ULF_PERIOD_RANGE
        failures = []
        row_count = 0
        with open(filename, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "station",
                    "time_utc",
                    "ULF_power_nT2",
                    "minimum_period_s",
                    "maximum_period_s",
                ]
            )
            for station_index in station_indices:
                code = str(self.data.station_codes[int(station_index)])
                try:
                    times, wave_power, _, _, band_note = self._station_ulf_power_series(
                        station_index, start, end
                    )
                    effective_minimum = float(band_note.split("–", 1)[0])
                    for sample_time, value in zip(times, wave_power):
                        writer.writerow(
                            [
                                code,
                                self._export_time_text(sample_time),
                                f"{float(value):.10g}",
                                f"{effective_minimum:g}",
                                f"{maximum_period:g}",
                            ]
                        )
                        row_count += 1
                except Exception as exc:
                    failures.append(f"{code}: {exc}")

        message = f"Exported {row_count:,} rows to:\n{filename}"
        if failures:
            message += "\n\nSkipped stations:\n" + "\n".join(failures[:10])
            viewer.messagebox.showwarning("ULF export", message)
        else:
            viewer.messagebox.showinfo("ULF export", message)

    def save_figure_png(self, figure: Optional[Figure], default_name: str) -> None:
        if figure is None:
            viewer.messagebox.showinfo("Save PNG", "There is no figure to save.")
            return
        path = viewer.filedialog.asksaveasfilename(
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
            viewer.messagebox.showerror(
                "Save PNG", f"Could not save the figure:\n\n{exc}"
            )

    def _new_figure_window(
        self,
        title: str,
        figure: Figure,
        default_filename: str,
        geometry: str = "1350x850",
        cursor_axes: Optional[list[Any]] = None,
        scalogram_cursor_data: Optional[
            dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = None,
        component_artists: Optional[dict[str, list[Any]]] = None,
        export_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        window = viewer.tk.Toplevel(self.root)
        window.title(title)
        window.geometry(geometry)
        window.minsize(900, 600)
        self._plot_windows.append(window)

        top = viewer.ttk.Frame(window, padding=5)
        top.pack(fill=viewer.tk.X)
        viewer.ttk.Button(
            top,
            text="Save figure PNG…",
            command=lambda: self.save_figure_png(figure, default_filename),
        ).pack(side=viewer.tk.RIGHT)

        if export_callback is not None:
            viewer.ttk.Button(
                top,
                text="Export data…",
                command=export_callback,
            ).pack(side=viewer.tk.RIGHT, padx=(0, 5))

        container = viewer.ttk.Frame(window)
        container.pack(fill=viewer.tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(figure, master=container)

        def refresh_legends() -> None:
            for axis in figure.axes:
                visible_lines = [
                    line
                    for line in axis.get_lines()
                    if line.get_visible()
                    and line.get_label()
                    and not line.get_label().startswith("_")
                ]
                legend = axis.get_legend()
                if visible_lines:
                    axis.legend(
                        visible_lines,
                        [line.get_label() for line in visible_lines],
                        loc="upper right",
                        ncol=min(3, len(visible_lines)),
                        fontsize=8,
                    )
                elif legend is not None:
                    legend.remove()

        if component_artists:
            visibility = {name: True for name in component_artists}
            button_text = {
                name: viewer.tk.StringVar(value=f"Hide {name}")
                for name in component_artists
            }

            def toggle_component(name: str) -> None:
                visibility[name] = not visibility[name]
                for artist in component_artists[name]:
                    artist.set_visible(visibility[name])
                button_text[name].set(
                    f"Hide {name}" if visibility[name] else f"Show {name}"
                )
                refresh_legends()
                canvas.draw_idle()

            # Pack in reverse so the visual order remains N/E/H (or SMU/SML/SME)
            # from left to right, immediately before Export and Save.
            for name in reversed(list(component_artists)):
                viewer.ttk.Button(
                    top,
                    textvariable=button_text[name],
                    command=lambda key=name: toggle_component(key),
                ).pack(side=viewer.tk.RIGHT, padx=(0, 5))

        canvas.draw()
        canvas.get_tk_widget().pack(fill=viewer.tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, container, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=viewer.tk.X)
        if cursor_axes:
            self._attach_time_value_cursor(
                canvas,
                cursor_axes,
                scalogram_data=scalogram_cursor_data,
            )

        def close_window() -> None:
            try:
                self._plot_windows.remove(window)
            except ValueError:
                pass
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)

    def show_wavelet_scalograms(self) -> None:
        """Create separate H, N, E and Z scalograms for each selected station."""
        if self.data is None:
            viewer.messagebox.showinfo(
                "Wavelet scalogram", "Open or download a magnetic data file first."
            )
            return
        if not self.selected_station_indices:
            viewer.messagebox.showinfo(
                "Wavelet scalogram", "Select one or more stations first."
            )
            return
        try:
            start, end = self._plot_interval()
            self._require_scipy()
        except (ValueError, RuntimeError) as exc:
            viewer.messagebox.showerror("Wavelet scalogram", str(exc))
            return

        failures = []
        selected = sorted(self.selected_station_indices)
        for number, station_index in enumerate(selected, start=1):
            code = str(self.data.station_codes[station_index])
            self.pi2_status_var.set(
                f"Computing H/N/E/Z wavelet scalograms for {code} "
                f"({number}/{len(selected)})…"
            )
            self.root.update_idletasks()
            try:
                times, values, cadence, labels, coordinate_name = (
                    self._prepare_station_scalogram_timeseries(
                        station_index, start, end
                    )
                )
                duration = float((times[-1] - times[0]) / np.timedelta64(1, "s"))
                periods = self._period_grid(
                    cadence,
                    duration,
                    self.SCALOGRAM_PERIOD_RANGE[0],
                    self.SCALOGRAM_PERIOD_RANGE[1],
                )
                powers = [
                    self._horizontal_wavelet_power(values[:, 1:3], cadence, periods),
                    self._morlet_power(values[:, 1:2], cadence, periods),
                    self._morlet_power(values[:, 2:3], cadence, periods),
                    self._morlet_power(values[:, 3:4], cadence, periods),
                ]
                color_limits = self._ask_scalogram_color_limits(labels, powers)
                if color_limits is None:
                    raise RuntimeError("Color-limit entry was cancelled.")

                figure = Figure(figsize=(14, 12), dpi=100)
                axes = [figure.add_subplot(4, 1, 1)]
                axes.extend(
                    figure.add_subplot(4, 1, row, sharex=axes[0])
                    for row in range(2, 5)
                )
                cursor_data = {}
                sources = (values[:, 1:3], values[:, 1:2], values[:, 2:3], values[:, 3:4])
                for axis, label, power, source, limits in zip(
                    axes, labels, powers, sources, color_limits
                ):
                    mesh = self._draw_scalogram(
                        axis, times, periods, power, source, cadence,
                        vmin=limits[0], vmax=limits[1]
                    )
                    self._configure_scalogram_axis(axis, visible_max_s=600.0)
                    axis.set_title(f"{label} wavelet scalogram")
                    axis.grid(True, alpha=0.18)
                    figure.colorbar(
                        mesh, ax=axis, pad=0.012,
                        label=f"{label} relative wavelet power (dB)"
                    )
                    cursor_data[axis] = (times, periods, power)
                for axis in axes[:-1]:
                    axis.tick_params(labelbottom=False)
                axes[-1].set_xlabel("UTC (HH:MM)")
                axes[-1].set_xlim(times[0], times[-1])
                self._configure_time_axis(axes[-1])
                figure.suptitle(
                    f"{code} component wavelet scalograms ({coordinate_name})\n"
                    f"{self._format_gui_time(start)} to "
                    f"{self._format_gui_time(end)} UTC",
                    fontsize=13,
                )
                figure.tight_layout(rect=(0, 0, 1, 0.96))
                self._new_figure_window(
                    f"Wavelet scalograms — {code}",
                    figure,
                    f"{code}_H_N_E_Z_wavelet_scalograms.png",
                    geometry="1450x950",
                    cursor_axes=axes,
                    scalogram_cursor_data=cursor_data,
                )
            except Exception as exc:
                failures.append(f"{code}: {exc}")

        successful = len(selected) - len(failures)
        self.pi2_status_var.set(
            f"Wavelet scalograms created for {successful}/{len(selected)} selected stations."
        )
        if failures:
            viewer.messagebox.showwarning(
                "Some scalograms failed", "\n\n".join(failures[:10])
            )

    def show_pi_pulsation_analysis(self) -> None:
        if self.data is None:
            viewer.messagebox.showinfo(
                "Pi pulsation analysis", "Open or download data file first."
            )
            return
        if not self.selected_station_indices:
            viewer.messagebox.showinfo(
                "Pi pulsation analysis", "Select one or more stations first."
            )
            return
        try:
            start, end = self._plot_interval()
            self._require_scipy()
        except (ValueError, RuntimeError) as exc:
            viewer.messagebox.showerror("Pi pulsation analysis", str(exc))
            return

        selected = sorted(self.selected_station_indices)
        failures = []
        for number, station_index in enumerate(selected, start=1):
            code = str(self.data.station_codes[station_index])
            self.pi2_status_var.set(
                f"Computing Pi bands for {code} ({number}/{len(selected)})…"
            )
            self.root.update_idletasks()
            try:
                times, values, cadence, labels, coordinate_name = (
                    self._prepare_station_timeseries(
                        station_index, start, end, include_horizontal_magnitude=True
                    )
                )
                figure = Figure(figsize=(14, 12), dpi=100)
                axes = [figure.add_subplot(5, 1, 1)]
                axes.extend(
                    figure.add_subplot(5, 1, row, sharex=axes[0])
                    for row in range(2, 6)
                )
                component_artists = {label: [] for label in labels}
                for component_index, label in enumerate(labels):
                    line = axes[0].plot(
                        times, values[:, component_index], linewidth=0.8, label=label
                    )[0]
                    component_artists[label].append(line)
                axes[0].set_title("Magnetic perturbation components")
                axes[0].set_ylabel("dB (nT)")
                axes[0].legend(loc="upper right", ncol=3, fontsize=8)
                axes[0].grid(True, alpha=0.25)
                axes[0].tick_params(labelbottom=False)
                for axis, (band_name, minimum_period, maximum_period) in zip(
                    axes[1:4], self.PI_BANDS
                ):
                    try:
                        filtered, resolution_note = self._pi_band_components(
                            values, cadence, minimum_period, maximum_period
                        )
                        for component_index, label in enumerate(labels):
                            line = axis.plot(
                                times, filtered[:, component_index],
                                linewidth=0.9, label=label
                            )[0]
                            component_artists[label].append(line)
                        axis.legend(loc="upper right", ncol=3, fontsize=8)
                        axis.set_title(f"{band_name}: {resolution_note}")
                    except ValueError as exc:
                        axis.text(
                            0.5, 0.5, str(exc), transform=axis.transAxes,
                            ha="center", va="center", wrap=True
                        )
                        nominal = (
                            f">{minimum_period:g} s" if maximum_period is None
                            else f"{minimum_period:g}–{maximum_period:g} s"
                        )
                        axis.set_title(f"{band_name}: {nominal}")
                    axis.set_ylabel("dB (nT)")
                    axis.grid(True, alpha=0.25)
                    axis.tick_params(labelbottom=False)

                duration = float((times[-1] - times[0]) / np.timedelta64(1, "s"))
                periods = self._period_grid(
                    cadence, duration,
                    self.SCALOGRAM_PERIOD_RANGE[0],
                    self.SCALOGRAM_PERIOD_RANGE[1],
                )
                h_power = self._horizontal_wavelet_power(
                    values[:, :2], cadence, periods
                )
                mesh = self._draw_scalogram(
                    axes[4], times, periods, h_power, values[:, :2], cadence
                )
                self._configure_scalogram_axis(axes[4], visible_max_s=600.0)
                axes[4].set_title("H wavelet scalogram")
                axes[4].set_xlabel("UTC (HH:MM)")
                axes[4].set_xlim(times[0], times[-1])
                self._configure_time_axis(axes[4])
                figure.colorbar(
                    mesh, ax=axes[4], pad=0.015, label="Relative wavelet power (dB)"
                )
                figure.suptitle(
                    f"{code} Pi pulsation analysis ({coordinate_name}; "
                    f"{labels[0]}/{labels[1]}/H)\n"
                    f"{self._format_gui_time(start)} to "
                    f"{self._format_gui_time(end)} UTC",
                    fontsize=13,
                )
                figure.tight_layout(rect=(0, 0, 1, 0.96))
                self._new_figure_window(
                    f"Pi pulsation analysis — {code}",
                    figure,
                    f"mag_{code}_pi_pulsation_analysis.png",
                    geometry="1450x950",
                    cursor_axes=axes,
                    scalogram_cursor_data={axes[4]: (times, periods, h_power)},
                    component_artists=component_artists,
                    export_callback=lambda selected=selected, start=start, end=end: (
                        self._export_pulsation_band_data("pi", selected, start, end)
                    ),
                )
            except Exception as exc:
                failures.append(f"{code}: {exc}")

        successful = len(selected) - len(failures)
        self.pi2_status_var.set(
            f"Pi pulsation figures created for {successful}/{len(selected)} selected stations."
        )
        if failures:
            viewer.messagebox.showwarning(
                "Some Pi analyses failed", "\n\n".join(failures[:10])
            )

    def show_pc_pulsation_analysis(self) -> None:
        if self.data is None:
            viewer.messagebox.showinfo(
                "PC pulsation analysis", "Open or download a magnetic data file first."
            )
            return
        if not self.selected_station_indices:
            viewer.messagebox.showinfo(
                "PC pulsation analysis", "Select one or more stations first."
            )
            return
        try:
            start, end = self._plot_interval()
            self._require_scipy()
        except (ValueError, RuntimeError) as exc:
            viewer.messagebox.showerror("PC pulsation analysis", str(exc))
            return

        selected = sorted(self.selected_station_indices)
        failures = []
        for number, station_index in enumerate(selected, start=1):
            code = str(self.data.station_codes[station_index])
            self.pi2_status_var.set(
                f"Computing PC bands for {code} ({number}/{len(selected)})…"
            )
            self.root.update_idletasks()
            try:
                times, values, cadence, labels, coordinate_name = (
                    self._prepare_station_timeseries(
                        station_index,
                        start,
                        end,
                        include_horizontal_magnitude=True,
                    )
                )
                figure = Figure(figsize=(14, 12), dpi=100)
                axes = [figure.add_subplot(6, 1, 1)]
                axes.extend(
                    figure.add_subplot(6, 1, row, sharex=axes[0])
                    for row in range(2, 7)
                )
                component_artists = {label: [] for label in labels}
                for component_index, label in enumerate(labels):
                    line = axes[0].plot(
                        times, values[:, component_index], linewidth=0.8, label=label
                    )[0]
                    component_artists[label].append(line)
                axes[0].set_title("Magnetic perturbation components")
                axes[0].set_ylabel("dB (nT)")
                axes[0].legend(loc="upper right", ncol=3, fontsize=8)
                axes[0].grid(True, alpha=0.25)
                axes[0].tick_params(labelbottom=False)
                for axis, (band_name, minimum_period, maximum_period) in zip(
                    axes[1:5], self.PC_BANDS
                ):
                    try:
                        filtered = self._bandpass_components(
                            values, cadence, minimum_period, maximum_period
                        )
                        for component_index, label in enumerate(labels):
                            line = axis.plot(
                                times,
                                filtered[:, component_index],
                                linewidth=0.9,
                                label=label,
                            )[0]
                            component_artists[label].append(line)
                        axis.legend(loc="upper right", ncol=3, fontsize=8)
                    except ValueError as exc:
                        axis.text(
                            0.5,
                            0.5,
                            str(exc),
                            transform=axis.transAxes,
                            ha="center",
                            va="center",
                            wrap=True,
                        )
                    axis.set_ylabel("dB (nT)")
                    axis.set_title(
                        f"{band_name}: {minimum_period:g}–{maximum_period:g} s"
                    )
                    axis.grid(True, alpha=0.25)
                    axis.tick_params(labelbottom=False)

                duration = float((times[-1] - times[0]) / np.timedelta64(1, "s"))
                periods = self._period_grid(
                    cadence,
                    duration,
                    self.SCALOGRAM_PERIOD_RANGE[0],
                    750.0,
                )
                wavelet_power = self._horizontal_wavelet_power(
                    values[:, :2], cadence, periods
                )
                mesh = self._draw_scalogram(
                    axes[5], times, periods, wavelet_power, values[:, :2], cadence
                )
                self._configure_scalogram_axis(
                    axes[5], visible_max_s=750.0, multiline_labels=True
                )
                axes[5].set_xlabel("UTC (HH:MM)")
                axes[5].set_title("H wavelet scalogram")
                axes[5].set_xlim(times[0], times[-1])
                self._configure_time_axis(axes[5])
                figure.colorbar(
                    mesh,
                    ax=axes[5],
                    pad=0.015,
                    label="Relative wavelet power (dB)",
                )
                figure.suptitle(
                    f"{code} PC pulsation analysis ({coordinate_name}; "
                    f"{labels[0]}/{labels[1]}/H)\n"
                    f"{self._format_gui_time(start)} to "
                    f"{self._format_gui_time(end)} UTC",
                    fontsize=13,
                )
                figure.tight_layout(rect=(0, 0, 1, 0.96))
                self._new_figure_window(
                    f"PC pulsation analysis — {code}",
                    figure,
                    f"mag_{code}_pc_pulsation_analysis.png",
                    geometry="1450x950",
                    cursor_axes=axes,
                    scalogram_cursor_data={
                        axes[5]: (times, periods, wavelet_power)
                    },
                    component_artists=component_artists,
                    export_callback=lambda selected=selected, start=start, end=end: (
                        self._export_pulsation_band_data(
                            "pc", selected, start, end
                        )
                    ),
                )
            except Exception as exc:
                failures.append(f"{code}: {exc}")

        successful = len(selected) - len(failures)
        self.pi2_status_var.set(
            f"PC pulsation figures created for {successful}/{len(selected)} "
            "selected stations."
        )
        if failures:
            viewer.messagebox.showwarning(
                "Some PC analyses failed", "\n\n".join(failures[:10])
            )

    def show_total_ulf_wave_power_map(self) -> None:
        if self.data is None:
            viewer.messagebox.showinfo(
                "Total ULF wave power", "Open or download a magnetic data file first."
            )
            return
        selected = [
            int(index)
            for index in sorted(self.selected_station_indices)
            if np.isfinite(self.data.glat[int(index)])
            and np.isfinite(self.data.glon[int(index)])
        ]
        if not selected:
            viewer.messagebox.showinfo(
                "Total ULF wave power",
                "Select one or more stations first. Use Select all stations to analyze every station.",
            )
            return
        if not self._confirm_small_region_map_analysis(
            "Total ULF wave power", selected
        ):
            self.pi2_status_var.set("Total ULF wave-power analysis cancelled.")
            return
        try:
            start, end = self._plot_interval()
            self._require_scipy()
        except (ValueError, RuntimeError) as exc:
            viewer.messagebox.showerror("Total ULF wave power", str(exc))
            return

        minimum_period, maximum_period = self.TOTAL_ULF_PERIOD_RANGE
        valid_indices = []
        powers = []
        resolved_band_notes = []
        failures = []
        station_count = len(selected)
        for number, station_index in enumerate(selected, start=1):
            code = str(self.data.station_codes[station_index])
            self.pi2_status_var.set(
                f"Computing total ULF power for {code} ({number}/{station_count})…"
            )
            self.root.update_idletasks()
            try:
                _, values, cadence, _, _ = self._prepare_station_timeseries(
                    station_index, start, end
                )
                filtered, _, band_note = self._ulf_bandpass_components(
                    values, cadence, minimum_period, maximum_period
                )
                mean_power = float(np.nanmean(np.sum(filtered ** 2, axis=1)))
                if np.isfinite(mean_power) and mean_power > 0.0:
                    valid_indices.append(station_index)
                    powers.append(mean_power)
                    resolved_band_notes.append(band_note)
            except Exception as exc:
                failures.append(f"{code}: {exc}")

        if not valid_indices:
            self.pi2_status_var.set("Total ULF wave-power calculation failed.")
            viewer.messagebox.showerror(
                "Total ULF wave power",
                "No station produced finite wave power.\n\n"
                + "\n\n".join(failures[:8]),
            )
            return

        power_values = np.asarray(powers, dtype=float)
        lats = np.asarray(self.data.glat[valid_indices], dtype=float)
        lons = np.asarray(viewer.normalize_longitude(self.data.glon[valid_indices]), dtype=float)
        center_lon, relative_lons, _ = viewer.choose_longitude_window(lons)

        low = float(np.nanpercentile(power_values, 5.0))
        high = float(np.nanpercentile(power_values, 95.0))
        if not np.isfinite(low) or low <= 0.0:
            low = float(np.nanmin(power_values[power_values > 0.0]))
        if not np.isfinite(high) or high <= low:
            high = float(np.nanmax(power_values))
        if high <= low:
            high = low * 1.001
        normalized = np.clip(
            (np.log10(power_values) - np.log10(low))
            / max(np.log10(high) - np.log10(low), 1.0e-12),
            0.0,
            1.0,
        )
        bubble_sizes = 45.0 + 360.0 * np.sqrt(normalized)

        window = viewer.tk.Toplevel(self.root)
        window.title("Total ULF wave power map")
        window.geometry("1250x820")
        window.minsize(900, 620)
        self._plot_windows.append(window)

        top = viewer.ttk.Frame(window, padding=5)
        top.pack(fill=viewer.tk.X)
        selection_status = viewer.tk.StringVar(
            value="No map stations selected. Left-click bubbles to select; double-click for station detail."
        )
        viewer.ttk.Label(
            top,
            textvariable=selection_status,
        ).pack(side=viewer.tk.LEFT)

        figure = Figure(figsize=(12, 7.5), dpi=100)
        if viewer.HAS_CARTOPY:
            projection = viewer.ccrs.PlateCarree(central_longitude=center_lon)
            axis = figure.add_subplot(111, projection=projection)
            axis.add_feature(viewer.cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
            axis.add_feature(viewer.cfeature.OCEAN, facecolor="white")
            axis.add_feature(viewer.cfeature.COASTLINE, linewidth=0.7)
            axis.add_feature(viewer.cfeature.BORDERS, linewidth=0.5, linestyle=":")
            axis.add_feature(viewer.cfeature.LAKES, alpha=0.45)
            grid = axis.gridlines(
                crs=viewer.ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.45,
                alpha=0.45,
                linestyle="--",
            )
            grid.top_labels = False
            grid.right_labels = False
            rel_min = float(np.nanmin(relative_lons))
            rel_max = float(np.nanmax(relative_lons))
            lat_min = float(np.nanmin(lats))
            lat_max = float(np.nanmax(lats))
            axis.set_extent(
                [
                    max(-180.0, rel_min - 6.0),
                    min(180.0, rel_max + 6.0),
                    max(-90.0, lat_min - 4.0),
                    min(90.0, lat_max + 4.0),
                ],
                crs=projection,
            )
            scatter = axis.scatter(
                lons,
                lats,
                s=bubble_sizes,
                c=power_values,
                cmap="viridis",
                norm=LogNorm(vmin=low, vmax=high),
                edgecolors="black",
                linewidths=0.7,
                alpha=0.85,
                picker=7,
                transform=viewer.ccrs.PlateCarree(),
                zorder=12,
            )
            selected_scatter = axis.scatter(
                [],
                [],
                s=125,
                c="red",
                marker="x",
                linewidths=2.0,
                transform=viewer.ccrs.PlateCarree(),
                zorder=20,
            )
            selection_x = lons
        else:
            axis = figure.add_subplot(111)
            scatter = axis.scatter(
                relative_lons,
                lats,
                s=bubble_sizes,
                c=power_values,
                cmap="viridis",
                norm=LogNorm(vmin=low, vmax=high),
                edgecolors="black",
                linewidths=0.7,
                alpha=0.85,
                picker=7,
                zorder=12,
            )
            selected_scatter = axis.scatter(
                [],
                [],
                s=125,
                c="red",
                marker="x",
                linewidths=2.0,
                zorder=20,
            )
            selection_x = relative_lons
            axis.set_xlabel(f"Longitude relative to {center_lon:.1f}°")
            axis.set_ylabel("Geographic latitude")
            axis.grid(True, alpha=0.3)
            axis.set_xlim(float(np.nanmin(relative_lons)) - 5.0, float(np.nanmax(relative_lons)) + 5.0)
            axis.set_ylim(float(np.nanmin(lats)) - 4.0, float(np.nanmax(lats)) + 4.0)

        unique_band_notes = list(dict.fromkeys(resolved_band_notes))
        band_display = (
            unique_band_notes[0]
            if len(unique_band_notes) == 1
            else "station cadence-dependent bands: " + "; ".join(unique_band_notes)
        )
        axis.set_title(
            f"Total horizontal ULF wave power ({band_display})\n"
            f"{self._format_gui_time(start)} to {self._format_gui_time(end)} UTC"
        )
        figure.colorbar(
            scatter,
            ax=axis,
            pad=0.02,
            label="Mean horizontal ULF wave power (nT²; logarithmic scale)",
        )
        figure.tight_layout()

        selected_local_indices: set[int] = set()

        def update_map_selection() -> None:
            ordered = sorted(selected_local_indices)
            if ordered:
                offsets = np.column_stack((selection_x[ordered], lats[ordered]))
            else:
                offsets = np.empty((0, 2), dtype=float)
            selected_scatter.set_offsets(offsets)
            codes = [str(self.data.station_codes[valid_indices[index]]) for index in ordered]
            if codes:
                preview = ", ".join(codes[:8])
                if len(codes) > 8:
                    preview += f", … (+{len(codes) - 8})"
                selection_status.set(f"Selected {len(codes)} station(s): {preview}")
            else:
                selection_status.set(
                    "No map stations selected. Left-click bubbles to select; double-click for station detail."
                )
            canvas.draw_idle()

        def plot_selected_map_stations() -> None:
            if not selected_local_indices:
                viewer.messagebox.showinfo(
                    "Multi-station ULF power",
                    "Select one or more station bubbles on the map first.",
                )
                return
            station_indices = [
                valid_indices[index] for index in sorted(selected_local_indices)
            ]
            self.plot_multi_station_ulf_power(station_indices, start, end)

        def clear_map_selection() -> None:
            selected_local_indices.clear()
            update_map_selection()

        # Packed first on the RIGHT so it appears immediately to the right of
        # the time-series button that is packed next.
        viewer.ttk.Button(
            top,
            text="Remove selection",
            command=clear_map_selection,
        ).pack(side=viewer.tk.RIGHT, padx=(5, 0))
        viewer.ttk.Button(
            top,
            text="Plot multi-station time series",
            command=plot_selected_map_stations,
        ).pack(side=viewer.tk.RIGHT, padx=(5, 0))
        viewer.ttk.Button(
            top,
            text="Save map PNG…",
            command=lambda: self.save_figure_png(figure, "mag_total_ulf_wave_power_map.png"),
        ).pack(side=viewer.tk.RIGHT)

        container = viewer.ttk.Frame(window)
        container.pack(fill=viewer.tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(figure, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=viewer.tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, container, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=viewer.tk.X)

        def on_pick(event) -> None:
            if event.artist is not scatter or not len(event.ind):
                return
            local_index = int(event.ind[0])
            if not 0 <= local_index < len(valid_indices):
                return
            if getattr(event.mouseevent, "dblclick", False):
                self.open_station_ulf_power_figure(valid_indices[local_index], start, end)
                return
            if local_index in selected_local_indices:
                selected_local_indices.remove(local_index)
            else:
                selected_local_indices.add(local_index)
            update_map_selection()

        canvas.mpl_connect("pick_event", on_pick)

        status = (
            f"Computed {len(valid_indices)} station(s); {len(failures)} failed or lacked "
            f"resolvable data. Band(s) used: {band_display}."
        )
        viewer.ttk.Label(window, text=status, padding=4).pack(fill=viewer.tk.X)
        self.pi2_status_var.set(status)

        def close_window() -> None:
            try:
                self._plot_windows.remove(window)
            except ValueError:
                pass
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)

    def _station_ulf_power_series(
        self,
        station_index: int,
        start: datetime,
        end: datetime,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, str], str, str]:
        minimum_period, maximum_period = self.TOTAL_ULF_PERIOD_RANGE
        times, values, cadence, labels, coordinate_name = (
            self._prepare_station_timeseries(station_index, start, end)
        )
        filtered, _, band_note = self._ulf_bandpass_components(
            values, cadence, minimum_period, maximum_period
        )
        instantaneous_power = np.sum(filtered ** 2, axis=1)
        averaging_samples = max(3, int(round(10.0 / cadence)))
        wave_power = self._rolling_mean(instantaneous_power, averaging_samples)
        return times, wave_power, labels, coordinate_name, band_note

    def plot_multi_station_ulf_power(
        self,
        station_indices: list[int],
        start: datetime,
        end: datetime,
    ) -> None:
        """Plot stacked station ULF power with editable time and y-axis limits."""
        if self.data is None or not station_indices:
            return

        figure = Figure(
            figsize=(14, max(6.0, 2.25 * len(station_indices) + 1.0)),
            dpi=100,
        )
        axes = [figure.add_subplot(len(station_indices), 1, 1)]
        axes.extend(
            figure.add_subplot(
                len(station_indices),
                1,
                row,
                sharex=axes[0],
            )
            for row in range(2, len(station_indices) + 1)
        )

        successful = 0
        axis_records: list[dict[str, Any]] = []
        available_starts: list[datetime] = []
        available_ends: list[datetime] = []
        for axis, station_index in zip(axes, station_indices):
            code = str(self.data.station_codes[int(station_index)])
            record: dict[str, Any] = {
                "axis": axis,
                "code": code,
                "station_index": int(station_index),
                "success": False,
            }
            try:
                times, wave_power, labels, coordinate_name, band_note = (
                    self._station_ulf_power_series(station_index, start, end)
                )
                axis.plot(times, wave_power, linewidth=1.0, label=code)
                axis.set_xlim(times[0], times[-1])
                record.update(
                    {
                        "success": True,
                        "times": times,
                        "wave_power": wave_power,
                        "coordinate_name": coordinate_name,
                        "labels": labels,
                        "band_note": band_note,
                    }
                )
                available_starts.append(self._datetime64_to_datetime(times[0]))
                available_ends.append(self._datetime64_to_datetime(times[-1]))
                successful += 1
            except Exception as exc:
                record["error"] = str(exc)
                axis.text(
                    0.5,
                    0.5,
                    str(exc),
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    wrap=True,
                )
            axis.set_ylabel(f"{code}\nULF power\n(nT²)")
            axis.grid(True, alpha=0.3)
            if axis is not axes[-1]:
                axis.tick_params(labelbottom=False)
            axis_records.append(record)

        axes[-1].set_xlabel("UTC (HH:MM)")
        self._configure_time_axis(axes[-1])
        minimum_period, maximum_period = self.TOTAL_ULF_PERIOD_RANGE
        used_bands = list(dict.fromkeys(
            str(record["band_note"])
            for record in axis_records if record.get("success")
        ))
        band_display = "; ".join(used_bands) if used_bands else "no resolvable band"

        available_start = min(available_starts) if available_starts else start
        available_end = max(available_ends) if available_ends else end
        displayed_interval = {"start": start, "end": end}

        def update_title() -> None:
            figure.suptitle(
                f"Multi-station running horizontal ULF wave power "
                f"({band_display})\n"
                f"{self._format_gui_time(displayed_interval['start'])} to "
                f"{self._format_gui_time(displayed_interval['end'])} UTC",
                fontsize=13,
            )

        update_title()
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        station_codes = "_".join(
            str(self.data.station_codes[index]) for index in station_indices[:6]
        )

        window = viewer.tk.Toplevel(self.root)
        window.title(
            f"Multi-station ULF wave power — "
            f"{successful}/{len(station_indices)} stations"
        )
        window.geometry("1500x950")
        window.minsize(1050, 650)
        self._plot_windows.append(window)

        action_row = viewer.ttk.Frame(window, padding=5)
        action_row.pack(fill=viewer.tk.X)
        viewer.ttk.Button(
            action_row,
            text="Save figure PNG…",
            command=lambda: self.save_figure_png(
                figure,
                f"{times[1]}_multi_station_ulf_power_{station_codes}.png",
            ),
        ).pack(side=viewer.tk.RIGHT)

        time_start_var = viewer.tk.StringVar(value=self._format_gui_time(start))
        time_end_var = viewer.tk.StringVar(value=self._format_gui_time(end))
        time_status_var = viewer.tk.StringVar(
            value=(
                f"Available plotted data: {self._format_gui_time(available_start)} to "
                f"{self._format_gui_time(available_end)} UTC"
            )
        )

        time_controls = viewer.ttk.LabelFrame(
            window, text="Displayed time range (UTC)", padding=5
        )
        time_controls.pack(fill=viewer.tk.X, padx=5, pady=(0, 5))
        viewer.ttk.Label(time_controls, text="Start").grid(
            row=0, column=0, sticky="w"
        )
        start_entry = viewer.ttk.Entry(
            time_controls, textvariable=time_start_var, width=21
        )
        start_entry.grid(row=0, column=1, sticky="w", padx=(4, 10))
        viewer.ttk.Label(time_controls, text="End").grid(
            row=0, column=2, sticky="w"
        )
        end_entry = viewer.ttk.Entry(
            time_controls, textvariable=time_end_var, width=21
        )
        end_entry.grid(row=0, column=3, sticky="w", padx=(4, 10))

        body = viewer.ttk.Frame(window)
        body.pack(fill=viewer.tk.BOTH, expand=True)

        y_outer = viewer.ttk.LabelFrame(body, text="Y-axis controls", padding=4)
        y_outer.pack(side=viewer.tk.LEFT, fill=viewer.tk.Y, padx=(5, 2), pady=(0, 5))
        y_canvas = viewer.tk.Canvas(
            y_outer,
            width=330,
            highlightthickness=0,
        )
        y_scrollbar = viewer.ttk.Scrollbar(
            y_outer, orient=viewer.tk.VERTICAL, command=y_canvas.yview
        )
        y_canvas.configure(yscrollcommand=y_scrollbar.set)
        y_scrollbar.pack(side=viewer.tk.RIGHT, fill=viewer.tk.Y)
        y_canvas.pack(side=viewer.tk.LEFT, fill=viewer.tk.BOTH, expand=True)
        y_inner = viewer.ttk.Frame(y_canvas)
        y_window_id = y_canvas.create_window((0, 0), window=y_inner, anchor="nw")

        def update_y_scrollregion(_event=None) -> None:
            y_canvas.configure(scrollregion=y_canvas.bbox("all"))

        def resize_y_inner(event) -> None:
            y_canvas.itemconfigure(y_window_id, width=event.width)

        y_inner.bind("<Configure>", update_y_scrollregion)
        y_canvas.bind("<Configure>", resize_y_inner)

        canvas_container = viewer.ttk.Frame(body)
        canvas_container.pack(
            side=viewer.tk.LEFT,
            fill=viewer.tk.BOTH,
            expand=True,
            padx=(2, 5),
            pady=(0, 5),
        )
        canvas = FigureCanvasTkAgg(figure, master=canvas_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=viewer.tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, canvas_container, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=viewer.tk.X)

        def parsed_display_interval() -> tuple[datetime, datetime]:
            new_start = self._parse_gui_datetime(
                time_start_var.get(), "Displayed start"
            )
            new_end = self._parse_gui_datetime(time_end_var.get(), "Displayed end")
            if new_end <= new_start:
                raise ValueError(
                    "Displayed end time must be later than displayed start time."
                )
            if new_end < available_start or new_start > available_end:
                raise ValueError(
                    "The entered time range does not overlap the plotted ULF data."
                )
            return new_start, new_end

        def apply_time_range(_event=None) -> None:
            try:
                new_start, new_end = parsed_display_interval()
            except ValueError as exc:
                viewer.messagebox.showerror(
                    "Multi-station ULF time range", str(exc), parent=window
                )
                return
            displayed_interval["start"] = new_start
            displayed_interval["end"] = new_end
            axes[-1].set_xlim(new_start, new_end)
            update_title()
            time_status_var.set(
                f"Displayed: {self._format_gui_time(new_start)} to "
                f"{self._format_gui_time(new_end)} UTC"
            )
            canvas.draw_idle()

        def reset_time_range() -> None:
            time_start_var.set(self._format_gui_time(start))
            time_end_var.set(self._format_gui_time(end))
            apply_time_range()

        viewer.ttk.Button(
            time_controls, text="Apply time range", command=apply_time_range
        ).grid(row=0, column=4, sticky="ew", padx=(0, 5))
        viewer.ttk.Button(
            time_controls, text="Reset", command=reset_time_range
        ).grid(row=0, column=5, sticky="ew")
        viewer.ttk.Label(
            time_controls,
            textvariable=time_status_var,
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(4, 0))
        start_entry.bind("<Return>", apply_time_range)
        end_entry.bind("<Return>", apply_time_range)

        def export_current_range() -> None:
            try:
                export_start, export_end = parsed_display_interval()
            except ValueError as exc:
                viewer.messagebox.showerror(
                    "Multi-station ULF export", str(exc), parent=window
                )
                return
            self._export_multi_station_ulf_data(
                list(station_indices), export_start, export_end
            )

        viewer.ttk.Button(
            action_row,
            text="Export displayed-range data…",
            command=export_current_range,
        ).pack(side=viewer.tk.RIGHT, padx=(0, 5))
        viewer.ttk.Label(
            action_row,
            text=(
                "Set the x-axis above; use the station controls at left for "
                "independent y-axis limits."
            ),
        ).pack(side=viewer.tk.LEFT)

        viewer.ttk.Label(y_inner, text="Station").grid(
            row=0, column=0, sticky="w", padx=(2, 4)
        )
        viewer.ttk.Label(y_inner, text="Y min").grid(
            row=0, column=1, sticky="w", padx=2
        )
        viewer.ttk.Label(y_inner, text="Y max").grid(
            row=0, column=2, sticky="w", padx=2
        )

        y_limit_controls: list[dict[str, Any]] = []

        def apply_y_limits(control: dict[str, Any], _event=None) -> None:
            axis = control["axis"]
            code = control["code"]
            low_text = control["low_var"].get().strip()
            high_text = control["high_var"].get().strip()
            try:
                current_low, current_high = axis.get_ylim()
                low = float(low_text) if low_text else float(current_low)
                high = float(high_text) if high_text else float(current_high)
                if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                    raise ValueError
            except ValueError:
                viewer.messagebox.showerror(
                    "ULF y-axis limits",
                    f"{code}: enter finite values with Y min < Y max.",
                    parent=window,
                )
                return
            axis.set_ylim(low, high)
            control["low_var"].set(f"{low:.6g}")
            control["high_var"].set(f"{high:.6g}")
            canvas.draw_idle()

        def auto_y_limits(control: dict[str, Any]) -> None:
            axis = control["axis"]
            # set_ylim() disables Matplotlib's y autoscaling. Re-enable it
            # explicitly so Auto continues to work after manual limits.
            axis.set_autoscaley_on(True)
            axis.relim(visible_only=True)
            axis.autoscale(enable=True, axis="y", tight=False)
            low, high = axis.get_ylim()
            control["low_var"].set(f"{low:.6g}")
            control["high_var"].set(f"{high:.6g}")
            canvas.draw_idle()

        for row, record in enumerate(axis_records, start=1):
            axis = record["axis"]
            code = record["code"]
            low, high = axis.get_ylim()
            low_var = viewer.tk.StringVar(value=f"{low:.6g}")
            high_var = viewer.tk.StringVar(value=f"{high:.6g}")
            control = {
                "axis": axis,
                "code": code,
                "low_var": low_var,
                "high_var": high_var,
            }
            y_limit_controls.append(control)
            viewer.ttk.Label(y_inner, text=code).grid(
                row=row, column=0, sticky="w", padx=(2, 4), pady=2
            )
            low_entry = viewer.ttk.Entry(
                y_inner, textvariable=low_var, width=10
            )
            low_entry.grid(row=row, column=1, sticky="ew", padx=2, pady=2)
            high_entry = viewer.ttk.Entry(
                y_inner, textvariable=high_var, width=10
            )
            high_entry.grid(row=row, column=2, sticky="ew", padx=2, pady=2)
            apply_button = viewer.ttk.Button(
                y_inner,
                text="Apply",
                command=lambda item=control: apply_y_limits(item),
            )
            apply_button.grid(row=row, column=3, sticky="ew", padx=2, pady=2)
            auto_button = viewer.ttk.Button(
                y_inner,
                text="Auto",
                command=lambda item=control: auto_y_limits(item),
            )
            auto_button.grid(row=row, column=4, sticky="ew", padx=(2, 4), pady=2)
            low_entry.bind(
                "<Return>", lambda event, item=control: apply_y_limits(item, event)
            )
            high_entry.bind(
                "<Return>", lambda event, item=control: apply_y_limits(item, event)
            )
            if not record["success"]:
                low_entry.configure(state="disabled")
                high_entry.configure(state="disabled")
                apply_button.configure(state="disabled")
                auto_button.configure(state="disabled")

        def auto_all_y_limits() -> None:
            for control, record in zip(y_limit_controls, axis_records):
                if record["success"]:
                    auto_y_limits(control)

        viewer.ttk.Separator(y_inner, orient=viewer.tk.HORIZONTAL).grid(
            row=len(axis_records) + 1,
            column=0,
            columnspan=5,
            sticky="ew",
            pady=(6, 4),
        )
        viewer.ttk.Button(
            y_inner, text="Auto-scale all Y axes", command=auto_all_y_limits
        ).grid(
            row=len(axis_records) + 2,
            column=0,
            columnspan=5,
            sticky="ew",
            padx=2,
            pady=(0, 4),
        )
        for column in (1, 2):
            y_inner.columnconfigure(column, weight=1)

        self._attach_time_value_cursor(canvas, axes)

        def close_window() -> None:
            try:
                self._plot_windows.remove(window)
            except ValueError:
                pass
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)

    def open_station_ulf_power_figure(
        self,
        station_index: int,
        start: datetime,
        end: datetime,
    ) -> None:
        if self.data is None:
            return
        code = str(self.data.station_codes[int(station_index)])
        minimum_period, maximum_period = self.TOTAL_ULF_PERIOD_RANGE
        try:
            times, wave_power, labels, coordinate_name, band_note = self._station_ulf_power_series(
                station_index, start, end
            )
            _, values, cadence, _, _ = self._prepare_station_timeseries(
                station_index, start, end
            )
            duration = float((times[-1] - times[0]) / np.timedelta64(1, "s"))
            periods = self._period_grid(
                cadence,
                duration,
                self.ULF_DETAIL_SCALOGRAM_RANGE[0],
                self.ULF_DETAIL_SCALOGRAM_RANGE[1],
            )
            scalogram_power = self._horizontal_wavelet_power(
                values[:, :2], cadence, periods
            )
        except Exception as exc:
            viewer.messagebox.showerror(
                "Station ULF wave power", f"Could not analyze {code}:\n\n{exc}"
            )
            return

        figure = Figure(figsize=(14, 9), dpi=100)
        power_axis = figure.add_subplot(2, 1, 1)
        scale_axis = figure.add_subplot(2, 1, 2, sharex=power_axis)
        power_axis.plot(times, wave_power, linewidth=1.0, label="ULF power")
        power_axis.set_ylabel("ULF power (nT²)")
        power_axis.set_title(
            f"{code} running horizontal {band_note} wave power "
            f"({coordinate_name} {labels[0]}/{labels[1]})"
        )
        power_axis.grid(True, alpha=0.3)
        power_axis.tick_params(labelbottom=False)

        mesh = self._draw_scalogram(
            scale_axis, times, periods, scalogram_power, values[:, :2], cadence
        )
        self._configure_scalogram_axis(scale_axis, visible_max_s=200.0)
        scale_axis.set_xlabel("UTC (HH:MM)")
        scale_axis.set_title("Station horizontal wavelet scalogram")
        scale_axis.grid(True, alpha=0.18)
        scale_axis.set_xlim(times[0], times[-1])
        self._configure_time_axis(scale_axis)
        figure.colorbar(
            mesh,
            ax=scale_axis,
            pad=0.015,
            label="Horizontal relative wavelet power (dB)",
        )
        figure.suptitle(
            f"{code} total ULF wave-power detail\n"
            f"{self._format_gui_time(start)} to {self._format_gui_time(end)} UTC",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        self._new_figure_window(
            f"Total ULF wave power — {code}",
            figure,
            f"{code}_total_ulf_wave_power.png",
            geometry="1450x900",
            cursor_axes=[power_axis, scale_axis],
            scalogram_cursor_data={scale_axis: (times, periods, scalogram_power)},
        )


    # ------------------------------------------------------------------
    # Pi2 analysis and result map
    # ------------------------------------------------------------------

    def _analysis_settings(
        self,
    ) -> tuple[datetime, datetime, datetime, datetime, Optional[datetime], str]:
        event_start = self._parse_required_time(self.pi2_start_var.get(), "Pi2 start")
        event_end = self._parse_required_time(self.pi2_end_var.get(), "Pi2 end")
        quiet_start = self._parse_required_time(
            self.quiet_start_var.get(), "Quiet/pre-event start"
        )
        quiet_end = self._parse_required_time(
            self.quiet_end_var.get(), "Quiet/pre-event end"
        )
        onset = self._parse_optional_onset(strict=True)
        coordinate = self.pi2_coordinate_var.get().strip().upper()
        if coordinate not in {"NEZ", "GEO"}:
            raise ValueError("Pi2 coordinates must be NEZ or GEO.")
        if event_end <= event_start:
            raise ValueError("Pi2 end time must be later than Pi2 start time.")
        if quiet_end <= quiet_start:
            raise ValueError("Quiet/pre-event end time must be later than its start time.")
        if max(event_start, quiet_start) <= min(event_end, quiet_end):
            raise ValueError("Pi2 and quiet/pre-event intervals must not overlap.")
        return event_start, event_end, quiet_start, quiet_end, onset, coordinate

    @staticmethod
    def _normalized_rotation_sense(value: Any) -> str:
        """Normalize the core result to a display-safe rotation label."""
        sense = str(value).strip().lower()
        if sense in {"clockwise", "cw"}:
            return "clockwise"
        if sense in {"counterclockwise", "anticlockwise", "ccw", "acw"}:
            return "counterclockwise"
        return "indeterminate"

    @classmethod
    def _apparent_map_rotation_sense(cls, value: Any) -> str:
        """Return the apparent sense after changing from hodogram to map axes.

        The literature-style horizontal hodogram has magnetic/geographic north
        on the x axis and east on the y axis.  A geographic map has east on x
        and north on y.  Interchanging the axes is a reflection, so the
        apparent clockwise/counterclockwise tracing is reversed.  Numerical
        results and colours continue to use the stated H-E/N-E hodogram
        convention; this helper is used only to explain the map view.
        """
        sense = cls._normalized_rotation_sense(value)
        if sense == "clockwise":
            return "counterclockwise"
        if sense == "counterclockwise":
            return "clockwise"
        return "indeterminate"

    @staticmethod
    def _ellipse_map_azimuth(
        longitude: float,
        latitude: float,
        result: StationPi2Result,
    ) -> tuple[float, Optional[float]]:
        """Return a geographic map bearing and the applied declination.

        Spectral azimuth is measured clockwise from the first horizontal
        component toward the second.  GEO X/Y already means geographic
        north/east. Magnetic NEZ is local magnetic north/east, so its bearing
        is rotated into geographic coordinates using station- and epoch-
        specific IGRF declination (positive east of true north).
        """
        spectral_azimuth = float(
            result.spectra["spectral_major_azimuth_deg"]
        ) % 180.0
        if str(result.coordinate_system).strip().upper() != "NEZ":
            return spectral_azimuth, None
        if ppigrf is None:
            raise RuntimeError(
                "Displaying NEZ polarization ellipses on a geographic map "
                "requires ppigrf (IGRF-14). Install it with: "
                "python -m pip install ppigrf"
            )

        event_times = np.asarray(result.times, dtype=float)[
            np.asarray(result.event_mask, dtype=bool)
        ]
        if not len(event_times) or not np.all(np.isfinite(event_times)):
            raise ValueError(
                f"{result.station_code}: cannot determine the Pi2 event epoch "
                "for the NEZ-to-GEO rotation."
            )
        epoch = datetime.fromtimestamp(
            float(0.5 * (event_times[0] + event_times[-1])),
            tz=timezone.utc,
        ).replace(tzinfo=None)
        east_field, north_field, _up_field = ppigrf.igrf(
            float(viewer.normalize_longitude(longitude)),
            float(latitude),
            0.0,
            epoch,
        )
        east_nt = float(np.asarray(east_field, dtype=float).reshape(-1)[0])
        north_nt = float(np.asarray(north_field, dtype=float).reshape(-1)[0])
        horizontal_nt = float(np.hypot(east_nt, north_nt))
        if not np.isfinite(horizontal_nt) or horizontal_nt <= 1.0:
            raise ValueError(
                f"{result.station_code}: IGRF returned an invalid horizontal "
                "field, so the NEZ ellipse cannot be oriented geographically."
            )
        declination_deg = float(
            np.degrees(np.arctan2(east_nt, north_nt))
        )
        return (spectral_azimuth + declination_deg) % 180.0, declination_deg

    @staticmethod
    def _ellipse_vertices(
        longitude: float,
        latitude: float,
        result: StationPi2Result,
        base_scale_degrees: float,
        amplitude_factor: float,
        map_azimuth_deg: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if map_azimuth_deg is None:
            map_azimuth_deg = float(
                result.spectra["spectral_major_azimuth_deg"]
            )
        azimuth = np.radians(float(map_azimuth_deg))
        ratio = float(result.spectra["spectral_minor_to_major_ratio"])
        if not np.isfinite(ratio):
            ratio = 0.0
        ratio = float(np.clip(ratio, 0.0, 1.0))
        major = base_scale_degrees * amplitude_factor
        minor = major * ratio
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        north = (
            major * np.cos(theta) * np.cos(azimuth)
            - minor * np.sin(theta) * np.sin(azimuth)
        )
        east = (
            major * np.cos(theta) * np.sin(azimuth)
            + minor * np.sin(theta) * np.cos(azimuth)
        )
        cosine_latitude = max(abs(np.cos(np.radians(latitude))), 0.20)
        ellipse_lon = viewer.normalize_longitude(longitude + east / cosine_latitude)
        ellipse_lat = latitude + north
        return np.asarray(ellipse_lon, dtype=float), np.asarray(ellipse_lat, dtype=float)

    @staticmethod
    def _major_axis_vertices(
        longitude: float,
        latitude: float,
        result: StationPi2Result,
        base_scale_degrees: float,
        amplitude_factor: float,
        map_azimuth_deg: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return both endpoints of the undirected full major axis.

        Polarization azimuth is axial, with a 180-degree ambiguity.  Drawing a
        one-sided ray incorrectly suggests a directed bearing, so the map uses
        the complete line through the station.
        """
        if map_azimuth_deg is None:
            map_azimuth_deg = float(
                result.spectra["spectral_major_azimuth_deg"]
            )
        azimuth = np.radians(float(map_azimuth_deg))
        major = base_scale_degrees * amplitude_factor
        north = major * np.cos(azimuth)
        east = major * np.sin(azimuth)
        cosine_latitude = max(abs(np.cos(np.radians(latitude))), 0.20)
        axis_lon = viewer.normalize_longitude(
            longitude + np.asarray([-east, east]) / cosine_latitude
        )
        axis_lat = latitude + np.asarray([-north, north])
        return np.asarray(axis_lon, dtype=float), np.asarray(axis_lat, dtype=float)

    def open_pi2_analysis_window(self) -> None:
        if self.data is None:
            viewer.messagebox.showinfo(
                "Pi2 analysis", "Open or download a magnetic data file first."
            )
            return
        selected = [
            index
            for index in sorted(self.selected_station_indices)
            if np.isfinite(self.data.glat[index]) and np.isfinite(self.data.glon[index])
        ]
        if not selected:
            viewer.messagebox.showinfo(
                "Pi2 analysis",
                "Select one or more stations with finite geographic coordinates first.",
            )
            return
        if not self._confirm_small_region_map_analysis("Pi2 analysis", selected):
            self.pi2_status_var.set("Pi2 map analysis cancelled.")
            return

        window = viewer.tk.Toplevel(self.root)
        window.title("Selected-station dominant-frequency Pi2 spectral ellipses")
        window.geometry("1280x900")
        window.minsize(980, 680)
        self._plot_windows.append(window)

        settings = viewer.ttk.LabelFrame(window, text="Pi2 analysis settings (UTC)", padding=6)
        settings.pack(fill=viewer.tk.X, padx=5, pady=(5, 2))
        for column in range(8):
            settings.columnconfigure(column, weight=0)
        settings.columnconfigure(7, weight=1)

        viewer.ttk.Label(settings, text="Pi2 start").grid(row=0, column=0, sticky="w")
        viewer.ttk.Entry(settings, textvariable=self.pi2_start_var, width=20).grid(
            row=0, column=1, sticky="w", padx=(4, 10)
        )
        viewer.ttk.Label(settings, text="Pi2 end").grid(row=0, column=2, sticky="w")
        viewer.ttk.Entry(settings, textvariable=self.pi2_end_var, width=20).grid(
            row=0, column=3, sticky="w", padx=(4, 10)
        )
        viewer.ttk.Label(settings, text="Optional onset").grid(row=0, column=4, sticky="w")
        viewer.ttk.Entry(settings, textvariable=self.pi2_onset_var, width=20).grid(
            row=0, column=5, sticky="w", padx=(4, 10)
        )
        viewer.ttk.Label(
            settings,
            text="Format: YYYY-MM-DD HH:mm:SS",
            foreground="#555555",
        ).grid(row=0, column=6, columnspan=2, sticky="w")

        viewer.ttk.Label(settings, text="Quiet/pre-event start").grid(
            row=1, column=0, sticky="w", pady=(5, 0)
        )
        viewer.ttk.Entry(settings, textvariable=self.quiet_start_var, width=20).grid(
            row=1, column=1, sticky="w", padx=(4, 10), pady=(5, 0)
        )
        viewer.ttk.Label(settings, text="Quiet/pre-event end").grid(
            row=1, column=2, sticky="w", pady=(5, 0)
        )
        viewer.ttk.Entry(settings, textvariable=self.quiet_end_var, width=20).grid(
            row=1, column=3, sticky="w", padx=(4, 10), pady=(5, 0)
        )
        viewer.ttk.Label(settings, text="Pi2 Coordinates").grid(
            row=1, column=4, sticky="w", pady=(5, 0)
        )
        viewer.ttk.Combobox(
            settings,
            textvariable=self.pi2_coordinate_var,
            values=("GEO", "NEZ"),
            state="readonly",
            width=8,
        ).grid(row=1, column=5, sticky="w", padx=(4, 10), pady=(5, 0))
        viewer.ttk.Checkbutton(
            settings,
            text="Confirm +north, +east and +Z downward signs",
            variable=self.pi2_confirm_signs_var,
        ).grid(row=1, column=6, columnspan=2, sticky="w", pady=(5, 0))

        viewer.ttk.Label(settings, text="Band period min (s)").grid(
            row=2, column=0, sticky="w", pady=(5, 0)
        )
        viewer.ttk.Entry(
            settings, textvariable=self.pi2_band_min_var, width=10
        ).grid(row=2, column=1, sticky="w", padx=(4, 10), pady=(5, 0))
        viewer.ttk.Label(settings, text="Band period max (s)").grid(
            row=2, column=2, sticky="w", pady=(5, 0)
        )
        viewer.ttk.Entry(
            settings, textvariable=self.pi2_band_max_var, width=10
        ).grid(row=2, column=3, sticky="w", padx=(4, 10), pady=(5, 0))
        viewer.ttk.Label(
            settings,
            text="The filter, spectra, ellipses, and map use this period band.",
            foreground="#555555",
        ).grid(row=2, column=4, columnspan=4, sticky="w", pady=(5, 0))

        utility_row = viewer.ttk.Frame(settings)
        utility_row.grid(row=3, column=0, columnspan=8, sticky="ew", pady=(6, 0))
        viewer.ttk.Button(
            utility_row,
            text="Use plot range as Pi2",
            command=self.use_plot_range_as_pi2,
        ).pack(side=viewer.tk.LEFT)
        viewer.ttk.Button(
            utility_row,
            text="Use plot range as quiet",
            command=self.use_plot_range_as_quiet,
        ).pack(side=viewer.tk.LEFT, padx=(5, 0))

        action_row = viewer.ttk.Frame(window, padding=(5, 2, 5, 3))
        action_row.pack(fill=viewer.tk.X)
        status_var = viewer.tk.StringVar(
            value=(
                f"{len(selected)} selected station(s). Enter intervals, then click "
                "Analyze selected stations."
            )
        )
        viewer.ttk.Label(action_row, textvariable=status_var).pack(side=viewer.tk.RIGHT)

        lats = np.asarray(self.data.glat[selected], dtype=float)
        lons = np.asarray(viewer.normalize_longitude(self.data.glon[selected]), dtype=float)
        codes = [str(self.data.station_codes[index]) for index in selected]
        center_lon, relative_lons, lon_span = viewer.choose_longitude_window(lons)
        lat_span = float(max(np.ptp(lats), 1.0))
        base_scale = float(np.clip(0.035 * max(lon_span, lat_span, 12.0), 0.45, 3.5))

        figure = Figure(figsize=(12, 7.2), dpi=100)
        if viewer.HAS_CARTOPY:
            projection = viewer.ccrs.PlateCarree(central_longitude=center_lon)
            axis = figure.add_subplot(111, projection=projection)
            axis.add_feature(viewer.cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
            axis.add_feature(viewer.cfeature.OCEAN, facecolor="white")
            axis.add_feature(viewer.cfeature.COASTLINE, linewidth=0.7)
            axis.add_feature(viewer.cfeature.BORDERS, linewidth=0.5, linestyle=":")
            axis.add_feature(viewer.cfeature.LAKES, alpha=0.45)
            grid = axis.gridlines(
                crs=viewer.ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.45,
                alpha=0.45,
                linestyle="--",
            )
            grid.top_labels = False
            grid.right_labels = False
            rel_min = float(np.nanmin(relative_lons))
            rel_max = float(np.nanmax(relative_lons))
            lon_pad = max(4.0, 0.15 * max(rel_max - rel_min, 1.0) + 3.0 * base_scale)
            lat_pad = max(3.0, 0.15 * lat_span + 2.0 * base_scale)
            axis.set_extent(
                [
                    max(-180.0, rel_min - lon_pad),
                    min(180.0, rel_max + lon_pad),
                    max(-90.0, float(np.nanmin(lats)) - lat_pad),
                    min(90.0, float(np.nanmax(lats)) + lat_pad),
                ],
                crs=projection,
            )
            scatter = axis.scatter(
                lons,
                lats,
                s=82,
                c="limegreen",
                edgecolors="black",
                linewidths=1.0,
                picker=7,
                transform=viewer.ccrs.PlateCarree(),
                zorder=15,
            )
        else:
            axis = figure.add_subplot(111)
            scatter = axis.scatter(
                relative_lons,
                lats,
                s=82,
                c="limegreen",
                edgecolors="black",
                linewidths=1.0,
                picker=7,
                zorder=15,
            )
            axis.set_xlabel(f"Longitude relative to {center_lon:.1f}°")
            axis.set_ylabel("Geographic latitude")
            axis.grid(True, alpha=0.3)
            axis.set_xlim(float(np.min(relative_lons)) - 5.0, float(np.max(relative_lons)) + 5.0)
            axis.set_ylim(float(np.min(lats)) - 4.0, float(np.max(lats)) + 4.0)

        station_labels = []
        for local_index, code in enumerate(codes):
            annotation_x = float(lons[local_index]) if viewer.HAS_CARTOPY else float(relative_lons[local_index])
            kwargs = dict(
                xy=(annotation_x, float(lats[local_index])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                zorder=16,
            )
            if viewer.HAS_CARTOPY:
                kwargs["transform"] = viewer.ccrs.PlateCarree()
            station_labels.append(axis.annotate(code, **kwargs))

        axis.set_title(
            "Selected stations for dominant-frequency Pi2 spectral-ellipse analysis\n"
            "Ellipses are calculated only after Analyze selected stations is clicked."
        )
        figure.tight_layout()

        container = viewer.ttk.Frame(window)
        container.pack(fill=viewer.tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(figure, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=viewer.tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, container, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=viewer.tk.X)

        ellipse_button_text = viewer.tk.StringVar(value="Hide ellipses")
        indeterminate_button_text = viewer.tk.StringVar(
            value="Hide indeterminate ellipses"
        )
        major_button_text = viewer.tk.StringVar(value="Hide major axis")
        state: dict[str, Any] = {
            "window": window,
            "figure": figure,
            "axis": axis,
            "canvas": canvas,
            "scatter": scatter,
            "indices": selected,
            "lats": lats,
            "lons": lons,
            "relative_lons": relative_lons,
            "center_lon": center_lon,
            "base_scale": base_scale,
            "station_labels": station_labels,
            "ellipse_artists": [],
            "indeterminate_ellipse_artists": [],
            "major_axis_artists": [],
            "ellipse_visible": True,
            "indeterminate_visible": True,
            "major_visible": True,
            "status_var": status_var,
            "ellipse_button_text": ellipse_button_text,
            "indeterminate_button_text": indeterminate_button_text,
            "major_button_text": major_button_text,
            "results": {},
        }

        analyze_button = viewer.ttk.Button(
            action_row,
            text="Analyze selected stations",
            command=lambda: self.analyze_selected_stations(state),
        )
        analyze_button.pack(side=viewer.tk.LEFT)

        def toggle_major_axis() -> None:
            state["major_visible"] = not bool(state["major_visible"])
            for artist in state["major_axis_artists"]:
                artist.set_visible(bool(state["major_visible"]))
            major_button_text.set(
                "Hide major axis" if state["major_visible"] else "Show major axis"
            )
            canvas.draw_idle()

        def apply_ellipse_visibility() -> None:
            indeterminate_artists = set(state["indeterminate_ellipse_artists"])
            for artist in state["ellipse_artists"]:
                visible = bool(state["ellipse_visible"])
                if artist in indeterminate_artists:
                    visible = visible and bool(state["indeterminate_visible"])
                artist.set_visible(visible)

        def toggle_ellipses() -> None:
            state["ellipse_visible"] = not bool(state["ellipse_visible"])
            apply_ellipse_visibility()
            ellipse_button_text.set(
                "Hide ellipses" if state["ellipse_visible"] else "Show ellipses"
            )
            canvas.draw_idle()

        def toggle_indeterminate_ellipses() -> None:
            state["indeterminate_visible"] = not bool(
                state["indeterminate_visible"]
            )
            apply_ellipse_visibility()
            indeterminate_button_text.set(
                "Hide indeterminate ellipses"
                if state["indeterminate_visible"]
                else "Display indeterminate ellipses"
            )
            canvas.draw_idle()

        major_button = viewer.ttk.Button(
            action_row,
            textvariable=major_button_text,
            command=toggle_major_axis,
            state="disabled",
        )
        major_button.pack(side=viewer.tk.LEFT, padx=(5, 0))
        ellipse_button = viewer.ttk.Button(
            action_row,
            textvariable=ellipse_button_text,
            command=toggle_ellipses,
            state="disabled",
        )
        ellipse_button.pack(side=viewer.tk.LEFT, padx=(5, 0))
        indeterminate_button = viewer.ttk.Button(
            action_row,
            textvariable=indeterminate_button_text,
            command=toggle_indeterminate_ellipses,
            state="disabled",
        )
        indeterminate_button.pack(side=viewer.tk.LEFT, padx=(5, 0))
        viewer.ttk.Button(
            action_row,
            text="Save ellipse map PNG…",
            command=lambda: self.save_figure_png(
                figure, "pi2_spectral_ellipse_map.png"
            ),
        ).pack(side=viewer.tk.LEFT, padx=(5, 0))
        state["major_button"] = major_button
        state["ellipse_button"] = ellipse_button
        state["indeterminate_button"] = indeterminate_button

        def on_pick(event) -> None:
            if event.artist is not scatter or not len(event.ind):
                return
            local = int(event.ind[0])
            if 0 <= local < len(selected):
                station_index = selected[local]
                result = state["results"].get(station_index)
                if result is not None:
                    self.open_pi2_station_figure(station_index, result=result)
                else:
                    status_var.set(
                        f"{codes[local]} has not been analyzed successfully yet."
                    )

        canvas.mpl_connect("pick_event", on_pick)

        def close_window() -> None:
            try:
                self._plot_windows.remove(window)
            except ValueError:
                pass
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)

    def analyze_selected_stations(self, map_state: Optional[dict[str, Any]] = None) -> None:
        if map_state is None:
            self.open_pi2_analysis_window()
            return
        if self.data is None:
            return
        selected = [int(index) for index in map_state["indices"]]
        try:
            event_start, event_end, quiet_start, quiet_end, onset, coordinate = (
                self._analysis_settings()
            )
            period_min = float(self.pi2_band_min_var.get())
            period_max = float(self.pi2_band_max_var.get())
            if not (np.isfinite(period_min) and np.isfinite(period_max)):
                raise ValueError("Pi2 band periods must be finite.")
            if period_min <= 0.0 or period_max <= period_min:
                raise ValueError(
                    "Pi2 band period minimum must be positive and less than its maximum."
                )
        except ValueError as exc:
            viewer.messagebox.showerror("Pi2 analysis settings", str(exc))
            return

        if coordinate == "GEO":
            if self.data.geo is None:
                viewer.messagebox.showerror(
                    "Pi2 analysis", "The loaded file does not contain GEO components. Use NEZ."
                )
                return
            coordinate_data = self.data.geo
            labels = ("X", "Y", "Z")
            coordinate_description = (
                "GEO: +X geographic north, +Y east, +Z downward"
            )
        else:
            if self.data.nez is None:
                viewer.messagebox.showerror(
                    "Pi2 analysis", "The loaded file does not contain NEZ components. Use GEO."
                )
                return
            coordinate_data = self.data.nez
            labels = ("N", "E", "Z")
            coordinate_description = (
                "NEZ: +N local magnetic north, +E east, +Z downward"
            )

        results: dict[int, StationPi2Result] = {}
        failures: dict[str, str] = {}
        warnings_by_station: dict[str, list[str]] = {}
        status_var = map_state["status_var"]
        for number, station_index in enumerate(selected, start=1):
            code = str(self.data.station_codes[station_index])
            message = f"Analyzing {code} ({number}/{len(selected)})…"
            status_var.set(message)
            self.pi2_status_var.set(message)
            self.root.update_idletasks()
            try:
                station_values = viewer.mask_bad_magnetometer_values(
                    coordinate_data[station_index, :, :]
                )
                result = run_pi2_analysis(
                    station_code=code,
                    datetimes=self.data.times,
                    values=station_values,
                    event_interval=(event_start, event_end),
                    background_interval=(quiet_start, quiet_end),
                    onset_time=onset,
                    coordinate_system=coordinate,
                    component_labels=labels,
                    coordinate_description=coordinate_description,
                    coordinate_signs_verified=bool(self.pi2_confirm_signs_var.get()),
                    pi2_period_range=(period_min, period_max),
                )
                results[station_index] = result
                if result.warnings:
                    warnings_by_station[code] = result.warnings
            except Exception as exc:
                failures[code] = str(exc)

        self._pi2_results = results
        map_state["results"] = results
        if not results:
            message = "No selected station could be analyzed."
            if failures:
                message += "\n\n" + "\n\n".join(
                    f"{code}: {reason}" for code, reason in list(failures.items())[:8]
                )
            message += (
                "\n\nFor the nominal 40–150 s Pi2 band, the cadence must be faster "
                "than 20 s; standard 60 s data are insufficient."
            )
            status_var.set("Pi2 analysis failed for all selected stations.")
            self.pi2_status_var.set(status_var.get())
            viewer.messagebox.showerror("Pi2 analysis", message)
            return

        self._render_pi2_results_on_map(map_state, results, event_start, event_end)
        summary = f"Analyzed {len(results)} station(s); {len(failures)} failed."
        status_var.set(summary + " Click a green station for its detailed figure.")
        self.pi2_status_var.set(summary)

        if failures:
            failure_text = "\n\n".join(
                f"{code}: {reason}" for code, reason in list(failures.items())[:8]
            )
            viewer.messagebox.showwarning(
                "Some Pi2 analyses failed",
                "The map contains ellipses for successful stations only.\n\n"
                + failure_text,
            )
        elif warnings_by_station:
            warning_text = "\n\n".join(
                f"{code}: {' | '.join(messages)}"
                for code, messages in list(warnings_by_station.items())[:8]
            )
            viewer.messagebox.showwarning("Pi2 analysis warnings", warning_text)

    def _render_pi2_results_on_map(
        self,
        map_state: dict[str, Any],
        results: dict[int, StationPi2Result],
        event_start: datetime,
        event_end: datetime,
    ) -> None:
        """Render normalized spectral ellipses using explicit Pi2 conventions.

        The map shows geographic orientation without a numerical axis-ratio
        label. All ellipses have the same displayed semi-major length so that station-to-station
        shape and azimuth are not confounded with an undocumented amplitude
        scale.  Rotation colours retain the conventional horizontal hodogram
        definition: first horizontal component on x, eastward component on y.
        """
        axis = map_state["axis"]
        canvas = map_state["canvas"]
        for artist in map_state["ellipse_artists"] + map_state["major_axis_artists"]:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        map_state["ellipse_artists"] = []
        map_state["indeterminate_ellipse_artists"] = []
        map_state["major_axis_artists"] = []
        map_state["ellipse_visible"] = True
        map_state["indeterminate_visible"] = True
        map_state["major_visible"] = True
        map_state["ellipse_button_text"].set("Hide ellipses")
        map_state["indeterminate_button_text"].set(
            "Hide indeterminate ellipses"
        )
        map_state["major_button_text"].set("Hide major axis")

        indices = map_state["indices"]
        lats = map_state["lats"]
        lons = map_state["lons"]
        center_lon = float(map_state["center_lon"])
        base_scale = float(map_state["base_scale"])
        applied_declinations: list[float] = []
        omitted_azimuth_count = 0

        for local_index, station_index in enumerate(indices):
            code = str(self.data.station_codes[station_index])
            label = map_state["station_labels"][local_index]
            result = results.get(station_index)
            if result is None:
                label.set_text(f"{code}\nfailed")
                continue

            period_s = float(result.spectra["dominant_period_s"])
            axis_ratio = float(result.spectra["spectral_minor_to_major_ratio"])
            if not np.isfinite(axis_ratio):
                axis_ratio = 0.0
            axis_ratio = float(np.clip(axis_ratio, 0.0, 1.0))
            label.set_text(f"{code}\n{period_s:.0f} s")

            map_azimuth_deg, declination_deg = self._ellipse_map_azimuth(
                float(lons[local_index]),
                float(lats[local_index]),
                result,
            )
            if declination_deg is not None:
                applied_declinations.append(float(declination_deg))

            # Equal major-axis length is intentional.  Amplitude remains a
            # numerical result in the station detail; it is not encoded by an
            # undocumented nonlinear map-size transform.
            amplitude_factor = 1.0
            ellipse_lon, ellipse_lat = self._ellipse_vertices(
                float(lons[local_index]),
                float(lats[local_index]),
                result,
                base_scale,
                amplitude_factor,
                map_azimuth_deg,
            )

            rotation_reliable = bool(result.spectra["rotation_reliable"])
            linestyle = "-" if rotation_reliable else "--"
            hodogram_sense = self._normalized_rotation_sense(
                result.spectra["rotation_sense"]
            )
            if hodogram_sense == "clockwise":
                ellipse_color = "red"
            elif hodogram_sense == "counterclockwise":
                ellipse_color = "blue"
            else:
                ellipse_color = "0.35"

            if viewer.HAS_CARTOPY:
                ellipse_line, = axis.plot(
                    ellipse_lon,
                    ellipse_lat,
                    color=ellipse_color,
                    linewidth=1.7,
                    linestyle=linestyle,
                    transform=viewer.ccrs.PlateCarree(),
                    zorder=12,
                )
            else:
                ellipse_line, = axis.plot(
                    viewer.normalize_longitude(ellipse_lon - center_lon),
                    ellipse_lat,
                    color=ellipse_color,
                    linewidth=1.7,
                    linestyle=linestyle,
                    zorder=12,
                )
            map_state["ellipse_artists"].append(ellipse_line)
            if not rotation_reliable:
                map_state["indeterminate_ellipse_artists"].append(ellipse_line)

            # Azimuth is not meaningful for a nearly circular ellipse.
            if axis_ratio < self.POLARIZATION_AZIMUTH_MAX_RATIO:
                axis_lon, axis_lat = self._major_axis_vertices(
                    float(lons[local_index]),
                    float(lats[local_index]),
                    result,
                    base_scale,
                    amplitude_factor,
                    map_azimuth_deg,
                )
                if viewer.HAS_CARTOPY:
                    major_line, = axis.plot(
                        axis_lon,
                        axis_lat,
                        color="black",
                        linewidth=1.2,
                        transform=viewer.ccrs.PlateCarree(),
                        zorder=13,
                    )
                else:
                    major_line, = axis.plot(
                        viewer.normalize_longitude(axis_lon - center_lon),
                        axis_lat,
                        color="black",
                        linewidth=1.2,
                        zorder=13,
                    )
                map_state["major_axis_artists"].append(major_line)
            else:
                omitted_azimuth_count += 1

        first_result = next(iter(results.values()))
        horizontal_labels = tuple(first_result.config.component_labels[:2])
        first_label = str(horizontal_labels[0]) if horizontal_labels else "N"
        second_label = str(horizontal_labels[1]) if len(horizontal_labels) > 1 else "E"
        orientation_note = (
            "GEO azimuths use geographic north/east."
            if first_result.coordinate_system == "GEO"
            else (
                "NEZ azimuths are rotated to geographic bearings with "
                "station/epoch IGRF-14 declination."
            )
        )
        if applied_declinations:
            orientation_note += (
                f" Applied D: {min(applied_declinations):+.1f}° to "
                f"{max(applied_declinations):+.1f}° (east-positive)."
            )
        if omitted_azimuth_count:
            orientation_note += (
                f" Major axis omitted for {omitted_azimuth_count} near-circular "
                "ellipse(s)."
            )

        axis.set_title(
            "Dominant-frequency Pi2 spectral ellipses — "
            f"{self._format_gui_time(event_start)} to "
            f"{self._format_gui_time(event_end)} UTC\n"
            f"{orientation_note} Azimuth is axial, clockwise from north toward east; "
            "ellipses have equal displayed major-axis length.\n"
            f"Colours report the {first_label}–{second_label} hodogram sense "
            f"(+{first_label} right, +{second_label} up): red CW, blue CCW; "
            "grey/dashed indeterminate. Apparent tracing on a north-up map is opposite."
        )
        legend_handles = [
            Line2D([0], [0], color="red", linewidth=1.7, label="CW hodogram"),
            Line2D([0], [0], color="blue", linewidth=1.7, label="CCW hodogram"),
            Line2D(
                [0], [0], color="0.35", linewidth=1.7, linestyle="--",
                label="Indeterminate rotation",
            ),
            Line2D([0], [0], color="black", linewidth=1.2, label="Full major axis"),
        ]
        axis.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=8,
            framealpha=0.90,
            title=f"{first_label}–{second_label} convention",
            title_fontsize=8,
        )
        map_state["major_button"].configure(
            state="normal" if map_state["major_axis_artists"] else "disabled"
        )
        map_state["ellipse_button"].configure(
            state="normal" if map_state["ellipse_artists"] else "disabled"
        )
        map_state["indeterminate_button"].configure(
            state=(
                "normal"
                if map_state["indeterminate_ellipse_artists"]
                else "disabled"
            )
        )
        canvas.draw_idle()


    def open_pi2_station_figure(
        self,
        station_index: int,
        result: Optional[StationPi2Result] = None,
    ) -> None:
        if result is None:
            result = self._pi2_results.get(int(station_index))
        if result is None:
            return

        labels = tuple(result.config.component_labels)
        first_label = str(labels[0]) if labels else "N"
        second_label = str(labels[1]) if len(labels) > 1 else "E"
        hodogram_sense = self._normalized_rotation_sense(
            result.spectra["rotation_sense"]
        )
        apparent_map_sense = self._apparent_map_rotation_sense(hodogram_sense)
        axis_ratio = float(result.spectra["spectral_minor_to_major_ratio"])
        if not np.isfinite(axis_ratio):
            axis_ratio = 0.0
        axis_ratio = float(np.clip(axis_ratio, 0.0, 1.0))
        azimuth_deg = float(result.spectra["spectral_major_azimuth_deg"]) % 180.0
        azimuth_reliable = axis_ratio < self.POLARIZATION_AZIMUTH_MAX_RATIO

        try:
            figure = create_polarization_figure(result)
            for axis in figure.axes:
                descriptor = " ".join(
                    (
                        axis.get_title(),
                        axis.get_xlabel(),
                        axis.get_ylabel(),
                    )
                ).lower()
                is_psd_axis = (
                    "psd" in descriptor
                    or "power spectral density" in descriptor
                    or "power spectrum" in descriptor
                    or "power spectra" in descriptor
                    or "/hz" in descriptor
                )
                if is_psd_axis:
                    axis.set_yscale("log", nonpositive="clip")
                    axis.relim()
                    axis.autoscale_view()

                # Hodograms and reconstructed ellipses must use equal data
                # scaling; otherwise the displayed |b/a| and azimuth are
                # visually distorted by the subplot aspect ratio.
                if "hodogram" in descriptor or "spectral ellipse" in descriptor:
                    axis.set_aspect("equal", adjustable="box")

                title = axis.get_title()
                if "spectral ellipse" in title.lower() and "right" not in title.lower():
                    axis.set_title(
                        title + f"\n(+{first_label} right, +{second_label} up)"
                    )

                # Make the unsigned ellipticity and viewpoint explicit even if
                # the imported core still uses the older terse annotations.
                for text_artist in axis.texts:
                    value = text_artist.get_text()
                    value = value.replace("b/a:", "|b/a|:")
                    value = value.replace("Sense:", "Hodogram sense:")
                    value = value.replace("Rotation:", "Hodogram sense:")
                    text_artist.set_text(value)

            figure.text(
                0.5,
                0.012,
                (
                    f"Convention: horizontal hodogram has +{first_label} to the right "
                    f"and +{second_label} upward, with +Z downward. CW/CCW refers to "
                    "that plotted hodogram. A geographic map has east right and north "
                    "up, so its apparent tracing sense is reversed. Azimuth is axial "
                    f"(0–180°); |b/a| is unsigned. Azimuth is suppressed on the map "
                    f"when |b/a| ≥ {self.POLARIZATION_AZIMUTH_MAX_RATIO:.2f}."
                ),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
            figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.94))
        except Exception as exc:
            viewer.messagebox.showerror(
                "Pi2 polarization figure", f"Could not create the figure:\n\n{exc}"
            )
            return

        window = viewer.tk.Toplevel(self.root)
        window.title(f"Pi2 polarization — {result.station_code}")
        window.geometry("1500x900")
        window.minsize(980, 650)
        self._plot_windows.append(window)

        top = viewer.ttk.Frame(window, padding=5)
        top.pack(fill=viewer.tk.X)
        azimuth_summary = (
            f"azimuth {azimuth_deg:.1f}° from +{first_label} toward +{second_label}"
            if azimuth_reliable
            else (
                f"azimuth poorly constrained (near-circular; "
                f"|b/a|={axis_ratio:.2f})"
            )
        )
        summary = (
            f"{result.station_code} | dominant period "
            f"{float(result.spectra['dominant_period_s']):.1f} s | "
            f"coherence {float(result.spectra['he_coherence']):.2f} | "
            f"{azimuth_summary} | |b/a| {axis_ratio:.2f} | "
            f"hodogram {hodogram_sense}; apparent north-up map {apparent_map_sense}"
        )
        viewer.ttk.Label(top, text=summary).pack(side=viewer.tk.LEFT)
        viewer.ttk.Button(
            top,
            text="Save figure PNG…",
            command=lambda: self.save_figure_png(
                figure, f"{result.station_code}_pi2_polarization.png"
            ),
        ).pack(side=viewer.tk.RIGHT)

        container = viewer.ttk.Frame(window)
        container.pack(fill=viewer.tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(figure, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=viewer.tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, container, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=viewer.tk.X)

        if result.warnings:
            viewer.ttk.Label(
                window,
                text="Warnings: " + " | ".join(result.warnings),
                foreground="#7a4d00",
                wraplength=1400,
                padding=4,
            ).pack(fill=viewer.tk.X)

        def close_window() -> None:
            try:
                self._plot_windows.remove(window)
            except ValueError:
                pass
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)


def main(argv: Optional[list[str]] = None) -> int:
    args = viewer.parse_arguments(argv)
    app = ExploreMagAnalyzer(initial_file=args.data_file)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
