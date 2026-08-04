#!/usr/bin/env python3
"""Reusable ground-magnetometer Pi2 polarization analysis engine.

This module supplies the numerical Pi2 workflow and publication-oriented
diagnostic figure used by ``ExploreMag.py``. It is independent of data download
libraries: callers provide UTC sample times and an ``N x 3`` magnetic-component
array in a declared coordinate system.

The analysis validates cadence, jitter, missing values, event/background
windows, filter padding, and component signs; applies a detrended zero-phase
Butterworth Pi2 bandpass; and calculates component statistics, horizontal
amplitude, covariance polarization, major-axis azimuth, axis ratio, and
ellipticity. Welch auto/cross spectra provide dominant frequency and period,
coherence, phase, spectral polarization, and qualified rotation sense.

Results are returned in ``StationPi2Result`` for programmatic use. The figure
builder produces component time series, event and background spectra,
time-coloured H-E/N-E hodograms, covariance and spectral polarization ellipses,
and textual quality/interpretation diagnostics. Ambiguous rotation and azimuth
are suppressed when coherence, linearity, circularity, coordinate-sign, or
filter-edge checks do not support a reliable interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import numpy as np
from scipy.signal import butter, csd, sosfiltfilt, welch
try:
    from scipy.integrate import trapezoid as scipy_trapezoid
except ImportError:
    scipy_trapezoid = None


# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class Config:
    # Ground-magnetometer station code.
    site: str = "OTT"

    # Load a padded interval. The event and background intervals must lie well
    # inside this range to reduce zero-phase filter edge contamination.
    load_time_range: tuple[str, str] = (
        "2024-05-12/03:55:00",
        "2024-05-12/04:50:00",
    )

    # Interval containing the individual Pi2 wave train used for hodograms,
    # amplitudes, spectra, and polarization estimates. Adjust after inspecting
    # the unfiltered data and a time-frequency representation.
    event_time_range: tuple[str, str] = (
        "2024-05-12/04:27:00",
        "2024-05-12/04:32:00",
    )

    # Quiet/pre-event interval used for an approximate event/background RMS
    # ratio and comparison power spectrum.
    background_time_range: tuple[str, str] = (
        "2024-05-12/04:05:00",
        "2024-05-12/04:15:00",
    )

    # Optional reference onset marker. This is not estimated by the filter.
    # Set to None to omit it.
    onset_time: str | None = "2024-05-12/04:28:06"

    # Nominal Pi2 period limits, in seconds.
    pi2_period_range: tuple[float, float] = (40.0, 150.0)

    # Butterworth prototype order. A bandpass transformation doubles the final
    # transfer-function order. Order 4 is retained for continuity with the
    # original script; always check sensitivity to filter settings.
    filter_order: int = 4

    # Component labels are supplied by the GUI for magnetic NEZ or GEO data.
    # Rotation reporting is controlled separately by the user's coordinate-sign
    # confirmation in ``run_pi2_analysis``.
    component_labels: tuple[str, str, str] = ("N", "E", "Z")
    require_verified_hez: bool = False

    # Apply sign corrections only when independently justified. Changing
    # exactly one horizontal sign reverses the inferred rotation sense.
    component_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Spectral polarization is averaged over the dominant Welch bin and
    # this many neighbouring bins on either side, restricted to the Pi2 band.
    dominant_band_half_width_bins: int = 1
    minimum_rotation_coherence: float = 0.50
    minimum_rotation_axis_ratio: float = 0.10
    linear_axis_ratio_threshold: float = 0.15
    circular_axis_ratio_threshold: float = 0.75

    # Sampling checks. Large gaps are rejected rather than interpolated because
    # interpolation across a gap can manufacture phase and polarization.
    max_gap_factor: float = 1.5
    cadence_jitter_tolerance: float = 0.05

    # Obvious fill-value guard. Earth's field is much smaller than this.
    max_abs_field_nt: float = 1.0e6

    # Filter-edge warning threshold in longest periods on each side of the
    # event interval. Three periods is a practical minimum, not a theorem.
    edge_padding_periods: float = 3.0

    # Figure output. Set save_figure=False to display only.
    save_figure: bool = True
    output_directory: str = "."
    output_dpi: int = 200
    show_figure: bool = True


PROGRAM_VERSION = "mag-analyzer-1.2-2026-08-02"
TIME_FORMAT = "%Y-%m-%d/%H:%M:%S"


# =============================================================================
# TIME AND DATA UTILITIES
# =============================================================================

def trapezoidal_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate y(x) without requiring ``numpy.trapezoid``.

    SciPy's implementation is preferred because some established PySPEDAS
    environments use NumPy versions that do not provide ``np.trapezoid``.
    An ``np.trapz`` fallback supports unusually old SciPy installations.
    """
    if scipy_trapezoid is not None:
        return float(scipy_trapezoid(y, x=x))
    return float(np.trapz(y, x=x))


def parse_utc(time_string: str) -> float:
    """Convert ``YYYY-mm-dd/HH:MM:SS`` UTC text to Unix seconds."""
    return datetime.strptime(time_string, TIME_FORMAT).replace(
        tzinfo=timezone.utc
    ).timestamp()


def unix_to_datetime(times: np.ndarray) -> list[datetime]:
    """Convert Unix seconds to timezone-aware UTC datetimes for Matplotlib."""
    return [datetime.fromtimestamp(float(t), tz=timezone.utc) for t in times]


def select_interval(
    times: np.ndarray,
    interval: Sequence[str],
    label: str,
) -> np.ndarray:
    """Return a Boolean mask for a closed UTC time interval."""
    start = parse_utc(interval[0])
    end = parse_utc(interval[1])
    if end <= start:
        raise ValueError(f"{label} end time must be later than its start time")

    mask = (times >= start) & (times <= end)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"{label} contains fewer than two samples: "
            f"{interval[0]} to {interval[1]}"
        )
    return mask


def validate_and_prepare_data(
    times: np.ndarray,
    values: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Validate shape, finite values, chronological order, cadence, and gaps.

    Small timestamp jitter is resampled onto a uniform grid. Any gap larger
    than ``max_gap_factor * median cadence`` is rejected.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    if times.ndim != 1:
        raise ValueError("Time array must be one-dimensional")
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(
            f"Expected an N x 3 magnetometer array; received shape {values.shape}"
        )
    if len(times) != values.shape[0]:
        raise ValueError("Time and magnetic-field arrays have unequal lengths")
    if len(times) < 20:
        raise ValueError("Too few samples for a reliable bandpass analysis")

    values = values[:, :3] * np.asarray(config.component_signs, dtype=float)

    if not np.all(np.isfinite(times)):
        raise ValueError("Non-finite timestamps are present")
    if not np.all(np.isfinite(values)):
        bad_count = np.size(values) - np.count_nonzero(np.isfinite(values))
        raise ValueError(
            f"Magnetic data contain {bad_count} NaN/inf values. Filter each "
            "continuous valid segment separately; do not bridge data gaps."
        )
    if np.any(np.abs(values) > config.max_abs_field_nt):
        raise ValueError(
            "Magnetic data contain values above the fill-value threshold "
            f"({config.max_abs_field_nt:g} nT)"
        )

    time_steps = np.diff(times)
    if np.any(time_steps <= 0.0):
        raise ValueError("Timestamps must be strictly increasing with no duplicates")

    median_dt = float(np.median(time_steps))
    if not np.isfinite(median_dt) or median_dt <= 0.0:
        raise ValueError("Could not determine a valid sampling cadence")

    largest_gap = float(np.max(time_steps))
    if largest_gap > config.max_gap_factor * median_dt:
        gap_index = int(np.argmax(time_steps))
        gap_start = datetime.fromtimestamp(times[gap_index], tz=timezone.utc)
        gap_end = datetime.fromtimestamp(times[gap_index + 1], tz=timezone.utc)
        raise ValueError(
            f"A {largest_gap:.3f} s data gap exceeds "
            f"{config.max_gap_factor:.2f} times the median cadence "
            f"({median_dt:.3f} s), between {gap_start.isoformat()} and "
            f"{gap_end.isoformat()}. Analyze continuous segments separately."
        )

    relative_jitter = np.max(np.abs(time_steps - median_dt)) / median_dt
    if relative_jitter > config.cadence_jitter_tolerance:
        uniform_times = np.arange(
            times[0],
            times[-1] + 0.5 * median_dt,
            median_dt,
            dtype=float,
        )
        uniform_values = np.column_stack(
            [np.interp(uniform_times, times, values[:, i]) for i in range(3)]
        )
        warnings.warn(
            "Sampling jitter exceeded the configured tolerance, but no large "
            "gap was present. Components were interpolated to a uniform grid.",
            RuntimeWarning,
        )
        times = uniform_times
        values = uniform_values

    return times, values, median_dt


# =============================================================================
# SIGNAL PROCESSING AND DIAGNOSTICS
# =============================================================================

def design_pi2_filter(
    sampling_frequency: float,
    period_range: Sequence[float],
    order: int,
) -> tuple[np.ndarray, float, float]:
    """Design a numerically stable Butterworth Pi2 bandpass in SOS form."""
    shortest_period, longest_period = sorted(float(p) for p in period_range)
    if shortest_period <= 0.0:
        raise ValueError("Periods must be positive")

    low_frequency = 1.0 / longest_period
    high_frequency = 1.0 / shortest_period
    nyquist = 0.5 * sampling_frequency

    if not 0.0 < low_frequency < high_frequency < nyquist:
        raise ValueError(
            "Pi2 band is incompatible with the data cadence: "
            f"band={low_frequency:.6f}-{high_frequency:.6f} Hz, "
            f"Nyquist={nyquist:.6f} Hz"
        )

    sos = butter(
        order,
        [low_frequency, high_frequency],
        btype="bandpass",
        fs=sampling_frequency,
        output="sos",
    )
    return sos, low_frequency, high_frequency


def filter_components(values: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Zero-phase filter already-baselined magnetic perturbations."""
    filtered_columns = []
    for component_index in range(3):
        component = values[:, component_index]
        try:
            filtered = sosfiltfilt(sos, component)
        except ValueError as exc:
            raise ValueError(
                "The loaded interval is too short for zero-phase filtering. "
                "Increase the padded load interval."
            ) from exc
        filtered_columns.append(filtered)
    return np.column_stack(filtered_columns)


def check_analysis_windows(
    times: np.ndarray,
    event_mask: np.ndarray,
    background_mask: np.ndarray,
    config: Config,
) -> None:
    """Warn about insufficient duration or filter-edge padding."""
    longest_period = max(config.pi2_period_range)
    event_times = times[event_mask]
    background_times = times[background_mask]

    for label, selected_times in (
        ("Event", event_times),
        ("Background", background_times),
    ):
        duration = selected_times[-1] - selected_times[0]
        if duration < 3.0 * longest_period:
            warnings.warn(
                f"{label} interval spans only {duration / longest_period:.2f} "
                "longest Pi2 periods. Spectral and RMS estimates may be unstable.",
                RuntimeWarning,
            )

    required_padding = config.edge_padding_periods * longest_period
    left_padding = event_times[0] - times[0]
    right_padding = times[-1] - event_times[-1]
    if left_padding < required_padding or right_padding < required_padding:
        warnings.warn(
            "Event interval is too close to a filter edge. Current padding is "
            f"{left_padding:.1f} s before and {right_padding:.1f} s after; "
            f"recommended minimum is {required_padding:.1f} s on each side.",
            RuntimeWarning,
        )

    if np.any(event_mask & background_mask):
        raise ValueError("Event and background intervals overlap")


def component_statistics(filtered_event: np.ndarray) -> dict[str, np.ndarray | float]:
    """Calculate event-only amplitudes from filtered components."""
    rms = np.sqrt(np.mean(filtered_event**2, axis=0))
    half_peak_to_peak = 0.5 * np.ptp(filtered_event, axis=0)
    horizontal_vector_rms = float(
        np.sqrt(np.mean(filtered_event[:, 0] ** 2 + filtered_event[:, 1] ** 2))
    )
    total_vector_rms = float(np.sqrt(np.mean(np.sum(filtered_event**2, axis=1))))
    return {
        "rms": rms,
        "half_peak_to_peak": half_peak_to_peak,
        "horizontal_vector_rms": horizontal_vector_rms,
        "total_vector_rms": total_vector_rms,
    }


def classify_polarization(
    axis_ratio: float,
    linear_threshold: float,
    circular_threshold: float,
) -> str:
    """Classify ellipse shape without assigning a propagation mode."""
    if axis_ratio < linear_threshold:
        return "near-linear"
    if axis_ratio >= circular_threshold:
        return "near-circular"
    return "elliptical"


def polarization_from_covariance(
    horizontal_event: np.ndarray,
    config: Config,
) -> dict[str, float | str | np.ndarray]:
    """Estimate the broadband Pi2-filtered horizontal covariance ellipse.

    Azimuth is axial in [0, 180 degrees), measured from +H toward +E. The
    signed hodogram area supplies an independent time-domain rotation check.
    """
    horizontal_event = np.asarray(horizontal_event, dtype=float)
    if horizontal_event.ndim != 2 or horizontal_event.shape[1] != 2:
        raise ValueError("Horizontal event data must have shape N x 2")

    centred = horizontal_event - np.mean(horizontal_event, axis=0)
    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if eigenvalues[0] <= 0.0:
        raise ValueError("Horizontal covariance has no positive variance")

    major_vector = eigenvectors[:, 0]
    major_azimuth = float(
        np.degrees(np.arctan2(major_vector[1], major_vector[0])) % 180.0
    )
    axis_ratio = float(np.sqrt(max(eigenvalues[1], 0.0) / eigenvalues[0]))

    x = centred[:, 0]
    y = centred[:, 1]
    signed_area = float(
        0.5 * np.sum(x[:-1] * y[1:] - y[:-1] * x[1:])
    )
    if np.isclose(signed_area, 0.0, atol=np.finfo(float).eps):
        area_rotation = "indeterminate"
    elif signed_area > 0.0:
        area_rotation = "counterclockwise"
    else:
        area_rotation = "clockwise"

    return {
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "major_azimuth_deg": major_azimuth,
        "minor_to_major_ratio": axis_ratio,
        "shape_classification": classify_polarization(
            axis_ratio,
            config.linear_axis_ratio_threshold,
            config.circular_axis_ratio_threshold,
        ),
        "signed_hodogram_area_nt2": signed_area,
        "area_rotation_sense": area_rotation,
    }

def spectral_diagnostics(
    event_raw: np.ndarray,
    background_raw: np.ndarray,
    sampling_frequency: float,
    low_frequency: float,
    high_frequency: float,
    config: Config,
    coordinates_verified: bool,
) -> dict[str, np.ndarray | float | str | bool]:
    """Compute PSD and frequency-local H-E polarization diagnostics.

    The H-E cross spectrum is defined by scipy.signal.csd(H, E) = conj(H) * E.
    Positive phase therefore means E leads H. In a verified north-east-down
    frame viewed along +Z (downward), positive phase gives counterclockwise
    rotation and negative phase gives clockwise rotation.
    """
    minimum_length = min(len(event_raw), len(background_raw))
    desired_segment = max(64, int(round(300.0 * sampling_frequency)))
    # A single Welch segment makes |P_HE|²/(P_HH P_EE) identically one at
    # every non-zero bin.  Use at least three 50%-overlapped segments so the
    # spectral matrix and coherence are estimates across independent windows.
    nperseg = min(desired_segment, minimum_length // 2)
    if nperseg < 16:
        raise ValueError("Event/background intervals are too short for spectra")
    noverlap = nperseg // 2
    segment_step = nperseg - noverlap
    segment_count = 1 + (minimum_length - nperseg) // segment_step
    if segment_count < 3:
        raise ValueError(
            "At least three Welch segments are required for polarization and "
            "coherence estimates. Increase the Pi2 and quiet intervals."
        )

    # Inputs are magnetic perturbations that were already baselined/detrended
    # by their data providers. Welch receives them without another trend fit.
    event_detrended = np.asarray(event_raw, dtype=float)
    background_detrended = np.asarray(background_raw, dtype=float)

    frequency, event_psd = welch(
        event_detrended,
        fs=sampling_frequency,
        axis=0,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )
    _, background_psd = welch(
        background_detrended,
        fs=sampling_frequency,
        axis=0,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    _, he_cross_spectrum = csd(
        event_detrended[:, 0],
        event_detrended[:, 1],
        fs=sampling_frequency,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    h_psd = event_psd[:, 0]
    e_psd = event_psd[:, 1]
    horizontal_event_psd = h_psd + e_psd
    horizontal_background_psd = background_psd[:, 0] + background_psd[:, 1]

    denominator = h_psd * e_psd
    coherence_spectrum = np.zeros_like(h_psd, dtype=float)
    valid_denominator = denominator > 0.0
    coherence_spectrum[valid_denominator] = (
        np.abs(he_cross_spectrum[valid_denominator]) ** 2
        / denominator[valid_denominator]
    )
    coherence_spectrum = np.clip(coherence_spectrum, 0.0, 1.0)
    phase_spectrum_deg = np.degrees(np.angle(he_cross_spectrum))

    band_mask = (frequency >= low_frequency) & (frequency <= high_frequency)
    if not np.any(band_mask):
        raise ValueError("Welch frequency grid contains no points in the Pi2 band")

    band_indices = np.flatnonzero(band_mask)
    dominant_index = band_indices[int(np.argmax(horizontal_event_psd[band_mask]))]
    dominant_frequency = float(frequency[dominant_index])
    dominant_period = float(1.0 / dominant_frequency)

    half_width = max(0, int(config.dominant_band_half_width_bins))
    local_start = max(band_indices[0], dominant_index - half_width)
    local_stop = min(band_indices[-1], dominant_index + half_width)
    local_indices = np.arange(local_start, local_stop + 1, dtype=int)
    local_mask = np.zeros_like(band_mask, dtype=bool)
    local_mask[local_indices] = True

    # Integrate the local spectral matrix over the selected bins. A coherent
    # band average is more stable than reporting one potentially noisy bin.
    local_frequency = frequency[local_mask]
    if len(local_frequency) == 1:
        p_h = float(h_psd[local_mask][0])
        p_e = float(e_psd[local_mask][0])
        cross = complex(he_cross_spectrum[local_mask][0])
    else:
        p_h = trapezoidal_integral(h_psd[local_mask], local_frequency)
        p_e = trapezoidal_integral(e_psd[local_mask], local_frequency)
        cross_real = trapezoidal_integral(
            he_cross_spectrum[local_mask].real, local_frequency
        )
        cross_imag = trapezoidal_integral(
            he_cross_spectrum[local_mask].imag, local_frequency
        )
        cross = complex(cross_real, cross_imag)

    local_coherence = (
        float(np.clip(np.abs(cross) ** 2 / (p_h * p_e), 0.0, 1.0))
        if p_h > 0.0 and p_e > 0.0
        else np.nan
    )
    he_phase_deg = float(np.degrees(np.angle(cross)))

    # Stokes-like parameters for the H-E spectral matrix. With C = H* E,
    # Positive S3 means E leads H and therefore counterclockwise rotation when
    # viewed along verified +Z (downward) in geographic/magnetic map view.
    s0 = p_h + p_e
    s1 = p_h - p_e
    s2 = 2.0 * cross.real
    s3 = 2.0 * cross.imag
    polarized_power = float(np.sqrt(s1**2 + s2**2 + s3**2))
    degree_of_polarization = (
        float(np.clip(polarized_power / s0, 0.0, 1.0)) if s0 > 0.0 else np.nan
    )
    spectral_azimuth = float(0.5 * np.degrees(np.arctan2(s2, s1)) % 180.0)

    if polarized_power > 0.0:
        ellipticity_angle = float(
            0.5 * np.degrees(
                np.arcsin(np.clip(s3 / polarized_power, -1.0, 1.0))
            )
        )
        spectral_axis_ratio = float(abs(np.tan(np.radians(ellipticity_angle))))
    else:
        ellipticity_angle = np.nan
        spectral_axis_ratio = np.nan

    spectral_shape = classify_polarization(
        spectral_axis_ratio,
        config.linear_axis_ratio_threshold,
        config.circular_axis_ratio_threshold,
    ) if np.isfinite(spectral_axis_ratio) else "indeterminate"

    rotation_reliable = bool(
        coordinates_verified
        and np.isfinite(local_coherence)
        and local_coherence >= config.minimum_rotation_coherence
        and np.isfinite(spectral_axis_ratio)
        and spectral_axis_ratio >= config.minimum_rotation_axis_ratio
    )
    if not coordinates_verified:
        rotation_sense = "not reported: HEZ signs unverified"
    elif not rotation_reliable:
        rotation_sense = "indeterminate: weak coherence or near-linear ellipse"
    elif he_phase_deg > 0.0:
        rotation_sense = "counterclockwise"
    elif he_phase_deg < 0.0:
        rotation_sense = "clockwise"
    else:
        rotation_sense = "indeterminate"

    event_band_power = trapezoidal_integral(
        horizontal_event_psd[band_mask], frequency[band_mask]
    )
    background_band_power = trapezoidal_integral(
        horizontal_background_psd[band_mask], frequency[band_mask]
    )
    band_power_ratio = (
        event_band_power / background_band_power
        if background_band_power > 0.0
        else np.inf
    )

    return {
        "frequency": frequency,
        "h_psd": h_psd,
        "e_psd": e_psd,
        "he_cross_spectrum": he_cross_spectrum,
        "coherence_spectrum": coherence_spectrum,
        "phase_spectrum_deg": phase_spectrum_deg,
        "horizontal_event_psd": horizontal_event_psd,
        "horizontal_background_psd": horizontal_background_psd,
        "dominant_frequency_hz": dominant_frequency,
        "dominant_period_s": dominant_period,
        "local_frequency_min_hz": float(frequency[local_indices[0]]),
        "local_frequency_max_hz": float(frequency[local_indices[-1]]),
        "local_frequency_mask": local_mask,
        "he_coherence": local_coherence,
        "e_relative_to_h_phase_deg": he_phase_deg,
        "spectral_major_azimuth_deg": spectral_azimuth,
        "spectral_minor_to_major_ratio": spectral_axis_ratio,
        "ellipticity_angle_deg": ellipticity_angle,
        "degree_of_polarization": degree_of_polarization,
        "spectral_shape_classification": spectral_shape,
        "rotation_sense": rotation_sense,
        "rotation_reliable": rotation_reliable,
        "stokes_s0": float(s0),
        "stokes_s1": float(s1),
        "stokes_s2": float(s2),
        "stokes_s3": float(s3),
        "local_h_power": float(p_h),
        "local_e_power": float(p_e),
        "local_cross_spectrum": cross,
        "event_band_power": event_band_power,
        "background_band_power": background_band_power,
        "band_power_ratio": float(band_power_ratio),
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "welch_segment_count": int(segment_count),
    }


# =============================================================================
# PLOTTING
# =============================================================================


def time_coloured_hodogram(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    elapsed_seconds: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    polarization: dict[str, Any] | None = None,
    arrow_count: int = 12,
) -> LineCollection:
    """Plot a time-coloured hodogram with trajectory arrows and ellipse axes."""
    if len(x) < 2:
        raise ValueError("At least two samples are required for a hodogram")

    points = np.column_stack((x, y)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    collection = LineCollection(segments, cmap="viridis", linewidth=1.4)
    collection.set_array(elapsed_seconds[:-1])
    axis.add_collection(collection)
    axis.autoscale_view()

    axis.scatter(x[0], y[0], marker="o", s=45, color="green", label="Start", zorder=5)
    axis.scatter(x[-1], y[-1], marker="s", s=45, color="red", label="End", zorder=5)

    if arrow_count > 0 and len(x) > 4:
        indices = np.unique(
            np.linspace(0, len(x) - 2, min(arrow_count, len(x) - 1), dtype=int)
        )
        axis.quiver(
            x[indices],
            y[indices],
            x[indices + 1] - x[indices],
            y[indices + 1] - y[indices],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            color="black",
            alpha=0.65,
            zorder=4,
        )

    if polarization is not None:
        eigenvalues = np.asarray(polarization["eigenvalues"], dtype=float)
        eigenvectors = np.asarray(polarization["eigenvectors"], dtype=float)
        centre = np.array([np.mean(x), np.mean(y)])
        azimuth = float(polarization["major_azimuth_deg"])
        ellipse = Ellipse(
            xy=centre,
            width=4.0 * np.sqrt(max(eigenvalues[0], 0.0)),
            height=4.0 * np.sqrt(max(eigenvalues[1], 0.0)),
            angle=azimuth,
            fill=False,
            edgecolor="black",
            linewidth=1.3,
            linestyle="--",
            label="2σ covariance ellipse",
            zorder=3,
        )
        axis.add_patch(ellipse)
        for vector_index, linestyle in ((0, "-"), (1, ":")):
            half_length = 2.0 * np.sqrt(max(eigenvalues[vector_index], 0.0))
            vector = eigenvectors[:, vector_index]
            endpoints = np.vstack(
                (centre - half_length * vector, centre + half_length * vector)
            )
            axis.plot(
                endpoints[:, 0],
                endpoints[:, 1],
                color="black",
                linestyle=linestyle,
                linewidth=1.2,
                zorder=4,
            )
        axis.text(
            0.02,
            0.98,
            f"Azimuth: {azimuth:.1f}°\n"
            f"b/a: {float(polarization['minor_to_major_ratio']):.2f}\n"
            f"Area sense: {polarization['area_rotation_sense']}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.6"),
        )

    axis.set_xlabel(f"{x_label} Pi2-band perturbation (nT)")
    axis.set_ylabel(f"{y_label} Pi2-band perturbation (nT)")
    axis.set_title(title)
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    return collection


def plot_spectral_ellipse(
    axis: plt.Axes,
    spectra: dict[str, Any],
    labels: Sequence[str],
    swap_horizontal_axes: bool = False,
) -> None:
    """Visualize the dominant-frequency H-E ellipse implied by the cross spectrum."""
    p_h = float(spectra["local_h_power"])
    p_e = float(spectra["local_e_power"])
    phase = np.radians(float(spectra["e_relative_to_h_phase_deg"]))
    h_amplitude = np.sqrt(max(p_h, 0.0))
    e_amplitude = np.sqrt(max(p_e, 0.0))
    normalizer = max(h_amplitude, e_amplitude, np.finfo(float).eps)
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    h = h_amplitude * np.cos(theta) / normalizer
    e = e_amplitude * np.cos(theta + phase) / normalizer

    x_values, y_values = (e, h) if swap_horizontal_axes else (h, e)
    x_label, y_label = (
        (labels[1], labels[0]) if swap_horizontal_axes else (labels[0], labels[1])
    )
    axis.plot(x_values, y_values, color="black", linewidth=1.5)
    indices = np.arange(0, len(theta) - 1, 45)
    axis.quiver(
        x_values[indices],
        y_values[indices],
        x_values[indices + 1] - x_values[indices],
        y_values[indices + 1] - y_values[indices],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        color="black",
        alpha=0.75,
    )
    axis.axhline(0.0, color="0.7", linewidth=0.8)
    axis.axvline(0.0, color="0.7", linewidth=0.8)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.set_xlabel(f"Normalized {x_label}")
    axis.set_ylabel(f"Normalized {y_label}")
    axis.set_title("Dominant-frequency spectral ellipse")
    axis.text(
        0.02,
        0.98,
        f"Period: {float(spectra['dominant_period_s']):.1f} s\n"
        f"Phase E−H: {float(spectra['e_relative_to_h_phase_deg']):.1f}°\n"
        f"Azimuth: {float(spectra['spectral_major_azimuth_deg']):.1f}°\n"
        f"b/a: {float(spectra['spectral_minor_to_major_ratio']):.2f}\n"
        f"Sense: {spectra['rotation_sense']}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="0.6"),
    )




@dataclass
class StationPi2Result:
    """All analysis products required by the GUI map and detail figure."""

    station_code: str
    coordinate_system: str
    coordinate_description: str
    times: np.ndarray
    raw_values: np.ndarray
    filtered_values: np.ndarray
    event_mask: np.ndarray
    background_mask: np.ndarray
    config: Config
    coordinates: dict[str, Any]
    sampling_interval_s: float
    low_frequency_hz: float
    high_frequency_hz: float
    event_stats: dict[str, Any]
    background_stats: dict[str, Any]
    polarization: dict[str, Any]
    spectra: dict[str, Any]
    warnings: list[str]


def _datetime_to_config_text(value: datetime) -> str:
    """Convert a naive/aware UTC datetime to the Config text format."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.strftime(TIME_FORMAT)


def _datetime_to_unix(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.timestamp()


def run_pi2_analysis(
    *,
    station_code: str,
    datetimes: Sequence[datetime],
    values: np.ndarray,
    event_interval: tuple[datetime, datetime],
    background_interval: tuple[datetime, datetime],
    onset_time: datetime | None,
    coordinate_system: str,
    component_labels: tuple[str, str, str],
    coordinate_description: str,
    coordinate_signs_verified: bool = True,
    pi2_period_range: tuple[float, float] = (40.0, 150.0),
    filter_order: int = 4,
) -> StationPi2Result:
    """Analyze one loaded ground-magnetometer station using the Pi2 workflow.

    The loaded/filter interval is automatically padded around both the event
    and quiet intervals. Internal NaNs are not bridged: removing invalid rows
    produces a timestamp gap that the validation routine rejects when it is
    larger than the configured cadence tolerance.
    """
    event_start, event_end = event_interval
    quiet_start, quiet_end = background_interval
    if event_end <= event_start:
        raise ValueError("Pi2 analysis end time must be later than its start time.")
    if quiet_end <= quiet_start:
        raise ValueError("Quiet/pre-event end time must be later than its start time.")
    if max(event_start, quiet_start) <= min(event_end, quiet_end):
        raise ValueError("Pi2 and quiet/pre-event intervals must not overlap.")

    period_min, period_max = sorted(float(value) for value in pi2_period_range)
    padding_seconds = 3.0 * period_max
    load_start_unix = min(_datetime_to_unix(event_start), _datetime_to_unix(quiet_start)) - padding_seconds
    load_end_unix = max(_datetime_to_unix(event_end), _datetime_to_unix(quiet_end)) + padding_seconds

    all_times = np.asarray([_datetime_to_unix(value) for value in datetimes], dtype=float)
    all_values = np.asarray(values, dtype=float)
    if all_values.ndim != 2 or all_values.shape[1] < 3:
        raise ValueError(f"Station values must have shape N x 3; received {all_values.shape}.")
    if len(all_times) != len(all_values):
        raise ValueError("Station time and data arrays have different lengths.")

    load_mask = (all_times >= load_start_unix) & (all_times <= load_end_unix)
    if np.count_nonzero(load_mask) < 20:
        raise ValueError(
            "The loaded file does not contain enough samples in the padded analysis interval."
        )
    times = all_times[load_mask]
    raw_values = all_values[load_mask, :3]

    finite_rows = np.isfinite(times) & np.all(np.isfinite(raw_values), axis=1)
    times = times[finite_rows]
    raw_values = raw_values[finite_rows]
    if len(times) < 20:
        raise ValueError("Too few finite three-component samples are available for this station.")

    config = Config(
        site=station_code.upper(),
        load_time_range=(
            datetime.fromtimestamp(float(times[0]), tz=timezone.utc).strftime(TIME_FORMAT),
            datetime.fromtimestamp(float(times[-1]), tz=timezone.utc).strftime(TIME_FORMAT),
        ),
        event_time_range=(
            _datetime_to_config_text(event_start),
            _datetime_to_config_text(event_end),
        ),
        background_time_range=(
            _datetime_to_config_text(quiet_start),
            _datetime_to_config_text(quiet_end),
        ),
        onset_time=_datetime_to_config_text(onset_time) if onset_time is not None else None,
        pi2_period_range=(period_min, period_max),
        filter_order=int(filter_order),
        component_labels=component_labels,
        require_verified_hez=False,
        component_signs=(1.0, 1.0, 1.0),
        save_figure=False,
        show_figure=False,
    )

    collected_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        times, raw_values, dt = validate_and_prepare_data(times, raw_values, config)
        sampling_frequency = 1.0 / dt
        event_mask = select_interval(times, config.event_time_range, "Pi2 interval")
        background_mask = select_interval(
            times, config.background_time_range, "Quiet/pre-event interval"
        )
        check_analysis_windows(times, event_mask, background_mask, config)
        sos, low_frequency, high_frequency = design_pi2_filter(
            sampling_frequency, config.pi2_period_range, config.filter_order
        )
        filtered_values = filter_components(raw_values, sos)
        event_filtered = filtered_values[event_mask]
        background_filtered = filtered_values[background_mask]
        event_stats = component_statistics(event_filtered)
        background_stats = component_statistics(background_filtered)
        polarization = polarization_from_covariance(event_filtered[:, :2], config)
        spectra = spectral_diagnostics(
            raw_values[event_mask],
            raw_values[background_mask],
            sampling_frequency,
            low_frequency,
            high_frequency,
            config,
            bool(coordinate_signs_verified),
        )
        collected_warnings = [str(record.message) for record in warning_records]

    coordinates = {
        "verified": bool(coordinate_signs_verified),
        "coordinate_system": coordinate_system,
        "axis_definitions": coordinate_description,
        "component_signs": (1.0, 1.0, 1.0),
        "view_direction": "+Z (downward toward Earth)",
    }
    return StationPi2Result(
        station_code=station_code.upper(),
        coordinate_system=coordinate_system,
        coordinate_description=coordinate_description,
        times=times,
        raw_values=raw_values,
        filtered_values=filtered_values,
        event_mask=event_mask,
        background_mask=background_mask,
        config=config,
        coordinates=coordinates,
        sampling_interval_s=dt,
        low_frequency_hz=low_frequency,
        high_frequency_hz=high_frequency,
        event_stats=event_stats,
        background_stats=background_stats,
        polarization=polarization,
        spectra=spectra,
        warnings=collected_warnings,
    )


def create_polarization_figure(result: StationPi2Result) -> Figure:
    """Create the requested two-row, four-column GUI polarization figure."""
    times = result.times
    raw_values = result.raw_values
    filtered_values = result.filtered_values
    event_mask = result.event_mask
    config = result.config
    polarization = result.polarization
    spectra = result.spectra
    coordinates = result.coordinates
    labels = config.component_labels
    event_times = times[event_mask]
    event_datetimes = unix_to_datetime(event_times)
    event_filtered = filtered_values[event_mask]
    elapsed = event_times - event_times[0]

    date_times = unix_to_datetime(times)
    raw_for_plot = raw_values - np.median(raw_values, axis=0)

    figure = Figure(figsize=(20, 10.5), dpi=100)
    axes = np.asarray(
        [figure.add_subplot(2, 4, index + 1) for index in range(8)],
        dtype=object,
    ).reshape(2, 4)

    def decorate_interval_axis(axis) -> None:
        event_start = datetime.fromtimestamp(float(event_times[0]), tz=timezone.utc)
        event_end = datetime.fromtimestamp(float(event_times[-1]), tz=timezone.utc)
        axis.axvspan(event_start, event_end, color="tab:orange", alpha=0.12)
        if config.onset_time is not None:
            onset_unix = parse_utc(config.onset_time)
            if times[0] <= onset_unix <= times[-1]:
                axis.axvline(
                    datetime.fromtimestamp(onset_unix, tz=timezone.utc),
                    color="red", linestyle="--", linewidth=1.1, label="Onset",
                )
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
        axis.set_xlabel("UTC")
        axis.grid(True, alpha=0.3)

    # Column 1: padded raw and filtered three-component records.
    for index, label in enumerate(labels):
        axes[0, 0].plot(date_times, raw_for_plot[:, index], linewidth=0.9, label=f"Δ{label}")
        axes[1, 0].plot(date_times, filtered_values[:, index], linewidth=1.0, label=label)
    axes[0, 0].set_title(f"Median-removed GMAG data — {result.station_code}")
    axes[0, 0].set_ylabel("Magnetic perturbation (nT)")
    axes[1, 0].set_title(
        f"Zero-phase Pi2-band output ({min(config.pi2_period_range):g}–"
        f"{max(config.pi2_period_range):g} s)"
    )
    axes[1, 0].set_ylabel("Filtered perturbation (nT)")
    for axis in (axes[0, 0], axes[1, 0]):
        decorate_interval_axis(axis)
        axis.legend(fontsize=7, loc="best")

    # Column 2: event-only horizontal waveforms and power check.
    axes[0, 1].plot(event_datetimes, event_filtered[:, 0], label=labels[0], linewidth=1.2)
    axes[0, 1].plot(event_datetimes, event_filtered[:, 1], label=labels[1], linewidth=1.2)
    if config.onset_time is not None:
        onset_unix = parse_utc(config.onset_time)
        if event_times[0] <= onset_unix <= event_times[-1]:
            axes[0, 1].axvline(
                datetime.fromtimestamp(onset_unix, tz=timezone.utc),
                color="red",
                linestyle="--",
                linewidth=1.25,
                label="Onset",
                zorder=5,
            )
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    axes[0, 1].set_xlabel("UTC")
    axes[0, 1].set_ylabel("Pi2-band perturbation (nT)")
    axes[0, 1].set_title("Horizontal filtered waveforms")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    # Column 3: east on x, north on y, as requested.
    collection = time_coloured_hodogram(
        axes[0, 2],
        event_filtered[:, 1],
        event_filtered[:, 0],
        elapsed,
        labels[1],
        labels[0],
        f"Filtered {labels[0]}-{labels[1]} hodogram (E/X, N/Y)",
        arrow_count=14,
    )
    figure.colorbar(collection, ax=axes[0, 2], label="Elapsed event time (s)")
    plot_spectral_ellipse(axes[1, 2], spectra, labels, swap_horizontal_axes=True)

    frequency_mhz = np.asarray(spectra["frequency"]) * 1000.0
    positive = frequency_mhz > 0.0
    axes[1, 1].plot(
        frequency_mhz[positive],
        np.asarray(spectra["horizontal_event_psd"])[positive],
        label="Event horizontal PSD",
    )
    axes[1, 1].plot(
        frequency_mhz[positive],
        np.asarray(spectra["horizontal_background_psd"])[positive],
        linestyle="--",
        label="Quiet horizontal PSD",
    )
    low_mhz = 1000.0 / max(config.pi2_period_range)
    high_mhz = 1000.0 / min(config.pi2_period_range)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].axvspan(low_mhz, high_mhz, color="0.85", alpha=0.6, label="Pi2 band")
    axes[1, 1].axvspan(
        float(spectra["local_frequency_min_hz"]) * 1000.0,
        float(spectra["local_frequency_max_hz"]) * 1000.0,
        color="0.65",
        alpha=0.45,
        label="Polarization average",
    )
    axes[1, 1].axvline(
        float(spectra["dominant_frequency_hz"]) * 1000.0,
        color="black",
        linestyle=":",
        label=f"Dominant {float(spectra['dominant_period_s']):.1f} s",
    )
    axes[1, 1].set_xlim(max(1.0, low_mhz / 3.0), min(200.0, high_mhz * 4.0))
    axes[1, 1].set_xlabel("Frequency (mHz)")
    axes[1, 1].set_ylabel(r"PSD (nT$^2$/Hz)")
    axes[1, 1].set_title("Horizontal power check")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    # Column 4: coherence and phase.
    axes[0, 3].plot(frequency_mhz, np.asarray(spectra["coherence_spectrum"]), linewidth=1.2)
    axes[0, 3].axvspan(low_mhz, high_mhz, color="0.85", alpha=0.6)
    axes[0, 3].axvspan(
        float(spectra["local_frequency_min_hz"]) * 1000.0,
        float(spectra["local_frequency_max_hz"]) * 1000.0,
        color="0.65",
        alpha=0.45,
    )
    axes[0, 3].axhline(
        config.minimum_rotation_coherence,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Rotation threshold",
    )
    axes[0, 3].scatter(
        [float(spectra["dominant_frequency_hz"]) * 1000.0],
        [float(spectra["he_coherence"])],
        color="black",
        zorder=5,
        label=f"Band average: {float(spectra['he_coherence']):.2f}",
    )
    axes[0, 3].set_xlim(low_mhz * 0.8, high_mhz * 1.2)
    axes[0, 3].set_ylim(0.0, 1.05)
    axes[0, 3].set_xlabel("Frequency (mHz)")
    axes[0, 3].set_ylabel("Magnitude-squared coherence")
    axes[0, 3].set_title(f"{labels[0]}-{labels[1]} coherence in the Pi2 band")
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].legend(fontsize=8)

    phase = np.asarray(spectra["phase_spectrum_deg"])
    axes[1, 3].plot(frequency_mhz, phase, linewidth=1.2)
    axes[1, 3].axvspan(low_mhz, high_mhz, color="0.85", alpha=0.6)
    axes[1, 3].axvspan(
        float(spectra["local_frequency_min_hz"]) * 1000.0,
        float(spectra["local_frequency_max_hz"]) * 1000.0,
        color="0.65",
        alpha=0.45,
    )
    axes[1, 3].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 3].scatter(
        [float(spectra["dominant_frequency_hz"]) * 1000.0],
        [float(spectra["e_relative_to_h_phase_deg"])],
        color="black",
        zorder=5,
    )
    axes[1, 3].set_xlim(low_mhz * 0.8, high_mhz * 1.2)
    axes[1, 3].set_ylim(-180.0, 180.0)
    axes[1, 3].set_yticks([-180, -90, 0, 90, 180])
    axes[1, 3].set_xlabel("Frequency (mHz)")
    axes[1, 3].set_ylabel(f"{labels[1]} phase relative to {labels[0]} (degrees)")
    axes[1, 3].set_title(f"{labels[0]}-{labels[1]} cross-spectral phase")
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].text(
        0.02,
        0.98,
        f"Signs verified: {coordinates['verified']}\n"
        f"Coherence: {float(spectra['he_coherence']):.2f}\n"
        f"Degree of polarization: {float(spectra['degree_of_polarization']):.2f}\n"
        f"Shape: {spectra['spectral_shape_classification']}\n"
        f"Rotation: {spectra['rotation_sense']}",
        transform=axes[1, 3].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="0.6"),
    )

    figure.suptitle(
        f"Horizontal Pi2 polarization — station {result.station_code}\n"
        f"{result.coordinate_description}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure
