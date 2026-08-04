# ExploreMag

ExploreMag is a desktop application for downloading, converting, inspecting, mapping, and analyzing ground-magnetometer data. Built around the SuperMAG API and the gmag Python package, it brings data from multiple sources into a single viewer. ExploreMag provides multi-station time-series plots, magnetic-field maps, preliminary results from wavelet and Pc-band analyses, total ULF wave-power maps, and Pi2 polarization analysis.


> **Scope and responsibility.** ExploreMag is an exploratory research tool,
> not an official SuperMAG, GMAG, THEMIS, CARISMA, IMAGE, or MagStar product.
> Confirm data coverage, coordinate conventions, processing metadata, and the
> applicable data acknowledgements before publication. A visually compelling
> filtered trace or scalogram ridge is evidence to investigate, not by itself
> proof of a physical mode or event onset. At all times, validate the results.

The main program is `ExploreMag.py`. Keep it in the same directory as:

- `mag_viewer.py` — download, NetCDF/CSV loading, map, and time-series viewer.
- `mag_analyzer.py` — reusable Pi2 polarization calculations and figures.
- `gmag_download_and_clean.py` — GMAG download, cadence conversion, cleaning,
  baselining, filtering, and NetCDF output.
- `20250712-supermag-stations.txt` — station metadata used by the SuperMAG
  download interface.

## Contents

- [Installation](#installation)
- [Running ExploreMag](#running-exploremag)
- [First-time orientation](#first-time-orientation)
- [A recommended event-exploration workflow](#a-recommended-event-exploration-workflow)
- [Download and conversion options](#download-and-conversion-options)
- [Common-cadence gridding and interpolation](#common-cadence-gridding-and-interpolation)
- [Cleaning and correction](#cleaning-and-correction)
- [GMAG baselining options](#gmag-baselining-options)
- [Opening data and selecting stations](#opening-data-and-selecting-stations)
- [Plot options](#plot-options)
- [Pulsation analysis prerequisites](#pulsation-analysis-prerequisites)
- [Wavelet scalograms](#wavelet-scalograms)
- [Pi-band analysis](#pi-band-analysis)
- [Pc-band analysis](#pc-band-analysis)
- [Total ULF wave power](#total-ulf-wave-power)
- [Pi2 ellipse and polarization analysis](#pi2-ellipse-and-polarization-analysis)
- [Saving and exporting](#saving-and-exporting)
- [Troubleshooting](#troubleshooting)
- [Rules of the Road](#exploremag-rules-of-the-road)
- [Literature and conventions](#literature-and-conventions)

## Installation

On Ubuntu, install Tkinter and create a virtual environment:

```bash
sudo apt update
sudo apt install git python3-tk python3-venv

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

NumPy, pandas, Matplotlib, SciPy, and netCDF4 are required. Cartopy and
mplcursors are recommended for geographic maps and interactive labels. GMAG
downloads additionally require the GMAG package and its data-loading
dependencies. The requirements file installs the official
[GMAG source repository](https://github.com/kylermurphy/gmag) automatically. Internet access and the `git` command are therefore required while installing ExploreMag.

Verify the installation from the activated virtual environment:

```bash
python -c "from gmag import utils; import gmag.arrays.themis; print('GMAG installation OK')"
```

If an earlier installation stopped at the PyPI error, update the checkout and
run the requirements command again. Pip will install GMAG from GitHub along
with the remaining dependencies; a separate manual clone is not needed.

## Running ExploreMag

```bash
python ExploreMag.py
```

An existing supported file can be opened at startup:

```bash
python ExploreMag.py path/to/magnetic_data.nc
```

Supported input includes suite-format NetCDF, native-cadence GMAG NetCDF4
files with `/stations/<code>` groups, SuperMAG custom block/vector NetCDF,
and the legacy magnetic CSV format.

At startup, acknowledge the SuperMAG rules-of-the-road dialog when it appears.
The main window then opens with download controls across the top, station and
plot controls on the left, the station map and active-station plot on the
right, and pulsation-analysis buttons below the main plot.

## First-time orientation

The interface is easiest to understand as four connected areas.

| Area | Purpose | First-time action |
| --- | --- | --- |
| **Ground-level magnetic field data download query (UTC)** | Download GMAG or SuperMAG data, or convert a custom SuperMAG file | For a first session, open an existing file before attempting a network download |
| **Station map and station lists** | Inspect coverage and select one or several stations | Click a map marker or use Ctrl/Shift in either ordered station list |
| **Plot mode** | Choose the shared UTC interval, coordinates, plot type, map quantity, cadence, and display options | Begin with **Single selected station**, then use component and derivative stacks |
| **Pulsation analysis** | Open wavelet, Pi, Pc, total-ULF, and Pi2-polarization products | Use only after checking raw traces, cadence, gaps, and the event interval |

### Selection language used by the program

- The **active station** is the most recently selected station. It drives the
  embedded single-station plot.
- **Selected stations** are the full multi-selection used by stack plots and
  pulsation tools. Selected markers appear green on the station map.
- Ctrl-click toggles individual list entries. Shift-click selects a range.
  Map clicks toggle stations individually.
- **Select all stations** is useful for a regional overview, but start with a
  small representative group for computationally expensive wavelet and Pi2
  analysis.
- **Clear station selection** resets the analysis selection.
- **Select station according to lat/lon** selects the stations inside the
  latitude and longitude entries under **Region selected for plots**. A west
  longitude greater than the east longitude selects across the ±180° meridian.

### Time and coordinate conventions

- All entered times are UTC. Analysis dialogs use
  `YYYY-MM-DD HH:mm:SS`; download entries may display date and time separately.
- **NEZ** means local magnetic north, local magnetic east, and vertical down.
- **GEO** means geographic north, geographic east, and vertical down, when the
  loaded file contains GEO components.
- `dBh = sqrt(dBn² + dBe²)` is the nonnegative horizontal magnitude derived
  from NEZ. Changing the component selector to GEO does not redefine `dBh`.
- Derivatives are calculated separately on continuous finite segments rather
  than across data gaps. Stacked derivatives use nT/sec for 1-second and 2-Hz
  products, and nT/min for 1-minute, 10-second, and native-cadence products.
  Custom SuperMAG derivative units are chosen from the detected cadence.

## A recommended event-exploration workflow

This sequence is designed for someone investigating a substorm or another
magnetospheric event for the first time.

1. **Define a scientific interval with context.** Include quiet context before
   the suspected event and enough post-event data to distinguish an isolated
   impulse from a sustained disturbance. For Pi2 work, load padding beyond the
   final event and background windows.
2. **Choose cadence before download.** Use 2 Hz or 1-second data when Pc2,
   short-period Pi1, detailed phase, or polarization is important. Ten-second
   data can resolve nominal Pi2 periods but cannot resolve the 5–10 s Pc2 band.
   One-minute SuperMAG data cannot resolve a 40–150 s Pi2 band.
3. **Download or open the file.** Confirm the loaded filename, station count,
   time range, and map coverage. Read the GMAG station report when available.
4. **Inspect raw components first.** Select one station near the event region,
   choose **Single selected station**, and scan N/E/Z or X/Y/Z plus dBh. Look
   for fill values, flat lines, steps, spikes, clipping, and remaining gaps.
5. **Compare stations.** Select stations spanning latitude and longitude. Use
   component, dB/dt, and dBh stacks to compare onset timing, phase, amplitude,
   and spatial coherence. Change station order to expose latitudinal or
   longitudinal progression.
6. **Examine maps and indices.** Plot dBz, dBh, derivatives, horizontal vectors,
   Pi3/Ps6 amplitude, or total ULF power. If relevant, open or download
   SuperMAG indices and compare their changes with the ground traces.
7. **Survey time-frequency structure.** Use **Wavelet scalogram** over a broad
   interval. Identify power localized in time and period, while discounting
   features inside the cone of influence or outside the 95% red-noise contour.
8. **Use category plots as a diagnostic.** Run **Pi pulsations** or
   **Pc pulsations** on a shorter interval chosen from the raw and wavelet
   views. Compare the filtered waveform with the unfiltered components; do not
   classify an event solely because a filter returns an oscillation.
9. **Map integrated short-period power.** Use **Total ULF wave power** to compare
   5–150 s horizontal power between stations. Treat interpolation and unequal
   station noise floors as possible spatial biases.
10. **Run the strict Pi2 workflow last.** Define non-overlapping Pi2 and quiet
    windows, optionally enter an independent onset marker, choose coordinates,
    confirm signs only when justified, and analyze selected stations. Inspect
    each detailed station figure before interpreting the ellipse map.
11. **Export an audit trail.** Save figures and CSV data, and retain the source
    NetCDF, station report, selected time windows, cadence, coordinate system,
    baseline method, and filter band with the analysis notes.

## Download and conversion options

The cadence selector in **Ground-level magnetic field data download query
(UTC)** determines the download service and output cadence. **Custom SuperMAG**
is a file-opening button beside the native-cadence option, not a cadence radio
button.

| ExploreMag option | Data source and download method | Output |
| --- | --- | --- |
| **SuperMAG 1-minute** | Uses the **SuperMAG web-services API** directly: `inventory.php` identifies stations and `data-api.php` downloads NEZ/GEO vectors. A valid SuperMAG logon is required. It does not use GMAG. | SuperMAG one-minute station data on the returned timestamps. |
| **gmag 1-second** | Uses **GMAG**, not the SuperMAG API. GMAG station loaders obtain THEMIS-hosted ground-magnetometer data, CARISMA data, and MagStar data. IMAGE stations prefer high-resolution THEMIS-hosted files and fall back to `gmag.arrays.image` when necessary. | All accepted stations on one common 1-second grid. |
| **gmag 2 Hz** | Uses the same **GMAG** sources and fallbacks described above; it does not call the SuperMAG data API. | All accepted stations on one common 0.5-second grid (2 Hz). |
| **gmag 10-second** | Uses the same **GMAG** sources and fallbacks; it does not call the SuperMAG data API. IMAGE native 10-second measurements are accepted, and every accepted station is placed on the common IMAGE-compatible 10-second grid. | All accepted stations on one common 10-second grid. |
| **native cadence** | Uses **GMAG**, not the SuperMAG API, and preserves each station's original cadence. | A NetCDF4 file with station-specific `/stations/<code>/time` and magnetic variables. ExploreMag can open this grouped layout. |
| **Custom SuperMAG** button | Opens a local SuperMAG custom block/vector NetCDF file, detects its cadence, aligns nominal sample times, and converts it to the suite's rectangular station/time/component format. Interior gaps shorter than five seconds are filled; longer gaps remain missing. | The converted file is saved beside the source as `<source-name>_converted_for_exploremag.nc`, then opened automatically. |

SuperMAG indices are a separate feature. **Download and save SuperMAG
indices** uses the SuperMAG indices API and is independent of the ground-field
source selected above.

## Common-cadence gridding and interpolation

GMAG data are cleaned at their measured native cadence before being placed on
a common output grid. The resulting NetCDF records each station's native and
output cadence, processing description, coverage, baseline method, and quality
flags.

### GMAG 2 Hz

- The output timestamps are spaced by 0.5 seconds.
- Native 0.5-second measurements retain their cadence after timestamp
  alignment.
- Native 1-second measurements are linearly interpolated onto the intervening
  half-second timestamps. Thus, some 2 Hz values are interpolated rather than
  independent measurements.
- Missing interior values are interpolated only when the gap is no longer than
  the configured short-gap limit. Longer gaps remain missing.

### GMAG 1-second

- The output timestamps are spaced by one second.
- Native 0.5-second data are averaged into populated one-second bins. This
  retains available high-rate observations and avoids phase-offset gaps.
- Native data slower than one second, or missing requested timestamps, require
  linear interpolation and are filled only within the configured short-gap
  limit. Longer gaps remain missing.

### GMAG 10-second

- The output timestamps are spaced by ten seconds and are shared by all
  accepted stations, matching the IMAGE cadence.
- Native higher-rate data are averaged into populated 10-second bins. This
  prevents continuous phase-offset input from appearing to contain regular
  gaps merely because samples do not fall exactly on grid timestamps.
- Missing requested timestamps are filled by linear interpolation only when
  allowed by the configured short-gap duration. A gap exceeding that duration
  remains NaN and is not silently bridged.

NetCDF `data_quality` values distinguish the result: `0` is observed/aligned,
`1` is interpolated or repaired, and `2` is missing.

## Cleaning and correction

The GMAG 2 Hz, 1-second, 10-second, and native-cadence paths apply vector-aware
quality control. Custom SuperMAG conversion and direct SuperMAG downloads also
apply the suite's cleaning before saving.

The processing sequence is:

1. Decode timestamps and N/E/Z components and mask fill, sentinel, invalid,
   and physically implausible values.
2. Detect isolated instrumental spikes with rolling-median/MAD tests. A spike
   found in any component is treated as a three-component vector event.
3. Repair confirmed short spike runs and fill eligible short interior gaps by
   time interpolation.
4. Run despiking again after gap filling so interpolation does not leave or
   introduce an uncorrected transient.
5. Preserve long gaps and sustained magnetic excursions rather than treating
   them as short instrumental spikes.

Because interpolation is intentionally limited, “cleaned” does not mean every
missing value is fabricated. Unrepairable or long gaps remain explicitly
missing and are visible through the quality flags and coverage metadata.

## GMAG baselining options

The GMAG **2 Hz**, **1-second**, and **10-second** download options produce
magnetic perturbations using one of two baselining choices. ExploreMag asks for
the choice when a GMAG download starts.

### 1. First-data baseline

Choose **No** when asked whether to use a user-defined quiet interval.

- For each station, the first complete N/E/Z vector in the requested active
  interval is used as its baseline.
- That station-specific vector is subtracted from the active measurements
  before the remaining cleaning, gap filling, and second despiking pass.
- If the exact first requested row is incomplete, ExploreMag may advance to
  the first complete vector within the allowed initial search period. It does
  not invent missing leading measurements.
- The actual baseline timestamp and N/E/Z values are written to the NetCDF and
  station report.

### 2. Quiet-time rolling-mean baseline

Choose **Yes** and enter the quiet-interval start and end in UTC.

- Quiet data remain at their native cadence while being cleaned and having
  only eligible short gaps filled.
- A centered rolling mean (30 minutes by default) suppresses short-period
  fluctuations while retaining the slowly varying background trend.
- For each N/E/Z component, the median of the valid rolling-mean background
  values inside the selected quiet interval becomes the scalar quiet baseline.
- This quiet background vector is subtracted from the active interval. The
  noisy quiet-time fluctuations are therefore not copied into the active
  perturbation data.
- The baseline vector, method, processing description, and quiet-data coverage
  are stored in the NetCDF and station report. The full quiet time series is
  not stored in the output file.

The native-cadence command-line downloader supports the same perturbation
baseline machinery. The GUI specifically presents these two choices for its
GMAG download modes.

## Opening data and selecting stations

Click **Open data file…** or pass a filename on the command line. After a file
loads, ExploreMag populates latitude- and longitude-ordered station lists,
draws the station map, initializes the full UTC range, selects an initial
station, and displays its components.

Before analysis, verify:

- the filename and requested interval;
- that station coordinates and regional coverage are plausible;
- that the selected plot coordinates exist in the file;
- sampling cadence and whether the file was interpolated to a common grid;
- the baseline method and `subtracted_baseline` metadata;
- station coverage and `data_quality` (`0` observed/aligned, `1` repaired or
  interpolated, `2` missing);
- whether event-scale features are visible at more than one station.

The two station lists contain the same network in different orders. Latitude
ordering is useful for poleward/equatorward propagation; longitude ordering is
useful for east-west progression and local-time comparisons. Optional
latitude/longitude filters restrict the stations included in automatic stacks
without changing the loaded file.

The same entries, labelled **Region selected for plots**, can be applied to the
pulsation-analysis selection with **Select station according to lat/lon**.

## Plot options

Set **Plot start**, **Plot end**, coordinate system, and any plot-specific
controls, choose a radio button under **Plot mode**, and click
**Create selected plot(s)**. The plot interval is also used by Wavelet, Pi,
Pc, and total-ULF tools unless their own dialog provides separate fields.

### Plot-mode summary

| Plot option | Result | Best first use | Important considerations |
| --- | --- | --- | --- |
| **Single selected station** | Embedded N/E/Z or X/Y/Z traces with dBh | Inspect quality and identify candidate event times | dBh always uses NEZ horizontal components |
| **Stacked components** | Three windows, one per component | Compare timing and morphology across stations | A positive offset median-centers and vertically shifts each trace |
| **Stacked component dB/dt** | Three derivative windows in cadence-aware nT/sec or nT/min units | Highlight rapid changes and possible onsets | Derivatives amplify noise and do not bridge missing runs |
| **Stacked dBh and dBh/dt** | Horizontal magnitude and derivative stacks | Compare disturbance strength independently of horizontal direction | dBh is nonnegative; it loses vector direction |
| **Plot maps** | Scalar and/or vector map at one time, or a timestamp sequence | Examine spatial structure and propagation | Sparse-station interpolation is a visualization, not a measurement between sites |
| **Plot SuperMAG indices** | Global, sunlit, dark, regional MLT, and ring-current index views | Compare local observations with regional/global activity | Uses the separately opened/downloaded indices file |

### Single selected station

Use this view before applying any pulsation filter. Select a station on the map
or in a list, choose NEZ or GEO, and adjust the shared time range. The three
components use the left axis; dBh uses a separate right axis. The most recently
selected station becomes active without clearing the other selected stations.

Questions to ask:

- Is the candidate oscillation present in one or both horizontal components?
- Is a large dBh feature caused by a smooth vector rotation, a step, or a
  single-component spike?
- Do the baseline and long-period trend make physical sense?
- Are gaps, interpolated runs, or edge effects close to the candidate event?

### Automatic component stacks

Select several stations and choose either raw components, component dB/dt, or
dBh/dBh-dt. Set station order, offsets, and optional geographic filters. With
a positive offset, each station is median-centered before vertical shifting;
this is useful for timing but hides absolute baseline differences. Set offset
to zero when absolute levels matter.

Use component stacks to look for phase reversals, coherent wave trains, and
latitudinal/longitudinal progression. Use derivative stacks to compare rapid
changes, but confirm every derivative feature in the raw traces because
differentiation enhances high-frequency noise.

### Manual station stack

The manual stack is independent of the plot-mode radio buttons. Select a
station, click **Add selected station to stack →**, and repeat in the desired
order. The manual interval can be changed after panels are added. It supports
component or dBh display, common or independent y scaling, point marking,
annotation lists, removal of selections, text export, and PNG output.

Use it to build a publication-oriented subset after the automatic stacks have
identified the most informative stations.

### Scalar and vector maps

Choose **Plot maps**, then select scalar and/or vector quantities. Available
scalar maps include dBn, dBe, dBz, dBh, derivatives, Pi3/Ps6 amplitude, and
total ULF power. Vector choices include horizontal H, rotated H, dH/dt, and
rotated dH/dt. A 90° clockwise rotated horizontal perturbation is often used
as a qualitative proxy for equivalent ionospheric-current direction, but the
relationship depends on ionospheric and Earth conductivity and must not be
treated as a direct current measurement.

- When start equals end, ExploreMag opens one interactive map.
- When start precedes end, it creates a sequence at **Plot cadence (s)**.
- **Unit vector scale** fixes a reference magnitude across frames; **Arrow
  size** changes display length without changing data.
- Choose a colormap and automatic or manual color limits. Fixed limits are
  preferred when comparing consecutive frames.
- Click **Halt process** to stop a long map sequence.

Interpolation between station locations is sensitive to array geometry,
coastlines, conductivity, outliers, and missing stations. Interpret features
supported by multiple nearby stations more strongly than extrapolated edges.

### SuperMAG indices

Use the top-row controls to choose, download, save, or open a SuperMAG indices
NetCDF file. Select **Plot SuperMAG indices** and create the plot. ExploreMag
can display SME/SML/SMU families, sunlit and dark-sector variants, regional
SMU_R/SML_R by MLT, and SMR. Index changes provide context; they do not replace
inspection of the contributing stations or establish a unique onset time.

## Pulsation analysis prerequisites

### Literature period convention used by ExploreMag

ExploreMag follows the IAGA classification introduced by
[Jacobs et al. (1964)](https://doi.org/10.1029/JZ069i001p00180). `Pc` denotes
more regular, continuous pulsations and `Pi` denotes irregular/impulsive
pulsations. The labels are morphological conventions, not guarantees of a
specific generation mechanism.

| Class | Period used by ExploreMag | Frequency equivalent | Minimum practical cadence implication |
| --- | ---: | ---: | --- |
| Pc2 | 5–10 s | 100–200 mHz | 2 Hz recommended; 1 s is close to the short-period edge |
| Pc3 | 10–45 s | 22.2–100 mHz | 1 s or 2 Hz preferred |
| Pc4 | 45–150 s | 6.67–22.2 mHz | 10 s can resolve the nominal band, but faster data improve waveform/phase work |
| Pc5 | 150–600 s | 1.67–6.67 mHz | Use a long interval containing several cycles |
| Pi1 | 1–40 s | 25 mHz–1 Hz | The shortest edge is cadence-limited; 2 Hz cannot fully resolve 1 s |
| Pi2 | 40–150 s | 6.67–25 mHz | Cadence must be faster than 20 s; 1 min is invalid |
| Pi3/Ps6 | >150 s | <6.67 mHz | Long trends and filter-edge sensitivity become important |

Nyquist is only a mathematical lower bound. Reliable amplitude, phase,
polarization, and waveform shape generally require substantially more than two
samples per shortest period. Select an interval long enough for several cycles
of the longest period, with additional filter padding.

### Data preparation shared by the exploratory pulsation tools

For each selected station, ExploreMag extracts the plot interval, masks invalid
values, checks sampling, and uses regularly sampled component arrays. Small
timestamp irregularities may be put on a regular grid; long gaps are not a
license to manufacture oscillations. The category plots use fourth-order
Butterworth filters in second-order-section form and forward-backward
zero-phase application.

Forward-backward filtering removes phase delay but is acausal: output before a
sharp change can contain filter response to later samples. Never use the
filtered trace alone to assign onset time. Compare it with raw data and an
independently defined onset marker.

## Wavelet scalograms

Click **Wavelet scalogram** after selecting stations and setting the plot
interval. One window per station contains four panels: total horizontal H,
first horizontal component, second horizontal component, and Z. The selected
coordinate system controls component labels. ExploreMag uses a complex Morlet
wavelet with nondimensional frequency `ω0 = 6` and the scale-to-Fourier-period
conversion of [Torrence and Compo (1998)](https://doi.org/10.1175/1520-0477%281998%29079%3C0061%3AAPGTWA%3E2.0.CO%3B2).

The requested survey range is 5–600 s, automatically restricted by cadence and
record duration. Power is displayed in dB relative to each period's median:

```text
display power = 10 log10(power / median power at that period)
```

### Reading a scalogram

- Time runs left to right; period is logarithmic, with short periods at top.
- Warm colors indicate power above that period's median, not an absolute nT²
  comparison between stations.
- The black contour is an approximate 95% AR(1) red-noise threshold.
- The white boundary and hatched region mark the cone of influence. Features
  inside the hatched edge region are vulnerable to finite-record effects.
- Dashed 40 s and 150 s lines show the nominal Pi2 boundaries.
- Move the cursor to read nearest UTC time, period, and power.

Use the scalogram to choose event and quiet windows, determine whether power is
localized or continuous, and see whether a ridge crosses formal band
boundaries. Do not interpret pixels outside the cone of influence as equally
reliable, and remember that the implemented red-noise contour is an
approximation rather than a multiple-testing-corrected event detector.

## Pi-band analysis

Click **Pi pulsations** to generate, for every selected station:

1. unfiltered H and three magnetic components;
2. Pi1, 1–40 s;
3. Pi2, 40–150 s;
4. Pi3/Ps6, periods longer than 150 s;
5. a 5–600 s horizontal Morlet scalogram.

Pi1 and Pi2 use zero-phase band-pass filters. Pi3/Ps6 uses a zero-phase
low-pass filter. When cadence cannot support a requested short-period edge,
ExploreMag reports or adjusts the effective edge rather than claiming the full
nominal range. Use **Export pulsation data** in the figure window to save the
filtered component series for all selected stations.

Literature convention treats Pi as irregular/impulsive morphology. Pi2 is
commonly an impulsive, often damped wave train associated with substorm onset,
but its absence at one station does not exclude a substorm and a Pi2-like
signal does not uniquely prove one. This limitation is emphasized by
[Rostoker and Olson (1978)](https://doi.org/10.5636/jgg.30.135) and the model
review of [Keiling and Takahashi (2011)](https://doi.org/10.1007/s11214-011-9818-4).

## Pc-band analysis

Click **Pc pulsations** to generate one six-panel window per selected station:

1. unfiltered components;
2. Pc2, 5–10 s;
3. Pc3, 10–45 s;
4. Pc4, 45–150 s;
5. Pc5, 150–600 s;
6. a horizontal Morlet scalogram extending to 750 s when the interval permits.

Each Pc category uses a fourth-order, zero-phase Butterworth band pass. A band
that violates Nyquist or is too long for stable filtering is annotated with an
error while other panels remain available. Component visibility controls and
CSV export help compare bands and stations.

In the literature, Pc labels indicate relatively continuous pulsations. Pc3–4
ground signals are often investigated as driven waves and field-line
resonances; classical field-line-resonance theory predicts characteristic
spatial amplitude, phase, and polarization structure
([Southwood, 1974](https://doi.org/10.1016/0032-0633%2874%2990078-6)). A peak
inside a Pc band is not sufficient to identify a field-line resonance. Look for
coherence, repeatable frequency, amplitude maxima and phase changes across a
latitudinal array, and appropriate local-time behavior.

## Total ULF wave power

ExploreMag's **Total ULF wave power** is not imported from SuperMag.

It is a derived product from the available data set, and it does not reflect
all possible ULF energy. For each station it:

1. applies the resolvable portion of the requested 5–150 s fourth-order
   zero-phase band pass to the two horizontal components; when cadence cannot
   resolve the short-period edge, ExploreMag raises that edge instead of
   rejecting the entire station (for example, 10-second data use approximately
   21.1–150 s);
2. calculates instantaneous horizontal power
   `P(t) = dB1_filtered² + dB2_filtered²` in nT²;
3. smooths it with a running mean of
   `max(3, round(10 seconds / cadence))` samples—approximately 10 seconds for
   1-second and 2-Hz data, 30 seconds for 10-second data, and three minutes for
   1-minute data;
4. displays station power series and a geographic station map.

The 5–150 s range combines Pc2, Pc3, Pc4, Pi1, and Pi2 timescales. Therefore
the product measures enhanced short-period horizontal activity but does not
separate continuous and impulsive classes or identify a wave mode. In wave
studies, power is normally interpreted together with spectra, coherence,
phase, polarization, duration, and spatial structure. Welch's foundational
method averages modified periodograms to reduce spectral-estimate variance
([Welch, 1967](https://research.ibm.com/publications/the-use-of-fast-fourier-transform-for-the-estimation-of-power-spectra-a-method-based-on-time-averaging-over-short-modified-periodograms)).

Validate the product for your use-case before using it for scientific
publication.

### Using the ULF window

1. Select multiple stations and set a plot interval.
2. Click **Total ULF wave power**.
3. Inspect the map for regional maxima, then click a station for its detailed
   raw components, filtered components, running power, and scalogram.
4. Select map bubbles and use the multi-station power stack to compare timing.
   **Remove selection** clears all selected map stations. Adjust displayed time
   and per-axis y limits without recomputing the underlying data; each **Auto**
   button and **Auto-scale all Y axes** restore data-based scaling after manual
   limits have been applied.
5. Export the multi-station power CSV and save the map/figures.

Compare stations cautiously: sensor response, cadence conversion, repaired
samples, local conductivity, baseline, and noise floor can change apparent
power. A smooth interpolated 2 Hz series does not contain the same independent
high-frequency information as native 2 Hz observations.

## Pi2 ellipse and polarization analysis

This is stricter than the general **Pi pulsations** plot. It estimates
event/background spectra and horizontal polarization for a user-defined Pi2
wave train, then maps the result at selected stations.

### Step-by-step use

1. Inspect raw and wavelet plots and select a continuous candidate Pi2 wave
   train. The default band is 40–150 s.
2. Select a **quiet/pre-event** interval with similar data quality and no
   overlap with the Pi2 interval. It is used for RMS and PSD comparison, not
   for the GMAG download baseline.
3. Select the stations to analyze and click **Pi2 ellipse/Polarization**.
4. Enter Pi2 start/end, quiet start/end, and an optional independent onset.
   **Use plot range as Pi2** and **Use plot range as quiet** copy the current
   shared interval into the corresponding fields.
5. Choose GEO for geographically interpretable azimuth, or NEZ for orientation
   relative to local magnetic north.
6. Check **Confirm +north, +east and +Z downward signs** only after verifying
   the file convention. One reversed horizontal sign reverses rotation sense.
7. Change minimum/maximum periods if scientifically justified; these values
   control the filter, spectra, ellipses, and map.
8. Click **Analyze selected stations**. Failed stations are reported rather
   than silently included.
9. Click a successful green station on the ellipse map to open its detailed
   diagnostic figure.

### What is calculated

- Data are required to be finite, chronological, nearly regular, and free of
  large gaps over the analysis input. Small timestamp jitter may be resampled;
  large gaps cause rejection.
- A fourth-order Butterworth band pass is applied with `sosfiltfilt`.
- The event hodogram and covariance matrix provide time-domain major/minor
  axes, axis ratio, azimuth, and signed-area rotation check.
- Event and quiet power spectral densities are estimated with Welch segment
  averaging; cross-spectral density gives phase and magnitude-squared
  coherence.
- Spectral Stokes parameters around the dominant Pi2 frequency give spectral
  azimuth, ellipticity, axis ratio, and rotation diagnostics.
- Event/background RMS and band-power ratios help judge whether the event
  exceeds the selected quiet interval.

Welch spectral averaging follows the variance-reduction principle of
[Welch (1967)](https://research.ibm.com/publications/the-use-of-fast-fourier-transform-for-the-estimation-of-power-spectra-a-method-based-on-time-averaging-over-short-modified-periodograms).
Covariance/coherency-matrix polarization has a long geophysical-wave tradition,
including [Means (1972)](https://ntrs.nasa.gov/citations/19720019517) and
[Samson and Olson (1980)](https://doi.org/10.1111/j.1365-246X.1980.tb04308.x).

### Ellipse-map conventions

- Every displayed ellipse has the same plotted semi-major length. Shape and
  azimuth are comparable; map size does **not** encode amplitude.
- The major/minor ratio describes linear versus circular polarization. Near a
  circular ellipse, the major-axis azimuth is intrinsically poorly defined;
  ExploreMag suppresses the major axis when `|b/a| > 0.8`.
- GEO azimuth is measured from geographic north toward east. NEZ azimuth is
  relative to local magnetic north and should not be interpreted as geographic
  direction without transformation.
- With horizontal component 1 on the x axis and eastward component on y,
  positive cross-spectral phase/Stokes `S3` is reported as counterclockwise and
  negative as clockwise. This is a stated plotting convention, not a universal
  sign convention across all papers and instruments.
- Rotation is not reported when coordinate signs are unverified, coherence is
  below 0.50, or the ellipse is too nearly linear (`|b/a| < 0.10`).
- **Indeterminate** is a valid quality result, not a failed computation.

The detailed station figure should be the basis of interpretation. Confirm
that event power exceeds background, coherence is meaningful, the dominant
frequency is resolved by the record length, time-domain and spectral ellipses
are broadly consistent, and results are not dominated by filter edges.

## Saving and exporting

- Use each figure's **Save PNG…** control for maps and analysis figures.
- Pulsation windows export filtered Pi or Pc component time series as CSV.
- Total ULF windows export time-resolved station power as CSV.
- The manual stack exports selected point annotations as text.
- GMAG downloads create a NetCDF plus a station report containing source,
  cadence, coverage, baseline, and processing outcomes.
- Keep the original NetCDF and reports. PNG and CSV products alone do not
  preserve all provenance and quality metadata.

For reproducibility, record the software version, source option, station list,
UTC intervals, coordinate system, baseline method, common/native cadence,
quality screening, filter periods, and any manual axis/color limits.

## Troubleshooting

### A pulsation band is rejected

The cadence violates Nyquist, the selected interval is too short for stable
forward-backward filtering, or it does not contain enough cycles. Use faster
data and/or a longer interval. Total ULF power is a special case: it uses and
reports the resolvable part of 5–150 s when a nonempty portion remains.

### A station disappears from Pi2 results

Check the warning dialog. Common causes are NaNs, a large gap, cadence jitter,
insufficient samples, overlapping event/quiet windows, weak coverage, or an
incompatible period band. Analyze continuous segments rather than bridging a
long outage.

### A wavelet feature appears only at an edge

It may lie inside the cone of influence. Load more padding and repeat the
analysis. A ridge should be treated more confidently when it persists outside
the hatched edge region and appears coherently at relevant nearby stations.

### Pc2 is blank for 1-second or 10-second data

The 5 s edge is too close to or above Nyquist. Use native or 2 Hz data. Ten
seconds cannot resolve Pc2 or much of Pc3.

### The map has smooth structure where there are no stations

The map interpolator fills a visual surface between sparse measurements. Check
the station markers and raw traces. Do not interpret extrapolated color as an
independent observation.

### GEO is unavailable

The loaded file does not contain a GEO variable. Use NEZ, or obtain/convert a
file that includes GEO components.

### A GMAG station is rejected

Inspect the station report for source-loader, cadence, component, coverage, or
baseline errors. IMAGE 10-second data are accepted by the 10-second and native
GUI modes, while faster common modes retain stricter native-cadence checks.

## ExploreMag Rules of the Road

ExploreMag is built around the SuperMAG API and the GMAG Python package,
which provide its core data-download capabilities. If you use ExploreMag to
download, explore, or analyze ground magnetic field data, please cite the
relevant references for SuperMAG, GMAG, and ExploreMag. Recommended references
are provided below.

> **Important:** SuperMAG is made possible by the generous contribution of
> data from numerous collaborators. To support their continued operation,
> users must follow these rules of the road. Data, plots, and derived data
> products are provided under the limitations of “fair use” and cannot be
> redistributed. Contact the relevant instrument PI and the SuperMAG PI if a
> proposed use conflicts with these restrictions.

Users are requested to acknowledge individual collaborators and SuperMAG when
original data, derived data, movies, or other data products are used in
publications or presentations.

### Contents

- [Using magnetometer data](#using-magnetometer-data)
- [Using substorm lists](#using-substorm-lists)
- [Using CARISMA data](#using-carisma-data)
- [Using IMAGE data](#using-image-data)
- [Using INTERMAGNET data](#using-intermagnet-data)
- [Using THEMIS data](#using-themis-data)
- [Core references](#core-references)
- [Collaborator references for SuperMag](#collaborator-references-for-supermag)
- [SuperMAG index references](#supermag-index-references)
- [Substorm-list references](#substorm-list-references)

### Using magnetometer data

#### Requirements for all uses of SuperMAG

- Include the acknowledgement listed on the SuperMAG website.
- Cite the appropriate technical papers for the stations used. See
  [Collaborator references for SuperMag](#collaborator-references-for-supermag).
- Cite the principal SuperMAG reference:
  Gjerloev, J. W. (2012), *The SuperMAG data processing technique*,
  *Journal of Geophysical Research*, 117, A09213,
  [doi:10.1029/2012JA017683](https://doi.org/10.1029/2012JA017683).

#### When a small number of stations are central to a study

If only a few stations play a key role and their data are central to the
scientific conclusions:

- Offer co-authorship to the PI or PIs of those stations.
- Cite the appropriate station or network paper listed under
  [Collaborator references for SuperMag](#collaborator-references-for-supermag).

### Using substorm lists

- If a substorm-onset list is central to the study, offer co-authorship to the
  authors of the technique used.
- Include the acknowledgements specified by SuperMAG.
- Cite the appropriate reference under
  [Substorm-list references](#substorm-list-references).
- Consult the [SuperMAG products description](https://supermag.jhuapl.edu/products/?tab=description)
  for details.
  
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
  

### Core references

#### GMAG Python package

Murphy, K. R., Rae, I. J., Halford, A. J., Engebretson, M., Russell, C. T.,
Matzka, J., ... & Tanskanen, E. (2022). GMAG: An open-source Python package
for ground-based magnetometers. *Frontiers in Astronomy and Space Sciences*,
9, 1005061, [doi:10.3389/fspas.2022.1005061](https://doi.org/10.3389/fspas.2022.1005061).

#### SuperMAG

Gjerloev, J. W. (2012), The SuperMAG data processing technique,
*Journal of Geophysical Research*, 117, A09213,
[doi:10.1029/2012JA017683](https://doi.org/10.1029/2012JA017683).

### Collaborator references for SuperMag

#### AUTUMNX

Connors, M., Schofield, I., Reiter, K., Chi, P. J., Rowe, K. M., & Russell, C. T. (2016). The AUTUMNX magnetometer meridian chain in Québec, Canada. *Earth, Planets and Space*, 68(1), 2, [doi:10.1186/s40623-015-0354-4](https://doi.org/10.1186/s40623-015-0354-4).

#### EMMA

Lichtenberger, J., M. Clilverd, B. Heilig, M. Vellante, J. Manninen,
C. Rodger, A. Collier, A. Jørgensen, J. Reda, R. Holzworth, and R. Friedel
(2013), The plasmasphere during a space weather event: First results from the
PLASMON project, *Journal of Space Weather and Space Climate*, 3, A23, [doi:10.1051/swsc/2013045](http://dx.doi.org/10.1051/swsc/2013045).

#### IMAGE Chain

Tanskanen, E. I. (2009), A comprehensive high-throughput analysis of substorms
observed by IMAGE magnetometer network: Years 1993–2003 examined, 114, A05204,
[doi:10.1029/2008JA013682](https://doi.org/10.1029/2008JA013682).

#### MACCS

Engebretson, M. J., W. J. Hughes, J. L. Alford, E. Zesta, L. J. Cahill Jr.,
R. L. Arnoldy, and G. D. Reeves (1995), Magnetometer array for cusp and cleft
studies observations of the spatial extent of broadband ULF magnetic
pulsations at cusp/cleft latitudes, *Journal of Geophysical Research*, 100,
19371–19386,
[doi:10.1029/95JA00768](https://doi.org/10.1029/95JA00768).

#### McMAC Chain

Chi, P. J., M. J. Engebretson, M. B. Moldwin, C. T. Russell, I. R. Mann,
M. R. Hairston, M. Reno, J. Goldstein, L. I. Winkler, J. L. Cruz-Abeyro,
D.-H. Lee, K. Yumoto, R. Dalrymple, B. Chen, and J. P. Gibson (2013), Sounding
of the plasmasphere by Mid-continent MAgnetoseismic Chain magnetometers,
*Journal of Geophysical Research: Space Physics*, 118,
[doi:10.1002/jgra.50274](https://doi.org/10.1002/jgra.50274).

#### MAGDAS / 210 Chain

Yumoto, K., and the CPMN Group (2001), Characteristics of Pi 2 magnetic
pulsations observed at the CPMN stations: A review of the STEP results,
*Earth, Planets and Space*, 53, 981–992, [doi:10.1186/BF03351695](https://doi.org/10.1186/BF03351695).

#### CARISMA

Mann, I. R., et al. (2008), The upgraded CARISMA magnetometer array in the
THEMIS era, *Space Science Reviews*, 141, 413–451,
[doi:10.1007/s11214-008-9457-6](https://doi.org/10.1007/s11214-008-9457-6).

#### AAL-PIP

Clauer, C. R., et al. (2014), An autonomous adaptive low-power instrument
platform (AAL-PIP) for remote high-latitude geospace data collection,
*Geoscientific Instrumentation, Methods and Data Systems*, 3, 211–227,
[doi:10.5194/gi-3-211-2014](https://doi.org/10.5194/gi-3-211-2014).

#### INTERMAGNET

Love, J. J., and A. Chulliat (2013), An international network of magnetic
observatories, *Eos*, 94(42), 373–374,
[doi:10.1002/2013EO420001](https://doi.org/10.1002/2013EO420001).

### SuperMAG index references

#### SML, SMU, and SME

Newell, P. T., and J. W. Gjerloev (2011), Evaluation of SuperMAG auroral
electrojet indices as indicators of substorms and auroral power,
*Journal of Geophysical Research*, 116, A12211,
[doi:10.1029/2011JA016779](https://doi.org/10.1029/2011JA016779).

#### SMLs, SMLd, SMUs, and SMUd

Gjerloev, J. W., R. A. Hoffman, S. Ohtani, J. Weygand, and R. Barnes (2010),
Response of the Auroral Electrojet Indices to Abrupt Southward IMF Turnings,
*Annales Geophysicae*, 28, 1167–1182.

#### SME-LT, SMU-LT, and SML-LT

Newell, P. T., and J. W. Gjerloev (2014), Local geomagnetic indices and the
prediction of auroral power, *Journal of Geophysical Research: Space Physics*,
119, [doi:10.1002/2014JA020524](https://doi.org/10.1002/2014JA020524).

#### SMR and SMR-LT

Newell, P. T., and J. W. Gjerloev (2012), SuperMAG-Based Partial Ring Current
Indices, *Journal of Geophysical Research*, 117,
[doi:10.1029/2012JA017586](https://doi.org/10.1029/2012JA017586).

### Substorm-list references

- Forsyth, C., Rae, I. J., Coxon, J. C., Freeman, M. P., Jackman, C. M.,
  Gjerloev, J., and Fazakerley, A. N. (2015), A new technique for determining
  Substorm Onsets and Phases from Indices of the Electrojet (SOPHIE),
  *Journal of Geophysical Research: Space Physics*, 120, 10,592–10,606,
  [doi:10.1002/2015JA021343](https://doi.org/10.1002/2015JA021343).
- Frey, H. U., Mende, S. B., Angelopoulos, V., and Donovan, E. F. (2004),
  Substorm onset observations by IMAGE-FUV, *Journal of Geophysical Research*,
  109, A10304,
  [doi:10.1029/2004JA010607](https://doi.org/10.1029/2004JA010607).
- Gjerloev, J. W. (2012), The SuperMAG data processing technique,
  *Journal of Geophysical Research*, 117, A09213,
  [doi:10.1029/2012JA017683](https://doi.org/10.1029/2012JA017683).
- Liou, K. (2010), Polar Ultraviolet Imager observation of auroral breakup,
  *Journal of Geophysical Research*, 115, A12219,
  [doi:10.1029/2010JA015578](https://doi.org/10.1029/2010JA015578).
- Newell, P. T., and J. W. Gjerloev (2011), Evaluation of SuperMAG auroral
  electrojet indices as indicators of substorms and auroral power,
  *Journal of Geophysical Research*, 116, A12211,
  [doi:10.1029/2011JA016779](https://doi.org/10.1029/2011JA016779).
- Newell, P. T., and J. W. Gjerloev (2011), Substorm and magnetosphere
  characteristic scales inferred from the SuperMAG auroral electrojet indices,
  *Journal of Geophysical Research*, 116, A12232,
  [doi:10.1029/2011JA016936](https://doi.org/10.1029/2011JA016936).
- Ohtani, S., and J. Gjerloev (2020), Is the Substorm Current Wedge an Ensemble
  of Wedgelets?: Revisit to Midlatitude Positive Bays, accepted,
  *Journal of Geophysical Research*.

## Literature and conventions

If you use ExploreMag to download data from SuperMag, please cite the SuperMag sources displayed when the software is first initialized, and contact the PIs of the individual instrument and magnetometer chains about the usage of their data.

If ExploreMag is used for downloading higher cadence data from THEMIS, CARISMA, and IMAGE chains, please also cite the gmag Python package that makes the download possible:

Murphy, K. R., Rae, I. J., Halford, A. J., Engebretson, M., Russell, C. T., Matzka, J., ... & Tanskanen, E. (2022). GMAG: An open-source python package for ground-based magnetometers. Frontiers in Astronomy and Space Sciences, 9, [doi:10.3389/fspas.2022.1005061](https://doi.org/10.3389/fspas.2022.1005061).

The following sources support the terminology and analysis conventions described above. They do not imply that ExploreMag reproduces every method or statistical assumption in each paper.

1. Jacobs, J. A., Kato, Y., Matsushita, S., and Troitskaya, V. A. (1964),
   “Classification of geomagnetic micropulsations,” *Journal of Geophysical
   Research*, 69, 180–181.
   [doi:10.1029/JZ069i001p00180](https://doi.org/10.1029/JZ069i001p00180).
2. Torrence, C., and Compo, G. P. (1998), “A Practical Guide to Wavelet
   Analysis,” *Bulletin of the American Meteorological Society*, 79, 61–78.
   [doi:10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2](https://doi.org/10.1175/1520-0477%281998%29079%3C0061%3AAPGTWA%3E2.0.CO%3B2).
3. Welch, P. D. (1967), “The Use of Fast Fourier Transform for the Estimation
   of Power Spectra: A Method Based on Time Averaging Over Short, Modified
   Periodograms,” *IEEE Transactions on Audio and Electroacoustics*, 15,
   70–73. [IBM publication record](https://research.ibm.com/publications/the-use-of-fast-fourier-transform-for-the-estimation-of-power-spectra-a-method-based-on-time-averaging-over-short-modified-periodograms).
4. Means, J. D. (1972), “Use of the Three-Dimensional Covariance Matrix in
   Analyzing the Polarization Properties of Plane Waves,” *Journal of
   Geophysical Research*, 77, 5551–5559.
   [NASA record](https://ntrs.nasa.gov/citations/19720019517).
5. Samson, J. C., and Olson, J. V. (1980), “Some comments on the descriptions
   of the polarization states of waves,” *Geophysical Journal International*,
   61, 115–129.
   [doi:10.1111/j.1365-246X.1980.tb04308.x](https://doi.org/10.1111/j.1365-246X.1980.tb04308.x).
6. Southwood, D. J. (1974), “Some features of field line resonances in the
   magnetosphere,” *Planetary and Space Science*, 22, 483–491.
   [doi:10.1016/0032-0633(74)90078-6](https://doi.org/10.1016/0032-0633%2874%2990078-6).
7. Rostoker, G., and Olson, J. V. (1978), “Pi2 Micropulsations as Indicators
   of Substorm Onsets and Intensifications,” *Journal of Geomagnetism and
   Geoelectricity*, 30, 135–147.
   [doi:10.5636/jgg.30.135](https://doi.org/10.5636/jgg.30.135).
8. Keiling, A., and Takahashi, K. (2011), “Review of Pi2 Models,” *Space
   Science Reviews*, 161, 63–148.
   [doi:10.1007/s11214-011-9818-4](https://doi.org/10.1007/s11214-011-9818-4).

Always inspect station coverage, quality flags, plots, and processing metadata before using downloaded or converted data in scientific results.
