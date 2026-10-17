# JSON file protocols

This reference documents the JSON files read or written by this repository's
`tool`, `lattice.py`, and `setup`. It describes the current implementation,
including differences between readers. Update this file whenever a file format
changes. The definitions in `tool` (`Settings`, `SnapshotReport`, `FitResult`,
`LiveAnalysis.export_bundle`, `generate_lattices`, and `simulate_experiment`) and
the parameter dataclasses in `lattice.py` are the implementation references.

## File inventory

Paths below are relative to an experiment unless a processed export is specified.
`settings.json` is a conventional name; callers can choose any filename.

| File | Producer | Consumer / purpose |
| --- | --- | --- |
| `settings.json` | `tool settings`, GUI Save settings | `--settings` and GUI preparation; analysis, lattice-generation, and simulation settings |
| `lattice_configs.json` | `tool lattice`, `tool simulate`, `lattice.save_lattice_config_list` | Analysis, lattice preview, simulation, `setup`, and `lattice.load_lattice_config_list` |
| `lattice_generation.json` | `tool lattice` | Generation settings and optics diagnostics for inspection |
| `sim_config.json` | `tool simulate` | Simulation provenance for inspection; its analysis/simulation sections can be loaded with `--settings` |
| `source_lattice_configs.json` | `tool simulate` | Exact source configuration before simulation overrides |
| `processed_*/config.json` | Analysis/replay CLI and GUI export | Full snapshot, results, and provenance; entry point for `tool replay` |
| `processed_*/results.json` | Same export | Convenience subset containing fits and units |
| `processed_*/lattice_configs.json` | Same export | Snapshot's configuration document, serialized again |
| `processed_*/inputs/lattice_configs.json` | Export with raw inputs | Exact original dependency bytes; checked during replay |
| `processed_*/projected/<config>/<sample_id>.json` | Same export | One processed projected measurement |
| `processed_*/sliced/<config>/<slice>/<sample_id>.json` | Same export | One processed slice measurement |
| `.processing-*/INCOMPLETE.json` | Failed export | Marks a diagnostic staging directory as unpublished |

The GUI preparation editor holds the `lattice` and `simulation` settings objects;
Save settings adds the current `analysis` object. Fit Diagnostics displays the fit
array defined below. The Lattice Preview tab's Twiss controls, configuration
selection, and coordinate mode are transient and have no JSON file protocol.
`BeamParameters.to_dict()` also has no standalone JSON file producer here.

## Shared conventions

- Objects use string keys; arrays preserve order. Integer-indexed Python maps
  become JSON objects with decimal string keys, including `"-1"` for projected
  measurements. Convert these keys explicitly when integer indices are needed.
- Configuration IDs are zero-based positions in the lattice configuration arrays.
  Slice indices are zero-based; `-1` denotes the projected region. With
  `num_slices=1`, only projected measurements are produced.
- JSON quantities use SI units unless the field name or table says otherwise.
  Emittances are in **m rad**, even when the GUI displays **mm mrad**:
  `1e-6 m rad = 1 mm mrad`. Beam energy is in GeV; phase advances are in radians.
- `R` is a finite 6×6 first-order Ocelot transport matrix, serialized as six row
  arrays in transverse/longitudinal order `(x, x', y, y', tau, relative momentum)`.
  Matrix indices in Python are zero-based: physical `R34` is `R[2][3]`.
- Screen arrays use `[x pixel, y pixel]` order. The second axis is the slicing /
  streak direction. Slices represent equal integrated intensity fractions per
  image, with no absolute time calibration.
- SHA-256 values are lowercase hexadecimal strings. File hashes refer to exact
  file bytes; reformatting equivalent JSON changes its file hash.
- Report timestamps (`*_utc`) are UTC ISO 8601 strings with a timezone offset.
  Generated sample/export names use local wall-clock timestamps, milliseconds,
  and an eight-character UUID suffix; they are not UTC timestamp fields.
- `tool.atomic_json` writes indented JSON with a trailing newline, rejects
  NaN/Infinity, flushes and fsyncs a temporary file in the destination directory,
  then replaces the destination. `lattice.save_lattice_config_list` instead uses
  a direct `json.dump(..., indent=4)` without this atomic publication guarantee.
- There is no separate `schema_version`. Reports currently write
  `tool_version: "1.0"`; readers validate structure rather than dispatching on
  this version. Distinguish an absent optional field from an explicitly allowed
  `null`; do not substitute `null` for required arrays, numbers, or objects.

## App settings: `settings.json`

The root object contains named settings sections:

```json
{
  "analysis": {},
  "lattice": {},
  "simulation": {}
}
```

Each section loader requires its own section to exist and be an object. Fields
omitted within that section use the defaults below. Unknown keys **inside a
settings section are rejected**; unrelated root sections are ignored by that
loader. JSON arrays restore tuple-valued settings (`roi`, `phases`, and
`screen_shape`). `tool settings --output settings.json` writes every current
default. Enum fields use the stored values below, not the GUI's descriptive labels.

### `analysis`

Used by GUI, analyze, inspect, replay, and simulation camera calibration; embedded
unchanged as `config.json.settings` and each processed sample's `processing`.

| Field | JSON type | Default | Meaning / units |
| --- | --- | --- | --- |
| `pixel_size_x` | number | `3.45e-6` | Metres per oriented x pixel |
| `pixel_size_y` | number | `3.45e-6` | Metres per oriented y pixel |
| `rotation` | integer | `0` | Counterclockwise display rotation: `0`, `90`, `180`, `270` |
| `flip_horizontal` | boolean | `false` | Flip oriented display horizontally after rotation |
| `flip_vertical` | boolean | `false` | Flip oriented display vertically after rotation |
| `num_slices` | integer | `20` | Positive slice count; `1` selects projected analysis |
| `fit_type` | string | `"RMS"` | `"RMS"`, `"Gaussian"`, or `"Gaussian-with-free-amplitude"` |
| `regression` | string | `"chi_squared"` | `"chi_squared"` or `"linear"` |
| `weighting` | string | `"shot_std"` | `"shot_std"` or `"standard_error"`; chi-squared weighting |
| `mask_downsampling` | integer | `8` | Positive downsampling factor for mask construction |
| `mask_filter` | string | `"gaussian"` | `"gaussian"` or `"uniform"` |
| `mask_filter_size` | integer | `9` | Positive uniform-filter size in original pixels, reduced by downsampling |
| `mask_sigma` | number | `20.0` | Positive Gaussian-filter sigma in original pixels, reduced by downsampling |
| `mask_cutoff` | number | `0.01` | Threshold of the smoothed, positive-peak-normalized mask image, in `[0,1]` |
| `mask_dilation` | integer | `0` | Nonnegative dilation iterations on the downsampled mask; zero disables dilation |
| `largest_patch` | boolean | `true` | Keep the largest connected mask component |
| `tail_extension` | boolean | `false` | Extend mask toward increasing image columns |
| `subtract_background` | boolean | `false` | Load and subtract `screen_background.npz` |
| `subtract_inverse_mask_mean` | boolean | `false` | Subtract mean outside the mask |
| `negative_intensity` | string | `"profile_threshold"` | `"profile_threshold"`, `"clip"`, or `"reject"` |
| `roi` | array of 4 integers/nulls | `[null,null,null,null]` | Half-open `[x_start,x_stop,y_start,y_stop]` in oriented pixels; null selects that image edge |
| `tail_sigma` | number or null | `null` | Projected region begins at fitted `mu_y + tail_sigma*sigma_y`; requires `num_slices=1` when set |
| `tail_fit_type` | string | `"Gaussian"` | Same three fit-type values as `fit_type` |
| `settle_seconds` | number | `1.0` | Nonnegative file-settling delay in seconds |
| `poll_seconds` | number | `30.0` | Positive live-refresh interval in seconds |
| `max_condition` | number | `1e10` | Positive maximum condition number of the scaled regression matrix |

Analysis validation requires finite positive pixel sizes, mask sigma, polling
interval, and condition limit; positive integer slice/downsampling/filter counts;
nonnegative integer dilation; finite nonnegative settling delay and tail sigma;
and true JSON booleans for boolean settings. ROI bounds must be nonnegative
integers or null; their order and image extent are checked when processing.
Both screen and background undergo rotation and flips before subtraction.
Pixel scales apply to the resulting axes without being swapped automatically.

`profile_threshold` retains negative pixels but width fits use positive profile
support; negative summed column intensity prevents monotone equal-charge slicing.
`clip` clips negative pixels before fitting; `reject` rejects a sample with any
negative intensity inside the mask. Per-region failures are recorded separately.

### `lattice`

These are generation controls, distinct from the per-configuration high-level
parameters in `lattice_configs.json`.

| Field | JSON type | Default | Meaning / units |
| --- | --- | --- | --- |
| `beam_energy` | number | `16.3` | Beam energy, GeV |
| `start` | string | `"STSEC.2473.T3"` | Lattice entrance element ID |
| `stop` | string | `"OTRC.2560.T3"` | Inclusive last element ID |
| `phase_start` | string | `"STSEC.2473.T3"` | Reference element for x phase matching |
| `phases` | array of numbers | See below | Ordered x phase targets, radians |
| `dispersion_x` | number | `0.0` | Screen horizontal dispersion target, m |
| `r34` | number or null | `-36.0` | Vertical transport target, m; null omits it |
| `max_beta` | number | `600.0` | Global x/y beta upper target, m |
| `min_screen_beta_x` | number | `20.0` | Screen horizontal beta lower target, m |
| `solver` | string | `"nelder-mead"` | Ocelot solver identifier |
| `max_iter` | integer | `1000` | Solver iteration limit |

Default `phases` are `[-pi/16, 0, pi/16, pi/8, 3*pi/16, pi/4, 5*pi/16, 3*pi/8]`
as JSON numbers. The generator checks that this list is nonempty and finite and
that energy is positive; additional feasibility depends on the lattice and solver.
Its phase residual wraps with period `pi`. Solver success and all reported target
constraints must pass before output is published. No fixed whitelist of solver
names is enforced by settings parsing.

### `simulation`

| Field | JSON type | Default | Meaning / units |
| --- | --- | --- | --- |
| `samples_per_config` | integer | `5` | Positive number of shots per configuration |
| `num_particles` | integer | `200000` | At least two macroparticles per shot |
| `normalized_emittance` | number | `1e-6` | Positive normalized emittance for both transverse planes, m rad |
| `charge` | number | `2.5e-10` | Positive bunch charge, C |
| `sigma_tau` | number | `1e-5` | Positive bunch length, m |
| `sigma_p` | number | `0.0` | Nonnegative relative momentum spread |
| `chirp` | number | `0.0` | Finite dimensionless chirp parameter passed to the beam generator |
| `twiss_noise` | number | `0.0` | In `[0,1)`; uniform additive alpha jitter and relative beta jitter per shot |
| `screen_shape` | array of 2 integers | `[3000,4096]` | Screen bin counts `[nx,ny]`, each at least two |
| `screen_noise` | number | `0.0` | Nonnegative additive Gaussian intensity-noise standard deviation |
| `screen_offset` | number | `0.0` | Nonnegative additive intensity offset |
| `screen_gain` | number | `1.0` | Positive peak-normalized image gain |
| `screen_gamma` | number | `1.0` | Positive image response exponent denominator (`image ** (1/screen_gamma)`) |
| `seed` | integer | `0` | NumPy random seed |
| `dechirper_enabled` | boolean or null | `null` | Override source enable flag; null retains source value |
| `dechirper_dist_y` | number or null | `null` | Positive plate-distance override, m; null retains source value |
| `dechirper_sigma_z` | number or null | `null` | Positive wake bunch-length override, m; null retains source value |

Physical positive/nonnegative quantities are checked for finiteness. Simulation
also validates its analysis settings and requires a new output directory. Explicit
dechirper overrides update both configuration arrays. The source high-level
`beam_energy` determines each tracked beam's energy; geometric emittance is
`normalized_emittance * 0.00051099895 / beam_energy`.

## Lattice configurations: `lattice_configs.json`

The root object has parallel arrays. Their order defines the configuration ID:

```json
{
  "high_level_params": [
    {"beam_energy": 14.0, "start": "VCCHIRP_S.2487.T3", "stop": "OTRC.2560.T3"}
  ],
  "low_level_params": [
    {
      "start": "VCCHIRP_S.2487.T3",
      "stop": "OTRC.2560.T3",
      "qe_2495_t3_k1": 0.1,
      "qe_2506_t3_k1": -0.1,
      "qe_2518_t3_k1": 0.1,
      "qh_2529_t3_k1": -0.1,
      "qh_2532_t3_k1": 0.1,
      "qh_2544_t3_k1": -0.1,
      "qm_2549_t3_k1": 0.1,
      "qm_2554_t3_k1": -0.1,
      "qm_2559_t3_k1": 0.1
    }
  ]
}
```

This example illustrates the analysis input shape; its arbitrary strengths are
not matched optics or machine settings. For `setup`, also provide a consistent
`target_doocs_params` array. Normal generators write all three arrays.

### High-level configuration object

| Field | JSON type | Meaning / units |
| --- | --- | --- |
| `beam_energy` | number | Required by analysis; energy in GeV |
| `start`, `stop` | string | Required by analysis; same endpoints as the corresponding low-level object |
| `mu_x_start` | string | Phase-target reference element ID |
| `dispersion_x` | number | Screen x dispersion target, m |
| `mu_x` | number | x phase target, radians |
| `global_max_beta` | number | Global beta target, m |
| `final_beta_min` | number or null | Final beta lower target, m |
| `r_34` | number or null | Vertical transport target, m |
| `minimize_second_order_t116` | boolean | Second-order minimization flag |
| `dechirper_enabled` | boolean | Enable wake tracking |
| `dechirper_dist_y` | number | Distance from plate, m |
| `dechirper_sigma_z` | number | Characteristic RMS bunch length for wake setup, m |

For `lattice.load_lattice_config_list`, omitted fields use
`LatticeHighLevelParams` defaults: energy `14.0`, start and `mu_x_start`
`"VCCHIRP_S.2487.T3"`, stop `"OTRC.2560.T3"`, dispersion and phase `0.0`,
global maximum beta `300`, final minimum beta `10.0`, `r_34=null`, both flags
`false`, plate distance `500e-6`, and wake length `10e-6`.
These differ from `tool`'s generation-settings defaults. Although the Python
`mu_x` annotation allows complex values, the JSON writer has no complex-number
encoding: use a real JSON number.

### Low-level configuration object

`start` and `stop` identify the actual lattice extent. Optional `mu_x_start` is
phase-reference metadata. All nine current magnet strengths below are in `m^-2`:

```text
qe_2495_t3_k1  qe_2506_t3_k1  qe_2518_t3_k1
qh_2529_t3_k1  qh_2532_t3_k1  qh_2544_t3_k1
qm_2549_t3_k1  qm_2554_t3_k1  qm_2559_t3_k1
```

Names encode element IDs, e.g. `qe_2495_t3_k1` → `QE.2495.T3`, parameter `k1`.
`LatticeLowLevelParams` gives omitted endpoints, reference, and strengths Python
defaults of `None`; this is not a usable complete optics configuration. Supply
finite strengths for the magnets used to construct the lattice. Its three
optional dechirper fields have defaults `false`, `500e-6 m`, and `10e-6 m`,
respectively. Low-level construction fixes the sextupole off; this JSON format
does not expose a sextupole strength. Unknown low-level keys fail when constructing
the parameter dataclass.

### `target_doocs_params`

An array of objects, one per configuration, mapping complete DOOCS channel names
to numeric setpoints. For example, the channel for a magnet is
`XFEL.MAGNETS/MAGNET.ML/QE.2495.T3/KICK_MRAD.SP`. Conversion uses the magnet
length `L` in metres:

```text
KICK_MRAD.SP = k1 * 1000 * L
k1 = KICK_MRAD.SP * 0.001 / L
```

The `setup` script selects this object by configuration ID and writes its channel
values. Analysis instead obtains the actual shot's setpoints from raw NPZ magnet
rows and converts them to low-level `k1` values. These are **setpoints**, not field
readbacks. Unrecorded non-magnet settings retain their target configuration values.

### Reader and consistency rules

`tool.configuration_document` requires nonempty, equally sized high/low arrays,
the same start/stop/energy for every configuration, matching high/low endpoints,
finite energy greater than electron rest energy (`0.00051099895 GeV`), and finite
values for low-level keys ending in `_k1`. This is a partial validator, not proof
of a complete, physically feasible, or matched lattice. Analysis does not require
or validate `target_doocs_params`, and does not verify target-vs-achieved optics.

The older `lattice.load_lattice_config_list` requires only `low_level_params`;
high-level and DOOCS arrays may be absent or null. Such files are not sufficient
for `tool` analysis. It constructs dataclasses directly and rejects unknown
parameter fields. `lattice.save_lattice_config_list` checks nonempty/equal list
lengths and derives DOOCS values from low-level parameters, but does not verify
that low-level settings achieve their high-level targets. `setup` expects the
DOOCS array and uses pandas to read the parallel arrays.

## Lattice generation: `lattice_generation.json`

```json
{
  "settings": {},
  "diagnostics": [
    {"R": [], "twiss": []}
  ]
}
```

The empty objects/arrays above indicate structure only. `settings` is the full
`lattice` settings object without an outer `lattice` wrapper. Each diagnostic
corresponds, in order, to one generated configuration and contains:

- `R`: its 6×6 transport matrix at the generation energy.
- `twiss`: an ordered array of objects with numeric `s` (m), `beta_x` (m),
  `beta_y` (m), `Dx` (m), `mux` (rad), and `muy` (rad). `s` and the phases are
  propagated Twiss coordinates; subtract the relevant initial values to get
  relative distance/phase advance.

This file is diagnostic output, not itself an app settings file. It is written
alongside `lattice_configs.json` and the non-JSON `lattice.pdf`.

## Simulation provenance

`sim_config.json` contains these required fields from its writer:

| Field | JSON type | Meaning |
| --- | --- | --- |
| `simulation` | object | Full simulation settings |
| `analysis` | object | Full analysis settings, including screen calibration |
| `source` | string | Absolute source experiment path |
| `lattice_sha256` | string | SHA-256 of the original source `lattice_configs.json` bytes |
| `dechirper_settings` | string | Explanatory text: source values retained unless explicitly overridden |

`source_lattice_configs.json` is a byte-for-byte copy of that source document.
The output `lattice_configs.json` is serialized after dechirper overrides and may
differ from the source, even when only JSON formatting changes. Consequently
`lattice_sha256` is the checksum of `source_lattice_configs.json`, not necessarily
the checksum of the output's active `lattice_configs.json`.

These files are written before particle tracking begins. Their presence alone
does not certify that every requested sample was generated successfully. Screen
samples are NPZ files under `samples_raw/`, with IDs beginning `MES_<config_id>_`.

## Processed export layout and publication

```text
processed_<local timestamp>-<uuid8>/
  config.json
  results.json
  lattice_configs.json
  tool
  projected/<config:02d>/<sample_id>.json
  sliced/<config:02d>/<slice:03d>/<sample_id>.json
  plots.png                       # when a figure was supplied
  plots.pdf                       # when a figure was supplied
  inputs/                         # only when raw inputs are included
    lattice_configs.json
    screen_background.npz         # only if used by processing
    samples_raw/<sample_id>.npz    # only accepted, nonduplicate shots
```

Numeric directory widths are minimum widths, not maximum ID lengths. Export
first builds a `.processing-*` staging directory, verifies unchanged input
signatures and copied-file hashes, and renames the directory only after success.
Newly arriving samples and excluded files may change without invalidating the
export; dependencies used by its snapshot must remain unchanged. The root
`lattice_configs.json` is a reserialization, whereas `inputs/lattice_configs.json`
retains the original bytes and matches the dependency checksum.

### Full report: `config.json`

Despite its filename, this is a complete analysis report, not an app settings
file. Unless marked optional below, fields are required by `Snapshot.parse_report`.

| Field | JSON type | Meaning |
| --- | --- | --- |
| `tool_version` | string | Currently `"1.0"` |
| `created_utc` | string | Snapshot creation time |
| `inventory_observed_utc` | string | Time the initial sample inventory was observed |
| `inventory` | array of strings | Raw `.npz` filenames observed at refresh start, including excluded files |
| `deferred_new_samples` | array of strings | Filenames that arrived after the initial inventory and were not processed in this snapshot |
| `experiment` | string | Absolute experiment path used for analysis |
| `settings` | object | Analysis settings, without an outer `analysis` wrapper |
| `energy_GeV` | number | Shared beam energy |
| `reference_start`, `reference_stop` | strings | Lattice endpoints for reconstruction |
| `design_twiss` | object or null; optional | Design reference described below |
| `design_twiss_error` | string or null; optional | Reason obtaining a design reference failed; null on success/no recorded error |
| `source_sha256` | object | Relative dependency filename → exact-byte checksum; includes `lattice_configs.json`, and background only when used |
| `units` | object of strings | Quantity → unit labels |
| `conventions` | object of strings | Recorded scientific and processing conventions |
| `samples` | object | Sample ID → inventory/status record |
| `shots` | array of objects | Accepted, deduplicated measurements described below |
| `fits` | array of objects | Fit records described below |
| `refresh_seconds` | number | Snapshot computation duration, s |
| `magnet_lengths` | object of numbers | Element ID → magnet length in m |
| `export` | object; optional in report type | Added to every current export; described below |
| `lattice_configurations` | object; optional in report type | Added to current exports; full lattice configuration document |
| `implementation_sha256` | string; optional | Hash of the exported `tool` source bytes |
| `runtime` | object of strings; optional | Python/NumPy versions plus available dependency versions |

`design_twiss`, when present and non-null, maps plane keys `"x"` and `"y"` to
objects containing `beta` (m) and `alpha` (dimensionless). It comes from
`make_twiss0(reference_start)` for live notebook optics. Older reports can omit
the reference; replay does not invent one. A design-reference failure does not
prevent emittance fitting, but mismatch is unavailable.

`export` contains `raw_inputs_included` (boolean), `folder` (published directory
basename), and `view` (object). Current GUI/CLI exports record `sample_id` and
`plane` (`"x"` or `"y"`) in `view`; an exporter without view data writes `{}`.
The view controls saved plot presentation, not the numerical fit schema.

`runtime` always includes `python` and `numpy`. The writer also looks up
`scipy`, `matplotlib`, `opencv-python`, and `ocelot-desy`, omitting distributions
not found under those exact names.

Current `units` keys are `variance` (`m^2`), `emittance` and
`normalized_emittance` (`m rad`), `beta` (`m`), `alpha` (`1`), `gamma` (`1/m`),
and `mismatch_zeta` (`1`). Current `conventions` entries are strings with these
keys and meanings:

| Key | Meaning |
| --- | --- |
| `image_axes` | `[x,y]` ordering and increasing-column slice order |
| `image_orientation` | Counterclockwise rotation, then horizontal and vertical flips; identical background transformation |
| `pixel_calibration` | Entered scales apply to oriented axes without automatic swapping |
| `roi_coordinates` | Half-open bounds in the oriented image |
| `magnet_source` | `KICK_MRAD.SP` setpoints, not measured fields |
| `negative_intensity` | Stored analysis negative-intensity policy value |
| `weighting` | Stored analysis weighting value |
| `chi_squared_optics` | Mean per-shot quadratic transport coefficients per configuration |
| `chi_squared_plot` | `chi_squared / sample_count`, rather than reduced chi-squared |
| `mismatch_zeta` | Mismatch formula and design reference source |
| `slice_alignment` | Equal intensity fractions per image, without absolute time calibration |
| `uncertainties` | Conditional statistical errors excluding calibration and optics systematics |

#### Sample status records: `samples[<sample_id>]`

`status` is one of `"pending"`, `"accepted"`, `"rejected"`, or `"duplicate"`.

| Status | Other current fields |
| --- | --- |
| `pending` | `reason`: unsettled-file explanation |
| `accepted` | `source_sha256`, integer `config_index`, and `slice_errors` object |
| `rejected` | `reason`; duplicates with conflicting optics also retain their hash, configuration, and slice errors |
| `duplicate` | Accepted-record fields plus `duplicate_of`, the retained sample ID |

Only accepted, nonduplicate shots enter `shots`. Identical images with matching
configuration and setpoints retain one shot; identical images under conflicting
optics reject all members of that group. A shot may be accepted while particular
projected/sliced fits fail; these failures are recorded in `slice_errors`.

#### Accepted measurement objects: `shots[]`

| Field | JSON type | Meaning / units |
| --- | --- | --- |
| `sample_id` | string | Raw filename stem, beginning `MES_<config_index>_` |
| `config_index` | integer | Configuration ID |
| `source_sha256` | string | Exact raw NPZ byte checksum |
| `image_sha256` | string | Checksum of `str(raw.shape).encode()` followed by contiguous float64 raw-image bytes |
| `recorded_setpoints` | object of numbers | Low-level parameter names → recorded `k1` values in `m^-2`, converted from DOOCS setpoints |
| `R` | 6×6 array | Transport matrix for target settings with recorded magnet values applied |
| `variances` | object of 2-number arrays | Region index string → `[variance_x, variance_y]`, both `m^2` |
| `slice_bounds` | array of integers | `num_slices+1` column boundaries relative to the oriented, cropped ROI |
| `slice_errors` | object of strings | Region index string → failure explanation |
| `negative_masked_pixel_fraction` | number | Fraction of masked pixels with negative intensity before clipping and ROI cropping |

Region keys use `"-1"` for projected and `"0"`, `"1"`, … for slices. Bounds are
half-open column intervals `[bounds[i],bounds[i+1])`. Invalid slicing can contain
repeated/placeholder boundaries; consult errors and variance availability rather
than assuming every interval is usable. The projected region can be a tail ROI
when `tail_sigma` is set; its tail-start value is not separately stored here.
No processed image pixels are embedded in JSON.

#### Fit records: `fits[]`

Every fit requires:

| Field | JSON type | Meaning |
| --- | --- | --- |
| `plane` | string | `"x"` or `"y"` |
| `slice_index` | integer | `-1` projected, otherwise slice ID |
| `status` | string | `"waiting"`, `"invalid"`, or `"valid"` |
| `sample_count` | integer | Shots with a usable variance for this region, before chi-squared groups lacking uncertainty are excluded |
| `excluded_slices` | object of strings | Sample ID → per-region error |
| `counts_per_configuration` | object of integers | Configuration ID string → usable-shot count |

Projected fits are emitted for x and y. When `num_slices>1`, additional sliced
fits are emitted for x only. `waiting` means insufficient configurations with
usable observations; `invalid` means attempted reconstruction failed checks.
At least three independent configurations are needed. Do not assume an invalid
fit lacks diagnostic numbers: failure can occur after solving the covariance.

The following diagnostics are optional and appear as their computation progresses:

| Field | JSON type | Meaning / units |
| --- | --- | --- |
| `reason` | string | Required for invalid fits; waiting fits may also include it |
| `unweighted_configurations` | array of integers | Chi-squared groups with fewer than two shots or no finite positive variance spread |
| `fit_configurations` | array of integers | Configuration ID for each observation, in observation order |
| `rank` | integer | Rank of the scaled regression matrix |
| `condition_number` | number or null | Scaled regression condition number; null for singular optics |
| `degrees_of_freedom` | integer | Number of observations minus three |
| `coefficients` | array of 3 numbers | Initial covariance terms `[sigma_uu, sigma_uu_prime, sigma_u_prime_u_prime]` |
| `coefficient_covariance` | 3×3 array or null | Covariance of fitted coefficients, in their product units |
| `observations` | array of numbers | Measured screen variances, `m^2` |
| `uncertainties` | array of numbers | Chi-squared observation standard deviations in `m^2`; linear mode stores unit weights `1.0` |
| `predictions` | array of numbers | Predicted screen variances, `m^2` |
| `residuals` | array of numbers | `(prediction-observation)/uncertainty`; dimensionless for chi-squared, unscaled variance residuals for linear unit weights |
| `r_squared` | number or null | Linear-regression coefficient of determination; null for zero observation variation |
| `chi_squared` | number | Chi-squared sum of squared normalized residuals |
| `reduced_chi_squared` | number or null | Chi-squared divided by degrees of freedom; null when degrees of freedom are not positive |

For chi-squared fits, each observation is one configuration's mean variance. Its
transport row is the **mean of each shot's quadratic transport coefficients**.
`shot_std` uses sample standard deviation (`ddof=1`); `standard_error` divides
that value by the square root of the group's shot count. Thus `sample_count` is
not necessarily the number of observations or the number of shots contributing
to the final fit. The GUI's chi-squared plot shows `chi_squared/sample_count`,
which differs from `reduced_chi_squared`.

Linear regression uses one observation per shot and that shot's transport map;
configuration IDs can repeat in `fit_configurations`. Coefficient covariance is
scaled by residual score/degrees of freedom, or null when that scale is undefined.

Valid fits require all of these reconstruction fields, even the nullable ones:

| Field | JSON type | Meaning / units |
| --- | --- | --- |
| `beta` | number | Initial beta, m |
| `alpha` | number | Initial alpha, dimensionless |
| `gamma` | number | `(1+alpha^2)/beta`, `1/m` |
| `emittance` | number | Geometric transverse emittance, m rad |
| `normalized_emittance` | number | `emittance * energy_GeV / 0.00051099895`, m rad |
| `transverse_covariance` | 2×2 array | Covariance in `(u,u')` for the fit's plane; entries have units `m^2`, `m rad`, `rad^2` |
| `errors` | object or null | One-sigma conditional errors: required numeric `beta`, `alpha`, `emittance` in the corresponding units |
| `normalized_emittance_error` | number or null | One-sigma normalized-emittance error, m rad |

`mismatch_zeta` is an optional dimensionless valid-fit field when a design
reference is available:

```text
gamma_design = (1 + alpha_design^2) / beta_design
zeta = (beta*gamma_design + gamma*beta_design - 2*alpha*alpha_design) / 2
```

Matched optics give zeta 1. Fit uncertainties are conditional statistical errors;
they exclude optics, calibration, setpoint/readback, and other systematic errors.
The fit does not separate dispersion contributions from transverse covariance.

### Convenience results: `results.json`

The root object has exactly the current writer's fields `fits` (the report's fit
array) and `units` (the report's unit object). It has no raw inputs or full
provenance and cannot replace `config.json` for replay.

### Processed projected and sliced measurement files

Both directory layouts use the same record:

| Field | JSON type | Meaning / units |
| --- | --- | --- |
| `sample_id` | string | Raw sample ID |
| `lattice_config_index` | integer | Configuration ID (named differently from `shots[].config_index`) |
| `slice_index` | integer | `-1` projected; zero-based for sliced records |
| `num_slices` | integer | **0 for projected files**; configured slice count for sliced files |
| `actual_doocs_lattice_params` | object of numbers | Recorded low-level `k1` values in `m^-2`, despite the DOOCS name; not raw channel values |
| `magnet_source` | string | `"KICK_MRAD.SP setpoints"` |
| `final_cov_xx` | number | Screen x variance, `m^2` |
| `final_cov_yy` | number | Screen y variance, `m^2` |
| `R` | 6×6 array | This shot's transport matrix |
| `source_sha256` | string | Raw NPZ checksum |
| `processing` | object | Full analysis settings, without outer `analysis` wrapper |

Only successfully processed regions have files. Their absence can mean a region
failed processing; consult `config.json` for status/errors. These records preserve
the notebook-style field names but are not reconstructed beam fit records.

### Replay requirements and compatibility

`tool replay <processed_directory>` reads and structurally validates `config.json`.
Declared report and fit fields are type-checked; declared floating-point fields
must be finite JSON numbers (integers are accepted as numbers, booleans are not).
Required fields and status-dependent valid/invalid fit requirements are enforced.
Unknown report/fit metadata is retained for compatibility. Nested acquisition
objects such as `shots` and `samples` have more permissive declared types and are
not exhaustively validated by this parser; matrix dimensions and scientific
validity are checked by the relevant processing code.

Replay additionally requires `export.raw_inputs_included=true`, the `inputs/`
files, and `lattice_configurations`. It verifies all archived shot byte hashes
and dependency hashes. The archived matrices are looked up by merged target
parameters and recorded setpoints; the energy must match `energy_GeV`. Archived
magnet lengths and design references are retained. If a report lacks design
Twiss, mismatch stays unavailable. Replay can use an overriding app settings file
but does not recompute the archived transport maps.

### Failed export marker: `INCOMPLETE.json`

```json
{
  "reason": "Export did not complete; not a published processing result"
}
```

This marker remains in a `.processing-*` directory after an export failure. It
does not contain the exception traceback or certify that other files are complete.
Readers should use successfully published `processed_*` bundles, not staging
directories.

## Non-JSON boundaries

Raw measurement files (`measure` and simulation), optional background images,
plots, and the exported `tool` source are not JSON protocols. JSON provenance
references the raw `samples_raw/MES_<config_index>_*.npz` files and optional
`screen_background.npz`; raw measurement NPZ files contain trusted local pickled
DOOCS response dictionaries, while background NPZ uses a numeric `data` array.
No independent beam-parameter JSON file, JSON Schema file, or saved lattice-preview
settings file is currently generated by the repository.
