#!/usr/bin/env python3
"""EuXFEL T3 optics-scan analysis, preparation, simulation, and inspection.

Disclaimer: This code is created with AI support.

Requires Python 3.13 or newer.

The Lattice Preview tab compares configuration phase advances on the initial
Twiss ellipses, using design optics at the configured lattice entrance by default.

Usage:
    python tool gui data/EXPERIMENT
    python tool analyze data/EXPERIMENT --output export_parent
    python tool settings --output settings.json
    python tool gui data/EXPERIMENT --settings settings.json
    python tool lattice data/NEW_EXPERIMENT --settings settings.json
    python tool simulate data/NEW_SIMULATION --source data/EXPERIMENT
    python tool inspect data/EXPERIMENT --sample MES_0_... --output image.png
    python tool replay data/EXPERIMENT/processed_TIMESTAMP
    python tool self-test

Paths on the command line are relative to the working directory.

Raw input is the trusted local NPZ format from 2_measure.py: screens and magnets
contain pickled DOOCS response dictionaries. Images use [x pixel, y pixel]; y
is the slicing/streak coordinate. No time calibration is inferred. Magnet
KICK_MRAD.SP values are SETPOINTS, converted to k1 = kick * 1e-3 / length.
Unrecorded lattice parameters retain the configuration's targets.

Scientific conventions:
* widths are intensity RMS or Gaussian fits, with independent x/y calibration;
* slices are equal integrated intensity bins in increasing image-column order;
* chi-squared weights default to shot-to-shot sample standard deviation (ddof=1),
  as in the notebooks, NOT the standard error of the mean;
* chi-squared optics use mean per-shot quadratic transport coefficients per
  configuration; linear regression uses each shot's R;
* reconstruction fits the three transverse covariance terms; it needs three
  independent optics configurations and does not separate dispersion contributions;
* uncertainties are conditional on the optics, pixel scales, mask and model.
  They do not include calibration, setpoint/readback, or systematic errors;
* normalized emittance uses E / electron rest energy, as in the notebooks.

The watcher defers unsettled files, retries failed reads, invalidates changed or
deleted samples, and hashes all loaded dependencies. Exports are atomic JSON
snapshots containing settings, source hashes, exclusions, and fit diagnostics.
It never deletes or rewrites acquired or notebook-processed data. Optional
preparation commands only create new experiment directories.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import math
import os
import queue
import re
import sys
import tempfile
import textwrap
import time
import tkinter as tk
import unittest
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import MethodType, NoneType, UnionType
from typing import (
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TypeAliasType,
    TypedDict,
    cast,
    get_args,
    get_origin,
    is_typeddict,
    runtime_checkable,
)
from unittest.mock import Mock, patch
from uuid import uuid4

import cv2
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Rectangle
import numpy as np
from numpy.typing import NDArray
from ocelot import MagneticLattice, SecondTM, Twiss, track, twiss
from ocelot.cpbd.matcher import MatchProblem, Target, TargetReport
from scipy import ndimage
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

# Notebook helpers must also resolve when this file is loaded as a module.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from beam import BeamParameters, make_beam_with_random_gaussian_mixture_current_profile
from lattice import (
    LatticeLowLevelParams,
    make_lattice,
    make_lattice_and_navigator,
    make_twiss0,
    show_lattice,
)
from ocelot_util import (
    get_lattice_element,
    get_lattice_param,
    make_sublattice,
    set_lattice_param,
)
import t3

type Array = NDArray[np.float64]
type Signature = tuple[int, int, int, int]
REST_ENERGY_GEV = 0.00051099895
VERSION = "1.0"


class FitType(StrEnum):
    RMS = "RMS"
    GAUSSIAN = "Gaussian"
    GAUSSIAN_FREE_AMPLITUDE = "Gaussian-with-free-amplitude"

    @property
    def label(self) -> str:
        match self:
            case FitType.GAUSSIAN_FREE_AMPLITUDE:
                return "Gaussian (free amplitude)"
            case _:
                return self.value


class RegressionType(StrEnum):
    CHI_SQUARED = "chi_squared"
    LINEAR = "linear"

    @property
    def label(self) -> str:
        return {
            self.CHI_SQUARED: "Chi-squared",
            self.LINEAR: "Linear least squares",
        }[self]


class Weighting(StrEnum):
    SHOT_STD = "shot_std"
    STANDARD_ERROR = "standard_error"

    @property
    def label(self) -> str:
        match self:
            case Weighting.SHOT_STD:
                return "Shot-to-shot standard deviation"
            case Weighting.STANDARD_ERROR:
                return "Standard error of the mean"


class MaskFilterType(StrEnum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"

    @property
    def label(self) -> str:
        return self.value.capitalize()


class ImageRotation(IntEnum):
    NONE = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270

    @property
    def label(self) -> str:
        match self:
            case ImageRotation.NONE:
                return "0° (unchanged)"
            case _:
                return f"{self.value}° counterclockwise"


class NegativeIntensityPolicy(StrEnum):
    PROFILE_THRESHOLD = "profile_threshold"
    CLIP = "clip"
    REJECT = "reject"

    @property
    def label(self) -> str:
        return {
            self.PROFILE_THRESHOLD: "Threshold summed profiles",
            self.CLIP: "Clip negative pixels to zero",
            self.REJECT: "Reject negative pixels",
        }[self]


class Plane(StrEnum):
    X = "x"
    Y = "y"

    @property
    def matrix_index(self) -> int:
        match self:
            case Plane.X:
                return 0
            case Plane.Y:
                return 2

    @property
    def variance_index(self) -> int:
        match self:
            case Plane.X:
                return 0
            case Plane.Y:
                return 1


class FitStatus(StrEnum):
    WAITING = "waiting"
    INVALID = "invalid"
    VALID = "valid"


class FitErrors(TypedDict):
    """Conditional standard deviations of the reconstructed beam parameters."""

    beta: float  # m
    alpha: float  # dimensionless
    emittance: float  # m rad


class FitResult(TypedDict):
    """Fit identity, diagnostics, and optional reconstructed beam parameters.

    Beam parameters are present for valid fits. The report loader checks these
    status-dependent requirements while keeping one shared result type.
    """

    plane: Plane
    slice_index: int
    status: FitStatus
    sample_count: int
    excluded_slices: dict[str, str]
    counts_per_configuration: dict[str, int]
    reason: NotRequired[str]
    unweighted_configurations: NotRequired[list[int]]
    fit_configurations: NotRequired[list[int]]
    rank: NotRequired[int]
    condition_number: NotRequired[float | None]
    degrees_of_freedom: NotRequired[int]
    coefficients: NotRequired[list[float]]
    coefficient_covariance: NotRequired[list[list[float]] | None]
    observations: NotRequired[list[float]]
    uncertainties: NotRequired[list[float]]
    predictions: NotRequired[list[float]]
    residuals: NotRequired[list[float]]
    r_squared: NotRequired[float | None]
    chi_squared: NotRequired[float]
    reduced_chi_squared: NotRequired[float | None]
    beta: NotRequired[float]  # m
    alpha: NotRequired[float]  # dimensionless
    gamma: NotRequired[float]  # 1/m
    emittance: NotRequired[float]  # m rad
    normalized_emittance: NotRequired[float]  # m rad
    transverse_covariance: NotRequired[list[list[float]]]
    errors: NotRequired[FitErrors | None]
    normalized_emittance_error: NotRequired[float | None]  # m rad
    mismatch_zeta: NotRequired[float]  # dimensionless


class TwissReference(TypedDict):
    """Design beta and alpha at the reconstruction start, for one plane."""

    beta: float  # m
    alpha: float  # dimensionless


type DesignTwiss = dict[str, TwissReference]


class SnapshotReport(TypedDict):
    """Typed reconstruction fields alongside extensible acquisition metadata."""

    tool_version: str
    created_utc: str
    inventory_observed_utc: str
    inventory: list[str]
    deferred_new_samples: list[str]
    experiment: str
    settings: dict[str, Any]
    energy_GeV: float
    reference_start: str
    reference_stop: str
    design_twiss: NotRequired[DesignTwiss | None]
    design_twiss_error: NotRequired[str | None]
    source_sha256: dict[str, str]
    units: dict[str, str]
    conventions: dict[str, str]
    samples: dict[str, Any]
    shots: list[dict[str, Any]]
    fits: list[FitResult]
    refresh_seconds: float
    magnet_lengths: dict[str, float]
    export: NotRequired[dict[str, Any]]
    lattice_configurations: NotRequired[dict[str, Any]]
    implementation_sha256: NotRequired[str]
    runtime: NotRequired[dict[str, str]]


class OpticsProvider(Protocol):
    """The minimal optics interface needed by reconstruction.

    Structural typing lets notebook optics, archived matrices, and test providers
    share this interface without inheriting a common implementation. Keeping
    Ocelot objects behind it makes analysis and replay independently testable.
    """

    @property
    def lengths(self) -> Mapping[str, float]:
        """Nominal magnet lengths in metres; consumers only read this mapping."""
        ...

    def matrix(
        self, target: dict[str, Any], recorded: dict[str, float], energy: float
    ) -> Array:
        """Return the 6×6 transport matrix using recorded setpoints and energy in GeV."""
        ...


@runtime_checkable
class DesignTwissProvider(Protocol):
    """An optional reference interface used only for mismatch plots.

    A provider can reconstruct emittance without a design reference. Separating
    this capability avoids inventing nominal optics for older archives; returning
    None explicitly means that mismatch is unavailable.
    """

    def design_twiss(self, start: str) -> DesignTwiss | None:
        """Return design Twiss at the requested start, or None when it was not recorded."""
        ...


class SampleStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class WorkerOperation(StrEnum):
    REFRESH = "refresh"
    PROCESSING_PREVIEW = "processing_preview"
    LATTICE_PREVIEW = "lattice_preview"
    EXPORT = "export"
    PREPARATION = "preparation"


class Command(StrEnum):
    GUI = "gui"
    ANALYZE = "analyze"
    INSPECT = "inspect"
    LATTICE = "lattice"
    SIMULATE = "simulate"
    SETTINGS = "settings"
    SELF_TEST = "self-test"
    REPLAY = "replay"


REGRESSION_CHOICES = {method.label: method for method in RegressionType}
type AnalysisChoice = (
    ImageRotation
    | FitType
    | RegressionType
    | Weighting
    | MaskFilterType
    | NegativeIntensityPolicy
)
ANALYSIS_ENUM_TYPES: dict[str, type[AnalysisChoice]] = {
    "rotation": ImageRotation,
    "fit_type": FitType,
    "tail_fit_type": FitType,
    "regression": RegressionType,
    "weighting": Weighting,
    "mask_filter": MaskFilterType,
    "negative_intensity": NegativeIntensityPolicy,
}


def parse_slice_count(value: str) -> int:
    """Accept the projected preset or an arbitrary positive integer."""
    value = value.strip()
    count = 1 if value == "1 = projected" else int(value)
    if count < 1:
        raise ValueError("Number of slices must be a positive integer (1 = projected)")
    return count


@dataclass(frozen=True)
class Settings:
    """Shared JSON conversion; each settings class names its section in app files."""

    section: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        """Return independent field values suitable for JSON export."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: object) -> Self:
        """Restore tuple fields and reject unknown keys before constructing settings."""
        if not isinstance(values, dict):
            raise ValueError(f"{cls.section} settings must be an object")
        settings_fields = {field.name: field for field in fields(cls)}
        unknown = values.keys() - settings_fields.keys()
        if unknown:
            raise ValueError(f"Unknown {cls.section} settings: {sorted(unknown)}")
        restored = {
            name: (
                tuple(value)
                if get_origin(settings_fields[name].type) is tuple
                else value
            )
            for name, value in values.items()
        }
        return cls(**restored)

    @classmethod
    def load(cls, path: Path | None) -> Self:
        """Read this settings section from an app file, or use defaults without a file."""
        if path is None:
            return cls()
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or cls.section not in document:
            raise ValueError(
                f"Settings file needs a '{cls.section}' object; use 'tool settings'"
            )
        return cls.from_dict(document[cls.section])


@dataclass(frozen=True)
class AnalysisSettings(Settings):
    section: ClassVar[str] = "analysis"

    pixel_size_x: float = 3.45e-6
    pixel_size_y: float = 3.45e-6
    rotation: ImageRotation = ImageRotation.NONE
    flip_horizontal: bool = False
    flip_vertical: bool = False
    num_slices: int = 20
    fit_type: FitType = FitType.RMS
    regression: RegressionType = RegressionType.CHI_SQUARED
    weighting: Weighting = Weighting.SHOT_STD
    mask_downsampling: int = 8
    mask_filter: MaskFilterType = MaskFilterType.GAUSSIAN
    mask_filter_size: int = 9
    mask_sigma: float = 20.0
    mask_cutoff: float = 0.01
    mask_dilation: int = 0
    largest_patch: bool = True
    tail_extension: bool = False
    subtract_background: bool = False
    subtract_inverse_mask_mean: bool = False
    negative_intensity: NegativeIntensityPolicy = (
        NegativeIntensityPolicy.PROFILE_THRESHOLD
    )
    # Half-open bounds in the oriented image [x_start, x_stop, y_start, y_stop].
    roi: tuple[int | None, ...] = (None, None, None, None)
    # Optional projected tail: begins at mu_y + tail_sigma * sigma_y.
    tail_sigma: float | None = None
    tail_fit_type: FitType = FitType.GAUSSIAN
    settle_seconds: float = 1.0
    poll_seconds: float = 30.0
    max_condition: float = 1e10

    def __post_init__(self) -> None:
        """Normalize JSON strings while preserving their serialized values."""
        for name, enum_type in ANALYSIS_ENUM_TYPES.items():
            object.__setattr__(self, name, enum_type(getattr(self, name)))

    def validate(self) -> None:
        """Reject invalid processing values before they can affect scientific results."""
        for name in (
            "flip_horizontal",
            "flip_vertical",
            "largest_patch",
            "tail_extension",
            "subtract_background",
            "subtract_inverse_mask_mean",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a JSON boolean")
        for name in (
            "pixel_size_x",
            "pixel_size_y",
            "mask_sigma",
            "poll_seconds",
            "max_condition",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("num_slices", "mask_downsampling", "mask_filter_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.mask_dilation) is not int or self.mask_dilation < 0:
            raise ValueError("mask_dilation must be a nonnegative integer")
        if not math.isfinite(self.settle_seconds) or self.settle_seconds < 0:
            raise ValueError("settle_seconds must be finite and nonnegative")
        if not math.isfinite(self.mask_cutoff) or not 0 <= self.mask_cutoff <= 1:
            raise ValueError("mask_cutoff must be in [0, 1]")
        if len(self.roi) != 4 or any(
            v is not None and (type(v) is not int or v < 0) for v in self.roi
        ):
            raise ValueError("roi needs four nonnegative pixel bounds or null")
        if self.tail_sigma is not None:
            if (
                not math.isfinite(self.tail_sigma)
                or self.tail_sigma < 0
                or self.num_slices != 1
            ):
                raise ValueError(
                    "tail_sigma requires a finite nonnegative value and num_slices=1"
                )

    def to_form_values(self) -> dict[str, str | bool]:
        """Represent settings as editable text, choice labels, and checkboxes."""
        values: dict[str, str | bool] = {}
        for _, group in PROCESSING_GROUPS:
            for field in group:
                if field.name in ROI_FIELDS:
                    bound = self.roi[ROI_FIELDS.index(field.name)]
                    values[field.name] = "" if bound is None else str(bound)
                    continue
                value = getattr(self, field.name)
                if field.name in PROCESSING_CHOICES:
                    values[field.name] = value.label
                elif isinstance(value, bool):
                    values[field.name] = value
                else:
                    values[field.name] = "" if value is None else str(value)
        return values

    @classmethod
    def parse_form_values(cls, values: dict[str, str | bool]) -> dict[str, Any]:
        """Parse processing controls into field values, excluding slices and regression.

        Call validate() on the completed settings to check scientific limits.
        """
        parsed: dict[str, Any] = {}
        roi: list[int | None] = []
        for _, group in PROCESSING_GROUPS:
            for field in group:
                value = values[field.name]
                try:
                    if field.name in ROI_FIELDS:
                        roi.append(
                            int(str(value).strip()) if str(value).strip() else None
                        )
                    elif field.name in CAMERA_RESOLUTION_PRESETS:
                        parsed[field.name] = parse_strictly_positive_finite_float(
                            str(value)
                        )
                    elif field.name in PROCESSING_CHOICES:
                        choices = PROCESSING_CHOICES[field.name]
                        parsed[field.name] = choices[str(value)]
                    elif cls.__annotations__[field.name] is bool:
                        if type(value) is not bool:
                            raise ValueError("expected a checkbox value")
                        parsed[field.name] = value
                    elif field.name == "tail_sigma":
                        parsed[field.name] = (
                            float(str(value)) if str(value).strip() else None
                        )
                    elif cls.__annotations__[field.name] is int:
                        parsed[field.name] = int(str(value))
                    else:
                        parsed[field.name] = float(str(value))
                except (ValueError, KeyError) as error:
                    raise ValueError(
                        f"{field.label}: invalid value {value!r}"
                    ) from error
        parsed["roi"] = tuple(roi)
        return parsed


@dataclass(frozen=True)
class ProcessingField:
    """Labels and numeric steps for one control in the processing form."""

    name: str
    label: str
    minimum: float = 0
    increment: float = 1
    maximum: float = 1e30


ROI_FIELDS = ("roi_x_start", "roi_x_stop", "roi_y_start", "roi_y_stop")
CAMERA_RESOLUTION_PRESETS = {
    "pixel_size_x": ("3.45e-6", "5.535e-6"),
    "pixel_size_y": ("3.45e-6", "5.521e-6"),
}


def parse_strictly_positive_finite_float(value: str) -> float:
    """Accept a positive finite real number."""
    resolution = float(value)
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Must be finite and strictly positive")
    return resolution


PROCESSING_CHOICES = {
    name: {choice.label: choice for choice in enum_type}
    for name, enum_type in ANALYSIS_ENUM_TYPES.items()
    if name != "regression"
}
PROCESSING_GROUPS = (
    (
        "Camera calibration — updates all plots automatically",
        (
            ProcessingField("pixel_size_x", "x resolution [m/pixel]"),
            ProcessingField("pixel_size_y", "y resolution [m/pixel]"),
        ),
    ),
    (
        "Image orientation — rotate, then flip (before processing)",
        (
            ProcessingField("rotation", "Rotation"),
            ProcessingField("flip_horizontal", "Flip horizontally (left ↔ right)"),
            ProcessingField("flip_vertical", "Flip vertically (bottom ↔ top)"),
        ),
    ),
    (
        "Profile fit and statistics",
        (
            ProcessingField("fit_type", "Profile fit"),
            ProcessingField("weighting", "Statistical weighting"),
        ),
    ),
    (
        "Signal mask",
        (
            ProcessingField("mask_filter", "Smoothing filter"),
            ProcessingField("mask_downsampling", "Downsampling factor", minimum=1),
            ProcessingField("mask_filter_size", "Uniform filter size [px]", minimum=1),
            ProcessingField(
                "mask_sigma", "Gaussian sigma [px]", minimum=1e-12, increment=0.1
            ),
            ProcessingField(
                "mask_cutoff", "Cutoff / peak intensity", increment=0.001, maximum=1
            ),
            ProcessingField("mask_dilation", "Dilation iterations (0 = off)"),
            ProcessingField("largest_patch", "Keep largest connected region"),
            ProcessingField(
                "tail_extension", "Tail extension (extend mask along +y and box it)"
            ),
        ),
    ),
    (
        "Background and negative intensity",
        (
            ProcessingField("subtract_background", "Subtract screen_background.npz"),
            ProcessingField("subtract_inverse_mask_mean", "Subtract inverse-mask mean"),
            ProcessingField("negative_intensity", "Negative intensity policy"),
        ),
    ),
    (
        "Oriented image region — blank bounds use the full image",
        (
            ProcessingField("roi_x_start", "x start [px]"),
            ProcessingField("roi_x_stop", "x stop [px] (excluded)"),
            ProcessingField("roi_y_start", "y start [px]"),
            ProcessingField("roi_y_stop", "y stop [px] (excluded)"),
        ),
    ),
    (
        "Tail selection — requires 1 = projected",
        (
            ProcessingField(
                "tail_sigma", "Tail start [sigma] (blank = off)", increment=0.1
            ),
            ProcessingField("tail_fit_type", "Head profile fit"),
        ),
    ),
    (
        "Live refresh and fit quality",
        (
            ProcessingField("settle_seconds", "File settling time [s]", increment=0.1),
            ProcessingField(
                "poll_seconds", "Refresh interval [s]", minimum=0.1, increment=0.1
            ),
            ProcessingField(
                "max_condition", "Maximum condition number", minimum=1, increment=1e8
            ),
        ),
    ),
)


@dataclass(frozen=True)
class LatticeSettings(Settings):
    section: ClassVar[str] = "lattice"

    beam_energy: float = 16.3
    start: str = "STSEC.2473.T3"
    stop: str = "OTRC.2560.T3"
    phase_start: str = "STSEC.2473.T3"
    phases: tuple[float, ...] = tuple(
        float(v) for v in np.linspace(0, np.pi / 2, 8, endpoint=False) - np.pi / 16
    )
    dispersion_x: float = 0.0
    r34: float | None = -36.0
    max_beta: float = 600.0
    min_screen_beta_x: float = 20.0
    # Ocelot solver identifiers remain extensible across dependency versions.
    solver: str = "nelder-mead"
    max_iter: int = 1000


@dataclass(frozen=True)
class SimulationSettings(Settings):
    section: ClassVar[str] = "simulation"

    samples_per_config: int = 5
    num_particles: int = 200_000
    normalized_emittance: float = 1e-6  # m rad
    charge: float = 250e-12
    sigma_tau: float = 10e-6
    sigma_p: float = 0.0
    chirp: float = 0.0
    twiss_noise: float = 0.0
    screen_shape: tuple[int, int] = (3000, 4096)
    screen_noise: float = 0.0
    screen_offset: float = 0.0
    screen_gain: float = 1.0
    screen_gamma: float = 1.0
    seed: int = 0
    # None retains source values; explicit values override BOTH configuration lists.
    dechirper_enabled: bool | None = None
    dechirper_dist_y: float | None = None
    dechirper_sigma_z: float | None = None


def signature(path: Path) -> Signature:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def stable_bytes(path: Path) -> tuple[bytes, Signature]:
    before = signature(path)
    content = path.read_bytes()
    if signature(path) != before:
        raise ValueError(f"File changed during reading: {path.name}")
    return content, before


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def atomic_json(path: Path, document: Any) -> None:
    """Write finite JSON to a temporary file and atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def configuration_document(content: bytes) -> dict[str, Any]:
    """Validate shared lattice endpoints, energy, and finite quadrupole strengths."""
    document = json.loads(content)
    low, high = document["low_level_params"], document["high_level_params"]
    if not low or len(low) != len(high):
        raise ValueError(
            "Lattice configuration lists must be nonempty and equally long"
        )
    reference = None
    for ll, hl in zip(low, high):
        energy = float(hl["beam_energy"])
        if not math.isfinite(energy) or energy <= REST_ENERGY_GEV:
            raise ValueError("Invalid lattice beam_energy in GeV")
        extent = ll["start"], ll["stop"], energy
        if reference is not None and extent != reference:
            raise ValueError(
                "All configurations must share start, stop and beam energy"
            )
        if extent[:2] != (hl["start"], hl["stop"]):
            raise ValueError("Low/high level lattice endpoints disagree")
        for key, value in ll.items():
            if key.endswith("_k1") and not math.isfinite(float(value)):
                raise ValueError(f"Nonfinite lattice parameter {key}")
        reference = extent
    return document


class NotebookOptics:
    """Use the same optics/tracking helpers as the notebooks."""

    def __init__(self) -> None:
        """Initialize nominal magnet lengths and the design-reference cache."""
        self.parameter_type = LatticeLowLevelParams
        self.make_lattice = make_lattice
        self.make_twiss0 = make_twiss0
        self.design_references: dict[str, DesignTwiss] = {}
        self.lengths = {e.id: float(e.l) for e in t3.cell if hasattr(e, "id")}

    def matrix(
        self, target: dict[str, Any], recorded: dict[str, float], energy: float
    ) -> Array:
        lattice = self.make_lattice(self.parameter_type(**(target | recorded)))
        return np.asarray(lattice.transfer_maps(energy=energy)[1], dtype=float)

    def design_twiss(self, start: str) -> DesignTwiss:
        """Use the notebook's nominal Twiss reference at the reconstruction start."""
        if start not in self.design_references:
            twiss = self.make_twiss0(start)
            self.design_references[start] = {
                plane.value: {
                    "beta": float(getattr(twiss, f"beta_{plane}")),
                    "alpha": float(getattr(twiss, f"alpha_{plane}")),
                }
                for plane in Plane
            }
        return self.design_references[start]


class ArchivedOptics:
    """Replay an export using its exact per-shot transfer matrices and lengths."""

    def __init__(self, document: SnapshotReport) -> None:
        """Index archived transfer matrices and retain the recorded energy and design reference."""
        self.lengths: dict[str, float] = document["magnet_lengths"]
        self.matrices: dict[str, Array] = {}
        lattice_configurations = document.get("lattice_configurations")
        if lattice_configurations is None:
            raise ValueError("Archived optics require lattice configurations")
        configurations = lattice_configurations["low_level_params"]
        for shot in document["shots"]:
            key = json.dumps(
                configurations[shot["config_index"]] | shot["recorded_setpoints"],
                sort_keys=True,
            )
            self.matrices[key] = np.asarray(shot["R"], dtype=float)
        self.energy = document["energy_GeV"]
        self.reference_start = document["reference_start"]
        self.design_reference: DesignTwiss | None = document.get("design_twiss")

    def design_twiss(self, start: str) -> DesignTwiss | None:
        """Retain the recorded reference; older archives may have no design Twiss."""
        if start != self.reference_start:
            raise ValueError("Archived design Twiss require the archived start")
        return self.design_reference

    def matrix(
        self, target: dict[str, Any], recorded: dict[str, float], energy: float
    ) -> Array:
        if energy != self.energy:
            raise ValueError("Archived optics require the archived beam energy")
        key = json.dumps(target | recorded, sort_keys=True)
        if key not in self.matrices:
            raise ValueError("No archived transfer matrix for this shot's setpoints")
        return self.matrices[key].copy()


def magnet_parameters(
    rows: Any, target: dict[str, Any], lengths: Mapping[str, float]
) -> dict[str, float]:
    """Require all quadrupoles in the target to have a recorded setpoint."""
    required = {key for key in target if key.endswith("_k1")}
    recorded: dict[str, float] = {}
    for row in rows:
        if len(row) != 5:
            raise ValueError("Expected five fields per DOOCS magnet row")
        eid = row[4].decode() if isinstance(row[4], bytes) else str(row[4])
        key = eid.lower().replace(".", "_") + "_k1"
        if key not in required:
            continue
        length, kick = lengths.get(eid, 0), float(row[1])
        if length <= 0 or not math.isfinite(kick):
            raise ValueError(f"Invalid length or magnet setpoint for {eid}")
        if key in recorded:
            raise ValueError(f"Repeated magnet entry for {eid}")
        recorded[key] = kick * 1e-3 / length
    if missing := required - recorded.keys():
        raise ValueError(f"Missing recorded magnet setpoints: {sorted(missing)}")
    return recorded


def make_mask(image: Array, settings: AnalysisSettings) -> NDArray[np.bool_]:
    """Threshold a smoothed image, extend its signal region, and restore its size."""
    factor = settings.mask_downsampling

    def prepare_filter_input() -> Array:
        """Downsample for mask construction and normalize by the positive peak."""
        work = image
        if factor > 1:
            shape = image.shape[1] // factor, image.shape[0] // factor
            if min(shape) < 1:
                raise ValueError("Image is smaller than mask_downsampling")
            # OpenCV preserves this float64 input's dtype when resizing.
            work = cast(Array, cv2.resize(image, shape, interpolation=cv2.INTER_AREA))
        peak = float(work.max())
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError("Screen image has no positive signal")
        return work / peak

    def threshold_signal(work: Array) -> NDArray[np.bool_]:
        """Apply the chosen smoothing filter and reject an empty signal mask."""
        match settings.mask_filter:
            case MaskFilterType.GAUSSIAN:
                filtered = ndimage.gaussian_filter(
                    work, sigma=settings.mask_sigma / factor
                )
            case MaskFilterType.UNIFORM:
                filtered = ndimage.uniform_filter(
                    work, size=max(1, settings.mask_filter_size // factor)
                )
        mask = filtered >= settings.mask_cutoff
        if not mask.any():
            raise ValueError("Mask is empty; adjust mask_cutoff")
        return mask

    def extend_signal_region(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Dilate, retain the largest patch, and optionally extend its tail."""
        structure = ndimage.generate_binary_structure(2, 1)
        if settings.mask_dilation > 0:
            mask = ndimage.binary_dilation(
                mask, structure=structure, iterations=settings.mask_dilation
            )
        if settings.largest_patch:
            labels, _ = ndimage.label(mask, structure=structure)
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            mask = labels == sizes.argmax()
        if settings.tail_extension:
            rows, cols = np.where(mask)
            mask[rows.min() : rows.max() + 1, cols.min() :] = True
        return mask

    work = prepare_filter_input()
    mask = extend_signal_region(threshold_signal(work))
    if factor > 1:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def profile_fit(profile: Array, fit_type: FitType | str) -> tuple[float, float]:
    """Return the profile mean and variance in pixels using the chosen method."""
    fit_type = FitType(fit_type)

    def rms_moments() -> tuple[float, float]:
        """Threshold the summed profile as in the notebook, then take moments."""
        weights = np.maximum(profile, 0)
        total = float(weights.sum())
        if profile.size == 0 or not math.isfinite(total) or total <= 0:
            raise ValueError("Profile is empty or contains no signal")
        x = np.arange(profile.size, dtype=float)
        mu = float(x @ weights / total)
        variance = float((x - mu) ** 2 @ weights / total)
        if variance < 0:
            raise ValueError("Profile has zero width")
        return mu, variance

    def gaussian_moments(mu: float, variance: float) -> tuple[float, float]:
        """Use RMS moments to initialize the bounded Gaussian profile fit."""
        if variance == 0:
            raise ValueError("Profile has zero width")
        x = np.arange(profile.size, dtype=float)
        nonnegative = profile >= 0
        normalized = profile / profile.max()

        def gaussian(x: Array, mean: float, sigma: float) -> Array:
            return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

        model: Callable[..., Array]
        initial: list[float]
        bounds: tuple[list[float], list[float]]
        match fit_type:
            case FitType.GAUSSIAN:
                model = gaussian
                initial = [mu, np.sqrt(variance)]
                bounds = ([0, 1e-12], [x.size, x.size])
            case FitType.GAUSSIAN_FREE_AMPLITUDE:

                def scaled_gaussian(
                    x: Array, amplitude: float, mean: float, sigma: float
                ) -> Array:
                    return amplitude * gaussian(x, mean, sigma)

                model = scaled_gaussian
                initial = [1, mu, np.sqrt(variance)]
                bounds = ([1e-3, 0, 1e-12], [1e3, x.size, x.size])
        parameters, _ = curve_fit(
            model, x[nonnegative], normalized[nonnegative], p0=initial, bounds=bounds
        )
        mu, sigma = parameters[-2:]
        return float(mu), float(sigma**2)

    mu, variance = rms_moments()
    match fit_type:
        case FitType.GAUSSIAN | FitType.GAUSSIAN_FREE_AMPLITUDE:
            mu, variance = gaussian_moments(mu, variance)
    if not np.isfinite([mu, variance]).all():
        raise ValueError("Nonfinite profile fit")
    return float(mu), float(variance)


@dataclass
class ProcessedImage:
    image: Array
    mask: NDArray[np.bool_]
    bounds: list[int]
    variances: dict[int, tuple[float, float]]
    slice_errors: dict[int, str]
    roi_origin: tuple[int, int]
    tail_start: int | None
    negative_fraction: float


def orient_image(image: Array, settings: AnalysisSettings) -> Array:
    """Rotate counterclockwise in the origin='lower' display, then flip its axes."""
    # Positive np.rot90 turns appear clockwise when row zero is displayed below.
    oriented = np.rot90(image, k=-(settings.rotation.value // 90))
    if settings.flip_horizontal:
        oriented = np.fliplr(oriented)
    if settings.flip_vertical:
        oriented = np.flipud(oriented)
    return oriented


def process_image(
    raw: Array, settings: AnalysisSettings, background: Array | None = None
) -> ProcessedImage:
    """Orient images, subtract background, mask, crop, and fit beam profiles."""

    def subtract_background() -> Array:
        if raw.ndim != 2 or min(raw.shape) < 2 or not np.isfinite(raw).all():
            raise ValueError("Expected a finite 2D image in [x, y] order")
        work = orient_image(np.asarray(raw, dtype=float), settings).copy()
        if background is not None:
            if background.shape != raw.shape or not np.isfinite(background).all():
                raise ValueError(
                    "Background and screen shapes must agree and be finite"
                )
            work -= orient_image(background, settings)
        return work

    def filter_signal(work: Array) -> tuple[Array, NDArray[np.bool_], float]:
        """Build the mask, estimate residual background, and apply the negative-pixel policy."""
        mask = make_mask(work, settings)
        if settings.subtract_inverse_mask_mean:
            if mask.all():
                raise ValueError("Cannot estimate background: inverse mask is empty")
            work -= float(work[~mask].mean())
        negative_fraction = float(
            np.count_nonzero((work < 0) & mask) / np.count_nonzero(mask)
        )
        match settings.negative_intensity:
            case NegativeIntensityPolicy.REJECT if negative_fraction > 0:
                raise ValueError("Negative intensity remains within the mask")
            case NegativeIntensityPolicy.CLIP:
                work = np.maximum(work, 0)
        work = work * mask
        return work, mask, negative_fraction

    def crop_roi(
        work: Array,
        mask: NDArray[np.bool_],
    ) -> tuple[Array, NDArray[np.bool_], tuple[int, int]]:
        x0, x1, y0, y1 = settings.roi
        height, width = work.shape
        x0, x1 = 0 if x0 is None else x0, height if x1 is None else x1
        y0, y1 = 0 if y0 is None else y0, width if y1 is None else y1
        if not (0 <= x0 < x1 <= height and 0 <= y0 < y1 <= width):
            raise ValueError("ROI is empty or outside the image")
        return work[x0:x1, y0:y1], mask[x0:x1, y0:y1], (x0, y0)

    def find_tail_start(profile: Array) -> int | None:
        tail_start = None
        if settings.tail_sigma is not None:
            mu, var = profile_fit(profile, settings.tail_fit_type)
            tail_start = int(mu + settings.tail_sigma * np.sqrt(var))
            if tail_start >= profile.size - 1:
                raise ValueError("Tail ROI contains fewer than two columns")
        return tail_start

    def find_slice_boundaries(profile: Array) -> tuple[list[int], bool]:
        """Locate equal-charge columns and flag profiles that cannot be sliced monotonically."""
        cumulative = np.cumsum(profile)
        invalid_slicing = bool(
            settings.num_slices > 1 and (cumulative[-1] <= 0 or np.any(profile < 0))
        )
        if invalid_slicing:
            bounds = [0] * settings.num_slices + [profile.size]
        else:
            bounds = (
                [0]
                + [
                    int(
                        np.searchsorted(
                            cumulative, cumulative[-1] * i / settings.num_slices
                        )
                    )
                    for i in range(1, settings.num_slices)
                ]
                + [profile.size]
            )
        return bounds, invalid_slicing

    def fit_regions(
        image: Array,
        bounds: list[int],
        tail_start: int | None,
        invalid_slicing: bool,
    ) -> tuple[dict[int, tuple[float, float]], dict[int, str]]:
        """Fit projected and sliced profiles, recording failures per region without stale widths."""
        variances: dict[int, tuple[float, float]] = {}
        errors: dict[int, str] = {}
        regions = {-1: (tail_start or 0, image.shape[1])}
        if settings.num_slices > 1:
            regions.update(enumerate(zip(bounds, bounds[1:])))
        for index, (start, stop) in regions.items():
            if index >= 0 and invalid_slicing:
                errors[index] = (
                    "Negative column intensity prevents monotone equal-charge slicing; adjust background/mask or choose pixel clipping"
                )
                continue
            try:
                piece = image[:, start:stop]
                _, vx = profile_fit(piece.sum(axis=1), settings.fit_type)
                _, vy = profile_fit(piece.sum(axis=0), settings.fit_type)
                variances[index] = (
                    vx * settings.pixel_size_x**2,
                    vy * settings.pixel_size_y**2,
                )
            except (ValueError, RuntimeError) as error:
                errors[index] = str(error)
        return variances, errors

    work = subtract_background()
    work, mask, negative_fraction = filter_signal(work)
    image, mask, roi_origin = crop_roi(work, mask)
    longitudinal_profile = image.sum(axis=0)
    tail_start = find_tail_start(longitudinal_profile)
    bounds, invalid_slicing = find_slice_boundaries(longitudinal_profile)
    variances, errors = fit_regions(image, bounds, tail_start, invalid_slicing)
    return ProcessedImage(
        image,
        mask,
        bounds,
        variances,
        errors,
        roi_origin,
        tail_start,
        negative_fraction,
    )


@dataclass
class Shot:
    sample_id: str
    config_index: int
    source_hash: str
    image_hash: str
    recorded_setpoints: dict[str, float]
    matrix: Array
    variances: dict[int, tuple[float, float]]
    slice_errors: dict[int, str]
    bounds: list[int]
    negative_fraction: float


def mismatch_zeta(
    beta: float, alpha: float, design_beta: float, design_alpha: float
) -> float:
    """Dimensionless Twiss mismatch; matched beams have zeta = 1."""
    if not all(
        math.isfinite(value) for value in (beta, alpha, design_beta, design_alpha)
    ):
        raise ValueError("Mismatch requires finite Twiss parameters")
    if beta <= 0 or design_beta <= 0:
        raise ValueError("Mismatch requires positive beta functions")
    # Equivalent to (beta*gamma_design + gamma*design_beta - 2*alpha*alpha_design)/2,
    # without subtracting large, nearly equal terms for matched beams.
    ratio = beta / design_beta
    return 0.5 * (ratio + 1 / ratio + (alpha - ratio * design_alpha) ** 2 / ratio)


def covariance_parameters(a: Array) -> tuple[Array, Array, Array]:
    """Return [beta, alpha, emit], Jacobian, and transverse covariance."""
    q, c, p = a
    covariance = np.array([[q, c], [c, p]])
    determinant = q * p - c * c
    if q <= 0 or p <= 0 or determinant <= 0 or not np.isfinite(covariance).all():
        raise ValueError("Reconstructed transverse covariance is not positive definite")
    emit = np.sqrt(determinant)
    gradient_emit = np.array([p / 2, -c, q / 2]) / emit
    gradient_beta = np.array([1, 0, 0]) / emit - q * gradient_emit / emit**2
    gradient_alpha = -np.array([0, 1, 0]) / emit + c * gradient_emit / emit**2
    jac = np.stack([gradient_beta, gradient_alpha, gradient_emit])
    params = np.array([q / emit, -c / emit, emit])
    return params, jac, covariance


def fit_scan(
    shots: list[Shot],
    slice_index: int,
    plane: Plane | str,
    energy: float,
    settings: AnalysisSettings,
) -> FitResult:
    """Independent fits; waiting/invalid results carry no numerical reconstruction."""
    plane = Plane(plane)
    usable = [shot for shot in shots if slice_index in shot.variances]
    groups: dict[int, list[Shot]] = {}
    for shot in usable:
        groups.setdefault(shot.config_index, []).append(shot)
    result: FitResult = {
        "plane": plane,
        "slice_index": slice_index,
        "status": FitStatus.WAITING,
        "sample_count": len(usable),
        "excluded_slices": {
            shot.sample_id: shot.slice_errors[slice_index]
            for shot in shots
            if slice_index in shot.slice_errors
        },
        "counts_per_configuration": {str(k): len(v) for k, v in sorted(groups.items())},
    }
    dimension = 3
    dim, var_dim = plane.matrix_index, plane.variance_index

    def design(r: Array) -> list[float]:
        r0, r1 = r[dim, dim], r[dim, dim + 1]
        return [r0 * r0, 2 * r0 * r1, r1 * r1]

    def mean_design_row(group: list[Shot]) -> list[float]:
        """Average transported covariance coefficients, which are quadratic in each shot's R."""
        return np.mean([design(shot.matrix) for shot in group], axis=0).tolist()

    def collect_observations() -> (
        tuple[list[list[float]], list[float], list[float], list[int]]
    ):
        """Build per-shot or weighted per-configuration variances for the selected regression."""
        rows, values, errors, indices = [], [], [], []
        match settings.regression:
            case RegressionType.LINEAR:
                rows = [design(shot.matrix) for shot in usable]
                values = [shot.variances[slice_index][var_dim] for shot in usable]
                errors = [1.0] * len(values)
                indices = [shot.config_index for shot in usable]
            case RegressionType.CHI_SQUARED:
                waiting = []
                for index, group in sorted(groups.items()):
                    variances = np.array(
                        [shot.variances[slice_index][var_dim] for shot in group]
                    )
                    std = float(variances.std(ddof=1)) if len(group) > 1 else 0.0
                    if len(group) < 2 or std <= 0 or not math.isfinite(std):
                        waiting.append(index)
                        continue
                    rows.append(mean_design_row(group))
                    values.append(float(variances.mean()))
                    match settings.weighting:
                        case Weighting.STANDARD_ERROR:
                            errors.append(std / np.sqrt(len(group)))
                        case Weighting.SHOT_STD:
                            errors.append(std)
                    indices.append(index)
                result["unweighted_configurations"] = waiting
        return rows, values, errors, indices

    rows, values, errors, indices = collect_observations()
    result["fit_configurations"] = indices
    if len(set(indices)) < dimension:
        result["reason"] = (
            f"Need {dimension} independent configurations with usable uncertainty; have {len(set(indices))}"
        )
        return result

    def solve_covariance() -> tuple[Array, Array | None] | None:
        """Solve the scaled least-squares problem and record fit diagnostics."""
        design_matrix = np.asarray(rows, dtype=float)
        observation, error = np.asarray(values), np.asarray(errors)
        weighted = design_matrix / error[:, None]
        scales = np.linalg.norm(weighted, axis=0)
        if np.any(scales == 0) or not np.isfinite(weighted).all():
            result.update(
                {
                    "status": FitStatus.INVALID,
                    "reason": "Zero or nonfinite design column",
                }
            )
            return None
        scaled = weighted / scales
        u, singular, vt = np.linalg.svd(scaled, full_matrices=False)
        rank = int(
            np.count_nonzero(
                singular > singular[0] * max(scaled.shape) * np.finfo(float).eps
            )
        )
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
        result.update(
            {
                "rank": rank,
                "condition_number": condition,
                "degrees_of_freedom": len(values) - dimension,
            }
        )
        if rank < dimension or condition is None or condition > settings.max_condition:
            result.update(
                {
                    "status": FitStatus.INVALID,
                    "reason": "Scan optics are rank deficient or ill conditioned",
                }
            )
            return None
        a = (vt.T @ ((u.T @ (observation / error)) / singular)) / scales
        cov_a = ((vt.T / singular**2) @ vt) / scales[:, None] / scales[None, :]
        residuals = (design_matrix @ a - observation) / error
        score = float(residuals @ residuals)
        dof = len(values) - dimension
        result.update(
            {
                "coefficients": a.tolist(),
                "coefficient_covariance": cov_a.tolist(),
                "observations": observation.tolist(),
                "uncertainties": error.tolist(),
                "predictions": (design_matrix @ a).tolist(),
                "residuals": residuals.tolist(),
            }
        )
        match settings.regression:
            case RegressionType.LINEAR:
                total = float(np.sum((observation - observation.mean()) ** 2))
                result["r_squared"] = 1 - score / total if total > 0 else None
                cov_a = cov_a * score / dof if dof > 0 else None
                result["coefficient_covariance"] = (
                    cov_a.tolist() if cov_a is not None else None
                )
            case RegressionType.CHI_SQUARED:
                result.update(
                    {
                        "chi_squared": score,
                        "reduced_chi_squared": score / dof if dof > 0 else None,
                    }
                )
        return a, cov_a

    def reconstruct_beam() -> None:
        """Convert fitted covariance into Twiss parameters, emittance, and propagated errors."""
        try:
            params, jac, covariance = covariance_parameters(a)
            parameter_errors = (
                np.sqrt(np.maximum(np.diag(jac @ cov_a @ jac.T), 0))
                if cov_a is not None
                else None
            )
            if not np.isfinite(params).all() or (
                parameter_errors is not None and not np.isfinite(parameter_errors).all()
            ):
                raise ValueError("Nonfinite reconstruction or uncertainty")
            beta, alpha, emit = map(float, params[:3])
            factor = energy / REST_ENERGY_GEV
            result.update(
                {
                    "status": FitStatus.VALID,
                    "beta": beta,
                    "alpha": alpha,
                    "gamma": (1 + alpha**2) / beta,
                    "emittance": emit,
                    "normalized_emittance": emit * factor,
                    "transverse_covariance": covariance.tolist(),
                }
            )
            result["errors"] = (
                {
                    "beta": float(parameter_errors[0]),
                    "alpha": float(parameter_errors[1]),
                    "emittance": float(parameter_errors[2]),
                }
                if parameter_errors is not None
                else None
            )
            result["normalized_emittance_error"] = (
                float(parameter_errors[2] * factor)
                if parameter_errors is not None
                else None
            )
        except ValueError as error:
            result.update(
                {
                    "status": FitStatus.INVALID,
                    "reason": str(error),
                }
            )

    solution = solve_covariance()
    if solution is None:
        return result
    a, cov_a = solution
    reconstruct_beam()
    return result


@dataclass
class Snapshot:
    document: SnapshotReport
    shots: list[Shot]
    configurations: dict[str, Any]
    background: Array | None
    signatures: dict[Path, Signature]
    raw_names: tuple[str, ...]

    @property
    def fits(self) -> list[FitResult]:
        """Expose the report's typed reconstruction records directly."""
        return self.document["fits"]

    @property
    def design_twiss(self) -> DesignTwiss | None:
        """Expose the typed optional design reference without changing the report schema."""
        return self.document.get("design_twiss")

    def to_dict(self) -> SnapshotReport:
        """Return a detached copy of the JSON-compatible analysis report."""
        return copy.deepcopy(self.document)

    @staticmethod
    def parse_report(document: object) -> SnapshotReport:
        """Check declared report fields and restore enums at the JSON boundary.

        Unknown metadata is retained for compatibility. Required keys and field
        types come directly from the TypedDict declarations.
        This validates record structure; scientific validity remains fit_scan's job.
        """

        def decode_fit(value: Any, path: str) -> Any:
            """Check the fields required by the fit's status after decoding their types."""
            result = decode_record(value, FitResult, path)
            required: tuple[str, ...]
            match result["status"]:
                case FitStatus.VALID:
                    required = (
                        "beta",
                        "alpha",
                        "gamma",
                        "emittance",
                        "normalized_emittance",
                        "transverse_covariance",
                        "errors",
                        "normalized_emittance_error",
                    )
                case FitStatus.INVALID:
                    required = ("reason",)
                case FitStatus.WAITING:
                    required = ()
            for name in required:
                if name not in result:
                    raise ValueError(f"{path}.{name}: required field is missing")
            return result

        def decode_record(value: Any, expected: Any, path: str) -> dict[str, Any]:
            """Validate declared keys while retaining unknown provenance metadata."""
            if not isinstance(value, dict):
                raise ValueError(f"{path}: expected an object")
            result = value.copy()
            for name, annotation in expected.__annotations__.items():
                if name not in value:
                    if name in expected.__required_keys__:
                        raise ValueError(f"{path}.{name}: required field is missing")
                    continue
                if get_origin(annotation) is NotRequired:
                    annotation = get_args(annotation)[0]
                result[name] = decode(value[name], annotation, f"{path}.{name}")
            return result

        def decode(value: Any, expected: Any, path: str) -> Any:
            """Walk the report's nested records, containers, enums, and scalar fields."""
            if isinstance(expected, TypeAliasType):
                return decode(value, expected.__value__, path)
            origin, arguments = get_origin(expected), get_args(expected)
            if expected is Any:
                return value
            if expected == FitResult:
                return decode_fit(value, path)
            if is_typeddict(expected):
                return decode_record(value, expected, path)
            if origin is UnionType:
                if value is not None:
                    arguments = tuple(
                        item for item in arguments if item is not NoneType
                    )
                if len(arguments) == 1:
                    return decode(value, arguments[0], path)
                for alternative in arguments:
                    try:
                        return decode(value, alternative, path)
                    except ValueError:
                        continue
                raise ValueError(f"{path}: expected {expected}")
            if origin is list and isinstance(value, list):
                return [
                    decode(item, arguments[0], f"{path}[{i}]")
                    for i, item in enumerate(value)
                ]
            elif origin is dict and isinstance(value, dict):
                return {
                    decode(key, arguments[0], f"{path} key"): decode(
                        item, arguments[1], f"{path}[{key!r}]"
                    )
                    for key, item in value.items()
                }
            elif isinstance(expected, type) and issubclass(expected, StrEnum):
                try:
                    return expected(value)
                except (TypeError, ValueError):
                    pass
            elif (
                expected is float
                and type(value) in (int, float)
                and math.isfinite(value)
            ):
                return value
            elif expected is str and isinstance(value, str):
                return value
            elif expected in (int, NoneType) and type(value) is expected:
                return value
            raise ValueError(f"{path}: expected {expected}")

        # The cast follows validation of every declared field, including valid-fit
        # completeness; it does not assert types over an unchecked JSON dictionary.
        return cast(SnapshotReport, decode(document, SnapshotReport, "report"))


class LiveAnalysis:
    """Single-worker incremental cache; failures never retain an older shot."""

    def __init__(
        self,
        experiment: Path,
        settings: AnalysisSettings,
        optics: OpticsProvider | None = None,
    ) -> None:
        settings.validate()
        self.experiment = experiment.resolve()
        self.settings = settings
        self.optics: OpticsProvider | None = optics
        self.cache: dict[str, tuple[Signature, Shot]] = {}
        self.observed: dict[str, tuple[Signature, float]] = {}
        self.dependency_hash: str | None = None

    def ready(self, path: Path, stat: Signature) -> bool:
        previous = self.observed.get(path.name)
        now = time.monotonic()
        if previous is None or previous[0] != stat:
            self.observed[path.name] = stat, now
        return now - self.observed[path.name][1] >= self.settings.settle_seconds

    def refresh(self) -> Snapshot:
        """Load stable shots, reject duplicates, reconstruct slices, and verify a consistent snapshot."""
        start = time.monotonic()

        def load_dependencies() -> (
            tuple[dict[str, Any], dict[str, str], dict[Path, Signature], Array | None]
        ):
            """Read consistent lattice settings and the optional background with source hashes."""
            config_path = self.experiment / "lattice_configs.json"
            content, config_sig = stable_bytes(config_path)
            configurations = configuration_document(content)
            dependencies = {"lattice_configs.json": digest(content)}
            signatures = {config_path: config_sig}
            background = None
            if self.settings.subtract_background:
                path = self.experiment / "screen_background.npz"
                content, stat = stable_bytes(path)
                with np.load(io.BytesIO(content), allow_pickle=False) as data:
                    background = np.asarray(data["data"], dtype=float)
                dependencies[path.name] = digest(content)
                signatures[path] = stat
            return configurations, dependencies, signatures, background

        configurations, dependencies, signatures, background = load_dependencies()
        dependency_hash = digest(json.dumps(dependencies, sort_keys=True).encode())
        if dependency_hash != self.dependency_hash:
            self.cache.clear()
            self.dependency_hash = dependency_hash
        raw_dir = self.experiment / "samples_raw"
        inventory_utc = datetime.now(UTC).isoformat()
        paths = sorted(raw_dir.glob("*.npz"))
        raw_names = tuple(path.name for path in paths)
        present = {path.stem for path in paths}
        self.cache = {key: value for key, value in self.cache.items() if key in present}
        self.observed = {
            key: value for key, value in self.observed.items() if key in raw_names
        }

        def read_shot(path: Path, stat: Signature) -> Shot:
            """Combine a stable image, recorded setpoints, transfer matrix, and processed widths."""
            sample_id = path.stem
            match = re.match(r"^MES_(\d+)_", sample_id)
            if match is None:
                raise ValueError("Sample filename must begin MES_<config_index>_")
            index = int(match[1])
            if index >= len(configurations["low_level_params"]):
                raise ValueError("Sample configuration index is out of range")
            content, loaded_stat = stable_bytes(path)
            if loaded_stat != stat:
                raise ValueError("File changed since directory scan")
            with np.load(io.BytesIO(content), allow_pickle=True) as data:
                raw = np.asarray(data["screens"].item()["data"])
                magnets = data["magnets"].item()["data"]
            if self.optics is None:
                self.optics = NotebookOptics()
            target = configurations["low_level_params"][index]
            recorded = magnet_parameters(magnets, target, self.optics.lengths)
            energy = float(configurations["high_level_params"][index]["beam_energy"])
            matrix = self.optics.matrix(target, recorded, energy)
            if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
                raise ValueError("Expected a finite 6x6 transfer matrix")
            processed = process_image(raw, self.settings, background)
            canonical = np.ascontiguousarray(raw, dtype=np.float64)
            image_hash = digest(str(canonical.shape).encode() + canonical.tobytes())
            return Shot(
                sample_id,
                index,
                digest(content),
                image_hash,
                recorded,
                matrix,
                processed.variances,
                processed.slice_errors,
                processed.bounds,
                processed.negative_fraction,
            )

        def collect_settled_shots() -> tuple[list[Shot], dict[str, Any]]:
            """Reuse unchanged shots and retry settled inputs while recording pending or failed files."""
            statuses: dict[str, Any] = {}
            candidates = []
            for path in paths:
                sample_id = path.stem
                try:
                    stat = signature(path)
                    if not self.ready(path, stat):
                        self.cache.pop(sample_id, None)
                        statuses[sample_id] = {
                            "status": SampleStatus.PENDING,
                            "reason": "File has not settled",
                        }
                        continue
                    cached = self.cache.get(sample_id)
                    if cached is not None and cached[0] == stat:
                        shot = cached[1]
                    else:
                        self.cache.pop(sample_id, None)
                        shot = read_shot(path, stat)
                        self.cache[sample_id] = stat, shot
                    signatures[path] = stat
                    candidates.append(shot)
                    statuses[sample_id] = {
                        "status": SampleStatus.ACCEPTED,
                        "source_sha256": shot.source_hash,
                        "config_index": shot.config_index,
                        "slice_errors": shot.slice_errors,
                    }
                except Exception as error:
                    # Acquisition writes non-atomically; retry even an unchanged bad file.
                    self.cache.pop(sample_id, None)
                    statuses[sample_id] = {
                        "status": SampleStatus.REJECTED,
                        "reason": f"{type(error).__name__}: {error}",
                    }
            return candidates, statuses

        def remove_duplicate_images(
            candidates: list[Shot],
            statuses: dict[str, Any],
        ) -> list[Shot]:
            """Keep one copy of each image and reject duplicate inputs with conflicting optics."""
            by_image: dict[str, list[Shot]] = {}
            for shot in candidates:
                by_image.setdefault(shot.image_hash, []).append(shot)
            shots = []
            for duplicates in by_image.values():
                first = duplicates[0]
                if any(
                    shot.config_index != first.config_index
                    or shot.recorded_setpoints != first.recorded_setpoints
                    for shot in duplicates[1:]
                ):
                    for shot in duplicates:
                        statuses[shot.sample_id].update(
                            status=SampleStatus.REJECTED,
                            reason="Identical image recorded under conflicting optics",
                        )
                else:
                    shots.append(first)
                    for shot in duplicates[1:]:
                        statuses[shot.sample_id].update(
                            status=SampleStatus.DUPLICATE, duplicate_of=first.sample_id
                        )
            shots.sort(key=lambda shot: shot.sample_id)
            return shots

        def build_report(
            shots: list[Shot],
            statuses: dict[str, Any],
            fits: list[FitResult],
        ) -> SnapshotReport:
            """Assemble results, source provenance, units, conventions, and design reference for export."""
            lengths = self.optics.lengths if self.optics is not None else {}
            magnet_lengths: dict[str, float] = {}
            for target in configurations["low_level_params"]:
                for key in target:
                    if key.endswith("_k1"):
                        name = key.removesuffix("_k1").upper().replace("_", ".")
                        if name in lengths:
                            magnet_lengths[name] = lengths[name]
            return {
                "tool_version": VERSION,
                "created_utc": datetime.now(UTC).isoformat(),
                "inventory_observed_utc": inventory_utc,
                "inventory": list(raw_names),
                "deferred_new_samples": sorted(
                    path.name
                    for path in raw_dir.glob("*.npz")
                    if path.name not in raw_names
                ),
                "experiment": str(self.experiment),
                "settings": self.settings.to_dict(),
                "energy_GeV": energy,
                "reference_start": configurations["low_level_params"][0]["start"],
                "reference_stop": configurations["low_level_params"][0]["stop"],
                "design_twiss": design_reference,
                "design_twiss_error": design_error,
                "source_sha256": dependencies,
                "units": {
                    "variance": "m^2",
                    "emittance": "m rad",
                    "normalized_emittance": "m rad",
                    "beta": "m",
                    "alpha": "1",
                    "gamma": "1/m",
                    "mismatch_zeta": "1",
                },
                "conventions": {
                    "image_axes": "[x, y]; slices increase in image-column order",
                    "image_orientation": "rotate counterclockwise in lower-origin display, then flip horizontally and vertically; transform background identically before subtraction",
                    "pixel_calibration": "entered x/y resolutions apply to oriented axes without swapping",
                    "roi_coordinates": "half-open pixel bounds in the oriented image",
                    "magnet_source": "KICK_MRAD.SP setpoints, not measured fields",
                    "negative_intensity": self.settings.negative_intensity,
                    "weighting": self.settings.weighting,
                    "chi_squared_optics": "mean per-shot quadratic transport coefficients per configuration",
                    "chi_squared_plot": "chi_squared / sample_count (usable shots per slice)",
                    "mismatch_zeta": "(beta*gamma_design + gamma*beta_design - 2*alpha*alpha_design)/2; design from lattice.make_twiss0(reference_start)",
                    "slice_alignment": "equal intensity fraction per image; no absolute time calibration",
                    "uncertainties": "conditional statistical errors; excludes calibration and optics systematics",
                },
                "samples": statuses,
                "shots": [
                    {
                        "sample_id": shot.sample_id,
                        "config_index": shot.config_index,
                        "source_sha256": shot.source_hash,
                        "image_sha256": shot.image_hash,
                        "recorded_setpoints": shot.recorded_setpoints,
                        "R": shot.matrix.tolist(),
                        "variances": shot.variances,
                        "slice_bounds": shot.bounds,
                        "slice_errors": shot.slice_errors,
                        "negative_masked_pixel_fraction": shot.negative_fraction,
                    }
                    for shot in shots
                ],
                "fits": fits,
                "refresh_seconds": time.monotonic() - start,
                "magnet_lengths": magnet_lengths,
            }

        def reconstruct_slices(shots: list[Shot], energy: float) -> list[FitResult]:
            """Fit both projected planes and each x slice using the current optics."""
            indices = (
                [-1, *range(self.settings.num_slices)]
                if self.settings.num_slices > 1
                else [-1]
            )
            return [
                fit_scan(shots, index, plane, energy, self.settings)
                for index in indices
                for plane in (tuple(Plane) if index == -1 else (Plane.X,))
            ]

        def load_design_reference() -> tuple[DesignTwiss | None, str | None]:
            """Validate nominal Twiss separately so its absence does not stop fitting."""
            design_reference: DesignTwiss | None = None
            design_error: str | None = None
            if isinstance(self.optics, DesignTwissProvider):
                try:
                    design_reference = self.optics.design_twiss(
                        configurations["low_level_params"][0]["start"]
                    )
                    if design_reference is not None:
                        for values in design_reference.values():
                            mismatch_zeta(
                                values["beta"],
                                values["alpha"],
                                values["beta"],
                                values["alpha"],
                            )
                except Exception as error:
                    design_reference = None
                    design_error = str(error)
            return design_reference, design_error

        def add_mismatch_parameters(
            fits: list[FitResult], design_reference: DesignTwiss | None
        ) -> None:
            """Attach zeta only when the fit and its design reference are available."""
            for result in fits:
                match result["status"]:
                    case FitStatus.VALID if design_reference is not None:
                        reference = design_reference.get(result["plane"])
                        if reference is not None:
                            result["mismatch_zeta"] = mismatch_zeta(
                                result["beta"],
                                result["alpha"],
                                reference["beta"],
                                reference["alpha"],
                            )

        candidates, statuses = collect_settled_shots()
        shots = remove_duplicate_images(candidates, statuses)
        energy = float(configurations["high_level_params"][0]["beam_energy"])
        fits = reconstruct_slices(shots, energy)
        design_reference, design_error = load_design_reference()
        add_mismatch_parameters(fits, design_reference)
        # Never publish a mixed snapshot assembled across an input change.
        self.verify(signatures)
        document = build_report(shots, statuses, fits)
        return Snapshot(
            document, shots, configurations, background, signatures, raw_names
        )

    def verify(self, signatures: dict[Path, Signature]) -> None:
        # A snapshot contains the settled inventory observed at the start of its
        # refresh. New files and changing excluded files are for the next refresh;
        # only dependencies that actually formed this result must remain stable.
        for path, stat in signatures.items():
            if signature(path) != stat:
                raise ValueError(f"Input changed during refresh: {path.name}; retry")

    def export_bundle(
        self,
        snapshot: Snapshot,
        parent: Path | None = None,
        figure: Figure | None = None,
        include_raw: bool = True,
        view: dict[str, Any] | None = None,
    ) -> Path:
        """Publish a complete timestamped folder only after all files are ready."""
        self.verify(snapshot.signatures)
        parent = self.experiment if parent is None else parent
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
        destination = parent / f"processed_{stamp}-{uuid4().hex[:8]}"
        staging = Path(tempfile.mkdtemp(prefix=".processing-", dir=parent))

        def build_export_report() -> SnapshotReport:
            """Extend a snapshot with export choices, implementation hash, and dependency versions."""
            report = snapshot.to_dict()
            report["export"] = {
                "raw_inputs_included": include_raw,
                "view": view or {},
                "folder": destination.name,
            }
            report["lattice_configurations"] = snapshot.configurations
            report["implementation_sha256"] = digest(Path(__file__).read_bytes())
            report["runtime"] = {"python": sys.version, "numpy": np.__version__}
            for package in ("scipy", "matplotlib", "opencv-python", "ocelot-desy"):
                try:
                    report["runtime"][package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    pass
            return report

        def copy_raw_inputs() -> None:
            """Copy only verified source files that formed this processing snapshot."""
            sources = staging / "inputs"
            sources.mkdir()
            (sources / "samples_raw").mkdir()

            def copy_verified(relative_path: Path, expected_hash: str) -> None:
                """Require both the original file signature and content hash before copying."""
                path = self.experiment / relative_path
                content, stat = stable_bytes(path)
                if (
                    stat != snapshot.signatures[path]
                    or digest(content) != expected_hash
                ):
                    raise ValueError(f"Input changed before export: {relative_path}")
                (sources / relative_path).write_bytes(content)

            for name, expected_hash in snapshot.document["source_sha256"].items():
                copy_verified(Path(name), expected_hash)
            for shot in snapshot.shots:
                copy_verified(
                    Path("samples_raw") / f"{shot.sample_id}.npz", shot.source_hash
                )

        def write_processed_samples() -> None:
            """Write notebook-compatible projected and sliced records with full processing settings."""
            for shot in snapshot.shots:
                for index, (vx, vy) in shot.variances.items():
                    subdirectory = (
                        staging
                        / ("projected" if index == -1 else "sliced")
                        / f"{shot.config_index:02d}"
                    )
                    if index != -1:
                        subdirectory /= f"{index:03d}"
                    record = {
                        "sample_id": shot.sample_id,
                        "lattice_config_index": shot.config_index,
                        "slice_index": index,
                        "num_slices": self.settings.num_slices if index != -1 else 0,
                        "actual_doocs_lattice_params": shot.recorded_setpoints,
                        "magnet_source": "KICK_MRAD.SP setpoints",
                        "final_cov_xx": vx,
                        "final_cov_yy": vy,
                        "R": shot.matrix.tolist(),
                        "source_sha256": shot.source_hash,
                        "processing": self.settings.to_dict(),
                    }
                    atomic_json(subdirectory / f"{shot.sample_id}.json", record)

        try:
            report = build_export_report()
            atomic_json(staging / "config.json", report)
            atomic_json(
                staging / "results.json",
                {"fits": report["fits"], "units": report["units"]},
            )
            atomic_json(staging / "lattice_configs.json", snapshot.configurations)
            # Include the implementation and exact data used, so future replacements
            # of acquisition files do not prevent reproducing the image processing.
            (staging / "tool").write_bytes(Path(__file__).read_bytes())
            if include_raw:
                copy_raw_inputs()
            write_processed_samples()
            if figure is not None:
                figure.savefig(staging / "plots.png", dpi=180, bbox_inches="tight")
                figure.savefig(staging / "plots.pdf", bbox_inches="tight")
            self.verify(snapshot.signatures)
            staging.rename(destination)
        except Exception:
            # Preserve an incomplete bundle for diagnosis; never present it as a result.
            atomic_json(
                staging / "INCOMPLETE.json",
                {
                    "reason": "Export did not complete; not a published processing result"
                },
            )
            raise
        return destination


def generate_lattices(
    output: Path, settings: LatticeSettings, progress: Callable[[str], None] = print
) -> None:
    """The notebook's dispersion match followed by bounded phase/R34 matching."""
    if output.exists():
        raise ValueError("Lattice generation requires a new experiment directory")
    if (
        not settings.phases
        or settings.beam_energy <= 0
        or not np.isfinite(settings.phases).all()
    ):
        raise ValueError("Need finite phase targets and positive beam energy")
    design = copy.deepcopy(MagneticLattice(t3.cell, method={"global": SecondTM}))
    set_lattice_param(design, "SAOX.2555.T3", "k2", 0.0)
    twiss0 = twiss(
        make_sublattice(design, stop=settings.start, inclusive=True), t3.twiss0
    )[-1]
    twiss0.E = settings.beam_energy
    default = make_sublattice(
        design, start=settings.start, stop=settings.stop, inclusive=True
    )
    free = ["QE.2518.T3", "QH.2529.T3", "QH.2532.T3", "QH.2544.T3"]
    quads = [
        "QE.2495.T3",
        "QE.2506.T3",
        *free,
        "QM.2549.T3",
        "QM.2554.T3",
        "QM.2559.T3",
    ]
    # Same conservative bounds as 04_make_lattice, including shared supplies.
    limits = {}
    for eid in quads:
        factor = (
            1.75 if eid == "QH.2544.T3" else 1.25 if eid.startswith("QM.") else 2.75
        )
        k = get_lattice_param(design, eid, "k1")
        limits[eid] = sorted((k * 0.05, k * factor))

    def match_screen_dispersion() -> None:
        """Match the screen dispersion using the last quadrupole before scanning phase."""
        problem = MatchProblem(default, twiss0, periodic=False)
        problem.vary_element(
            get_lattice_element(default, "QM.2554.T3"),
            "k1",
            limits=limits["QM.2554.T3"],
        )
        problem.target_twiss(
            get_lattice_element(default, settings.stop),
            "Dx",
            settings.dispersion_x,
            weight=1e6,
        )
        progress("Matching screen dispersion")
        result = problem.solve(solver="ls_trf", max_iter=3000)
        if not result.success:
            raise ValueError(f"Dispersion matching failed: {result.message}")

    class PhaseTarget(Target):

        def __init__(self, start: Any, end: Any, value: float) -> None:
            super().__init__(name="phase_advance_x", weight=1e1, tol=np.pi / 20)
            self.start, self.end, self.value = start, end, value

        def actual(self, state: Any) -> float:
            return float(state.twiss_at(self.end).mux - state.twiss_at(self.start).mux)

        def residuals(self, state: Any) -> Array:
            distance = abs(
                (self.actual(state) - self.value + np.pi / 2) % np.pi - np.pi / 2
            )
            return np.array([0.0 if distance <= self.tol else distance])

        def report(self, state: Any) -> Any:
            """Expose the phase target value, periodic residual, and tolerance result to the optimizer."""
            residual = float(self.residuals(state)[0])
            return TargetReport(
                name=self.name,
                residual_norm=residual,
                met=residual < 1e-12,
                weight=self.weight,
                details={
                    "actual": self.actual(state),
                    "target": self.value,
                    "periodicity": float(np.pi),
                },
            )

    document: dict[str, list[Any]] = {
        "low_level_params": [],
        "high_level_params": [],
        "target_doocs_params": [],
    }
    diagnostics = []

    def match_phase(index: int, phase: float) -> MagneticLattice:
        """Solve one phase target with bounded quadrupoles and screen-width constraints."""
        lattice = MagneticLattice(copy.deepcopy(default.sequence))
        problem = MatchProblem(lattice, twiss0, periodic=False)
        for eid in free:
            problem.vary_element(
                get_lattice_element(lattice, eid), "k1", limits=limits[eid]
            )
        problem.add_target(
            PhaseTarget(
                get_lattice_element(lattice, settings.phase_start),
                lattice.sequence[-1],
                phase,
            )
        )
        if settings.r34 is not None:
            problem.target_rmatrix(
                get_lattice_element(lattice, settings.start),
                lattice.sequence[-1],
                i=2,
                j=3,
                value=settings.r34,
                relation="==",
                name="R34_fixed",
                weight=1e-1,
                tol=0.1 * abs(settings.r34),
            )
        for axis in Plane:
            problem.target_global(
                quantity=f"beta_{axis}",
                value=settings.max_beta,
                relation="<=",
                name=f"beta_{axis}_max",
                tol=0.01 * settings.max_beta,
            )
        problem.target_twiss(
            lattice.sequence[-1],
            "beta_x",
            value=settings.min_screen_beta_x,
            relation=">=",
            name="screen_beta_x_min",
            weight=1e-1,
            tol=0.01 * settings.min_screen_beta_x,
        )
        result = problem.solve(
            solver=settings.solver,
            max_iter=settings.max_iter,
            tol=1e-8,
            restore_if_fail=False,
        )
        if not result.success or any(
            not target.met for target in result.target_reports
        ):
            raise ValueError(
                f"Optics {index} failed or missed a constraint: {result.message}; {result.target_reports}"
            )
        return lattice

    def record_configuration(lattice: MagneticLattice, phase: float) -> None:
        """Record optimized strengths, matching targets, and DOOCS setpoints together."""
        low: dict[str, Any] = {
            "start": settings.start,
            "stop": settings.stop,
            "mu_x_start": settings.phase_start,
        }
        doocs = {}
        for eid in quads:
            k = float(get_lattice_param(lattice, eid, "k1"))
            low[eid.lower().replace(".", "_") + "_k1"] = k
            doocs[f"XFEL.MAGNETS/MAGNET.ML/{eid}/KICK_MRAD.SP"] = (
                k * 1e3 * get_lattice_param(lattice, eid, "l")
            )
        high = {
            "beam_energy": settings.beam_energy,
            "start": settings.start,
            "stop": settings.stop,
            "mu_x_start": settings.phase_start,
            "dispersion_x": settings.dispersion_x,
            "mu_x": phase,
            "global_max_beta": settings.max_beta,
            "final_beta_min": settings.min_screen_beta_x,
            "r_34": settings.r34,
            "minimize_second_order_t116": False,
        }
        document["low_level_params"].append(low)
        document["high_level_params"].append(high)
        document["target_doocs_params"].append(doocs)
        optics = twiss(lattice, twiss0)
        diagnostics.append(
            {
                "R": np.asarray(
                    lattice.transfer_maps(energy=settings.beam_energy)[1]
                ).tolist(),
                "twiss": [
                    {
                        key: float(getattr(t, key))
                        for key in ("s", "beta_x", "beta_y", "Dx", "mux", "muy")
                    }
                    for t in optics
                ],
            }
        )

    match_screen_dispersion()
    for index, phase in enumerate(settings.phases):
        progress(
            f"Matching optics {index + 1}/{len(settings.phases)}: phase={phase:.4g} rad"
        )
        lattice = match_phase(index, phase)
        record_configuration(lattice, phase)
    configuration_document(json.dumps(document).encode())
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "lattice_configs.json", document)
    atomic_json(
        output / "lattice_generation.json",
        {
            "settings": settings.to_dict(),
            "diagnostics": diagnostics,
        },
    )

    def save_optics_plot() -> None:
        """Save beta and dispersion along the optimized lattice for review."""
        figure = Figure(figsize=(9, 6))
        axes = figure.subplots(2, 1)
        for index, diagnostic in enumerate(diagnostics):
            ts = diagnostic["twiss"]
            axes[0].plot(
                [t["s"] for t in ts], [t["beta_x"] for t in ts], label=str(index)
            )
            axes[1].plot([t["s"] for t in ts], [t["Dx"] for t in ts], label=str(index))
        axes[0].set_ylabel(r"$\beta_x$ [m]")
        axes[1].set_ylabel(r"$D_x$ [m]")
        axes[1].set_xlabel(r"$s$ [m]")
        axes[0].legend(title="configuration")
        figure.savefig(output / "lattice.pdf", bbox_inches="tight")

    save_optics_plot()
    progress(f"Saved {len(diagnostics)} lattice configurations to {output}")


def simulate_experiment(
    source: Path,
    output: Path,
    settings: SimulationSettings,
    analysis: AnalysisSettings,
    progress: Callable[[str], None] = print,
) -> None:
    """Particle tracking with the notebook screen convention and DOOCS NPZ schema."""

    def validate_inputs() -> None:
        """Check simulation dimensions, particle counts, and physical input limits."""
        if output.exists():
            raise ValueError("Simulation requires a new output experiment directory")
        analysis.validate()
        if (
            settings.samples_per_config < 1
            or settings.num_particles < 2
            or min(settings.screen_shape) < 2
        ):
            raise ValueError(
                "Need positive sample count, >=2 particles, and a 2D screen"
            )
        if any(
            not math.isfinite(v) or v <= 0
            for v in (
                settings.normalized_emittance,
                settings.sigma_tau,
                settings.charge,
                settings.screen_gamma,
                settings.screen_gain,
            )
        ):
            raise ValueError(
                "Invalid simulation emittance, charge, bunch length, gain or gamma"
            )
        if any(
            not math.isfinite(v) or v < 0
            for v in (
                settings.screen_noise,
                settings.screen_offset,
                settings.twiss_noise,
                settings.sigma_p,
            )
        ):
            raise ValueError(
                "Noise, offset, and spreads must be finite and nonnegative"
            )
        if settings.twiss_noise >= 1 or not math.isfinite(settings.chirp):
            raise ValueError("twiss_noise must be below 1 and chirp must be finite")
        if (
            settings.dechirper_enabled is not None
            and type(settings.dechirper_enabled) is not bool
        ):
            raise ValueError("dechirper_enabled must be a JSON boolean or null")

    def make_particles(initial: Any, energy: float) -> Any:
        """Jitter a fresh Twiss copy and generate particles at the recorded beam energy."""
        seed_twiss = copy.deepcopy(initial)  # avoid cumulative jitter drift
        for axis in Plane:
            setattr(
                seed_twiss,
                f"alpha_{axis}",
                getattr(seed_twiss, f"alpha_{axis}")
                + rng.uniform(-1, 1) * settings.twiss_noise,
            )
            setattr(
                seed_twiss,
                f"beta_{axis}",
                getattr(seed_twiss, f"beta_{axis}")
                * (1 + rng.uniform(-1, 1) * settings.twiss_noise),
            )
        emit = settings.normalized_emittance * REST_ENERGY_GEV / energy
        beam_params = BeamParameters(
            energy=energy,
            charge=settings.charge,
            emit_x=emit,
            emit_y=emit,
            alpha_x=seed_twiss.alpha_x,
            alpha_y=-seed_twiss.alpha_y,
            beta_x=seed_twiss.beta_x,
            beta_y=seed_twiss.beta_y,
            sigma_tau=settings.sigma_tau,
            sigma_p=settings.sigma_p,
            chirp=settings.chirp,
        )
        particles = make_beam_with_random_gaussian_mixture_current_profile(
            beam_params, num_particles=settings.num_particles
        )
        return particles

    def make_screen_image(final: Any, half_x: float, half_y: float) -> Array:
        """Bin tracked particles with the notebook orientation and camera response."""
        image = np.histogram2d(
            final.x(),
            final.y(),
            bins=settings.screen_shape,
            range=[[-half_x, half_x], [-1.5 * half_y, 0.5 * half_y]],
        )[0][:, ::-1]
        if image.max() <= 0:
            raise ValueError("No tracked particles fall on the screen")
        image *= settings.screen_gain / image.max()
        image += settings.screen_offset
        if settings.screen_noise > 0:
            image = np.clip(
                image + rng.normal(0, settings.screen_noise, image.shape), 0, 1
            )
        image **= 1 / settings.screen_gamma
        return image

    def save_sample(
        index: int,
        low: dict[str, Any],
        lattice: MagneticLattice,
        image: Array,
    ) -> None:
        """Atomically save the simulated image and magnet setpoints in the acquisition schema."""
        magnets = []
        for key in low:
            if key.endswith("_k1"):
                eid = key.removesuffix("_k1").upper().replace("_", ".")
                kick = (
                    get_lattice_param(lattice, eid, "k1")
                    * 1e3
                    * get_lattice_param(lattice, eid, "l")
                )
                magnets.append([0, kick, kick, 0, eid])
        sample_id = f"MES_{index}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]}-{uuid4().hex[:8]}"
        destination = output / "samples_raw" / f"{sample_id}.npz"
        temporary = destination.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                screens=np.array({"data": image}, dtype=object),
                magnets=np.array({"data": magnets}, dtype=object),
            )
        temporary.replace(destination)

    def track_configuration(
        index: int, low: dict[str, Any], high: dict[str, Any]
    ) -> None:
        """Use a fresh lattice, navigator, and beam for every simulated shot."""
        for sample in range(settings.samples_per_config):
            progress(
                f"Tracking configuration {index}, sample {sample + 1}/{settings.samples_per_config}"
            )
            lattice, navigator = make_lattice_and_navigator(
                LatticeLowLevelParams(**low)
            )
            particles = make_particles(initial, float(high["beam_energy"]))
            _, final = track(
                lattice, particles.copy(), navi=navigator, print_progress=False
            )
            image = make_screen_image(final, half_x, half_y)
            save_sample(index, low, lattice, image)

    def prepare_output() -> dict[str, Any]:
        """Apply explicit source overrides and record all simulation inputs before tracking."""
        source_content, _ = stable_bytes(source / "lattice_configs.json")
        document = configuration_document(source_content)
        for key in ("dechirper_dist_y", "dechirper_sigma_z"):
            value = getattr(settings, key)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive, or null")
        for low, high in zip(
            document["low_level_params"], document["high_level_params"]
        ):
            for key in ("dechirper_enabled", "dechirper_dist_y", "dechirper_sigma_z"):
                value = getattr(settings, key)
                if value is not None:
                    low[key] = high[key] = value
        output.mkdir(parents=True, exist_ok=False)
        (output / "samples_raw").mkdir()
        atomic_json(output / "lattice_configs.json", document)
        (output / "source_lattice_configs.json").write_bytes(source_content)
        atomic_json(
            output / "sim_config.json",
            {
                "simulation": settings.to_dict(),
                "analysis": analysis.to_dict(),
                "source": str(source.resolve()),
                "lattice_sha256": digest(source_content),
                "dechirper_settings": "Source values retained unless explicitly overridden in simulation settings",
            },
        )
        return document

    validate_inputs()
    document = prepare_output()
    random_state = np.random.get_state()
    np.random.seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    try:
        initial = make_twiss0(document["low_level_params"][0]["start"])
        nx, ny = settings.screen_shape
        half_x, half_y = nx * analysis.pixel_size_x / 2, ny * analysis.pixel_size_y / 2
        for index, (low, high) in enumerate(
            zip(document["low_level_params"], document["high_level_params"])
        ):
            track_configuration(index, low, high)
    finally:
        np.random.set_state(random_state)
    progress(f"Simulation completed: {output}")


@dataclass
class ProcessingPreview:
    sample_id: str
    settings: AnalysisSettings
    raw: Array
    processed: ProcessedImage
    source_sha256: str


def preview_processing_sample(
    experiment: Path, sample_id: str, settings: AnalysisSettings
) -> ProcessingPreview:
    """Preview current controls using stable raw inputs, independently of optics."""
    settings.validate()
    if Path(sample_id).name != sample_id:
        raise ValueError("Expected a sample filename stem")
    path = experiment / "samples_raw" / f"{sample_id}.npz"
    content, stat = stable_bytes(path)
    source_hash = digest(content)
    with np.load(io.BytesIO(content), allow_pickle=True) as data:
        raw = np.asarray(data["screens"].item()["data"], dtype=float)
    signatures = {path: stat}
    background = None
    if settings.subtract_background:
        background_path = experiment / "screen_background.npz"
        content, stat = stable_bytes(background_path)
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            background = np.asarray(data["data"], dtype=float)
        signatures[background_path] = stat
    processed = process_image(raw, settings, background)
    for path, stat in signatures.items():
        if signature(path) != stat:
            raise ValueError(f"Preview input changed while processing: {path.name}")
    return ProcessingPreview(sample_id, settings, raw, processed, source_hash)


def plot_processing_preview(
    figure: Figure, preview: ProcessingPreview, slice_index: int = -1
) -> None:
    """Draw a full-resolution processing result with compact image displays."""
    figure.clear()
    raw_axis, signal_axis, mask_axis, profile_axis = figure.subplots(2, 2).flat
    processed = preview.processed
    x_mm_per_pixel = preview.settings.pixel_size_x * 1e3
    y_mm_per_pixel = preview.settings.pixel_size_y * 1e3
    x0, y0 = processed.roi_origin
    height, width = processed.image.shape
    if slice_index >= preview.settings.num_slices or preview.settings.num_slices == 1:
        slice_index = -1

    def show_image(
        axis: Axes,
        image: NDArray[Any],
        title: str,
        origin: tuple[int, int],
        cmap: str,
        use_pixel_units: bool = False,
    ) -> None:
        # Downsampling affects display only; process_image always uses the full array.
        """Downsample only the display while preserving calibrated coordinates and ROI origin."""
        step = max(1, math.ceil(max(image.shape) / 900))
        x, y = origin
        x_scale = 1.0 if use_pixel_units else x_mm_per_pixel
        y_scale = 1.0 if use_pixel_units else y_mm_per_pixel
        unit = "px" if use_pixel_units else "mm"
        axis.imshow(
            image[::step, ::step],
            origin="lower",
            cmap=cmap,
            aspect="auto",
            extent=(
                (y - 0.5) * y_scale,
                (y + image.shape[1] - 0.5) * y_scale,
                (x - 0.5) * x_scale,
                (x + image.shape[0] - 0.5) * x_scale,
            ),
        )
        axis.set(title=title, xlabel=rf"$y$ [{unit}]", ylabel=rf"$x$ [{unit}]")

    show_image(
        raw_axis,
        orient_image(preview.raw, preview.settings),
        "oriented input — orange ROI",
        (0, 0),
        "viridis",
        use_pixel_units=True,
    )
    raw_axis.add_patch(
        Rectangle(
            (y0 - 0.5, x0 - 0.5),
            width,
            height,
            fill=False,
            edgecolor="orange",
            linewidth=1.2,
        )
    )
    show_image(signal_axis, processed.image, "processed signal", (x0, y0), "viridis")
    show_image(mask_axis, processed.mask, "signal mask", (x0, y0), "gray")
    for boundary in processed.bounds[1:-1]:
        signal_axis.axvline(
            (y0 + boundary - 0.5) * y_mm_per_pixel,
            color="white",
            alpha=0.7,
            linewidth=0.6,
        )
    start, stop = processed.tail_start or 0, width
    if slice_index >= 0:
        start, stop = processed.bounds[slice_index : slice_index + 2]
    if slice_index >= 0 or processed.tail_start is not None:
        signal_axis.axvspan(
            (y0 + start - 0.5) * y_mm_per_pixel,
            (y0 + stop - 0.5) * y_mm_per_pixel,
            color="red",
            alpha=0.12,
        )
    selected = processed.image[:, start:stop]
    for axis, name, mm_per_pixel in (
        (1, "x", x_mm_per_pixel),
        (0, "y", y_mm_per_pixel),
    ):
        profile = selected.sum(axis=axis)
        if profile.size and profile.max() > 0:
            profile_axis.plot(
                np.arange(profile.size) * mm_per_pixel,
                profile / profile.max(),
                label=rf"${name}$",
            )
    if profile_axis.lines:
        profile_axis.legend()
    region = "projected" if slice_index < 0 else f"slice {slice_index}"
    if slice_index in processed.variances:
        vx, vy = processed.variances[slice_index]
        title = (
            rf"{region}: $\sigma_x = {np.sqrt(vx) * 1e6:.4g}\,\mathrm{{\mu m}}$"
            "\n"
            rf"$\sigma_y = {np.sqrt(vy) * 1e6:.4g}\,\mathrm{{\mu m}}$"
        )
    else:
        title = f"{region}: profile fit unavailable"
    profile_axis.set(
        title=title,
        xlabel="position within selected region [mm]",
        ylabel="intensity / peak",
    )
    # Pack title fragments so only the preview label is emphasized.
    heading = HPacker(
        children=[
            TextArea(
                "preview of current controls",
                textprops={"color": "red", "weight": "bold", "size": 9},
            ),
            TextArea(f" — {preview.sample_id}", textprops={"size": 9}),
        ],
        align="center",
        pad=0,
        sep=0,
    )
    figure.add_artist(
        AnchoredOffsetbox(
            loc="upper center",
            child=heading,
            frameon=False,
            pad=0,
            borderpad=0,
            bbox_to_anchor=(0.5, 0.98),
            bbox_transform=figure.transFigure,
        )
    )
    figure.suptitle(
        f"profile fit: {preview.settings.fit_type.label}; full-resolution processing",
        fontsize=9,
        y=0.95,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))


@dataclass
class SliceBeamWidths:
    """Mean screen widths and sample standard deviations, both in micrometres."""

    slice_index: int
    configuration_indices: list[int]
    mean_um: Array
    std_um: Array


def beam_widths_at_screen(
    snapshot: Snapshot, plane: Plane | str
) -> list[SliceBeamWidths]:
    """Match notebook aggregation: average shot widths, then compute std(ddof=1)."""
    plane = Plane(plane)
    num_slices = snapshot.document["settings"]["num_slices"]
    slice_indices = {-1} if num_slices == 1 else range(num_slices)
    groups: dict[int, dict[int, list[float]]] = {}
    for shot in snapshot.shots:
        for index, variances in shot.variances.items():
            if index not in slice_indices:
                continue
            variance = variances[plane.variance_index]
            if math.isfinite(variance) and variance >= 0:
                configurations = groups.setdefault(index, {})
                configurations.setdefault(shot.config_index, []).append(
                    np.sqrt(variance) * 1e6
                )

    results: list[SliceBeamWidths] = []
    for index, configurations in sorted(groups.items()):
        indices = sorted(configurations)
        widths = [np.asarray(configurations[config]) for config in indices]
        results.append(
            SliceBeamWidths(
                slice_index=index,
                configuration_indices=indices,
                mean_um=np.array([values.mean() for values in widths]),
                # A single shot has no measured spread; omit its error bar.
                std_um=np.array(
                    [
                        values.std(ddof=1) if len(values) > 1 else np.nan
                        for values in widths
                    ]
                ),
            )
        )
    return results


def plot_snapshot(
    figure: Figure,
    snapshot: Snapshot,
    plane: Plane | str = Plane.X,
) -> None:
    """Plot all slices, or the projected beam in one-slice mode, for live views and exports."""
    plane = Plane(plane)
    figure.clear()
    grid = figure.add_gridspec(2, 3)
    width_axis = figure.add_subplot(grid[0, 0])
    width_3d_axis = figure.add_subplot(grid[0, 1], projection="3d")
    ellipse_axis = figure.add_subplot(grid[0, 2])
    emit_axis, beta_axis, mismatch_axis = (
        figure.add_subplot(grid[1, column]) for column in range(3)
    )
    chi_axis = emit_axis.twinx()
    alpha_axis = beta_axis.twinx()
    count = snapshot.document["settings"]["num_slices"]
    mode = "projected" if count == 1 else "sliced"
    colors = colormaps["coolwarm"](np.linspace(0, 1, count))
    indices = {-1} if count == 1 else range(count)
    valid_fits = sorted(
        (
            result
            for result in snapshot.fits
            if result["status"] == FitStatus.VALID
            and result["plane"] == plane
            and result["slice_index"] in indices
        ),
        key=lambda result: result["slice_index"],
    )
    reconstruction_indices = [max(0, result["slice_index"]) for result in valid_fits]

    def label_reconstruction_axis(axis: Axes) -> None:
        if count == 1:
            axis.set_xticks([0], ["projected"])
        else:
            axis.set_xlabel("equal-charge slice index")

    def show_reconstruction_placeholder(axis: Axes) -> None:
        """Explain why the selected plane has no valid reconstruction to display."""
        match plane:
            case Plane.Y if count > 1:
                reason = r"sliced $y$ reconstruction is not performed"
            case _:
                reason = "waiting for valid reconstructions; see Fit diagnostics"
        axis.text(
            0.5,
            0.5,
            reason,
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=8,
            wrap=True,
        )

    def draw_beam_widths() -> None:
        """Show mean slice widths and shot spread in the notebook 2D and waterfall views."""
        series = beam_widths_at_screen(snapshot, plane)
        # Match the notebook's back-to-front painting order and waterfall view.
        for widths in reversed(series):
            projected = widths.slice_index < 0
            index = 0 if projected else widths.slice_index
            color = colors[index]
            label = "projected" if projected else f"slice {index}"
            x = np.asarray(widths.configuration_indices)
            y = np.full(len(x), index, dtype=float)
            z, error = widths.mean_um, widths.std_um
            width_axis.errorbar(x, z, yerr=error, fmt="o-", label=label, color=color)
            width_3d_axis.plot(x, y, z, "-o", lw=2, color=color)
            for xi, yi, zi, ei in zip(x, y, z, error, strict=True):
                if math.isfinite(ei):
                    width_3d_axis.plot(
                        [xi, xi], [yi, yi], [zi - ei, zi + ei], color="k", lw=0.8
                    )
            width_3d_axis.plot(x, y, np.zeros_like(z), "--", color=color, alpha=0.3)
        if series:
            width_axis.legend(fontsize=6, ncol=max(1, math.ceil(len(series) / 4)))
            width_axis.margins(y=0.3)
        else:
            width_axis.text(
                0.5,
                0.5,
                "waiting for processed widths",
                ha="center",
                va="center",
                transform=width_axis.transAxes,
            )
        width_axis.set(
            title=r"screen widths — mean $\pm$ shot std",
            xlabel="configuration index",
            ylabel=rf"$\sigma_{{{plane}}}$ [$\mathrm{{\mu m}}$]",
        )
        width_3d_axis.set(
            title="screen widths per slice (3D)",
            xlabel="configuration index",
            ylabel="slice index",
            zlabel=rf"$\sigma_{{{plane}}}$ [$\mathrm{{\mu m}}$]",
        )
        if count == 1:
            width_3d_axis.set_yticks([0], ["projected"])
        width_3d_axis.view_init(elev=20, azim=-65)

    def draw_emittance() -> None:
        """Plot normalized emittance with conditional errors and the notebook chi-squared score."""
        if valid_fits:
            emit_axis.errorbar(
                reconstruction_indices,
                [result["normalized_emittance"] * 1e6 for result in valid_fits],
                yerr=[
                    (
                        np.nan
                        if result["normalized_emittance_error"] is None
                        else result["normalized_emittance_error"] * 1e6
                    )
                    for result in valid_fits
                ],
                fmt="o-",
                color="green",
                capsize=2,
            )
        else:
            show_reconstruction_placeholder(emit_axis)

        # The notebook divides chi-squared by the usable shot count, not fit DoF.
        scores = [
            (
                result["chi_squared"] / result["sample_count"]
                if "chi_squared" in result and result["sample_count"] > 0
                else np.nan
            )
            for result in valid_fits
        ]
        if any(math.isfinite(score) for score in scores):
            chi_axis.plot(reconstruction_indices, scores, "o-", color="black")
        else:
            match snapshot.document["settings"]["regression"]:
                case RegressionType.LINEAR:
                    chi_axis.text(
                        0.98,
                        0.02,
                        r"$\chi^2$ unavailable for linear regression",
                        ha="right",
                        transform=chi_axis.transAxes,
                        fontsize=7,
                    )
        emit_axis.set_title(
            rf"reconstruction: {mode} $\varepsilon_{{n,{plane}}}$", fontsize=9
        )
        emit_axis.set_ylabel(rf"$\varepsilon_{{n,{plane}}}$ [mm mrad]", color="green")
        emit_axis.tick_params(axis="y", labelcolor="green")
        emit_axis.set_ylim(bottom=0)
        chi_axis.set_ylabel(r"$\chi^2/N_{\mathrm{samples}}$", color="black")
        chi_axis.set_ylim(bottom=0)
        emit_axis.grid(False)
        chi_axis.grid(False)
        label_reconstruction_axis(emit_axis)

    def draw_twiss_parameters() -> None:
        """Show reconstructed beta and alpha with separate axes and propagated errors."""

        def draw_parameter(
            axis: Axes, name: Literal["beta", "alpha"], color: str
        ) -> None:
            """Plot one fitted Twiss parameter with its matching uncertainty and axis color."""
            if valid_fits:
                axis.errorbar(
                    reconstruction_indices,
                    [result[name] for result in valid_fits],
                    yerr=[
                        (
                            errors.get(name, np.nan)
                            if (errors := result.get("errors")) is not None
                            else np.nan
                        )
                        for result in valid_fits
                    ],
                    fmt="o-",
                    color=color,
                    capsize=2,
                )
            axis.tick_params(axis="y", labelcolor=color)
            axis.grid(False)

        draw_parameter(beta_axis, "beta", "blue")
        draw_parameter(alpha_axis, "alpha", "red")
        beta_axis.set_ylabel(rf"$\beta_{{{plane}}}$ [m]", color="blue")
        alpha_axis.set_ylabel(rf"$\alpha_{{{plane}}}$", color="red")
        beta_axis.set_ylim(bottom=0)
        beta_axis.set_title(
            rf"reconstruction: {mode} $\beta_{{{plane}}}$ & $\alpha_{{{plane}}}$",
            fontsize=9,
        )
        label_reconstruction_axis(beta_axis)
        if not valid_fits:
            show_reconstruction_placeholder(beta_axis)

    def draw_slice_ellipses() -> None:
        """Overlay fitted RMS ellipses, as in notebook show_slice_elippses."""
        theta = np.linspace(0, 2 * np.pi, 100)
        for result in valid_fits:
            covariance = np.asarray(result["transverse_covariance"])
            ellipse = np.linalg.cholesky(covariance) @ np.stack(
                [np.cos(theta), np.sin(theta)]
            )
            index = result["slice_index"]
            ellipse_axis.plot(
                ellipse[0] * 1e3,
                ellipse[1] * 1e3,
                color=colors[max(0, index)],
                label="projected" if index < 0 else f"slice {index}",
            )
        if valid_fits:
            ellipse_axis.legend(
                fontsize=6,
                ncol=max(1, math.ceil(len(valid_fits) / 4)),
                loc="upper center",
            )
            # Reserve room above the ellipses for up to four legend rows.
            ellipse_axis.margins(x=0.1, y=0.1)
            bottom, top = ellipse_axis.get_ylim()
            legend_rows = min(4, len(valid_fits))
            ellipse_axis.set_ylim(bottom, top + (top - bottom) * 0.1 * legend_rows)
        else:
            match plane:
                case Plane.Y if count > 1:
                    reason = r"sliced $y$ reconstruction is not performed"
                case _ if count == 1 and projected_fit is not None:
                    reason = (
                        f"{projected_fit['status']}: {projected_fit.get('reason', '')}"
                    )
                case _:
                    reason = "no valid slice reconstructions yet; see Fit diagnostics"
            ellipse_axis.text(
                0.5,
                0.5,
                textwrap.fill(reason, 42),
                ha="center",
                va="center",
                transform=ellipse_axis.transAxes,
                fontsize=9,
            )
        title = "reconstruction: " + (
            rf"projected ${plane}$ beam ellipse"
            if count == 1
            else rf"slice beam ${plane}$ ellipses — ({len(valid_fits)}/{count} valid)"
        )
        ellipse_axis.set_title(title, fontsize=9)
        ellipse_axis.set(
            xlabel=rf"${plane}$ [mm]",
            ylabel=rf"${plane}^\prime$ [mrad]",
        )

    def draw_mismatch() -> None:
        """Compare fitted Twiss to the archived design reference, with unity indicating a match."""
        reference = (snapshot.design_twiss or {}).get(plane)
        mismatch_axis.set_title(
            rf"reconstruction: {mode} ${plane}$ mismatch", fontsize=9
        )
        mismatch_axis.set_ylabel(r"$\zeta$")
        label_reconstruction_axis(mismatch_axis)
        if reference is None:
            reason = "design Twiss unavailable at the reconstruction start"
            if snapshot.document.get("design_twiss_error"):
                reason += f"\n{snapshot.document['design_twiss_error']}"
            mismatch_axis.text(
                0.5,
                0.5,
                reason,
                ha="center",
                va="center",
                transform=mismatch_axis.transAxes,
                fontsize=8,
                wrap=True,
            )
        elif valid_fits:
            values = [
                mismatch_zeta(
                    result["beta"],
                    result["alpha"],
                    reference["beta"],
                    reference["alpha"],
                )
                for result in valid_fits
            ]
            mismatch_axis.plot(reconstruction_indices, values, "o-", color="tab:purple")
            mismatch_axis.axhline(
                1,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                label=r"matched ($\zeta = 1$)",
            )
            mismatch_axis.legend(fontsize=8)
        else:
            show_reconstruction_placeholder(mismatch_axis)

    projected_fit = next(
        (
            result
            for result in snapshot.fits
            if result["plane"] == plane and result["slice_index"] == -1
        ),
        None,
    )
    draw_beam_widths()
    draw_emittance()
    draw_twiss_parameters()
    draw_slice_ellipses()
    draw_mismatch()
    slice_label = "projected" if count == 1 else f"{count} slices"
    figure.suptitle(
        f"experiment: {Path(snapshot.document['experiment']).name} | {snapshot.document['created_utc']}\n"
        f"{snapshot.document['settings']['regression']} | {slice_label} | "
        rf"$E = {snapshot.document['energy_GeV']:g}$ GeV | {len(snapshot.shots)} unique shots",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))


@dataclass(frozen=True)
class LatticePreview:
    """Target lattices and optics propagated from one shared entrance state."""

    initial: Twiss
    lattices: list[MagneticLattice]
    profiles: list[list[Twiss]]
    phases: Array  # configuration, plane; radians modulo 2 pi


def lattice_design_values(initial: Twiss) -> dict[str, float]:
    """Resolve entrance design values; use the simulation emittance if unspecified."""
    values: dict[str, float] = {}
    for plane in Plane:
        normalized = (
            float(getattr(initial, f"emit_{plane}")) * initial.E / REST_ENERGY_GEV
        )
        values[f"emit_{plane}"] = (
            normalized if normalized > 0 else SimulationSettings.normalized_emittance
        ) * 1e6  # GUI units: mm mrad
        for name in ("beta", "alpha"):
            values[f"{name}_{plane}"] = float(getattr(initial, f"{name}_{plane}"))
    return values


def lattice_preview_initial(
    design: Twiss, energy: float, controls: Mapping[str, str]
) -> Twiss:
    """Apply editable design/numeric selectors without changing the design reference."""
    initial = copy.deepcopy(design)
    initial.E = energy
    defaults = lattice_design_values(design)
    for name, default in defaults.items():
        text = controls.get(name, "Design").strip()
        value = default if text == "Design" else float(text)
        if not math.isfinite(value) or (not name.startswith("alpha") and value <= 0):
            raise ValueError(
                f"{name} must be finite"
                + (" and positive" if not name.startswith("alpha") else "")
            )
        if name.startswith("emit"):
            value *= 1e-6 * REST_ENERGY_GEV / energy
        setattr(initial, name, value)
    return initial


def make_lattice_preview(document: dict[str, Any], initial: Twiss) -> LatticePreview:
    """Propagate all target configurations and derive phase advance from their maps.

    The atan2 convention follows 04_make_lattice.ipynb and remains valid for
    user-supplied, unmatched Twiss values. Phases are measured from the actual
    entrance, independently of the design state's accumulated phase.
    """
    lattices: list[MagneticLattice] = []
    profiles: list[list[Twiss]] = []
    phases = np.empty((len(document["low_level_params"]), 2))
    for index, target in enumerate(document["low_level_params"]):
        lattice = make_lattice(LatticeLowLevelParams(**target))
        profile = twiss(lattice, copy.deepcopy(initial))
        matrix = lattice.transfer_maps(energy=initial.E)[1]
        for column, plane in enumerate(Plane):
            offset = plane.matrix_index
            beta0 = getattr(initial, f"beta_{plane}")
            beta1 = getattr(profile[-1], f"beta_{plane}")
            alpha0 = getattr(initial, f"alpha_{plane}")
            sine = matrix[offset, offset + 1] / np.sqrt(beta0 * beta1)
            cosine = matrix[offset, offset] * np.sqrt(beta0 / beta1) - alpha0 * sine
            phases[index, column] = np.arctan2(sine, cosine) % (2 * np.pi)
        lattices.append(lattice)
        profiles.append(profile)
    if not np.isfinite(phases).all():
        raise ValueError("Nonfinite phase advance in lattice optics")
    return LatticePreview(copy.deepcopy(initial), lattices, profiles, phases)


def plot_lattice_phases(
    axis: Axes,
    preview: LatticePreview,
    plane: Plane,
    selected: int,
    normalized: bool,
) -> int:
    """Compare every configuration on its one-sigma entrance ellipse."""
    beta = getattr(preview.initial, f"beta_{plane}")
    alpha = getattr(preview.initial, f"alpha_{plane}")
    emittance = getattr(preview.initial, f"emit_{plane}")

    def coordinates(phase: Array) -> tuple[Array, Array]:
        """Use physical mm/mrad or dimensionless Courant–Snyder coordinates."""
        if normalized:
            return np.cos(phase), np.sin(phase)
        return (
            1e3 * np.sqrt(emittance * beta) * np.cos(phase),
            1e3 * np.sqrt(emittance / beta) * (-alpha * np.cos(phase) + np.sin(phase)),
        )

    theta = np.linspace(0, 2 * np.pi, 361)
    axis.plot(*coordinates(theta), color="0.4", label="Initial Twiss ellipse")
    axis.scatter(
        *coordinates(np.array([0.0])),
        marker="x",
        color="black",
        s=65,
        label="Δμ = 0",
        zorder=4,
    )
    column = 0 if plane == Plane.X else 1
    phases = preview.phases[:, column]
    positions = coordinates(phases)
    colors = colormaps["viridis"](np.linspace(0, 1, len(phases)))
    points = axis.scatter(*positions, c=colors, s=45, zorder=3)
    current = (positions[0][selected], positions[1][selected])
    axis.annotate(
        str(selected), current, xytext=(7, 7), textcoords="offset points", fontsize=9
    )
    axis.scatter(
        *current,
        color="tab:red",
        marker="*",
        s=210,
        edgecolors="black",
        zorder=5,
        label=f"ID {selected}: Δμ = {phases[selected]:.3f} rad ({np.degrees(phases[selected]):.1f}°)",
    )
    axis.plot(
        [0, current[0]], [0, current[1]], color="tab:red", alpha=0.5, linestyle="--"
    )
    axis.set_title(rf"$\Delta\mu_{plane}$ at lattice start")
    axis.set_xlabel(rf"$u_{plane}$" if normalized else rf"${plane}$ [mm]")
    axis.set_ylabel(rf"$u'_{plane}$" if normalized else rf"${plane}'$ [mrad]")
    if normalized:
        axis.set_aspect("equal", adjustable="box")
    axis.margins(0.15)
    axis.axhline(0, color="gray", alpha=0.2)
    axis.axvline(0, color="gray", alpha=0.2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=8)
    tooltip = axis.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.95},
        fontsize=8,
        zorder=10,
    )
    tooltip.set_visible(False)

    def hover(event: MouseEvent) -> None:
        """Identify all overlapping configurations without crowding the ellipse."""
        hit, info = points.contains(event) if event.inaxes is axis else (False, {})
        was_visible = tooltip.get_visible()
        if hit:
            indices = info["ind"]
            index = int(indices[0])
            tooltip.xy = (positions[0][index], positions[1][index])
            tooltip.set_text(
                "\n".join(
                    f"ID {i}: {phases[i]:.4f} rad ({np.degrees(phases[i]):.2f}°)"
                    for i in indices
                )
            )
        tooltip.set_visible(hit)
        if hit or was_visible:
            axis.figure.canvas.draw_idle()

    return axis.figure.canvas.mpl_connect("motion_notify_event", hover)


class LatticePreviewPanel:
    """Own the lattice controls and plots; queue optics work on the desktop worker."""

    def __init__(self, parent: ttk.Frame, experiment: Path) -> None:
        self.experiment = experiment
        self.document: dict[str, Any] | None = None
        self.design: Twiss | None = None
        self.preview: LatticePreview | None = None
        self.revision = 0
        self.pending: tuple[int, dict[str, Any], Twiss] | None = None
        self.timer: str | None = None
        self.hover_callbacks: set[int] = set()
        self.parent = parent
        self.config_id = tk.StringVar(parent, value="0")
        self.coordinates = tk.StringVar(parent, value="Original")
        self.status = tk.StringVar(parent, value="Waiting for lattice configurations")
        self.variables: dict[str, tk.StringVar] = {}
        self.boxes: dict[str, ttk.Combobox] = {}
        self.design_labels: dict[str, ttk.Label] = {}
        self._build()

    def _build(self) -> None:
        """Place the six editable design selectors beside a three-panel plot grid."""
        sidebar = ttk.Frame(self.parent, padding=10, width=270)
        sidebar.pack(side="left", fill="y")
        plots = ttk.Frame(self.parent)
        plots.pack(side="right", fill="both", expand=True)
        ttk.Label(sidebar, text="Lattice configuration ID").pack(anchor="w")
        self.config_box = ttk.Combobox(
            sidebar, textvariable=self.config_id, state="readonly", width=24
        )
        self.config_box.pack(fill="x", pady=(3, 12))
        ttk.Label(sidebar, text="Initial Twiss at lattice start").pack(anchor="w")
        for plane in Plane:
            group = ttk.LabelFrame(sidebar, text=f"{plane.upper()} plane", padding=6)
            group.pack(fill="x", pady=6)
            for name, label in (
                ("emit", "Normalized ε [mm mrad]"),
                ("beta", "β [m]"),
                ("alpha", "α"),
            ):
                key = f"{name}_{plane}"
                variable = tk.StringVar(self.parent, value="Design")
                ttk.Label(group, text=label).pack(anchor="w")
                box = ttk.Combobox(
                    group, textvariable=variable, values=("Design",), width=24
                )
                box.pack(fill="x")
                design_label = ttk.Label(group, text="Design: —")
                design_label.pack(anchor="w", pady=(0, 4))
                self.variables[key], self.boxes[key], self.design_labels[key] = (
                    variable,
                    box,
                    design_label,
                )
                variable.trace_add("write", self.schedule)
        ttk.Label(sidebar, text="Phase space coordinates").pack(
            anchor="w", pady=(10, 0)
        )
        ttk.Combobox(
            sidebar,
            textvariable=self.coordinates,
            values=("Original", "Normalized"),
            state="readonly",
            width=24,
        ).pack(fill="x", pady=3)
        ttk.Label(
            sidebar,
            text="Choose Design or type a value.\nNormalized coordinates use a unit circle.\nWhen design emittance is unspecified,\nεₙ = 1 mm mrad is used.",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=10)
        ttk.Label(plots, textvariable=self.status, wraplength=1050, padding=6).pack(
            fill="x"
        )
        self.figure = Figure(figsize=(11, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plots)
        toolbar = NavigationToolbar2Tk(self.canvas, plots, pack_toolbar=False)
        toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.config_id.trace_add("write", lambda *args: self.draw())
        self.coordinates.trace_add("write", lambda *args: self.draw())

    def refresh_document(self) -> None:
        """Reload independently of sample availability, retaining valid draft controls."""
        try:
            content, _ = stable_bytes(self.experiment / "lattice_configs.json")
            document = configuration_document(content)
            if document == self.document:
                return
            self.design = make_twiss0(document["low_level_params"][0]["start"])
            self.document = document
            defaults = lattice_design_values(self.design)
            for name, value in defaults.items():
                self.boxes[name].configure(values=("Design", f"{value:.8g}"))
                self.design_labels[name].configure(text=f"Design: {value:.6g}")
            ids = tuple(
                str(index) for index in range(len(document["low_level_params"]))
            )
            self.config_box.configure(values=ids)
            if self.config_id.get() not in ids:
                self.config_id.set(ids[0])
            self.schedule()
        except Exception as error:
            self.document = None
            self.invalidate(f"Lattice preview: {error}")

    def invalidate(self, message: str) -> None:
        """Discard obsolete work and plots while input is changing or invalid."""
        self.revision += 1
        self.pending = None
        self.preview = None
        self.status.set(message)
        self.clear()
        self.canvas.draw_idle()

    def clear(self) -> None:
        """Disconnect Ocelot hover handlers before replacing their axes."""
        for callback in self.hover_callbacks:
            self.canvas.mpl_disconnect(callback)
        self.hover_callbacks.clear()
        self.figure.clear()

    def schedule(self, *args: str) -> None:
        """Debounce numerical edits and invalidate results from earlier controls."""
        self.invalidate("Updating lattice preview…")
        if self.timer is not None:
            self.parent.after_cancel(self.timer)
        self.timer = self.parent.after(350, self.queue)

    def queue(self) -> None:
        """Validate draft values before submitting propagation to the worker."""
        self.timer = None
        if self.document is None or self.design is None:
            return
        try:
            initial = lattice_preview_initial(
                self.design,
                float(self.document["high_level_params"][0]["beam_energy"]),
                {name: variable.get() for name, variable in self.variables.items()},
            )
            self.pending = (self.revision, self.document, initial)
        except Exception as error:
            self.invalidate(f"Lattice preview: {error}")

    def display(self, revision: int | None, preview: LatticePreview) -> None:
        """Accept only a result for the current configuration document and controls."""
        if revision == self.revision:
            self.preview = preview
            self.draw()

    def draw(self) -> None:
        """Use the selected lattice above the two phase-space comparison panels."""
        if self.preview is None:
            return
        self.clear()
        selected = int(self.config_id.get())
        upper, lower = self.figure.subfigures(2, 1, height_ratios=(1.1, 1))
        lattice = self.preview.lattices[selected]
        upper.suptitle(
            f"Configuration {selected}: {lattice.sequence[0].id} → {lattice.sequence[-1].id}",
            fontsize=10,
        )
        before = set(self.canvas.callbacks.callbacks.get("motion_notify_event", {}))
        show_lattice(
            lattice,
            self.preview.initial,
            figure=upper,
            twisses=self.preview.profiles[selected],
            show=False,
            diagnostics=False,
        )
        self.hover_callbacks = (
            set(self.canvas.callbacks.callbacks.get("motion_notify_event", {})) - before
        )
        axes = lower.subplots(1, 2)
        lower.subplots_adjust(left=0.09, right=0.97, bottom=0.30, top=0.90, wspace=0.32)
        for axis, plane in zip(axes, Plane):
            callback = plot_lattice_phases(
                axis,
                self.preview,
                plane,
                selected,
                self.coordinates.get() == "Normalized",
            )
            self.hover_callbacks.add(callback)
        self.status.set(
            f"Target lattice configurations | {len(self.preview.lattices)} IDs | E = {self.preview.initial.E:g} GeV | Red star: selected ID. Hover points for all IDs and phase advances."
        )
        self.canvas.draw_idle()

    def close(self) -> None:
        """Cancel edits queued for widgets about to be destroyed."""
        if self.timer is not None:
            self.parent.after_cancel(self.timer)


@dataclass(frozen=True)
class PreviewRequest:
    """A sample and draft configuration captured when a preview is queued."""

    revision: int
    sample_id: str
    settings: AnalysisSettings


@dataclass(frozen=True)
class WorkerJob:
    """A future and the exact analysis/revision that owns its result."""

    operation: WorkerOperation
    future: Future[Any]
    analysis: LiveAnalysis
    preview_revision: int | None = None


@dataclass
class AppState:
    """Mutable desktop state, independent of Tk widgets and their draft input values."""

    analysis: LiveAnalysis
    snapshot: Snapshot | None = None
    job: WorkerJob | None = None
    selected_sample: str = ""
    pending_refresh: bool = True
    last_refresh: float = -math.inf
    closing: bool = False
    slice_update_timer: str | None = None
    resolution_update_timer: str | None = None
    processing_preview_timer: str | None = None
    processing_preview_revision: int = 0
    pending_processing_preview: PreviewRequest | None = None


@dataclass(frozen=True)
class DesktopView:
    """Typed Tk controls and Matplotlib canvases used by the desktop controller."""

    root: tk.Tk
    status: tk.StringVar
    regression: tk.StringVar
    slices: tk.StringVar
    plane: tk.StringVar
    watching: tk.BooleanVar
    include_raw: tk.BooleanVar
    processing_sample: tk.StringVar
    follow_newest: tk.BooleanVar
    figure: Figure
    canvas: FigureCanvasTkAgg
    samples_tree: ttk.Treeview
    fit_text: tk.Text
    form_variables: dict[str, tk.StringVar | tk.BooleanVar]
    processing_sample_box: ttk.Combobox
    processing_figure: Figure
    processing_canvas: FigureCanvasTkAgg
    processing_status: tk.StringVar
    preparation: tk.Text
    output_path: tk.StringVar
    lattice_preview: LatticePreviewPanel | None = None


class DesktopController:
    """Coordinate GUI actions and one worker while keeping result ownership explicit."""

    def __init__(
        self, state: AppState, view: DesktopView, executor: Executor | None = None
    ) -> None:
        """Keep the applied state, UI handles, and injectable worker in separate objects."""
        self.state = state
        self.view = view
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="emittance")
            if executor is None
            else executor
        )
        self.messages: queue.Queue[str] = queue.Queue()

    def submit(
        self,
        operation: WorkerOperation,
        function: Callable[..., Any],
        *args: Any,
        preview_revision: int | None = None,
    ) -> None:
        """Start one job and capture its owner before GUI settings can change."""
        if self.state.closing:
            raise RuntimeError("The desktop is closing")
        if self.state.job is not None:
            raise RuntimeError("Wait for the active operation before starting another")
        analysis = self.state.analysis
        future = self.executor.submit(function, *args)
        self.state.job = WorkerJob(operation, future, analysis, preview_revision)

    def current_settings(self) -> AnalysisSettings:
        """Parse all draft processing controls into a validated analysis configuration."""
        values = AnalysisSettings.parse_form_values(
            {
                name: variable.get()
                for name, variable in self.view.form_variables.items()
            }
        )
        values.update(
            regression=REGRESSION_CHOICES[self.view.regression.get()],
            num_slices=parse_slice_count(self.view.slices.get()),
        )
        config = AnalysisSettings(**values)
        config.validate()
        return config

    def show_processing_placeholder(self, message: str) -> None:
        """Replace the preview with a clear waiting or error message."""
        self.view.processing_figure.clear()
        axis = self.view.processing_figure.add_subplot(111)
        axis.text(
            0.5,
            0.5,
            message[:1].lower() + message[1:],
            ha="center",
            va="center",
            transform=axis.transAxes,
            wrap=True,
        )
        axis.set_axis_off()
        self.view.processing_canvas.draw_idle()

    def queue_processing_preview(self) -> None:
        """Queue the current sample and validated draft settings after the debounce delay."""
        self.state.processing_preview_timer = None
        if self.state.closing:
            return
        try:
            config = self.current_settings()
        except Exception as error:
            self.state.pending_processing_preview = None
            self.view.processing_status.set(f"Preview: {error}")
            self.show_processing_placeholder(
                "Correct the processing controls to preview this sample"
            )
            return
        sample_id = self.view.processing_sample.get()
        if not sample_id:
            self.view.processing_status.set("Waiting for a sample")
            self.show_processing_placeholder("Waiting for sample data")
            return
        self.state.pending_processing_preview = PreviewRequest(
            self.state.processing_preview_revision, sample_id, config
        )
        self.view.processing_status.set(
            "Preview queued — current controls; Apply updates scan results"
        )

    def schedule_processing_preview(self, *args: str) -> None:
        """Invalidate older previews and debounce a new preview of the current controls."""
        if self.state.closing:
            return
        self.state.processing_preview_revision += 1
        self.state.pending_processing_preview = None
        self.view.processing_status.set("Updating preview of current controls…")
        self.show_processing_placeholder("Updating preview…")
        self.state.processing_preview_timer = self.restart_timer(
            self.state.processing_preview_timer, 450, self.queue_processing_preview
        )

    def restart_timer(
        self, timer: str | None, delay_ms: int, callback: Callable[[], None]
    ) -> str:
        """Replace a pending GUI callback so repeated edits trigger one update."""
        if timer is not None:
            self.view.root.after_cancel(timer)
        return self.view.root.after(delay_ms, callback)

    def processing_sample_changed(self, *args: str) -> None:
        self.state.selected_sample = self.view.processing_sample.get()
        self.schedule_processing_preview()

    def update_processing_sample(self) -> None:
        """Refresh sample choices and follow the newest stable shot when requested."""
        snapshot = self.state.snapshot
        if snapshot is None:
            return
        available = tuple(snapshot.document["samples"])
        self.view.processing_sample_box.configure(values=available)
        candidate = self.view.processing_sample.get()
        if (self.view.follow_newest.get() or candidate not in available) and available:
            if snapshot.shots:
                newest = max(
                    snapshot.shots,
                    key=lambda shot: snapshot.signatures[
                        self.state.analysis.experiment
                        / "samples_raw"
                        / f"{shot.sample_id}.npz"
                    ][2],
                )
                candidate = newest.sample_id
            else:
                candidate = available[-1]
        if not available:
            candidate = ""
        if candidate != self.view.processing_sample.get():
            self.view.processing_sample.set(candidate)
        else:
            self.schedule_processing_preview()

    def redraw(self) -> None:
        if self.state.snapshot is not None:
            plot_snapshot(self.view.figure, self.state.snapshot, self.view.plane.get())
            self.view.canvas.draw_idle()

    def install_settings(self, config: AnalysisSettings) -> None:
        """Invalidate displayed results and queue reprocessing with a new configuration."""
        config.validate()
        self.state.analysis = LiveAnalysis(self.state.analysis.experiment, config)
        self.state.pending_refresh = True
        self.clear_reconstruction()
        mode = (
            "projected analysis"
            if config.num_slices == 1
            else f"{config.num_slices} slices"
        )
        self.view.status.set(f"Reprocessing {mode}; waiting for a current snapshot")
        self.schedule_processing_preview()

    def apply(self) -> None:
        try:
            config = self.current_settings()
            self.install_settings(config)
        except Exception as error:
            messagebox.showerror("Invalid processing settings", str(error))

    def commit_slice_count(self, event: tk.Event | None = None) -> None:
        """Apply a valid slice count automatically while retaining other unapplied edits."""
        if self.state.slice_update_timer is not None:
            self.view.root.after_cancel(self.state.slice_update_timer)
            self.state.slice_update_timer = None
        try:
            count = parse_slice_count(self.view.slices.get())
            if count != self.state.analysis.settings.num_slices:
                # Apply only this control; other edits still require Apply.
                self.install_settings(
                    replace(self.state.analysis.settings, num_slices=count)
                )
        except (ValueError, TypeError) as error:
            self.view.status.set(f"Slice count not applied: {error}")

    def commit_resolution(self, event: tk.Event | None = None) -> None:
        """Apply both camera scales automatically while retaining other unapplied edits."""
        if self.state.resolution_update_timer is not None:
            self.view.root.after_cancel(self.state.resolution_update_timer)
            self.state.resolution_update_timer = None
        try:
            values = {
                name: parse_strictly_positive_finite_float(
                    str(self.view.form_variables[name].get())
                )
                for name in CAMERA_RESOLUTION_PRESETS
            }
            if any(
                value != getattr(self.state.analysis.settings, name)
                for name, value in values.items()
            ):
                # Reprocess with the applied settings; leave other draft edits alone.
                self.install_settings(
                    replace(
                        self.state.analysis.settings,
                        pixel_size_x=values["pixel_size_x"],
                        pixel_size_y=values["pixel_size_y"],
                    )
                )
        except (ValueError, TypeError) as error:
            self.view.status.set(f"Camera calibration not applied: {error}")

    def schedule_resolution(self, *args: str) -> None:
        self.view.status.set(
            "Camera calibration changed; reprocessing will start when the input is valid"
        )
        self.state.resolution_update_timer = self.restart_timer(
            self.state.resolution_update_timer, 400, self.commit_resolution
        )

    def schedule_slice_count(self, *args: str) -> None:
        self.view.status.set(
            "Slice count changed; reprocessing will start when the input is valid"
        )
        self.state.slice_update_timer = self.restart_timer(
            self.state.slice_update_timer, 400, self.commit_slice_count
        )

    def select_sample(self, event: tk.Event | None = None) -> None:
        selection = self.view.samples_tree.selection()
        if selection:
            self.state.selected_sample = selection[0]
            self.view.follow_newest.set(False)
            if self.view.processing_sample.get() != self.state.selected_sample:
                self.view.processing_sample.set(self.state.selected_sample)

    def refresh_now(self) -> None:
        self.state.pending_refresh = True

    def choose_export(self) -> None:
        """Check applied settings, capture the displayed plane, and queue an atomic export."""

        def can_export() -> bool:
            """Check again after the directory dialog, which can run other GUI actions."""
            if self.state.snapshot is None:
                messagebox.showinfo(
                    "No current result", "Wait for a stable input snapshot"
                )
                return False
            if self.state.job is not None:
                self.view.status.set("Wait for the active operation before exporting")
                return False
            return True

        if not can_export():
            return
        try:
            if self.current_settings() != self.state.analysis.settings:
                raise ValueError(
                    "Processing controls have unapplied changes; click Apply before exporting"
                )
        except Exception as error:
            messagebox.showerror("Apply processing settings", str(error))
            return
        directory = filedialog.askdirectory(
            title="Parent directory for timestamped processed export",
            initialdir=self.state.analysis.experiment,
        )
        if not directory or not can_export():
            return
        snapshot = self.state.snapshot
        assert snapshot is not None  # can_export rechecked it after the dialog.
        view = {
            "sample_id": self.state.selected_sample,
            "plane": self.view.plane.get(),
        }
        # Render a separate figure now; the worker does not access the live canvas.
        exported_figure = Figure(figsize=(13, 7))
        plot_snapshot(exported_figure, snapshot, view["plane"])
        self.submit(
            WorkerOperation.EXPORT,
            self.state.analysis.export_bundle,
            snapshot,
            Path(directory),
            exported_figure,
            self.view.include_raw.get(),
            view,
        )
        self.view.status.set("Exporting timestamped data, configuration, and plots")

    def save_settings(self) -> None:
        """Save the current analysis and preparation settings to a user-selected JSON file."""

        try:
            document = json.loads(self.view.preparation.get("1.0", "end"))
            document["analysis"] = self.current_settings().to_dict()
            filename = filedialog.asksaveasfilename(
                title="Save app settings",
                defaultextension=".json",
                initialfile="settings.json",
            )
            if filename:
                atomic_json(Path(filename), document)
        except Exception as error:
            messagebox.showerror("Cannot save settings", str(error))

    def prepare(self, kind: Command) -> None:
        """Validate a new destination and queue optional lattice generation or simulation."""

        if self.state.job is not None:
            self.view.status.set("Wait for the active operation before preparation")
            return
        try:
            config = json.loads(self.view.preparation.get("1.0", "end"))
            output = Path(self.view.output_path.get()).expanduser().resolve()
            if output.exists():
                raise ValueError("Enter a new output experiment directory")
            match kind:
                case Command.LATTICE:
                    lattice = LatticeSettings.from_dict(config["lattice"])
                    self.submit(
                        WorkerOperation.PREPARATION,
                        generate_lattices,
                        output,
                        lattice,
                        self.messages.put,
                    )
                case Command.SIMULATE:
                    simulation = SimulationSettings.from_dict(config["simulation"])
                    self.submit(
                        WorkerOperation.PREPARATION,
                        simulate_experiment,
                        self.state.analysis.experiment,
                        output,
                        simulation,
                        self.current_settings(),
                        self.messages.put,
                    )
            self.view.status.set(f"Running {kind} preparation")
        except Exception as error:
            messagebox.showerror("Cannot start preparation", str(error))

    def display_snapshot(self, result: Snapshot) -> None:
        """Replace the sample table, diagnostics, and displayed snapshot together."""

        def sample_note(info: dict[str, Any]) -> str:
            """Prefer the exclusion reason, then duplicate or slice details."""
            if "reason" in info:
                return info["reason"]
            if "duplicate_of" in info:
                return info["duplicate_of"]
            errors = info.get("slice_errors")
            return str(errors) if errors else ""

        self.state.snapshot = result
        self.view.samples_tree.delete(*self.view.samples_tree.get_children())
        counts: dict[str, int] = {}
        for sample_id, info in result.document["samples"].items():
            sample_status = SampleStatus(info["status"]).value
            counts[sample_status] = counts.get(sample_status, 0) + 1
            self.view.samples_tree.insert(
                "",
                "end",
                iid=sample_id,
                text=sample_id,
                values=(
                    info["status"],
                    info.get("config_index", ""),
                    sample_note(info),
                ),
            )
        if self.state.selected_sample not in result.document["samples"]:
            self.state.selected_sample = (
                result.shots[0].sample_id if result.shots else ""
            )
        self.view.fit_text.delete("1.0", "end")
        self.view.fit_text.insert("end", json.dumps(result.fits, indent=2))
        mode = (
            "projected"
            if self.state.analysis.settings.num_slices == 1
            else f"{self.state.analysis.settings.num_slices} slices"
        )
        self.view.status.set(
            f"Snapshot {result.document['created_utc']} | {mode} | {counts} | "
            f"{result.document['refresh_seconds']:.2f} s | SP magnet setpoints | slices in image order"
        )
        self.update_processing_sample()

    def display_processing_preview(
        self, revision: int | None, sample_preview: ProcessingPreview
    ) -> None:
        """Publish a preview only while its sample and draft settings are still selected."""
        if (
            revision != self.state.processing_preview_revision
            or sample_preview.sample_id != self.view.processing_sample.get()
        ):
            return
        plot_processing_preview(self.view.processing_figure, sample_preview)
        self.view.processing_canvas.draw_idle()
        region_error = sample_preview.processed.slice_errors.get(-1)
        self.view.processing_status.set(
            f"Current controls — {sample_preview.sample_id}. Apply updates scan results."
            + (f" Profile unavailable: {region_error}" if region_error else "")
        )

    def show_worker_error(self, job: WorkerJob, error: Exception) -> None:
        """Clear current failed reconstructions and keep obsolete jobs from clearing newer results."""
        match job.operation:
            case WorkerOperation.LATTICE_PREVIEW:
                panel = self.view.lattice_preview
                if panel is not None and job.preview_revision == panel.revision:
                    panel.invalidate(f"Lattice preview: {error}")
                return
            case WorkerOperation.PROCESSING_PREVIEW:
                if job.preview_revision == self.state.processing_preview_revision:
                    self.view.processing_status.set(f"Preview: {error}")
                    self.show_processing_placeholder(
                        "Preview unavailable — see the message above"
                    )
                return
            case WorkerOperation.REFRESH if job.analysis is self.state.analysis:
                self.clear_reconstruction()
        self.view.status.set(f"{job.operation}: {type(error).__name__}: {error}")

    def consume_finished_operation(self) -> None:
        """Dispatch a completed job using its captured owner, then release the worker slot."""
        job = self.state.job
        if self.state.closing or job is None or not job.future.done():
            return
        try:
            result = job.future.result()
            match job.operation:
                case WorkerOperation.EXPORT:
                    self.view.status.set(f"Export complete: {result}")
                case WorkerOperation.PREPARATION:
                    self.view.status.set(
                        f"Preparation complete: {self.view.output_path.get()}"
                    )
                case WorkerOperation.PROCESSING_PREVIEW:
                    self.display_processing_preview(job.preview_revision, result)
                case WorkerOperation.LATTICE_PREVIEW:
                    if self.view.lattice_preview is not None:
                        self.view.lattice_preview.display(job.preview_revision, result)
                case WorkerOperation.REFRESH if job.analysis is self.state.analysis:
                    self.display_snapshot(result)
                    self.redraw()
        except Exception as error:
            self.show_worker_error(job, error)
        finally:
            self.state.job = None

    def start_pending_operation(self) -> None:
        """Prioritize a queued draft preview, then run an explicit or due live refresh."""
        if self.state.closing or self.state.job is not None:
            return
        now = time.monotonic()
        request = self.state.pending_processing_preview
        if request is not None:
            self.state.pending_processing_preview = None
            self.submit(
                WorkerOperation.PROCESSING_PREVIEW,
                preview_processing_sample,
                self.state.analysis.experiment,
                request.sample_id,
                request.settings,
                preview_revision=request.revision,
            )
            self.view.processing_status.set("Processing preview of current controls…")
        elif (
            self.view.lattice_preview is not None
            and self.view.lattice_preview.pending is not None
        ):
            panel = self.view.lattice_preview
            revision, document, initial = panel.pending
            panel.pending = None
            self.submit(
                WorkerOperation.LATTICE_PREVIEW,
                make_lattice_preview,
                document,
                initial,
                preview_revision=revision,
            )
        elif self.state.pending_refresh or (
            self.view.watching.get()
            and now - self.state.last_refresh
            >= self.state.analysis.settings.poll_seconds
        ):
            if self.view.lattice_preview is not None:
                self.view.lattice_preview.refresh_document()
            self.submit(WorkerOperation.REFRESH, self.state.analysis.refresh)
            self.state.pending_refresh = False
            self.state.last_refresh = now

    def tick(self) -> None:
        """Deliver worker progress and completions on the Tk thread, then schedule the next tick."""
        if self.state.closing:
            return
        while not self.messages.empty():
            self.view.status.set(self.messages.get_nowait())
        self.consume_finished_operation()
        self.start_pending_operation()
        self.view.root.after(150, self.tick)

    def clear_reconstruction(self) -> None:
        """Remove the snapshot and every displayed reconstruction value together."""
        self.state.snapshot = None
        self.view.figure.clear()
        self.view.canvas.draw_idle()
        self.view.fit_text.delete("1.0", "end")
        self.view.samples_tree.delete(*self.view.samples_tree.get_children())

    def close(self) -> None:
        """Stop timers and queued work before destroying the desktop widgets."""
        if self.state.closing:
            return
        self.state.closing = True
        if self.view.lattice_preview is not None:
            self.view.lattice_preview.close()
        for timer in (
            self.state.processing_preview_timer,
            self.state.slice_update_timer,
            self.state.resolution_update_timer,
        ):
            if timer is not None:
                self.view.root.after_cancel(timer)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.view.root.destroy()


def desktop(experiment: Path, settings_path: Path | None) -> None:
    """Tk controls and a Matplotlib canvas; all analysis runs in one worker."""

    settings = AnalysisSettings.load(settings_path)
    settings.validate()
    analysis = LiveAnalysis(experiment, settings)
    root = tk.Tk()
    root.title("EuXFEL T3 — live emittance analysis")
    root.geometry("1450x950")
    controls = ttk.Frame(root, padding=6)
    controls.pack(fill="x")
    status = tk.StringVar(value="Waiting for first stable data snapshot")
    regression = tk.StringVar(value=settings.regression.label)
    slices = tk.StringVar(
        value="1 = projected" if settings.num_slices == 1 else str(settings.num_slices)
    )
    plane = tk.StringVar(value=Plane.X)
    watching = tk.BooleanVar(value=True)
    include_raw = tk.BooleanVar(value=True)
    processing_sample = tk.StringVar()
    follow_newest = tk.BooleanVar(value=True)
    resolution_boxes: dict[str, ttk.Combobox] = {}

    def embed_plot(
        parent: ttk.Frame, size: tuple[int, int]
    ) -> tuple[Figure, FigureCanvasTkAgg]:
        """Embed a resizable Matplotlib canvas with its plot-export toolbar."""
        figure = Figure(figsize=size)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return figure, canvas

    def build_controls() -> None:
        ttk.Checkbutton(controls, text="Live refresh", variable=watching).grid(
            row=0, column=4
        )
        ttk.Checkbutton(
            controls, text="Include raw inputs in export", variable=include_raw
        ).grid(row=0, column=5)

    def build_tabs() -> tuple[ttk.Frame, ...]:
        """Create the processing, optics, reconstruction, and preparation pages."""
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        (
            reconstruction_tab,
            samples_tab,
            fits_tab,
            settings_tab,
            preparation_tab,
            lattice_tab,
        ) = (ttk.Frame(notebook) for _ in range(6))
        for tab, title in [
            (settings_tab, "Image Processing"),
            (lattice_tab, "Lattice Preview"),
            (reconstruction_tab, "Reconstruction"),
            (samples_tab, "Samples"),
            (fits_tab, "Fit Diagnostics"),
            (preparation_tab, "Simulation"),
        ]:
            notebook.add(tab, text=title)
        return (
            reconstruction_tab,
            samples_tab,
            fits_tab,
            settings_tab,
            preparation_tab,
            lattice_tab,
        )

    def build_reconstruction_tab() -> tuple[Figure, FigureCanvasTkAgg, ttk.Combobox]:
        """Place reconstruction controls above the shared plots and export toolbar."""
        header = ttk.Frame(reconstruction_tab, padding=6)
        header.pack(fill="x")
        boxes: dict[str, ttk.Combobox] = {}
        choices: list[tuple[str, str, tk.StringVar, list[str], str]] = [
            (
                "num_slices",
                "Slices (1 = projected)",
                slices,
                ["1 = projected", "20"],
                "normal",
            ),
            ("plane", "Plane", plane, [item.value for item in Plane], "readonly"),
            (
                "regression",
                "Regression type",
                regression,
                list(REGRESSION_CHOICES),
                "readonly",
            ),
        ]
        for column, (name, label, variable, values, state) in enumerate(choices):
            ttk.Label(header, text=label).grid(row=0, column=column, sticky="w", padx=5)
            box = ttk.Combobox(
                header,
                name=name,
                textvariable=variable,
                values=values,
                state=state,
                width=28,
            )
            box.grid(row=1, column=column, sticky="ew", padx=5, pady=4)
            boxes[name] = box
        ttk.Label(
            header,
            text="Slice count applies automatically. Plane changes the displayed results. "
            "Click Apply processing settings to use a new regression type.",
            wraplength=1300,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=5)
        figure, canvas = embed_plot(reconstruction_tab, (13, 7))
        return figure, canvas, boxes["num_slices"]

    def build_samples_tab() -> ttk.Treeview:
        """Build the selectable sample inventory with processing status and exclusion reasons."""
        samples_tree = ttk.Treeview(
            samples_tab,
            columns=("status", "configuration", "reason"),
            show="tree headings",
            selectmode="browse",
        )
        samples_tree.heading("#0", text="Sample ID (select to inspect)")
        samples_tree.column("#0", width=510)
        for key, title in [
            ("status", "Status"),
            ("configuration", "Configuration"),
            ("reason", "Reason / exclusions"),
        ]:
            samples_tree.heading(key, text=title)
        samples_scroll = ttk.Scrollbar(
            samples_tab, orient="vertical", command=samples_tree.yview
        )
        samples_tree.configure(yscrollcommand=samples_scroll.set)
        samples_scroll.pack(side="right", fill="y")
        samples_tree.pack(fill="both", expand=True)
        return samples_tree

    def build_fit_tab() -> tk.Text:
        fit_text = tk.Text(fits_tab, wrap="none", font=("Courier", 11))
        fit_text.pack(fill="both", expand=True)
        return fit_text

    def build_processing_tab() -> tuple[
        dict[str, tk.StringVar | tk.BooleanVar],
        ttk.Combobox,
        Figure,
        FigureCanvasTkAgg,
        tk.StringVar,
    ]:
        """Build typed processing controls beside an independent live sample preview."""

        def build_panels() -> tuple[ttk.Frame, ttk.Frame]:
            """Place the scrollable processing form beside its sample preview."""
            panes = ttk.Panedwindow(settings_tab, orient="horizontal")
            panes.pack(fill="both", expand=True)
            form_panel = ttk.Frame(panes, width=470, padding=6)
            preview_panel = ttk.Frame(panes, padding=6)
            panes.add(form_panel, weight=0)
            panes.add(preview_panel, weight=1)
            return form_panel, preview_panel

        def build_scrollable_form() -> tuple[ttk.Frame, tk.Canvas]:
            """Keep processing controls accessible when the form exceeds the window height."""
            ttk.Label(
                form_panel,
                text="Camera calibration updates the preview and scan results automatically.\n"
                "For other processing edits, Apply updates all scan results.\n"
                "Slices, plane and regression are in the Reconstruction plots tab.\n"
                "The preview shows projected profiles and all slice boundaries.",
                justify="left",
                wraplength=450,
            ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
            form_panel.rowconfigure(1, weight=1)
            form_panel.columnconfigure(0, weight=1)
            scroller = tk.Canvas(form_panel, highlightthickness=0, width=460)
            scrollbar = ttk.Scrollbar(
                form_panel, orient="vertical", command=scroller.yview
            )
            scroller.configure(yscrollcommand=scrollbar.set)
            scroller.grid(row=1, column=0, sticky="nsew")
            scrollbar.grid(row=1, column=1, sticky="ns")
            form = ttk.Frame(scroller)
            window = scroller.create_window((0, 0), window=form, anchor="nw")
            form.bind(
                "<Configure>",
                lambda event: scroller.configure(scrollregion=scroller.bbox("all")),
            )
            scroller.bind(
                "<Configure>",
                lambda event: scroller.itemconfigure(window, width=event.width),
            )
            return form, scroller

        initial_values = settings.to_form_values()
        variables: dict[str, tk.StringVar | tk.BooleanVar] = {}

        def build_field_input(
            parent: ttk.LabelFrame, field: ProcessingField, variable: tk.StringVar
        ) -> ttk.Widget:
            """Choose an editable calibration, a discrete choice, or a numeric input."""
            presets = CAMERA_RESOLUTION_PRESETS.get(field.name)
            choices = presets or PROCESSING_CHOICES.get(field.name)
            if choices is not None:
                widget = ttk.Combobox(
                    parent,
                    name=field.name,
                    textvariable=variable,
                    values=list(choices),
                    state="normal" if presets else "readonly",
                    width=28,
                )
                if presets:
                    resolution_boxes[field.name] = widget
                return widget
            is_integer = (
                field.name in ROI_FIELDS
                or AnalysisSettings.__annotations__[field.name] is int
            )
            options = (
                {"from_": int(field.minimum), "to": 2**31 - 1, "increment": 1}
                if is_integer
                else {
                    "from_": field.minimum,
                    "to": field.maximum,
                    "increment": field.increment,
                    "format": "%.6g",
                }
            )
            return ttk.Spinbox(
                parent, name=field.name, textvariable=variable, width=16, **options
            )

        def add_field(parent: ttk.LabelFrame, field: ProcessingField, row: int) -> None:
            """Connect each labeled widget to the processing form's draft value."""
            initial = initial_values[field.name]
            variable: tk.StringVar | tk.BooleanVar
            widget: ttk.Widget
            if isinstance(initial, bool):
                variable = tk.BooleanVar(value=initial)
                widget = ttk.Checkbutton(
                    parent, name=field.name, text=field.label, variable=variable
                )
                widget.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=4)
            else:
                variable = tk.StringVar(value=initial)
                ttk.Label(parent, text=field.label, wraplength=205).grid(
                    row=row, column=0, sticky="w", padx=5, pady=4
                )
                widget = build_field_input(parent, field, variable)
                widget.grid(row=row, column=1, sticky="ew", padx=5, pady=4)
            variables[field.name] = variable

        def build_form_fields() -> None:
            """Populate the processing groups and add their Apply action and units hint."""
            for title, group in PROCESSING_GROUPS:
                section = ttk.LabelFrame(form, text=title, padding=3)
                section.pack(fill="x", pady=5)
                section.columnconfigure(1, weight=1)
                for row, field in enumerate(group):
                    add_field(section, field, row)
            ttk.Button(
                form, text="Apply to all samples", command=lambda: controller.apply()
            ).pack(fill="x", pady=8)
            ttk.Label(
                form,
                text="Pixel scales apply to the oriented x/y axes. ROI bounds use oriented pixels.\n"
                "Images use [x, y] order. Slice indices are not calibrated times.",
                wraplength=450,
            ).pack(anchor="w", pady=6)

        def scroll_form(event: tk.Event) -> str:
            steps = -int(event.delta / 120) if abs(event.delta) >= 120 else -event.delta
            scroller.yview_scroll(steps, "units")
            return "break"

        def bind_scroll(widget: tk.Misc) -> None:
            widget.bind("<MouseWheel>", scroll_form)
            for child in widget.winfo_children():
                bind_scroll(child)

        def build_preview_panel() -> (
            tuple[ttk.Combobox, Figure, FigureCanvasTkAgg, tk.StringVar]
        ):
            """Build sample selection, preview status, and the exportable Matplotlib canvas."""
            header = ttk.Frame(preview_panel)
            header.pack(fill="x")
            header.columnconfigure(1, weight=1)
            ttk.Label(header, text="Preview sample").grid(
                row=0, column=0, padx=5, sticky="w"
            )
            sample_box = ttk.Combobox(
                header,
                name="preview_sample",
                textvariable=processing_sample,
                state="readonly",
                width=48,
            )
            sample_box.grid(row=0, column=1, sticky="ew", padx=5)
            ttk.Checkbutton(
                header,
                text="Follow newest sample",
                variable=follow_newest,
                command=lambda: controller.update_processing_sample(),
            ).grid(row=1, column=1, sticky="w", padx=5, pady=4)
            preview_status = tk.StringVar(value="Waiting for a sample")
            ttk.Label(preview_panel, textvariable=preview_status, wraplength=850).pack(
                fill="x", pady=6
            )
            preview_figure, preview_canvas = embed_plot(preview_panel, (8, 7))
            return sample_box, preview_figure, preview_canvas, preview_status

        form_panel, preview_panel = build_panels()
        form, scroller = build_scrollable_form()
        build_form_fields()
        bind_scroll(form)
        sample_box, preview_figure, preview_canvas, preview_status = (
            build_preview_panel()
        )

        return (
            variables,
            sample_box,
            preview_figure,
            preview_canvas,
            preview_status,
        )

    def build_preparation_tab() -> tuple[tk.Text, ttk.Frame, tk.StringVar]:
        """Provide optional lattice and simulation settings with a new experiment destination."""
        ttk.Label(
            preparation_tab,
            text="Optional preparation uses new output directories. Source dechirper settings are retained in simulation.\nEdit the lattice/simulation JSON, enter a new output path, then run the selected step.",
            padding=8,
        ).pack(anchor="w")
        preparation = tk.Text(
            preparation_tab, wrap="none", font=("Courier", 11), height=20
        )
        preparation.insert(
            "1.0",
            json.dumps(
                {
                    "lattice": LatticeSettings.load(settings_path).to_dict(),
                    "simulation": SimulationSettings.load(settings_path).to_dict(),
                },
                indent=2,
            ),
        )
        preparation.pack(fill="both", expand=True)
        prep_controls = ttk.Frame(preparation_tab, padding=6)
        prep_controls.pack(fill="x")
        output_path = tk.StringVar(value=str(experiment.parent / "NEW_EXPERIMENT"))
        ttk.Label(prep_controls, text="New output experiment:").pack(side="left")
        ttk.Entry(prep_controls, textvariable=output_path, width=70).pack(
            side="left", padx=6
        )
        return preparation, prep_controls, output_path

    build_controls()

    (
        reconstruction_tab,
        samples_tab,
        fits_tab,
        settings_tab,
        preparation_tab,
        lattice_tab,
    ) = build_tabs()

    figure, canvas, slice_count_box = build_reconstruction_tab()

    samples_tree = build_samples_tab()

    fit_text = build_fit_tab()

    (
        form_variables,
        processing_sample_box,
        processing_figure,
        processing_canvas,
        processing_status,
    ) = build_processing_tab()

    preparation, prep_controls, output_path = build_preparation_tab()
    lattice_preview = LatticePreviewPanel(lattice_tab, experiment)
    ttk.Label(root, textvariable=status, padding=6, wraplength=1400).pack(fill="x")
    view = DesktopView(
        root=root,
        status=status,
        regression=regression,
        slices=slices,
        plane=plane,
        watching=watching,
        include_raw=include_raw,
        processing_sample=processing_sample,
        follow_newest=follow_newest,
        figure=figure,
        canvas=canvas,
        samples_tree=samples_tree,
        fit_text=fit_text,
        form_variables=form_variables,
        processing_sample_box=processing_sample_box,
        processing_figure=processing_figure,
        processing_canvas=processing_canvas,
        processing_status=processing_status,
        preparation=preparation,
        output_path=output_path,
        lattice_preview=lattice_preview,
    )
    controller = DesktopController(AppState(analysis=analysis), view)

    def bind_controls() -> None:
        """Connect actions, selectors, and debounced edits to their update handlers."""
        for column, (title, callback) in enumerate(
            [
                ("Apply processing settings", controller.apply),
                ("Refresh now", controller.refresh_now),
                ("Export data + plots", controller.choose_export),
                ("Save settings", controller.save_settings),
            ]
        ):
            ttk.Button(controls, text=title, command=callback).grid(
                row=0, column=column, padx=5, pady=6
            )
        ttk.Button(
            prep_controls,
            text="Generate lattices",
            command=lambda: controller.prepare(Command.LATTICE),
        ).pack(side="left", padx=6)
        ttk.Button(
            prep_controls,
            text="Simulate",
            command=lambda: controller.prepare(Command.SIMULATE),
        ).pack(side="left")
        samples_tree.bind("<<TreeviewSelect>>", controller.select_sample)
        plane.trace_add("write", lambda *args: controller.redraw())
        regression.trace_add(
            "write",
            lambda *args: status.set(
                "Processing controls changed; click Apply to recompute results"
            ),
        )
        regression.trace_add("write", controller.schedule_processing_preview)
        slices.trace_add("write", controller.schedule_slice_count)
        slices.trace_add("write", controller.schedule_processing_preview)
        slice_count_box.bind("<<ComboboxSelected>>", controller.commit_slice_count)
        slice_count_box.bind("<Return>", controller.commit_slice_count)
        slice_count_box.bind("<FocusOut>", controller.commit_slice_count)
        for variable in form_variables.values():
            variable.trace_add("write", controller.schedule_processing_preview)
        for name, box in resolution_boxes.items():
            form_variables[name].trace_add("write", controller.schedule_resolution)
            box.bind("<<ComboboxSelected>>", controller.commit_resolution)
            box.bind("<Return>", controller.commit_resolution)
            box.bind("<FocusOut>", controller.commit_resolution)
        processing_sample.trace_add("write", controller.processing_sample_changed)
        processing_sample_box.bind(
            "<<ComboboxSelected>>", lambda event: follow_newest.set(False)
        )

    bind_controls()
    controller.show_processing_placeholder("Waiting for sample data")
    root.protocol("WM_DELETE_WINDOW", controller.close)
    root.after(0, controller.tick)
    root.mainloop()


def self_test() -> None:
    """Scientific and acquisition regression checks without machine services."""

    class Value:
        """A minimal get/set variable for controller tests that need no Tk root."""

        def __init__(self, value: Any) -> None:
            self.value = value

        def get(self) -> Any:
            return self.value

        def set(self, value: Any) -> None:
            self.value = value

    class ManualExecutor(Executor):
        """Record submitted work and let tests decide when each future completes."""

        def __init__(self) -> None:
            self.calls: list[
                tuple[Callable[..., Any], tuple[Any, ...], Future[Any]]
            ] = []
            self.shutdown_options: tuple[bool, bool] | None = None

        def submit[**P, T](
            self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
        ) -> Future[T]:
            future: Future[T] = Future()
            self.calls.append((function, args, future))
            return future

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            self.shutdown_options = wait, cancel_futures

    class FakeOptics:
        lengths = {"Q.T3": 0.5}

        def matrix(
            self, target: dict[str, Any], recorded: dict[str, float], energy: float
        ) -> Array:
            matrix = np.eye(6)
            matrix[0, 1] = recorded["q_t3_k1"]
            return matrix

    class Tests(unittest.TestCase):

        def setUp(self) -> None:
            """Create a temporary experiment, simple optics, and Gaussian image for each regression check."""
            self.temporary = tempfile.TemporaryDirectory()
            self.addCleanup(self.temporary.cleanup)
            self.path = Path(self.temporary.name) / "experiment"
            self.path.mkdir()
            (self.path / "samples_raw").mkdir()
            self.settings = AnalysisSettings(
                num_slices=2,
                mask_downsampling=1,
                mask_sigma=1.0,
                mask_cutoff=0,
                mask_dilation=0,
                largest_patch=False,
                tail_extension=False,
                settle_seconds=0,
            )
            self.document: dict[str, Any] = {
                "low_level_params": [
                    {"start": "START", "stop": "SCREEN", "q_t3_k1": 1.0}
                ],
                "high_level_params": [
                    {"start": "START", "stop": "SCREEN", "beam_energy": 14.0}
                ],
                "target_doocs_params": [{}],
            }
            atomic_json(self.path / "lattice_configs.json", self.document)
            x, y = np.indices((40, 48))
            self.image = np.exp(-0.5 * (((x - 20) / 4) ** 2 + ((y - 24) / 5) ** 2))
            self.live = LiveAnalysis(self.path, self.settings, FakeOptics())

        def write_sample(
            self,
            name: str = "MES_0_one",
            image: Array | None = None,
            kick: float = 500.0,
        ) -> Path:
            """Write a test acquisition shot with configurable image and magnet setpoint."""
            path = self.path / "samples_raw" / f"{name}.npz"
            np.savez(
                path,
                screens=np.array(
                    {"data": self.image if image is None else image}, dtype=object
                ),
                magnets=np.array({"data": [[0, kick, kick, 0, "Q.T3"]]}, dtype=object),
            )
            return path

        def make_controller(self) -> DesktopController:
            """Use fake widgets and a manually completed worker to test desktop transitions."""
            view = DesktopView(
                **{field.name: Mock(name=field.name) for field in fields(DesktopView)}
            )
            values = {
                "status": "",
                "regression": self.settings.regression.label,
                "slices": str(self.settings.num_slices),
                "plane": Plane.X,
                "watching": True,
                "include_raw": True,
                "processing_sample": "MES_0_one",
                "follow_newest": False,
                "processing_status": "",
                "output_path": str(self.path / "new_experiment"),
            }
            view = replace(
                view,
                **{name: Mock(wraps=Value(value)) for name, value in values.items()},
            )
            view = replace(
                view,
                lattice_preview=None,
                form_variables={
                    name: Mock(wraps=Value(value))
                    for name, value in self.settings.to_form_values().items()
                },
            )
            cast(Mock, view.root.after).return_value = "timer"
            cast(Mock, view.samples_tree.get_children).return_value = ()
            controller = DesktopController(AppState(self.live), view, ManualExecutor())
            self.addCleanup(controller.close)
            return controller

        def test_lattice_preview_initial_design_and_overrides(self) -> None:
            """Resolve design at different entrances and preserve normalized emittance at scan energy."""
            for start in (t3.cell[0].id, "VCCHIRP_S.2487.T3"):
                design = make_twiss0(start)
                before = copy.deepcopy(design.__dict__)
                initial = lattice_preview_initial(design, 14.0, {})
                self.assertEqual(initial.beta_x, design.beta_x)
                self.assertEqual(initial.alpha_y, design.alpha_y)
                self.assertAlmostEqual(initial.emit_x * 14 / REST_ENERGY_GEV, 1e-6)
                custom = lattice_preview_initial(
                    design,
                    14.0,
                    {
                        "emit_x": "2.5",
                        "beta_x": "7.5",
                        "alpha_x": "-1.2",
                        "emit_y": "0.5",
                        "beta_y": "13",
                        "alpha_y": "2.1",
                    },
                )
                self.assertEqual((custom.beta_x, custom.alpha_x), (7.5, -1.2))
                self.assertEqual((custom.beta_y, custom.alpha_y), (13, 2.1))
                self.assertAlmostEqual(custom.emit_x * 14 / REST_ENERGY_GEV, 2.5e-6)
                self.assertAlmostEqual(custom.emit_y * 14 / REST_ENERGY_GEV, 0.5e-6)
                self.assertEqual(design.__dict__, before)
                for name in lattice_design_values(design):
                    for invalid in (
                        ("nan", "inf", "bad")
                        if name.startswith("alpha")
                        else ("nan", "inf", "0", "-1", "bad")
                    ):
                        with (
                            self.subTest(name=name, invalid=invalid),
                            self.assertRaises(ValueError),
                        ):
                            lattice_preview_initial(design, 14.0, {name: invalid})
            design.emit_x = 3e-11
            expected = design.emit_x * design.E / REST_ENERGY_GEV
            self.assertAlmostEqual(
                lattice_design_values(design)["emit_x"] * 1e-6, expected
            )

        def test_lattice_preview_phase_and_ellipse_coordinates(self) -> None:
            """Check phases against analytic drift optics and ellipse points against covariance."""
            from ocelot import Drift
            from matplotlib.backend_bases import MouseEvent
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            initial = lattice_preview_initial(
                make_twiss0(),
                14.0,
                {
                    "emit_x": "2",
                    "beta_x": "3",
                    "alpha_x": "-0.7",
                    "emit_y": "0.5",
                    "beta_y": "11",
                    "alpha_y": "0.4",
                },
            )
            initial.mux, initial.muy = 8.5, 17.0
            lattices = [
                MagneticLattice((Drift(l=length, eid="D"),)) for length in (1.0, 5.0)
            ]
            document = {"low_level_params": [{}, {}]}
            with patch.object(
                sys.modules[__name__], "make_lattice", side_effect=lattices
            ):
                preview = make_lattice_preview(document, initial)
            for column, plane in enumerate(Plane):
                beta = getattr(initial, f"beta_{plane}")
                alpha = getattr(initial, f"alpha_{plane}")
                emit = getattr(initial, f"emit_{plane}")
                expected = np.arctan2(
                    np.array([1.0, 5.0]), beta - alpha * np.array([1.0, 5.0])
                ) % (2 * np.pi)
                np.testing.assert_allclose(preview.phases[:, column], expected)
                covariance = emit * np.array(
                    [[beta, -alpha], [-alpha, (1 + alpha**2) / beta]]
                )
                for normalized in (False, True):
                    figure = Figure(figsize=(6, 5))
                    canvas = FigureCanvasAgg(figure)
                    axis = figure.add_subplot()
                    plot_lattice_phases(axis, preview, plane, 1, normalized)
                    points = np.asarray(axis.lines[0].get_data())
                    if normalized:
                        np.testing.assert_allclose(
                            np.sum(points**2, axis=0), 1.0, atol=1e-12
                        )
                    else:
                        points *= 1e-3
                        np.testing.assert_allclose(
                            np.einsum(
                                "in,ij,jn->n", points, np.linalg.inv(covariance), points
                            ),
                            1.0,
                            atol=1e-12,
                        )
                    selected = np.asarray(axis.collections[2].get_offsets())[0]
                    all_points = np.asarray(axis.collections[1].get_offsets())
                    np.testing.assert_allclose(selected, all_points[1])
                    canvas.draw()
                    x, y = axis.transData.transform(all_points[0])
                    event = MouseEvent("motion_notify_event", canvas, x, y)
                    canvas.callbacks.process("motion_notify_event", event)
                    self.assertTrue(axis.texts[-1].get_visible())
                    self.assertIn("ID 0:", axis.texts[-1].get_text())

        def test_show_lattice_embedded_without_pyplot_windows(self) -> None:
            """Render the lattice on a SubFigure using a custom seed and no standalone window."""
            from ocelot import Drift, Quadrupole
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.backend_bases import MouseEvent
            import matplotlib.pyplot as plt

            initial = lattice_preview_initial(make_twiss0(), 14, {"beta_x": "7"})
            lattice = MagneticLattice(
                (Quadrupole(l=0.3, k1=0.2, eid="Q"), Drift(l=2, eid="D"))
            )
            figure = Figure(figsize=(8, 6))
            canvas = FigureCanvasAgg(figure)
            embedded = figure.subfigures(1, 1)
            before = plt.get_fignums()
            with patch.object(plt, "show") as show:
                show_lattice(
                    lattice, initial, figure=embedded, show=False, diagnostics=False
                )
                canvas.draw()
                self.assertEqual(embedded.axes[1].lines[0].get_ydata()[0], 7)
                axis = embedded.axes[2]
                x, y = axis.transData.transform((initial.s + 0.15, 0.2))
                event = MouseEvent("motion_notify_event", canvas, x, y)
                canvas.callbacks.process("motion_notify_event", event)
                self.assertTrue(any(text.get_visible() for text in axis.texts))
                show.assert_not_called()
            self.assertEqual(plt.get_fignums(), before)

        def test_controller_lattice_preview_ownership(self) -> None:
            """Queue propagation ahead of scans and discard an outdated result or error."""
            controller = self.make_controller()
            panel = Mock(spec=LatticePreviewPanel)
            panel.revision = 3
            panel.pending = (3, self.document, Twiss())
            controller.view = replace(controller.view, lattice_preview=panel)
            controller.start_pending_operation()
            job = controller.state.job
            assert job is not None
            self.assertEqual(job.operation, WorkerOperation.LATTICE_PREVIEW)
            self.assertIsNone(panel.pending)
            result = Mock(spec=LatticePreview)
            job.future.set_result(result)
            controller.consume_finished_operation()
            panel.display.assert_called_once_with(3, result)
            # The panel owns revision checks; stale results must not redraw it.
            actual_panel = LatticePreviewPanel.__new__(LatticePreviewPanel)
            actual_panel.revision = 4
            actual_panel.preview = None
            actual_panel.draw = Mock()
            actual_panel.display(3, result)
            self.assertIsNone(actual_panel.preview)
            actual_panel.draw.assert_not_called()
            actual_panel.display(4, result)
            actual_panel.draw.assert_called_once()
            panel.revision = 4
            controller.show_worker_error(job, ValueError("old error"))
            panel.invalidate.assert_not_called()
            panel.revision = 3
            controller.show_worker_error(job, ValueError("current error"))
            panel.invalidate.assert_called_once()

        def test_lattice_panel_reload_and_redraw(self) -> None:
            """Reload an entrance without samples and keep redraw callbacks bounded."""
            from ocelot import Drift
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            panel = LatticePreviewPanel.__new__(LatticePreviewPanel)
            panel.experiment = self.path
            panel.parent = Mock()
            panel.parent.after.return_value = "timer"
            panel.document, panel.design, panel.preview = None, None, None
            panel.pending, panel.timer = None, None
            panel.revision = 0
            panel.hover_callbacks = set()
            panel.config_id, panel.coordinates = Value("9"), Value("Original")
            panel.status, panel.config_box = Value(""), Mock()
            initial = lattice_preview_initial(make_twiss0(), 14, {})
            defaults = lattice_design_values(initial)
            panel.variables = {name: Value("Design") for name in defaults}
            panel.boxes = {name: Mock() for name in defaults}
            panel.design_labels = {name: Mock() for name in defaults}
            panel.figure = Figure(figsize=(11, 8))
            panel.canvas = FigureCanvasAgg(panel.figure)
            with patch.object(
                sys.modules[__name__], "make_twiss0", return_value=initial
            ) as design:
                panel.refresh_document()
                design.assert_called_once_with("START")
                self.assertEqual(panel.config_id.get(), "0")
                panel.queue()
                assert panel.pending is not None
                self.assertEqual(panel.pending[2].beta_x, initial.beta_x)
                revision = panel.revision
                panel.refresh_document()
                self.assertEqual(panel.revision, revision)
                changed = copy.deepcopy(self.document)
                for params in (
                    changed["low_level_params"],
                    changed["high_level_params"],
                ):
                    params[0]["start"] = "ANOTHER_START"
                atomic_json(self.path / "lattice_configs.json", changed)
                panel.refresh_document()
                self.assertEqual(design.call_args.args, ("ANOTHER_START",))
                self.assertGreater(panel.revision, revision)
                panel.queue()
            lattice = MagneticLattice((Drift(l=2, eid="D"),))
            preview = LatticePreview(
                initial, [lattice], [twiss(lattice, initial)], np.array([[0.5, 1.5]])
            )
            initial_callbacks = len(
                panel.canvas.callbacks.callbacks.get("motion_notify_event", {})
            )
            for coordinates in ("Original", "Normalized", "Original"):
                panel.coordinates.set(coordinates)
                panel.display(panel.revision, preview)
                panel.canvas.draw()
                self.assertEqual(len(panel.figure.axes), 5)
                self.assertEqual(
                    len(panel.canvas.callbacks.callbacks["motion_notify_event"]),
                    initial_callbacks + 3,
                )
            (self.path / "lattice_configs.json").write_text("incomplete")
            panel.refresh_document()
            self.assertIsNone(panel.preview)
            self.assertIsNone(panel.pending)
            self.assertFalse(panel.figure.axes)
            panel.close()

        def test_controller_settings_discard_old_refresh(self) -> None:
            """Changing applied settings keeps an older completed scan from repopulating the GUI."""
            self.write_sample()
            snapshot = self.live.refresh()
            controller = self.make_controller()
            controller.start_pending_operation()
            old_job = controller.state.job
            assert old_job is not None
            self.assertIs(old_job.analysis, self.live)
            controller.install_settings(replace(self.settings, num_slices=3))
            self.assertIsNone(controller.state.snapshot)
            old_job.future.set_result(snapshot)
            controller.consume_finished_operation()
            self.assertIsNone(controller.state.snapshot)
            self.assertIsNone(controller.state.job)
            cast(Mock, controller.view.fit_text.insert).assert_not_called()
            controller.start_pending_operation()
            new_job = controller.state.job
            assert new_job is not None
            self.assertIs(new_job.analysis, controller.state.analysis)
            self.assertEqual(new_job.analysis.settings.num_slices, 3)
            function = cast(ManualExecutor, controller.executor).calls[-1][0]
            assert isinstance(function, MethodType)
            self.assertIs(function.__self__, new_job.analysis)

        def test_controller_current_refresh_and_busy_slot(self) -> None:
            """Only one operation runs at a time, and a current scan publishes its exact snapshot."""
            self.write_sample()
            snapshot = self.live.refresh()
            controller = self.make_controller()
            controller.start_pending_operation()
            job = controller.state.job
            assert job is not None
            self.assertEqual(job.operation, WorkerOperation.REFRESH)
            controller.consume_finished_operation()
            self.assertIs(controller.state.job, job)
            with self.assertRaises(RuntimeError):
                controller.submit(WorkerOperation.EXPORT, lambda: None)
            self.assertEqual(len(cast(ManualExecutor, controller.executor).calls), 1)
            job.future.set_result(snapshot)
            with patch.object(sys.modules[__name__], "plot_snapshot") as draw:
                controller.consume_finished_operation()
                draw.assert_called_once_with(controller.view.figure, snapshot, Plane.X)
            self.assertIs(controller.state.snapshot, snapshot)
            self.assertIsNone(controller.state.job)
            self.assertFalse(controller.state.pending_refresh)

        def test_controller_preview_ownership_and_priority(self) -> None:
            """Queued previews capture their inputs and stale revisions or samples never publish."""
            preview = ProcessingPreview(
                "MES_0_one",
                self.settings,
                self.image,
                process_image(self.image, self.settings),
                "hash",
            )
            for stale in ("current", "revision", "sample"):
                with self.subTest(stale=stale):
                    controller = self.make_controller()
                    controller.state.processing_preview_revision = 4
                    request = PreviewRequest(4, preview.sample_id, self.settings)
                    controller.state.pending_processing_preview = request
                    controller.start_pending_operation()
                    job = controller.state.job
                    assert job is not None
                    self.assertEqual(job.operation, WorkerOperation.PROCESSING_PREVIEW)
                    function, args, _ = cast(ManualExecutor, controller.executor).calls[
                        -1
                    ]
                    self.assertIs(function, preview_processing_sample)
                    self.assertEqual(
                        args, (self.live.experiment, preview.sample_id, self.settings)
                    )
                    self.assertIsNone(controller.state.pending_processing_preview)
                    if stale == "revision":
                        controller.state.processing_preview_revision += 1
                    elif stale == "sample":
                        controller.view.processing_sample.set("another_sample")
                    job.future.set_result(preview)
                    with patch.object(
                        sys.modules[__name__], "plot_processing_preview"
                    ) as draw:
                        controller.consume_finished_operation()
                        self.assertEqual(draw.call_count, int(stale == "current"))
                    self.assertIsNone(controller.state.job)
                    controller.start_pending_operation()
                    assert controller.state.job is not None
                    self.assertEqual(
                        controller.state.job.operation, WorkerOperation.REFRESH
                    )

        def test_controller_refresh_errors_preserve_newer_results(self) -> None:
            """A current scan failure clears old values while an obsolete failure preserves new ones."""
            self.write_sample()
            snapshot = self.live.refresh()
            for stale in (False, True):
                with self.subTest(stale=stale):
                    controller = self.make_controller()
                    controller.state.snapshot = snapshot
                    controller.start_pending_operation()
                    job = controller.state.job
                    assert job is not None
                    if stale:
                        controller.state.analysis = LiveAnalysis(
                            self.path, self.settings
                        )
                    job.future.set_exception(ValueError("injected scan failure"))
                    controller.consume_finished_operation()
                    self.assertIsNone(controller.state.job)
                    if stale:
                        self.assertIs(controller.state.snapshot, snapshot)
                        cast(Mock, controller.view.figure.clear).assert_not_called()
                    else:
                        self.assertIsNone(controller.state.snapshot)
                        cast(Mock, controller.view.figure.clear).assert_called_once()
                        cast(Mock, controller.view.fit_text.delete).assert_called_once()
                        cast(
                            Mock, controller.view.samples_tree.delete
                        ).assert_called_once()

        def test_controller_preview_errors_respect_revision(self) -> None:
            """An obsolete preview error cannot replace the current preview with an error message."""
            for stale in (False, True):
                with self.subTest(stale=stale):
                    controller = self.make_controller()
                    controller.state.pending_processing_preview = PreviewRequest(
                        0, "MES_0_one", self.settings
                    )
                    controller.start_pending_operation()
                    job = controller.state.job
                    assert job is not None
                    before = controller.view.processing_status.get()
                    if stale:
                        controller.state.processing_preview_revision += 1
                    job.future.set_exception(ValueError("injected preview failure"))
                    controller.consume_finished_operation()
                    self.assertIsNone(controller.state.job)
                    if stale:
                        self.assertEqual(
                            controller.view.processing_status.get(), before
                        )
                        cast(
                            Mock, controller.view.processing_figure.clear
                        ).assert_not_called()
                    else:
                        self.assertIn(
                            "injected preview failure",
                            controller.view.processing_status.get(),
                        )
                        cast(
                            Mock, controller.view.processing_figure.clear
                        ).assert_called_once()

        def test_controller_partial_settings_and_preview_queue(self) -> None:
            """Automatic slicing/calibration retain draft edits until Apply and previews use those drafts."""
            controller = self.make_controller()
            cast(Mock, controller.view.form_variables["mask_sigma"]).set("2")
            controller.view.regression.set(RegressionType.LINEAR.label)
            cast(Mock, controller.view.form_variables["pixel_size_x"]).set("4e-6")
            controller.commit_resolution()
            self.assertEqual(controller.state.analysis.settings.pixel_size_x, 4e-6)
            self.assertEqual(
                controller.state.analysis.settings.mask_sigma, self.settings.mask_sigma
            )
            controller.view.slices.set("3")
            controller.commit_slice_count()
            self.assertEqual(controller.state.analysis.settings.num_slices, 3)
            self.assertEqual(
                controller.state.analysis.settings.regression,
                RegressionType.CHI_SQUARED,
            )
            controller.apply()
            self.assertEqual(controller.state.analysis.settings.mask_sigma, 2)
            self.assertEqual(
                controller.state.analysis.settings.regression, RegressionType.LINEAR
            )
            controller.queue_processing_preview()
            request = controller.state.pending_processing_preview
            assert request is not None
            self.assertEqual(request.settings, controller.current_settings())
            self.assertEqual(
                request.revision, controller.state.processing_preview_revision
            )
            cast(Mock, controller.view.form_variables["mask_sigma"]).set("invalid")
            controller.queue_processing_preview()
            self.assertIsNone(controller.state.pending_processing_preview)
            self.assertEqual(controller.state.analysis.settings.mask_sigma, 2)

        def test_controller_close_stops_timers_and_publication(self) -> None:
            """Closing cancels timers and prevents a running job from updating destroyed widgets."""
            controller = self.make_controller()
            controller.state.processing_preview_timer = "preview_timer"
            controller.state.slice_update_timer = "slice_timer"
            controller.state.resolution_update_timer = "resolution_timer"
            controller.start_pending_operation()
            job = controller.state.job
            assert job is not None
            controller.close()
            self.assertTrue(controller.state.closing)
            self.assertEqual(
                cast(ManualExecutor, controller.executor).shutdown_options,
                (False, True),
            )
            self.assertEqual(
                [
                    call.args[0]
                    for call in cast(
                        Mock, controller.view.root.after_cancel
                    ).call_args_list
                ],
                ["preview_timer", "slice_timer", "resolution_timer"],
            )
            job.future.set_result(None)
            controller.consume_finished_operation()
            controller.start_pending_operation()
            controller.tick()
            controller.close()
            self.assertEqual(len(cast(ManualExecutor, controller.executor).calls), 1)
            cast(Mock, controller.view.root.destroy).assert_called_once()
            cast(Mock, controller.view.root.after).assert_not_called()
            with self.assertRaises(RuntimeError):
                controller.submit(WorkerOperation.REFRESH, self.live.refresh)

        def test_controller_export_owns_snapshot_and_rechecks_dialog(self) -> None:
            """Exports capture their snapshot and cannot overwrite a job started during the directory dialog."""
            self.write_sample()
            snapshot = self.live.refresh()
            destination = self.path / "exports"
            for change in ("none", "worker_started", "snapshot_cleared"):
                with self.subTest(change=change):
                    controller = self.make_controller()
                    controller.state.snapshot = snapshot

                    def choose_directory(**kwargs: Any) -> str:
                        if change == "worker_started":
                            controller.start_pending_operation()
                        elif change == "snapshot_cleared":
                            controller.clear_reconstruction()
                        return str(destination)

                    with (
                        patch(
                            "tkinter.filedialog.askdirectory",
                            side_effect=choose_directory,
                        ),
                        patch("tkinter.messagebox.showinfo"),
                        patch.object(sys.modules[__name__], "plot_snapshot"),
                    ):
                        controller.choose_export()
                    match change:
                        case "none":
                            job = controller.state.job
                            assert job is not None
                            self.assertEqual(job.operation, WorkerOperation.EXPORT)
                            function, args, _ = cast(
                                ManualExecutor, controller.executor
                            ).calls[-1]
                            assert isinstance(function, MethodType)
                            self.assertIs(function.__self__, self.live)
                            self.assertIs(args[0], snapshot)
                            self.assertEqual(args[1], destination)
                            job.future.set_result(destination)
                            controller.consume_finished_operation()
                            self.assertIs(controller.state.snapshot, snapshot)
                            self.assertIsNone(controller.state.job)
                        case "worker_started":
                            assert controller.state.job is not None
                            self.assertEqual(
                                controller.state.job.operation, WorkerOperation.REFRESH
                            )
                            self.assertEqual(
                                len(cast(ManualExecutor, controller.executor).calls), 1
                            )
                        case "snapshot_cleared":
                            self.assertIsNone(controller.state.job)
                            self.assertIsNone(controller.state.snapshot)

        def test_controller_preparation_dispatch(self) -> None:
            """Both preparation actions share worker ownership and report completion without altering results."""
            for command in (Command.LATTICE, Command.SIMULATE):
                with self.subTest(command=command):
                    controller = self.make_controller()
                    preparation_json = json.dumps(
                        {
                            "lattice": LatticeSettings().to_dict(),
                            "simulation": SimulationSettings().to_dict(),
                        }
                    )
                    cast(Mock, controller.view.preparation.get).return_value = (
                        preparation_json
                    )
                    controller.prepare(command)
                    job = controller.state.job
                    assert job is not None
                    self.assertEqual(job.operation, WorkerOperation.PREPARATION)
                    self.assertIs(job.analysis, self.live)
                    function = cast(ManualExecutor, controller.executor).calls[-1][0]
                    match command:
                        case Command.LATTICE:
                            self.assertIs(function, generate_lattices)
                        case Command.SIMULATE:
                            self.assertIs(function, simulate_experiment)
                    job.future.set_result(None)
                    controller.consume_finished_operation()
                    self.assertIsNone(controller.state.job)
                    self.assertIn("Preparation complete", controller.view.status.get())

        def synthetic(self) -> tuple[list[Shot], Array]:
            """Generate known covariance scans with controlled shot-to-shot variance for fit checks."""
            a = np.array([4e-10, 1e-10, 2e-10])
            shots = []
            for index in range(6):
                matrix = np.eye(6)
                phase = index * np.pi / 6
                matrix[0, 0], matrix[0, 1] = (
                    np.cos(phase),
                    np.sin(phase),
                )
                r0, r1 = matrix[0, 0], matrix[0, 1]
                row = [r0 * r0, 2 * r0 * r1, r1 * r1]
                value = np.dot(row, a)
                for repeat, offset in enumerate([-0.01, 0, 0.01]):
                    shots.append(
                        Shot(
                            f"{index}-{repeat}",
                            index,
                            "",
                            "",
                            {},
                            matrix,
                            {-1: (value * (1 + offset), 2e-10)},
                            {},
                            [],
                            0,
                        )
                    )
            return shots, a

        def test_reconstruct_known_covariance(self) -> None:
            shots, expected = self.synthetic()
            result = fit_scan(shots, -1, "x", 14, self.settings)
            self.assertEqual(result["status"], "valid")
            np.testing.assert_allclose(result["coefficients"], expected, rtol=1e-10)
            self.assertAlmostEqual(
                result["gamma"], (1 + result["alpha"] ** 2) / result["beta"]
            )
            self.assertLess(result["chi_squared"], 1e-20)

        def test_linear_reconstruction(self) -> None:
            shots, expected = self.synthetic()
            settings = replace(self.settings, regression=RegressionType.LINEAR)
            result = fit_scan(shots, -1, "x", 14, settings)
            self.assertEqual(result["status"], FitStatus.VALID)
            np.testing.assert_allclose(
                result["coefficients"], expected, rtol=1e-9, atol=1e-22
            )

        def test_fit_report_json_round_trip(self) -> None:
            """Restore all fit states and enum interfaces without changing exported JSON."""
            shots, _ = self.synthetic()
            waiting = fit_scan([], -1, Plane.X, 14, self.settings)
            invalid = fit_scan(
                [replace(shot, matrix=np.eye(6)) for shot in shots],
                -1,
                Plane.X,
                14,
                self.settings,
            )
            valid = fit_scan(shots, -1, Plane.X, 14, self.settings)
            self.assertEqual(
                [result["status"] for result in (waiting, invalid, valid)],
                [FitStatus.WAITING, FitStatus.INVALID, FitStatus.VALID],
            )
            snapshot = self.live.refresh()
            snapshot.document["fits"] = [waiting, invalid, valid]
            snapshot.document["design_twiss"] = {"x": {"beta": 4.0, "alpha": 1.0}}
            self.assertEqual(
                Snapshot.parse_report(snapshot.document), snapshot.document
            )
            serialized = json.loads(json.dumps(snapshot.document))
            serialized["future_metadata"] = {"value": [1, 2, 3]}
            restored = replace(snapshot, document=Snapshot.parse_report(serialized))
            self.assertEqual(json.loads(json.dumps(restored.document)), serialized)
            self.assertIs(restored.fits[0]["plane"], Plane.X)
            self.assertIs(restored.fits[1]["status"], FitStatus.INVALID)
            self.assertEqual(restored.fits[0]["plane"].matrix_index, 0)
            self.assertEqual(restored.design_twiss, snapshot.design_twiss)
            # Older archives can reconstruct emittance without a design reference.
            del serialized["design_twiss"]
            restored = replace(snapshot, document=Snapshot.parse_report(serialized))
            self.assertIsNone(restored.design_twiss)

        def test_valid_fit_requires_complete_reconstruction(self) -> None:
            """Reject incomplete valid records at loading, before any plot can access them."""
            shots, _ = self.synthetic()
            report = self.live.refresh().document
            report["fits"] = [fit_scan(shots, -1, Plane.X, 14, self.settings)]
            serialized = json.loads(json.dumps(report))
            beam_fields = {
                "beta",
                "alpha",
                "gamma",
                "emittance",
                "normalized_emittance",
                "transverse_covariance",
                "errors",
                "normalized_emittance_error",
            }
            for name in beam_fields:
                with self.subTest(field=name):
                    incomplete = copy.deepcopy(serialized)
                    del incomplete["fits"][0][name]
                    with self.assertRaisesRegex(
                        ValueError, rf"fits\[0\].{name}: required"
                    ):
                        Snapshot.parse_report(incomplete)

        def test_report_rejects_invalid_field_types(self) -> None:
            """Check enum, scalar, and nested uncertainty validation with useful field paths."""
            shots, _ = self.synthetic()
            report = self.live.refresh().document
            report["fits"] = [fit_scan(shots, -1, Plane.X, 14, self.settings)]
            serialized = json.loads(json.dumps(report))
            for name, value in (
                ("plane", "z"),
                ("status", "unknown"),
                ("sample_count", True),
                ("beta", "4.0"),
                ("normalized_emittance", None),
                ("emittance", float("nan")),
                ("errors", {"beta": 0.1}),
                ("transverse_covariance", [[1.0, "wrong"], [0.0, 1.0]]),
            ):
                with self.subTest(field=name, value=value):
                    malformed = copy.deepcopy(serialized)
                    malformed["fits"][0][name] = value
                    with self.assertRaisesRegex(ValueError, rf"fits\[0\].{name}"):
                        Snapshot.parse_report(malformed)
            serialized["fits"][0]["status"] = "invalid"
            with self.assertRaisesRegex(ValueError, r"fits\[0\].reason: required"):
                Snapshot.parse_report(serialized)

        def test_fit_report_preserves_unavailable_uncertainties(self) -> None:
            """A determined linear fit is valid even when its error estimates are unavailable."""
            shots, _ = self.synthetic()
            settings = replace(self.settings, regression=RegressionType.LINEAR)
            result = fit_scan(shots[1:8:3], -1, Plane.X, 14, settings)
            self.assertEqual(result["status"], FitStatus.VALID)
            self.assertIsNone(result["errors"])
            self.assertIsNone(result["normalized_emittance_error"])
            snapshot = self.live.refresh()
            snapshot.document["fits"] = [result]
            restored = Snapshot.parse_report(json.loads(json.dumps(snapshot.document)))
            self.assertEqual(restored["fits"], [result])

        def test_replay_cli_validates_report(self) -> None:
            """Exercise archive decoding through CLI replay and reject incomplete valid fits."""
            self.write_sample()
            snapshot = self.live.refresh()
            destination = self.live.export_bundle(snapshot)
            arguments = ["tool", "replay", str(destination)]
            output = io.StringIO()
            with patch.object(sys, "argv", arguments), patch("sys.stdout", output):
                main()
            replay_path = Path(output.getvalue().splitlines()[0])
            replay = Snapshot.parse_report(
                json.loads((replay_path / "config.json").read_text())
            )
            self.assertEqual(replay["fits"], snapshot.fits)
            report_path = destination / "config.json"
            malformed = json.loads(report_path.read_text())
            malformed["fits"][0]["status"] = FitStatus.VALID
            atomic_json(report_path, malformed)
            with patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(ValueError, r"fits\[0\].beta: required"):
                    main()

        def test_inspect_cli_requires_only_screen_inputs(self) -> None:
            """Export a screen preview without constructing optics or needing lattice files."""
            source = self.write_sample()
            experiment = self.path / "preview_only"
            raw_dir = experiment / "samples_raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / source.name).write_bytes(source.read_bytes())
            settings_path = self.path / "settings.json"
            atomic_json(settings_path, {"analysis": self.settings.to_dict()})
            output_path = self.path / "preview.png"
            arguments = [
                "tool",
                "inspect",
                str(experiment),
                "--sample",
                source.stem,
                "--settings",
                str(settings_path),
                "--output",
                str(output_path),
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch("sys.stdout", io.StringIO()),
                patch.object(
                    sys.modules[__name__],
                    "LiveAnalysis",
                    side_effect=AssertionError("Inspection must not start a scan"),
                ),
            ):
                main()
            self.assertGreater(output_path.stat().st_size, 1000)

        def test_weighted_reconstruction_with_varying_optics(self) -> None:
            """Recover known projected and slice covariances when each shot has different optics."""
            covariance = np.array([[4e-10, 1e-10], [1e-10, 2e-10]])
            expected = np.array([covariance[0, 0], covariance[0, 1], covariance[1, 1]])
            expected_emit = np.sqrt(np.linalg.det(covariance))

            def make_variable_optics_shots(plane: Plane) -> list[Shot]:
                """Propagate one known beam through three rotations at each scan configuration."""
                shots = []
                dim = plane.matrix_index
                for index in range(6):
                    for repeat, offset in enumerate((-0.15, 0, 0.15)):
                        phase = index * np.pi / 6 + offset
                        vector = np.array([np.cos(phase), np.sin(phase)])
                        matrix = np.eye(6)
                        matrix[dim : dim + 2, dim : dim + 2] = (
                            (vector[0], vector[1]),
                            (-vector[1], vector[0]),
                        )
                        variance = float(vector @ covariance @ vector)
                        shots.append(
                            Shot(
                                f"{index}-{repeat}",
                                index,
                                "",
                                "",
                                {},
                                matrix,
                                {
                                    -1: (variance, variance),
                                    0: (0.75 * variance, 0.75 * variance),
                                },
                                {},
                                [],
                                0,
                            )
                        )
                return shots

            for plane in Plane:
                shots = make_variable_optics_shots(plane)
                for weighting in Weighting:
                    settings = replace(self.settings, weighting=weighting)
                    for slice_index, scale in ((-1, 1), (0, 0.75)):
                        with self.subTest(
                            plane=plane, weighting=weighting, slice=slice_index
                        ):
                            result = fit_scan(shots, slice_index, plane, 14, settings)
                            self.assertEqual(result["status"], FitStatus.VALID)
                            np.testing.assert_allclose(
                                result["coefficients"], expected * scale, rtol=1e-10
                            )
                            np.testing.assert_allclose(
                                result["transverse_covariance"],
                                covariance * scale,
                                rtol=1e-10,
                            )
                            np.testing.assert_allclose(
                                result["normalized_emittance"],
                                expected_emit * scale * 14 / REST_ENERGY_GEV,
                                rtol=1e-10,
                            )
                            spreads = np.array(
                                [
                                    np.std(
                                        [
                                            shot.variances[slice_index][
                                                plane.variance_index
                                            ]
                                            for shot in shots
                                            if shot.config_index == index
                                        ],
                                        ddof=1,
                                    )
                                    for index in range(6)
                                ]
                            )
                            match weighting:
                                case Weighting.SHOT_STD:
                                    expected_errors = spreads
                                case Weighting.STANDARD_ERROR:
                                    expected_errors = spreads / np.sqrt(3)
                            np.testing.assert_allclose(
                                result["uncertainties"], expected_errors
                            )

        def test_uncertainty_jacobian(self) -> None:
            """Compare analytic covariance-to-Twiss derivatives with finite differences."""
            for a in (
                np.array([4.0, 0.5, 2.0]),
                np.array([5.0, -0.9, 1.5]),
            ):
                _, jac, _ = covariance_parameters(a)
                numeric = []
                for index in range(len(a)):
                    delta = np.zeros_like(a)
                    delta[index] = 1e-6
                    plus = covariance_parameters(a + delta)[0]
                    minus = covariance_parameters(a - delta)[0]
                    numeric.append((plus - minus) / 2e-6)
                np.testing.assert_allclose(
                    jac, np.array(numeric).T, rtol=1e-6, atol=1e-8
                )

        def test_weights_preserve_shot_std(self) -> None:
            """Verify default shot spread and optional standard-error weighting remain distinct."""
            shots, _ = self.synthetic()
            std = fit_scan(shots, -1, "x", 14, self.settings)
            sem = fit_scan(
                shots,
                -1,
                "x",
                14,
                replace(self.settings, weighting=Weighting.STANDARD_ERROR),
            )
            np.testing.assert_allclose(
                std["uncertainties"], np.array(sem["uncertainties"]) * np.sqrt(3)
            )
            std_errors, sem_errors = std["errors"], sem["errors"]
            assert std_errors is not None and sem_errors is not None
            np.testing.assert_allclose(
                std_errors["emittance"], sem_errors["emittance"] * np.sqrt(3)
            )

        def test_mismatch_zeta_physics_and_units(self) -> None:
            """Verify mismatch symmetry, unit independence, matched limits, and invalid inputs."""
            for beta, alpha, design_beta, design_alpha in (
                (5.0, -2.0, 5.0, -2.0),
                (10.0, 0.0, 5.0, 0.0),
                (7.0, 2.0, 5.0, -1.0),
            ):
                gamma = (1 + alpha**2) / beta
                design_gamma = (1 + design_alpha**2) / design_beta
                expected = (
                    beta * design_gamma + gamma * design_beta - 2 * alpha * design_alpha
                ) / 2
                actual = mismatch_zeta(beta, alpha, design_beta, design_alpha)
                self.assertAlmostEqual(actual, expected)
                self.assertGreaterEqual(actual, 1)
                self.assertAlmostEqual(
                    actual, mismatch_zeta(design_beta, design_alpha, beta, alpha)
                )
                self.assertAlmostEqual(
                    actual,
                    mismatch_zeta(beta * 1e3, alpha, design_beta * 1e3, design_alpha),
                )
            self.assertEqual(mismatch_zeta(5, -2, 5, -2), 1)
            self.assertEqual(mismatch_zeta(10, 0, 5, 0), 1.25)
            self.assertEqual(mismatch_zeta(5, 1e8, 5, 1e8), 1)
            for invalid in (0.0, -1.0, np.nan, np.inf):
                with self.assertRaises(ValueError):
                    mismatch_zeta(invalid, 0, 5, 0)

        def test_mismatch_design_export_replay_and_plots(self) -> None:
            """Check design-reference provenance, replay, and sliced or projected mismatch plots."""
            reference: DesignTwiss = {
                "x": {"beta": 4.0, "alpha": 1.0},
                "y": {"beta": 5.0, "alpha": -1.0},
            }

            class DesignOptics(FakeOptics):
                def design_twiss(self, start: str) -> DesignTwiss:
                    return reference

            document = copy.deepcopy(self.document)
            document["low_level_params"] *= 6
            document["high_level_params"] *= 6
            document["target_doocs_params"] *= 6
            atomic_json(self.path / "lattice_configs.json", document)
            x, y = np.indices((256, 64))
            for config in range(6):
                variance = 4e-10 + 2e-10 * config + 2e-10 * config**2
                for repeat, jitter in enumerate((0.99, 1.0, 1.01)):
                    sigma = np.sqrt(variance) / self.settings.pixel_size_x * jitter
                    image = np.exp(
                        -0.5 * (((x - 128) / sigma) ** 2 + ((y - 32) / 5) ** 2)
                    )
                    self.write_sample(
                        f"MES_{config}_{repeat}", image=image, kick=config * 500.0
                    )
            live = LiveAnalysis(self.path, self.settings, DesignOptics())
            snapshot = live.refresh()
            self.assertEqual(snapshot.document["design_twiss"], reference)
            fits = [
                result
                for result in snapshot.fits
                if result["status"] == FitStatus.VALID
            ]
            self.assertEqual(len(fits), 3)
            for result in fits:
                self.assertEqual(
                    result["mismatch_zeta"],
                    mismatch_zeta(result["beta"], result["alpha"], 4, 1),
                )
            figure = Figure(figsize=(13, 7))
            plot_snapshot(figure, snapshot)
            axis = figure.axes[5]
            np.testing.assert_allclose(
                np.asarray(axis.lines[0].get_xdata(), dtype=float), [0, 1]
            )
            np.testing.assert_allclose(
                np.asarray(axis.lines[0].get_ydata(), dtype=float),
                [
                    result["mismatch_zeta"]
                    for result in fits
                    if result["slice_index"] >= 0
                ],
            )
            np.testing.assert_allclose(
                np.asarray(axis.lines[1].get_ydata(), dtype=float), [1, 1]
            )
            self.assertEqual(axis.get_ylabel(), r"$\zeta$")
            destination = live.export_bundle(snapshot, figure=figure)
            archived = json.loads((destination / "config.json").read_text())
            self.assertEqual(archived["design_twiss"], reference)
            self.assertEqual(archived["units"]["mismatch_zeta"], "1")
            replay = LiveAnalysis(
                destination / "inputs", self.settings, ArchivedOptics(archived)
            ).refresh()
            self.assertEqual(replay.document["design_twiss"], reference)
            np.testing.assert_allclose(
                [
                    result["mismatch_zeta"]
                    for result in replay.fits
                    if result["status"] == FitStatus.VALID
                ],
                [result["mismatch_zeta"] for result in fits],
            )

            older_archive = archived.copy()
            older_archive.pop("design_twiss")
            replay = LiveAnalysis(
                destination / "inputs", self.settings, ArchivedOptics(older_archive)
            ).refresh()
            plot_snapshot(figure, replay)
            self.assertFalse(figure.axes[5].lines)
            self.assertIn("unavailable", figure.axes[5].texts[0].get_text())
            projected = replace(
                snapshot,
                document=snapshot.document
                | {
                    "settings": replace(self.settings, num_slices=1).to_dict(),
                },
            )
            plot_snapshot(figure, projected)
            np.testing.assert_allclose(
                np.asarray(figure.axes[5].lines[0].get_xdata(), dtype=float), [0]
            )
            self.assertEqual(
                figure.axes[5].get_xticklabels()[0].get_text(), "projected"
            )

        def test_rank_and_early_scan(self) -> None:
            shots, _ = self.synthetic()
            self.assertEqual(
                fit_scan(shots[:3], -1, "x", 14, self.settings)["status"], "waiting"
            )
            for shot in shots:
                shot.matrix = np.eye(6)
            result = fit_scan(shots, -1, "x", 14, self.settings)
            self.assertEqual(result["status"], "invalid")
            self.assertNotIn("emittance", result)

        def test_nonphysical_fit(self) -> None:
            with self.assertRaises(ValueError):
                covariance_parameters(np.array([1.0, 2.0, 1.0]))
            with self.assertRaises(ValueError):
                covariance_parameters(np.array([1.0, 0.0, -1.0]))

        def test_rms_axes_and_calibration(self) -> None:
            """Check image-axis ordering and the squared pixel-scale dependence of variances."""
            processed = process_image(self.image, self.settings)
            vx, vy = processed.variances[-1]
            self.assertAlmostEqual(vx / self.settings.pixel_size_x**2, 16, places=3)
            self.assertAlmostEqual(vy / self.settings.pixel_size_y**2, 25, places=2)
            self.assertEqual(processed.bounds, [0, 24, 48])
            calibrated = process_image(
                self.image,
                replace(self.settings, pixel_size_x=self.settings.pixel_size_x * 2),
            )
            np.testing.assert_allclose(calibrated.variances[-1], [4 * vx, vy])

        def test_image_orientation_order_and_defaults(self) -> None:
            """Verify rotation direction, flip order, defaults, and configuration validation."""
            raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            original = raw.copy()
            expected_rotations = {
                ImageRotation.NONE: [[1, 2, 3], [4, 5, 6]],
                ImageRotation.ROTATE_90: [[4, 1], [5, 2], [6, 3]],
                ImageRotation.ROTATE_180: [[6, 5, 4], [3, 2, 1]],
                ImageRotation.ROTATE_270: [[3, 6], [2, 5], [1, 4]],
            }
            for rotation, expected in expected_rotations.items():
                for horizontal in (False, True):
                    for vertical in (False, True):
                        with self.subTest(
                            rotation=rotation, horizontal=horizontal, vertical=vertical
                        ):
                            settings = replace(
                                self.settings,
                                rotation=rotation,
                                flip_horizontal=horizontal,
                                flip_vertical=vertical,
                            )
                            oriented = expected
                            if horizontal:
                                oriented = [row[::-1] for row in oriented]
                            if vertical:
                                oriented = oriented[::-1]
                            np.testing.assert_array_equal(
                                orient_image(raw, settings), oriented
                            )
                            np.testing.assert_array_equal(
                                process_image(raw, settings).image, oriented
                            )
                            np.testing.assert_array_equal(raw, original)
            with self.assertRaises(ValueError):
                AnalysisSettings.from_dict({"rotation": 45})
            with self.assertRaises(ValueError):
                AnalysisSettings.from_dict(
                    self.settings.to_dict() | {"flip_horizontal": "true"}
                ).validate()

        def test_orientation_background_roi_and_calibration(self) -> None:
            """Check identical background orientation and calibrated fitting after ROI cropping."""
            background = np.arange(self.image.size).reshape(self.image.shape) * 0.001
            raw = self.image + background
            original = raw.copy()
            settings = replace(
                self.settings,
                rotation=ImageRotation.ROTATE_90,
                flip_horizontal=True,
                flip_vertical=True,
                pixel_size_x=2e-6,
                pixel_size_y=5e-6,
                roi=(4, 44, 3, 35),
            )
            processed = process_image(raw, settings, background)
            np.testing.assert_allclose(
                processed.image,
                orient_image(self.image, settings)[4:44, 3:35],
                atol=1e-14,
            )
            self.assertEqual(processed.roi_origin, (4, 3))
            np.testing.assert_array_equal(raw, original)
            full = process_image(
                self.image,
                replace(settings, num_slices=1, roi=(None, None, None, None)),
            )
            # The rotated beam widths swap axes; the entered calibration does not.
            np.testing.assert_allclose(
                full.variances[-1], [25 * (2e-6) ** 2, 16 * (5e-6) ** 2], rtol=1e-3
            )
            self.assertEqual(settings.pixel_size_x, 2e-6)
            self.assertEqual(settings.pixel_size_y, 5e-6)

        def test_orientation_preview_export_and_replay(self) -> None:
            """Verify oriented previews and reproducible export of processing transformations."""
            self.write_sample()
            settings = replace(
                self.settings,
                rotation=ImageRotation.ROTATE_270,
                flip_horizontal=True,
                flip_vertical=True,
                pixel_size_x=2e-6,
                pixel_size_y=5e-6,
                roi=(1, 46, 2, 39),
            )
            live = LiveAnalysis(self.path, settings, FakeOptics())
            snapshot = live.refresh()
            preview = preview_processing_sample(self.path, "MES_0_one", settings)
            self.assertEqual(preview.processed.variances, snapshot.shots[0].variances)
            figure = Figure(figsize=(8, 7))
            plot_processing_preview(figure, preview)
            np.testing.assert_array_equal(
                figure.axes[0].images[0].get_array(), orient_image(self.image, settings)
            )
            np.testing.assert_allclose(
                figure.axes[0].images[0].get_extent(), [-0.5, 39.5, -0.5, 47.5]
            )
            np.testing.assert_allclose(
                figure.axes[1].images[0].get_extent(), [0.0075, 0.1925, 0.001, 0.091]
            )
            destination = live.export_bundle(snapshot)
            document = json.loads((destination / "config.json").read_text())
            self.assertEqual(document["settings"]["rotation"], 270)
            self.assertTrue(document["settings"]["flip_horizontal"])
            self.assertTrue(document["settings"]["flip_vertical"])
            restored = AnalysisSettings.from_dict(document["settings"])
            self.assertEqual(restored, settings)
            replay = LiveAnalysis(
                destination / "inputs", restored, ArchivedOptics(document)
            ).refresh()
            self.assertEqual(replay.shots[0].variances, snapshot.shots[0].variances)

        def test_negative_rms_profile_matches_notebook_threshold(self) -> None:
            mean, variance = profile_fit(np.array([-1.0, 1.0, 3.0, 1.0, -1.0]), "RMS")
            self.assertEqual(mean, 2)
            self.assertAlmostEqual(variance, 0.4)

        def test_negative_intensity_policies_and_slicing(self) -> None:
            """Check profile thresholding, clipping, rejection, and nonmonotone slice failures."""
            raw = self.image.copy()
            raw[:, -1] = -1
            config = replace(self.settings, tail_extension=True, mask_cutoff=0.1)
            preserved = process_image(raw, config)
            self.assertIn(-1, preserved.variances)
            self.assertIn(0, preserved.slice_errors)
            clipped = process_image(
                raw, replace(config, negative_intensity=NegativeIntensityPolicy.CLIP)
            )
            self.assertIn(0, clipped.variances)
            self.assertTrue((clipped.image >= 0).all())
            with self.assertRaises(ValueError):
                process_image(
                    raw,
                    replace(config, negative_intensity=NegativeIntensityPolicy.REJECT),
                )

        def test_single_column_slice_preserves_x_width(self) -> None:
            profile = self.image[:, 24]
            one_column = np.zeros_like(self.image)
            one_column[:, 24] = profile
            processed = process_image(one_column, self.settings)
            self.assertGreater(processed.variances[1][0], 0)
            self.assertEqual(processed.variances[1][1], 0)

        def test_zero_dilation_is_noop(self) -> None:
            mask = make_mask(self.image, replace(self.settings, mask_cutoff=0.5))
            self.assertFalse(mask.all())
            self.assertTrue(mask.any())

        def test_background_before_mask(self) -> None:
            config = replace(
                self.settings, mask_cutoff=0.4, subtract_inverse_mask_mean=True
            )
            processed = process_image(self.image + 0.2, config)
            self.assertEqual(float(processed.image[~processed.mask].sum()), 0.0)
            self.assertLess(processed.image.max(), 1.05)
            with self.assertRaises(ValueError):
                process_image(self.image, self.settings, np.zeros((3, 3)))

        def test_tail_is_used(self) -> None:
            tail = process_image(
                self.image, replace(self.settings, num_slices=1, tail_sigma=1.0)
            )
            full = process_image(self.image, self.settings)
            self.assertIsNotNone(tail.tail_start)
            self.assertLess(tail.variances[-1][1], full.variances[-1][1])
            self.assertEqual(set(tail.variances), {-1})

        def test_slice_count_input(self) -> None:
            self.assertEqual(parse_slice_count("1 = projected"), 1)
            self.assertEqual(parse_slice_count("1"), 1)
            self.assertEqual(parse_slice_count(" 20 "), 20)
            self.assertEqual(parse_slice_count("7"), 7)
            for value in ("", "0", "-1", "2.5", "bad"):
                with self.assertRaises(ValueError):
                    parse_slice_count(value)

        def test_enum_settings_keep_json_compatibility(self) -> None:
            """Check enum normalization, settings round trips, and rejection of unsupported methods."""
            settings = AnalysisSettings.from_dict(
                {
                    "fit_type": "Gaussian",
                    "regression": "linear",
                    "weighting": "standard_error",
                    "mask_filter": "uniform",
                    "negative_intensity": "clip",
                }
            )
            self.assertIs(settings.fit_type, FitType.GAUSSIAN)
            self.assertIs(settings.regression, RegressionType.LINEAR)
            self.assertIs(settings.weighting, Weighting.STANDARD_ERROR)
            self.assertIs(settings.mask_filter, MaskFilterType.UNIFORM)
            self.assertIs(settings.negative_intensity, NegativeIntensityPolicy.CLIP)
            document = json.loads(json.dumps(settings.to_dict()))
            self.assertEqual(document["regression"], "linear")
            self.assertEqual(document["fit_type"], "Gaussian")
            path = self.path / "settings.json"
            atomic_json(path, {"analysis": document})
            self.assertEqual(AnalysisSettings.load(path), settings)
            with self.assertRaises(ValueError):
                AnalysisSettings.from_dict({"regression": "unknown"})
            with self.assertRaises(ValueError):
                AnalysisSettings.from_dict(
                    {"regression": "chi_squared_with_dispersion"}
                )

        def test_settings_loading_for_all_sections(self) -> None:
            """Preserve settings, tuple fields, and defaults through file and replay parsing."""
            expected = {
                "analysis": replace(self.settings, roi=(2, 30, 3, None)),
                "lattice": LatticeSettings(phases=(0.1, 0.4, 0.8)),
                "simulation": SimulationSettings(screen_shape=(120, 160)),
            }
            document = json.loads(
                json.dumps(
                    {
                        section: settings.to_dict()
                        for section, settings in expected.items()
                    }
                )
            )
            unchanged = copy.deepcopy(document)
            path = self.path / "settings.json"
            atomic_json(path, document)
            for section, settings in expected.items():
                with self.subTest(section=section):
                    cls = type(settings)
                    self.assertEqual(cls.load(path), settings)
                    self.assertEqual(cls.from_dict(document[section]), settings)
                    self.assertEqual(cls.from_dict({}), cls())
                    with self.assertRaisesRegex(ValueError, f"Unknown {section}"):
                        cls.from_dict({"unknown": 1})
                    with self.assertRaisesRegex(ValueError, "must be an object"):
                        cls.from_dict([])
            self.assertEqual(document, unchanged)

        def test_processing_form_round_trip_and_coverage(self) -> None:
            """Verify all processing fields survive conversion between settings and GUI values."""
            alternatives = replace(
                self.settings,
                fit_type=FitType.GAUSSIAN_FREE_AMPLITUDE,
                weighting=Weighting.STANDARD_ERROR,
                mask_filter=MaskFilterType.UNIFORM,
                negative_intensity=NegativeIntensityPolicy.CLIP,
                roi=(2, 30, 3, None),
                num_slices=1,
                tail_sigma=1.5,
                largest_patch=True,
                subtract_background=True,
                rotation=ImageRotation.ROTATE_90,
                flip_horizontal=True,
                flip_vertical=True,
            )
            for settings in (self.settings, alternatives):
                values = settings.to_form_values()
                parsed = AnalysisSettings.parse_form_values(values)
                rebuilt = AnalysisSettings(**(settings.to_dict() | parsed))
                rebuilt.validate()
                self.assertEqual(rebuilt, settings)
                self.assertEqual(
                    set(parsed),
                    {field.name for field in fields(AnalysisSettings)}
                    - {
                        "regression",
                        "num_slices",
                    },
                )

        def test_camera_resolution_controls(self) -> None:
            """Check calibration presets, numeric overrides, and rejection of invalid scales."""
            values = self.settings.to_form_values()
            parsed = AnalysisSettings.parse_form_values(
                values
                | {
                    "pixel_size_x": "5.535e-6",
                    "pixel_size_y": "5.521e-6",
                }
            )
            self.assertEqual(parsed["pixel_size_x"], 5.535e-6)
            self.assertEqual(parsed["pixel_size_y"], 5.521e-6)
            self.assertEqual(parse_strictly_positive_finite_float("3.45e-6"), 3.45e-6)
            self.assertEqual(parse_strictly_positive_finite_float(" 2e-6 "), 2e-6)
            for name in CAMERA_RESOLUTION_PRESETS:
                for invalid in ("", "bad", "0", "-1e-6", "nan", "inf"):
                    with (
                        self.subTest(name=name, invalid=invalid),
                        self.assertRaises(ValueError),
                    ):
                        AnalysisSettings.parse_form_values(values | {name: invalid})

        def test_processing_form_invalid_numeric_input(self) -> None:
            """Ensure malformed, missing, and unknown form values are reported clearly."""
            values = self.settings.to_form_values()
            for name, value in (
                ("mask_downsampling", "2.5"),
                ("mask_sigma", "bad"),
                ("roi_x_start", "bad"),
            ):
                with self.assertRaises(ValueError):
                    AnalysisSettings.parse_form_values(values | {name: value})
            parsed = AnalysisSettings.parse_form_values(values | {"mask_cutoff": "1.5"})
            with self.assertRaises(ValueError):
                AnalysisSettings(**parsed).validate()

        def test_processing_preview_matches_applied_processing(self) -> None:
            """Compare draft-preview widths and boundaries with the main processing pipeline."""
            self.write_sample()
            snapshot = self.live.refresh()
            draft = replace(self.settings, mask_cutoff=0.2, mask_sigma=1.2)
            preview = preview_processing_sample(self.path, "MES_0_one", draft)
            expected = process_image(self.image, draft)
            np.testing.assert_array_equal(preview.processed.image, expected.image)
            self.assertEqual(preview.processed.variances, expected.variances)
            self.assertEqual(preview.settings, draft)
            self.assertEqual(
                snapshot.document["settings"]["mask_cutoff"], self.settings.mask_cutoff
            )
            self.assertEqual(
                snapshot.shots[0].variances,
                process_image(self.image, self.settings).variances,
            )

        def test_processing_preview_loads_draft_background(self) -> None:
            """Verify preview background subtraction uses the current draft configuration."""
            self.write_sample()
            background = np.full_like(self.image, 0.001)
            np.savez(self.path / "screen_background.npz", data=background)
            draft = replace(
                self.settings,
                subtract_background=True,
                negative_intensity=NegativeIntensityPolicy.CLIP,
            )
            preview = preview_processing_sample(self.path, "MES_0_one", draft)
            expected = process_image(self.image, draft, background)
            np.testing.assert_array_equal(preview.processed.image, expected.image)
            self.assertEqual(preview.processed.variances, expected.variances)

        def test_processing_preview_plot_export(self) -> None:
            self.write_sample()
            preview = preview_processing_sample(self.path, "MES_0_one", self.settings)
            figure = Figure(figsize=(8, 7))
            plot_processing_preview(figure, preview, slice_index=0)
            output = self.path / "processing-preview.png"
            figure.savefig(output)
            self.assertGreater(output.stat().st_size, 1000)
            self.assertEqual(len(figure.axes), 4)

        def test_processing_preview_calibration(self) -> None:
            """Check raw pixels, processed millimetres, profiles, and calibration export consistency."""
            self.write_sample()
            settings = replace(
                self.settings, pixel_size_x=2e-6, pixel_size_y=5e-6, roi=(3, 35, 4, 44)
            )
            preview = preview_processing_sample(self.path, "MES_0_one", settings)
            figure = Figure(figsize=(8, 7))
            plot_processing_preview(figure, preview, slice_index=0)
            raw_axis, signal_axis, mask_axis, profile_axis = figure.axes
            np.testing.assert_allclose(
                raw_axis.images[0].get_extent(), [-0.5, 47.5, -0.5, 39.5]
            )
            self.assertEqual(raw_axis.get_xlabel(), r"$y$ [px]")
            self.assertEqual(raw_axis.get_ylabel(), r"$x$ [px]")
            for axis in (signal_axis, mask_axis):
                np.testing.assert_allclose(
                    axis.images[0].get_extent(), [0.0175, 0.2175, 0.005, 0.069]
                )
            roi = raw_axis.patches[0]
            assert isinstance(roi, Rectangle)
            np.testing.assert_allclose(roi.get_xy(), [3.5, 2.5])
            self.assertEqual(roi.get_width(), 40)
            self.assertEqual(roi.get_height(), 32)
            np.testing.assert_allclose(
                np.asarray(signal_axis.lines[0].get_xdata(), dtype=float),
                (4 + preview.processed.bounds[1] - 0.5) * 0.005,
            )
            for line, scale in zip(profile_axis.lines, (0.002, 0.005), strict=True):
                np.testing.assert_allclose(
                    np.asarray(line.get_xdata(), dtype=float),
                    np.arange(len(np.asarray(line.get_ydata(), dtype=float))) * scale,
                )
            changed = replace(settings, pixel_size_x=4e-6, pixel_size_y=15e-6)
            updated = preview_processing_sample(self.path, "MES_0_one", changed)
            snapshot = LiveAnalysis(self.path, changed, FakeOptics()).refresh()
            plot_processing_preview(figure, updated, slice_index=0)
            np.testing.assert_allclose(
                figure.axes[1].images[0].get_extent(), [0.0525, 0.6525, 0.01, 0.138]
            )
            for index, variances in preview.processed.variances.items():
                expected = np.asarray(variances) * [4, 9]
                np.testing.assert_allclose(updated.processed.variances[index], expected)
                np.testing.assert_allclose(snapshot.shots[0].variances[index], expected)
            self.assertEqual(snapshot.document["settings"]["pixel_size_x"], 4e-6)
            self.assertEqual(snapshot.document["settings"]["pixel_size_y"], 15e-6)

        def test_projected_mode_and_slice_count_reprocessing(self) -> None:
            """Verify projected mode and new slice counts replace cached processing results."""
            self.write_sample()
            projected = LiveAnalysis(
                self.path, replace(self.settings, num_slices=1), FakeOptics()
            ).refresh()
            self.assertEqual(set(projected.shots[0].variances), {-1})
            self.assertEqual(
                [(fit["plane"], fit["slice_index"]) for fit in projected.fits],
                [("x", -1), ("y", -1)],
            )
            four = LiveAnalysis(
                self.path, replace(self.settings, num_slices=4), FakeOptics()
            ).refresh()
            self.assertEqual(len(four.shots[0].bounds), 5)
            self.assertEqual(set(four.shots[0].variances), {-1, 0, 1, 2, 3})
            self.assertNotEqual(
                projected.shots[0].variances[-1], four.shots[0].variances[0]
            )

        def test_missing_magnet_is_rejected(self) -> None:
            with self.assertRaises(ValueError):
                magnet_parameters([], {"q_t3_k1": 1}, FakeOptics.lengths)

        def test_incomplete_retry_replacement_and_deletion(self) -> None:
            """Check retries and cache invalidation as acquisition files change or disappear."""
            path = self.path / "samples_raw" / "MES_0_one.npz"
            path.write_bytes(b"incomplete zip")
            snapshot = self.live.refresh()
            self.assertEqual(
                snapshot.document["samples"][path.stem]["status"], "rejected"
            )
            self.write_sample()
            snapshot = self.live.refresh()
            self.assertEqual(len(snapshot.shots), 1)
            old_hash = snapshot.shots[0].source_hash
            self.write_sample(image=self.image**2)
            snapshot = self.live.refresh()
            self.assertNotEqual(snapshot.shots[0].source_hash, old_hash)
            path.write_bytes(b"bad again")
            self.assertEqual(len(self.live.refresh().shots), 0)
            path.unlink()
            self.assertEqual(self.live.refresh().document["samples"], {})

        def test_duplicates_and_conflicting_optics(self) -> None:
            """Verify duplicate images cannot distort the fit or silently change their optics."""
            self.write_sample("MES_0_a")
            self.write_sample("MES_0_b")
            snapshot = self.live.refresh()
            self.assertEqual(len(snapshot.shots), 1)
            self.assertEqual(
                snapshot.document["samples"]["MES_0_b"]["status"], "duplicate"
            )
            self.write_sample("MES_0_b", kick=600)
            snapshot = self.live.refresh()
            self.assertEqual(len(snapshot.shots), 0)
            self.assertTrue(
                all(
                    info["status"] == "rejected"
                    for info in snapshot.document["samples"].values()
                )
            )

        def test_new_arrivals_do_not_block_export(self) -> None:
            self.write_sample("MES_0_a")
            snapshot = self.live.refresh()
            self.write_sample("MES_0_b", image=self.image**2)
            destination = self.live.export_bundle(snapshot)
            archived = json.loads((destination / "config.json").read_text())
            self.assertEqual(len(archived["shots"]), 1)
            self.assertEqual(len(self.live.refresh().shots), 2)

        def test_changing_rejected_files_do_not_block_export(self) -> None:
            self.write_sample("MES_0_a")
            bad = self.path / "samples_raw" / "MES_0_bad.npz"
            bad.write_bytes(b"partial")
            snapshot = self.live.refresh()
            bad.write_bytes(b"still writing")
            destination = self.live.export_bundle(snapshot)
            self.assertTrue((destination / "config.json").exists())

        def test_dependency_invalidation_and_export(self) -> None:
            """Check dependency changes clear caches and exports retain exact inputs and settings."""
            self.write_sample()
            snapshot = self.live.refresh()
            previous_matrix = snapshot.shots[0].matrix.copy()
            self.document["high_level_params"][0]["beam_energy"] = 16.3
            atomic_json(self.path / "lattice_configs.json", self.document)
            with self.assertRaises(ValueError):
                self.live.export_bundle(snapshot)
            snapshot = self.live.refresh()
            self.assertEqual(snapshot.document["energy_GeV"], 16.3)
            np.testing.assert_allclose(snapshot.shots[0].matrix, previous_matrix)
            destination = self.live.export_bundle(snapshot)
            config = json.loads((destination / "config.json").read_text())
            self.assertEqual(config["settings"]["weighting"], "shot_std")
            self.assertTrue(
                (destination / "inputs" / "samples_raw" / "MES_0_one.npz").exists()
            )
            self.assertTrue(
                (destination / "sliced" / "00" / "000" / "MES_0_one.json").exists()
            )
            self.assertEqual(
                digest((destination / "inputs" / "lattice_configs.json").read_bytes()),
                config["source_sha256"]["lattice_configs.json"],
            )
            replay = LiveAnalysis(
                destination / "inputs", self.settings, ArchivedOptics(config)
            ).refresh()
            np.testing.assert_allclose(replay.shots[0].matrix, snapshot.shots[0].matrix)
            self.assertEqual(replay.shots[0].variances, snapshot.shots[0].variances)
            self.assertEqual(len(self.live.refresh().shots), 1)

        def test_unsettled_files_do_not_retain_old_results(self) -> None:
            """Ensure changing files lose their old results until a stable replacement is available."""
            self.write_sample()
            self.assertEqual(len(self.live.refresh().shots), 1)
            self.live.settings = replace(self.settings, settle_seconds=100)
            self.write_sample(image=self.image**2)
            snapshot = self.live.refresh()
            self.assertEqual(len(snapshot.shots), 0)
            self.assertEqual(
                snapshot.document["samples"]["MES_0_one"]["status"], "pending"
            )

        def test_background_change_invalidates_cache(self) -> None:
            """Verify a changed background forces shot reprocessing and dependency rehashing."""
            self.write_sample()
            self.live = LiveAnalysis(
                self.path,
                replace(self.settings, subtract_background=True),
                FakeOptics(),
            )
            path = self.path / "screen_background.npz"
            np.savez(path, data=np.zeros_like(self.image))
            before = self.live.refresh()
            np.savez(path, data=self.image * 0.1)
            after = self.live.refresh()
            self.assertNotEqual(
                before.document["source_sha256"][path.name],
                after.document["source_sha256"][path.name],
            )
            self.assertEqual(len(after.shots), 1)

        def test_live_plot_layout_and_width_calibration(self) -> None:
            """Check reconstruction panel positions, units, shared axes, and calibrated screen widths."""
            self.write_sample()
            self.live = LiveAnalysis(
                self.path,
                replace(
                    self.settings,
                    pixel_size_x=2e-6,
                    pixel_size_y=5e-6,
                ),
                FakeOptics(),
            )
            snapshot = self.live.refresh()
            figure = Figure(figsize=(13, 7))
            plot_snapshot(figure, snapshot)
            width_axis, width_3d_axis, ellipse_axis = figure.axes[:3]
            self.assertEqual(len(figure.axes), 8)
            for index, axis in enumerate(figure.axes[:6]):
                spec = axis.get_subplotspec()
                assert spec is not None
                self.assertEqual(
                    (spec.rowspan.start, spec.colspan.start), divmod(index, 3)
                )
            self.assertTrue(all(not axis.images for axis in figure.axes))
            expected = np.sqrt(snapshot.shots[0].variances[0][0]) * 1e6
            np.testing.assert_allclose(
                np.asarray(width_axis.lines[-1].get_ydata(), dtype=float), [expected]
            )
            self.assertIn(r"[$\mathrm{\mu m}$]", width_axis.get_ylabel())
            self.assertEqual(width_3d_axis.name, "3d")
            assert isinstance(width_3d_axis, Axes3D)
            self.assertIn(r"[$\mathrm{\mu m}$]", width_3d_axis.get_zlabel())
            self.assertIn("[mrad]", ellipse_axis.get_ylabel())
            self.assertEqual(figure.axes[5].get_ylabel(), r"$\zeta$")
            self.assertTrue(
                figure.axes[3]
                .get_shared_x_axes()
                .joined(figure.axes[3], figure.axes[6])
            )
            self.assertTrue(
                figure.axes[4]
                .get_shared_x_axes()
                .joined(figure.axes[4], figure.axes[7])
            )

        def test_beam_width_statistics_match_notebook(self) -> None:
            """Verify per-slice means, sample standard deviations, projected mode, and empty scans."""
            self.write_sample()
            snapshot = self.live.refresh()
            shot = snapshot.shots[0]
            snapshot.shots = [
                replace(
                    shot,
                    config_index=4,
                    variances={
                        -1: (100e-12, 100e-12),
                        0: (1e-12, 16e-12),
                        1: (9e-12, 4e-12),
                    },
                ),
                replace(
                    shot,
                    config_index=4,
                    variances={
                        0: (9e-12, 64e-12),
                        1: (25e-12, 16e-12),
                    },
                ),
                replace(shot, config_index=7, variances={0: (16e-12, 36e-12)}),
            ]
            widths = beam_widths_at_screen(snapshot, Plane.X)
            self.assertEqual([result.slice_index for result in widths], [0, 1])
            self.assertEqual(widths[0].configuration_indices, [4, 7])
            np.testing.assert_allclose(widths[0].mean_um, [2, 4])
            np.testing.assert_allclose(widths[0].std_um, [np.sqrt(2), np.nan])
            np.testing.assert_allclose(widths[1].mean_um, [4])
            np.testing.assert_allclose(widths[1].std_um, [np.sqrt(2)])
            y_widths = beam_widths_at_screen(snapshot, Plane.Y)
            np.testing.assert_allclose(y_widths[0].mean_um, [6, 6])
            np.testing.assert_allclose(y_widths[0].std_um, [2 * np.sqrt(2), np.nan])

            figure = Figure(figsize=(13, 7))
            plot_snapshot(figure, snapshot)
            width_axis, width_3d_axis = figure.axes[:2]
            np.testing.assert_allclose(
                np.asarray(width_axis.lines[1].get_ydata(), dtype=float), [2, 4]
            )
            container = width_axis.containers[1]
            assert isinstance(container, ErrorbarContainer)
            segment = container.lines[2][0].get_segments()[0]
            np.testing.assert_allclose(
                segment, [[4, 2 - np.sqrt(2)], [4, 2 + np.sqrt(2)]]
            )
            curves = [line for line in width_3d_axis.lines if line.get_marker() == "o"]
            np.testing.assert_allclose(
                np.asarray(cast(Line3D, curves[1]).get_data_3d(), dtype=float),
                [[4, 7], [0, 0], [2, 4]],
            )

            projected = replace(
                snapshot,
                document=snapshot.document
                | {
                    "settings": replace(self.settings, num_slices=1).to_dict(),
                },
            )
            widths = beam_widths_at_screen(projected, Plane.X)
            self.assertEqual([result.slice_index for result in widths], [-1])
            np.testing.assert_allclose(widths[0].mean_um, [10])
            self.assertTrue(np.isnan(widths[0].std_um[0]))
            plot_snapshot(figure, projected)
            self.assertEqual(
                figure.axes[0].get_legend_handles_labels()[1], ["projected"]
            )
            self.assertEqual(
                figure.axes[1].get_yticklabels()[0].get_text(), "projected"
            )
            empty = replace(
                snapshot, shots=[], document=snapshot.document | {"fits": []}
            )
            self.assertEqual(beam_widths_at_screen(empty, Plane.X), [])
            plot_snapshot(figure, empty)
            self.assertFalse(figure.axes[0].lines)
            self.assertFalse(figure.axes[1].lines)

        def test_slice_ellipse_overlay(self) -> None:
            """Compare slice ellipses with fitted covariance and reject invalid or unsupported results."""
            self.write_sample()
            snapshot = self.live.refresh()
            settings = replace(self.settings, num_slices=3)
            shots, _ = self.synthetic()
            projected_fit = fit_scan(shots, -1, Plane.X, 14, settings)
            slice_fits = []
            for index, factor in ((0, 1), (1, 2)):
                slice_shots = [
                    replace(
                        shot,
                        variances={
                            index: (
                                shot.variances[-1][0] * factor,
                                shot.variances[-1][1],
                            ),
                        },
                    )
                    for shot in shots
                ]
                result = fit_scan(slice_shots, index, Plane.X, 14, settings)
                self.assertEqual(result["status"], FitStatus.VALID)
                slice_fits.append(result)
            invalid_fit: FitResult = {
                "plane": Plane.X,
                "slice_index": 2,
                "status": FitStatus.INVALID,
                "reason": "Nonphysical covariance",
                "sample_count": 0,
                "excluded_slices": {},
                "counts_per_configuration": {},
            }
            snapshot.document["settings"] = settings.to_dict()
            snapshot.document["fits"] = [
                projected_fit,
                invalid_fit,
                slice_fits[1],
                slice_fits[0],
            ]
            figure = Figure(figsize=(13, 7))
            colors = colormaps["coolwarm"](np.linspace(0, 1, settings.num_slices))

            # All valid slices appear together; projected and invalid fits stay excluded.
            plot_snapshot(figure, snapshot)
            axis = figure.axes[2]
            self.assertEqual(
                axis.get_legend_handles_labels()[1], ["slice 0", "slice 1"]
            )
            self.assertIn("2/3 valid", axis.get_title())
            self.assertIn("[mm]", axis.get_xlabel())
            self.assertIn("[mrad]", axis.get_ylabel())
            for line, result in zip(axis.lines, slice_fits, strict=True):
                points = np.asarray(line.get_data(), dtype=float) * 1e-3
                covariance = np.asarray(result["transverse_covariance"])
                self.assertEqual(points.shape, (2, 100))
                # Every plotted point lies on the fitted one-sigma ellipse.
                np.testing.assert_allclose(
                    np.einsum("in,ij,jn->n", points, np.linalg.inv(covariance), points),
                    1,
                    rtol=1e-12,
                )
                np.testing.assert_allclose(
                    np.asarray(line.get_color(), dtype=float),
                    colors[result["slice_index"]],
                )

            # A fit disappearing from a later snapshot removes its ellipse immediately.
            snapshot.document["fits"] = [projected_fit, slice_fits[0]]
            plot_snapshot(figure, snapshot)
            self.assertEqual(len(figure.axes[2].lines), 1)
            snapshot.document["fits"] = [invalid_fit]
            plot_snapshot(figure, snapshot)
            self.assertFalse(figure.axes[2].lines)
            self.assertIn("0/3 valid", figure.axes[2].get_title())
            plot_snapshot(figure, snapshot, plane=Plane.Y)
            self.assertFalse(figure.axes[2].lines)
            self.assertIn(r"sliced $y$", figure.axes[2].texts[0].get_text())

            snapshot.document["settings"] = replace(settings, num_slices=1).to_dict()
            snapshot.document["fits"] = [projected_fit, *slice_fits]
            plot_snapshot(figure, snapshot)
            self.assertEqual(
                figure.axes[2].get_legend_handles_labels()[1], ["projected"]
            )

        def test_emittance_chi_squared_and_twiss_plots(self) -> None:
            """Check emittance, score, Twiss errors, and projected or unavailable plot states."""
            self.write_sample()
            snapshot = self.live.refresh()
            shots, _ = self.synthetic()
            result = fit_scan(shots, -1, Plane.X, 14, self.settings)
            first = result | {"slice_index": 0, "chi_squared": 18.0}
            second = result | {
                "slice_index": 1,
                "chi_squared": 36.0,
                "beta": result["beta"] * 1.5,
                "alpha": result["alpha"] * 2,
            }
            snapshot.document["fits"] = [second, first]
            figure = Figure(figsize=(13, 7))
            plot_snapshot(figure, snapshot)
            emit_axis, beta_axis = figure.axes[3:5]
            chi_axis, alpha_axis = figure.axes[6:8]
            np.testing.assert_allclose(
                np.asarray(emit_axis.lines[0].get_ydata(orig=False), dtype=float),
                [result["normalized_emittance"] * 1e6] * 2,
            )
            np.testing.assert_allclose(
                np.asarray(chi_axis.lines[0].get_data(), dtype=float),
                [[0, 1], [1, 2]],
            )
            self.assertEqual(chi_axis.get_ylabel(), r"$\chi^2/N_{\mathrm{samples}}$")
            np.testing.assert_allclose(
                np.asarray(beta_axis.lines[0].get_ydata(orig=False), dtype=float),
                [first["beta"], second["beta"]],
            )
            np.testing.assert_allclose(
                np.asarray(alpha_axis.lines[0].get_ydata(orig=False), dtype=float),
                [first["alpha"], second["alpha"]],
            )
            parameters: tuple[tuple[Axes, Literal["beta", "alpha"]], ...] = (
                (beta_axis, "beta"),
                (alpha_axis, "alpha"),
            )
            errors = result["errors"]
            assert errors is not None
            for axis, name in parameters:
                container = axis.containers[0]
                assert isinstance(container, ErrorbarContainer)
                segment = container.lines[2][0].get_segments()[0]
                error = errors[name]
                np.testing.assert_allclose(
                    segment, [[0, first[name] - error], [0, first[name] + error]]
                )
            self.assertEqual(beta_axis.get_ylabel(), r"$\beta_{x}$ [m]")
            self.assertEqual(alpha_axis.get_ylabel(), r"$\alpha_{x}$")

            # Waiting/invalid fits remove their values from both axes together.
            snapshot.document["fits"] = [first, second | {"status": FitStatus.WAITING}]
            plot_snapshot(figure, snapshot)
            for index in (3, 4, 6, 7):
                np.testing.assert_allclose(
                    np.asarray(
                        figure.axes[index].lines[0].get_xdata(orig=False), dtype=float
                    ),
                    [0],
                )
            plot_snapshot(figure, snapshot, plane=Plane.Y)
            for index in (3, 4, 6, 7):
                self.assertFalse(figure.axes[index].lines)

            snapshot.document["settings"] = replace(
                self.settings, num_slices=1
            ).to_dict()
            snapshot.document["fits"] = [result]
            plot_snapshot(figure, snapshot)
            for index in (3, 4):
                self.assertEqual(
                    figure.axes[index].get_xticklabels()[0].get_text(), "projected"
                )

            snapshot.document["settings"]["regression"] = RegressionType.LINEAR
            linear_result = result.copy()
            linear_result.pop("chi_squared", None)
            linear_result["errors"] = None
            linear_result["normalized_emittance_error"] = None
            snapshot.document["fits"] = [linear_result]
            plot_snapshot(figure, snapshot)
            self.assertFalse(figure.axes[6].lines)
            self.assertIn("unavailable", figure.axes[6].texts[0].get_text())
            self.assertTrue(figure.axes[4].lines)

        def test_plot_export(self) -> None:
            self.write_sample()
            snapshot = self.live.refresh()
            figure = Figure(figsize=(13, 7))
            plot_snapshot(figure, snapshot)
            destination = self.live.export_bundle(snapshot, figure=figure)
            self.assertGreater((destination / "plots.png").stat().st_size, 1000)
            self.assertGreater((destination / "plots.pdf").stat().st_size, 1000)

        def test_export_rechecks_contents_while_copying(self) -> None:
            """Keep changed dependency and shot bytes out of published export bundles."""
            source = self.write_sample()
            snapshot = self.live.refresh()
            parent = self.path / "exports"
            read_bytes = stable_bytes
            for relative_path in (
                Path("lattice_configs.json"),
                Path("samples_raw") / source.name,
            ):
                target = self.live.experiment / relative_path

                def changed_bytes(path: Path) -> tuple[bytes, Signature]:
                    content, stat = read_bytes(path)
                    if path == target:
                        content += b"changed"
                    return content, stat

                with (
                    self.subTest(path=relative_path),
                    patch.object(sys.modules[__name__], "stable_bytes", changed_bytes),
                    self.assertRaisesRegex(ValueError, "Input changed before export"),
                ):
                    self.live.export_bundle(snapshot, parent=parent)
            self.assertEqual(list(parent.glob("processed_*")), [])
            incomplete = list(parent.glob(".processing-*"))
            self.assertEqual(len(incomplete), 2)
            self.assertTrue(
                all((path / "INCOMPLETE.json").is_file() for path in incomplete)
            )

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    """Parse the requested command and dispatch analysis, preparation, replay, or the desktop app."""

    def build_parser() -> argparse.ArgumentParser:
        """Define CLI commands and their required experiment, settings, and output arguments."""
        parser = argparse.ArgumentParser(
            description="EuXFEL T3 live optics-scan emittance tool (originally AI generated)"
        )
        commands = parser.add_subparsers(dest="command", required=True)

        def add_experiment_command(command: Command, help_text: str) -> None:
            """Add shared experiment arguments and the selected command options."""
            subparser = commands.add_parser(command, help=help_text)
            experiment_help = (
                "Experiment directory containing lattice_configs.json and samples_raw/"
            )
            match command:
                case Command.REPLAY:
                    experiment_help = "Timestamped processed export folder"
            subparser.add_argument(
                "experiment",
                type=Path,
                help=experiment_help,
            )
            subparser.add_argument(
                "--settings",
                type=Path,
                help="App settings JSON from the 'settings' command or GUI",
            )
            match command:
                case Command.ANALYZE | Command.REPLAY:
                    subparser.add_argument(
                        "--output",
                        type=Path,
                        help="Parent for timestamped processed folder (default: input experiment directory)",
                    )
                    subparser.add_argument(
                        "--no-raw-copy",
                        action="store_true",
                        help="Export source hashes without copying raw inputs",
                    )
                case Command.INSPECT:
                    subparser.add_argument(
                        "--sample", required=True, help="Sample filename stem"
                    )
                    subparser.add_argument(
                        "--output",
                        type=Path,
                        required=True,
                        help="Plot path (.png, .pdf, .svg, ...)",
                    )
                case Command.SIMULATE:
                    subparser.add_argument(
                        "--source",
                        type=Path,
                        required=True,
                        help="Source lattice experiment",
                    )

        for command, help_text in [
            (Command.GUI, "Desktop app"),
            (Command.ANALYZE, "Process current data and export a timestamped folder"),
            (Command.INSPECT, "Export a selected screen inspection"),
            (Command.LATTICE, "Generate new bounded optics configurations"),
            (Command.SIMULATE, "Track particles into a new experiment"),
            (
                Command.REPLAY,
                "Reprocess an export with archived raw inputs and transfer matrices",
            ),
        ]:
            add_experiment_command(command, help_text)
        settings_parser = commands.add_parser(
            Command.SETTINGS, help="Write all default settings to JSON"
        )
        settings_parser.add_argument("--output", type=Path, required=True)
        commands.add_parser(
            Command.SELF_TEST, help="Run scientific and live-file regression tests"
        )
        return parser

    def load_verified_archive() -> tuple[SnapshotReport, AnalysisSettings]:
        """Check archived source hashes before replaying recorded data and optics."""
        archive = Snapshot.parse_report(
            json.loads((args.experiment / "config.json").read_text())
        )
        export = archive.get("export")
        if export is None or not export.get("raw_inputs_included"):
            raise ValueError("Replay requires an export that includes raw inputs")
        settings = (
            AnalysisSettings.load(args.settings)
            if args.settings
            else AnalysisSettings.from_dict(archive["settings"])
        )
        input_dir = args.experiment / "inputs"
        for shot in archive["shots"]:
            if (
                digest(
                    (
                        input_dir / "samples_raw" / f"{shot['sample_id']}.npz"
                    ).read_bytes()
                )
                != shot["source_sha256"]
            ):
                raise ValueError(
                    f"Archived sample checksum failed: {shot['sample_id']}"
                )
        for name, expected_hash in archive["source_sha256"].items():
            if digest((input_dir / name).read_bytes()) != expected_hash:
                raise ValueError(f"Archived dependency checksum failed: {name}")
        return archive, settings

    def inspect_and_export() -> None:
        """Export one processing preview without constructing optics or fitting the scan."""
        settings.validate()
        preview = preview_processing_sample(args.experiment, args.sample, settings)
        figure = Figure(figsize=(13, 7))
        plot_processing_preview(figure, preview)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180, bbox_inches="tight")
        print(args.output.resolve())

    def analyze_and_export() -> None:
        """Read a settled batch and export reconstruction results and their source data."""
        settings.validate()
        # One batch observation establishes the settle interval, then loads current data.
        live = (
            LiveAnalysis(args.experiment / "inputs", settings, ArchivedOptics(archive))
            if archive
            else LiveAnalysis(args.experiment, settings)
        )
        live.refresh()
        time.sleep(settings.settle_seconds)
        snapshot = live.refresh()

        figure = Figure(figsize=(13, 7))
        sample_id = snapshot.shots[0].sample_id if snapshot.shots else ""
        plot_snapshot(figure, snapshot)
        destination = live.export_bundle(
            snapshot,
            args.output or args.experiment,
            figure,
            not args.no_raw_copy,
            {"sample_id": sample_id, "plane": "x"},
        )
        print(destination)
        for result in snapshot.fits:
            if result["slice_index"] == -1:
                print(
                    result["plane"],
                    result["status"],
                    result.get(
                        "reason",
                        f"εn={result.get('normalized_emittance', 0) * 1e6:.5g} mm mrad",
                    ),
                )

    args = build_parser().parse_args()
    args.command = Command(args.command)
    archive = None
    match args.command:
        case Command.SELF_TEST:
            self_test()
        case Command.SETTINGS:
            atomic_json(
                args.output,
                {
                    "analysis": AnalysisSettings().to_dict(),
                    "lattice": LatticeSettings().to_dict(),
                    "simulation": SimulationSettings().to_dict(),
                },
            )
            print(args.output.resolve())
        case Command.GUI:
            desktop(args.experiment, args.settings)
        case Command.LATTICE:
            generate_lattices(
                args.experiment,
                LatticeSettings.load(args.settings),
            )
        case Command.REPLAY:
            archive, settings = load_verified_archive()
            analyze_and_export()
        case Command.SIMULATE:
            settings = AnalysisSettings.load(args.settings)
            simulate_experiment(
                args.source,
                args.experiment,
                SimulationSettings.load(args.settings),
                settings,
            )
        case Command.INSPECT:
            settings = AnalysisSettings.load(args.settings)
            inspect_and_export()
        case Command.ANALYZE:
            settings = AnalysisSettings.load(args.settings)
            analyze_and_export()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
