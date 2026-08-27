# %% Imports and publication settings
r"""Faithful, consolidated generator for the publication figures in ``figures.py``.

The plotting code in this file is ported from the final original figure blocks:

1. pooled experimental/simulation :math:`\Delta S_a` scatter and linear fits;
2. six-panel experimental/simulation :math:`\Delta S_a` histories;
3. radial height PSD and PSD gain by strain group;
4/5. bracketed mean log10 PSD gain and radial ACF change by strain group;
6. four-panel endpoint experimental/simulation radial PSD and ACF comparison;
7. the same endpoint comparison with the experiment kept as one full crop after
   resampling to 1 um (a sectioning sensitivity check).

Only Figure 6 adds the manuscript-style matched height-field sectioning.  All
other experimental height calculations use the requested crop as one complete
field.  Raw experimental and simulation inputs are read once into the four
notebook-facing caches ``exp_df``, ``sim_df``, ``exp_heights``, and
``sim_heights``.  Every ACF curve uses the same linear, overlap-corrected
estimator reduced over radial annuli.
"""

from __future__ import annotations

import hashlib
import json
import warnings
import zipfile
from contextlib import AbstractContextManager
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import fftconvolve

from utils.config import (
    DATA_DIR,
    MICROSTRUCTURE_DIR,
    PROFILOMETRY_SPACING_UM,
    RC_PARAMS,
    RESULTS_DIR,
    VOXELSIZE,
)
from utils.data_utils import SimResults


# The imported project style is the only global Matplotlib style source.
mpl.rcParams.update(RC_PARAMS)

PROJECT_ROOT = Path(DATA_DIR).resolve().parent
OUTPUT_DIR = Path(RESULTS_DIR) / "publication_figures"
CACHE_DIR = Path(RESULTS_DIR) / "publication_figure_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

POLISH = "polished"
MAGNIFICATION = "10x"
STRAIN_COLUMN = "epav33"

EXPERIMENTS = (
    (475, "int"),
    (500, "unint"),
    (525, "int"),
    (530, "unint"),
    (575, "int"),
    (588, "unint"),
)
INTERRUPTED_CASES = ((475, "int"), (525, "int"), (575, "int"))
UNINTERRUPTED_CASES = ((500, "unint"), (530, "unint"), (588, "unint"))
PLOT_GRID = (INTERRUPTED_CASES, UNINTERRUPTED_CASES)

# Requested fixed experiment colors, shared by experimental and simulation data.
LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

EXP_NATIVE_SPACING_UM = float(PROFILOMETRY_SPACING_UM[MAGNIFICATION])
SIM_SPACING_UM = float(VOXELSIZE)
EXP_ANALYSIS_CROP = (slice(50, -50), slice(50, 750))
EXPECTED_EXP_RAW_SHAPE = (768, 1024)
EXPECTED_CROP_FIRST_MATCHED_FIELDS = 21
EXCLUDED_DIRECTORY_NAMES = {"bad"}

# Figure 3: original long-wavelength radial PSD presentation.
RADIAL_WAVELENGTH_MIN_UM = 3.0
RADIAL_WAVELENGTH_MAX_UM = 400.0
RADIAL_WAVELENGTH_BINS = 90
RADIAL_MIN_MODES = 1

# Figures 4/5: original corrected 20-um brackets and ACF range.
BRACKET_WAVELENGTH_MIN_UM = 4.0 * EXP_NATIVE_SPACING_UM
BRACKET_WAVELENGTH_MAX_UM = 300.0
BRACKET_WIDTH_UM = 20.0
ACF_CHANGE_BIN_WIDTH_UM = 2.0 * EXP_NATIVE_SPACING_UM
ACF_CHANGE_MAX_LAG_UM = 128.0

# Shared strain grouping used by Figures 3--5.  Each positive-strain
# observation belongs to exactly one load-pair group; the displayed strain
# level is the mean of the same observations used for the plotted group mean.
INITIAL_TIME_TOL_H = 1.0e-8
INITIAL_STRAIN_TOL_PERCENT = 0.05
STRAIN_LOAD_GROUPS = ((475, 500), (525, 530), (575, 588))

# Figure 6: one matched 256-by-128 um sectioning operator for both sources.
# Figure 7 changes only the experimental support: it keeps the resampled crop
# as one field while reusing the Figure 6 simulation curves unchanged.
MATCHED_TARGET_SPACING_UM = 1.0
MATCHED_FIELD_SHAPE_UM = (256.0, 128.0)
ENDPOINT_WAVELENGTH_MIN_UM = 4.0 * EXP_NATIVE_SPACING_UM
ENDPOINT_WAVELENGTH_MAX_UM = 128.0
ENDPOINT_WAVELENGTH_BINS = 16
ENDPOINT_MIN_MODES = 4
ENDPOINT_ACF_BIN_WIDTH_UM = 2.0
ENDPOINT_ACF_MAX_LAG_UM = 128.0

MICRO_RUNS = tuple(
    {
        "micro_id": f"micro{i}",
        "sim_root": PROJECT_ROOT
        / "hpc_downloads"
        / "gtdebru"
        / f"micro{i}_production",
        "microstructure": Path(MICROSTRUCTURE_DIR)
        / "production"
        / f"micro{i}_production.dat",
    }
    for i in (1, 2, 3)
)

# Exact requested cache names, both as file stems and notebook-facing variables.
CACHE_SCHEMA_VERSION = 2
EXP_DF_CACHE = CACHE_DIR / "exp_df.pkl"
SIM_DF_CACHE = CACHE_DIR / "sim_df.pkl"
EXP_HEIGHTS_CACHE = CACHE_DIR / "exp_heights.npz"
SIM_HEIGHTS_CACHE = CACHE_DIR / "sim_heights.npz"
CACHE_MANIFEST = CACHE_DIR / "publication_cache_manifest.json"


def load_color(load_mpa: int | float) -> str:
    return LOAD_COLORS[int(load_mpa)]


def strain_group_colors(n_groups: int) -> np.ndarray:
    if n_groups < 1:
        return np.empty((0, 4))
    return mpl.colormaps["viridis"](np.linspace(0.18, 0.90, n_groups))


def add_panel_label(axis, label: str) -> None:
    axis.text(
        0.03,
        0.95,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )


def configure_log_wavelength_axis(
    axis,
    minimum_wavelength_um: float,
    maximum_wavelength_um: float,
) -> None:
    """Use numeric decade labels and append only the configured upper limit."""
    axis.set_xscale("log")
    axis.set_xlim(minimum_wavelength_um, maximum_wavelength_um)
    first_decade = int(np.ceil(np.log10(minimum_wavelength_um)))
    last_decade = int(np.floor(np.log10(maximum_wavelength_um)))
    ticks = 10.0 ** np.arange(first_decade, last_decade + 1)
    if not np.any(np.isclose(ticks, maximum_wavelength_um)):
        ticks = np.append(ticks, float(maximum_wavelength_um))
    ticks = np.unique(np.sort(ticks))
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{tick:g}" for tick in ticks])
    axis.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


# %% One-time experimental and simulation caches
def _numeric_file_sort_key(path: Path) -> tuple[int, float, str]:
    try:
        return 0, float(path.stem), str(path)
    except ValueError:
        return 1, np.inf, str(path)


def _array_key(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return "h_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


class StreamingNpzWriter(AbstractContextManager["StreamingNpzWriter"]):
    """Write a compressed NPZ incrementally without retaining all maps in RAM."""

    def __init__(self, destination: Path):
        self.destination = Path(destination)
        self.temporary = self.destination.with_suffix(self.destination.suffix + ".tmp")
        self.archive: zipfile.ZipFile | None = None

    def __enter__(self) -> "StreamingNpzWriter":
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        self.archive = zipfile.ZipFile(
            self.temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        )
        return self

    def write(self, key: str, values: np.ndarray) -> None:
        if self.archive is None:
            raise RuntimeError("StreamingNpzWriter is not open.")
        with self.archive.open(f"{key}.npy", mode="w", force_zip64=True) as stream:
            np.lib.format.write_array(
                stream,
                np.ascontiguousarray(values),
                allow_pickle=False,
            )

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.archive is not None:
            self.archive.close()
        if exc_type is None:
            self.temporary.replace(self.destination)
        else:
            self.temporary.unlink(missing_ok=True)
        return False


def _write_dataframe_cache(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_pickle(temporary)
    temporary.replace(destination)


def _cache_configuration() -> dict[str, object]:
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "data_dir": str(Path(DATA_DIR).resolve()),
        "microstructure_dir": str(Path(MICROSTRUCTURE_DIR).resolve()),
        "experiments": [[int(load), sample_type] for load, sample_type in EXPERIMENTS],
        "polish": POLISH,
        "magnification": MAGNIFICATION,
        "experimental_spacing_um": EXP_NATIVE_SPACING_UM,
        "simulation_spacing_um": SIM_SPACING_UM,
        "micro_runs": [
            {
                "micro_id": str(run["micro_id"]),
                "sim_root": str(Path(run["sim_root"])),
                "microstructure": str(Path(run["microstructure"])),
            }
            for run in MICRO_RUNS
        ],
    }


def _write_cache_manifest() -> None:
    temporary = CACHE_MANIFEST.with_suffix(CACHE_MANIFEST.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_cache_configuration(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(CACHE_MANIFEST)


@lru_cache(maxsize=1)
def _cache_fingerprint() -> str:
    payload = json.dumps(_cache_configuration(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_manifest_is_current() -> bool:
    try:
        recorded = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return recorded == _cache_configuration()


def _read_current_dataframe_cache(
    path: Path,
    *,
    manifest_current: bool | None = None,
) -> pd.DataFrame | None:
    if manifest_current is None:
        manifest_current = _cache_manifest_is_current()
    if not path.exists() or not manifest_current:
        return None
    try:
        frame = pd.read_pickle(path)
    except Exception:
        return None
    valid = (
        len(frame) > 0
        and "cache_schema_version" in frame
        and frame["cache_schema_version"].eq(CACHE_SCHEMA_VERSION).all()
        and "cache_fingerprint" in frame
        and frame["cache_fingerprint"].eq(_cache_fingerprint()).all()
    )
    return frame if valid else None


def raw_height_csv(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def fill_missing_nearest(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    missing = ~np.isfinite(values)
    if not np.any(missing):
        return values
    if np.all(missing):
        raise ValueError("Height field contains no finite values.")
    nearest = distance_transform_edt(
        missing,
        return_distances=False,
        return_indices=True,
    )
    return values[tuple(nearest)]


def _read_strain_table(load_mpa: int, sample_type: str) -> pd.DataFrame:
    path = Path(DATA_DIR) / f"creep_{sample_type}_{POLISH}_{load_mpa}" / "strain.csv"
    if not path.exists():
        warnings.warn(f"Missing strain table: {path}")
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "time_h" not in frame:
        warnings.warn(f"No time_h column in {path}.")
        return pd.DataFrame()
    for column in frame:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _interpolate_series(time: np.ndarray, values: np.ndarray, query: float) -> float:
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(time) & np.isfinite(values)
    if not np.any(valid):
        return np.nan
    time = time[valid]
    values = values[valid]
    order = np.argsort(time)
    return float(np.interp(query, time[order], values[order]))


def _experimental_strain_at_time(
    strain_df: pd.DataFrame,
    sample_id: str,
    time_h: float,
) -> tuple[float, str]:
    if strain_df.empty:
        return np.nan, "none"
    candidates = [column for column in strain_df.columns if column != "time_h"]
    lookup = {str(column).strip().lower(): column for column in candidates}
    match = lookup.get(str(sample_id).strip().lower())
    if match is not None:
        value = _interpolate_series(
            strain_df["time_h"].to_numpy(dtype=float),
            strain_df[match].to_numpy(dtype=float),
            time_h,
        )
        if np.isfinite(value):
            return value, "sample"
    if candidates:
        mean_strain = np.nanmean(strain_df[candidates].to_numpy(dtype=float), axis=1)
        value = _interpolate_series(
            strain_df["time_h"].to_numpy(dtype=float),
            mean_strain,
            time_h,
        )
        if np.isfinite(value):
            return value, "experiment_mean"
    return np.nan, "none"


def build_experimental_cache() -> tuple[pd.DataFrame, Mapping[str, np.ndarray]]:
    """Read each experimental height CSV once and cache its raw full map."""
    rows: list[dict[str, object]] = []
    with StreamingNpzWriter(EXP_HEIGHTS_CACHE) as writer:
        for load_mpa, sample_type in EXPERIMENTS:
            root = (
                Path(DATA_DIR)
                / f"creep_{sample_type}_{POLISH}_{load_mpa}"
                / "profilometry"
                / MAGNIFICATION
            )
            if not root.exists():
                warnings.warn(f"Missing profilometry directory: {root}")
                continue
            strain_df = _read_strain_table(load_mpa, sample_type)
            for path in sorted(root.rglob("*.csv"), key=_numeric_file_sort_key):
                relative = path.relative_to(root)
                if len(relative.parts) < 2:
                    continue
                if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
                    continue
                try:
                    time_h = float(path.stem)
                except ValueError:
                    continue
                sample_id = str(relative.parts[0]).strip()
                try:
                    raw = fill_missing_nearest(raw_height_csv(path))
                except Exception as exc:
                    warnings.warn(f"Skipping experimental map {path}: {exc}")
                    continue
                height_key = _array_key(
                    "exp", load_mpa, sample_type, sample_id, time_h, path
                )
                writer.write(height_key, raw)
                strain, strain_source = _experimental_strain_at_time(
                    strain_df,
                    sample_id,
                    time_h,
                )
                rows.append(
                    {
                        "cache_schema_version": CACHE_SCHEMA_VERSION,
                        "cache_fingerprint": _cache_fingerprint(),
                        "source": "exp",
                        "load_mpa": int(load_mpa),
                        "sample_type": sample_type,
                        "sample_id": sample_id,
                        "sample": sample_id,
                        "time_h": float(time_h),
                        "bulk_z_strain": float(strain),
                        "bulk_z_strain_percent": 100.0 * float(strain),
                        "strain_source": strain_source,
                        "height_key": height_key,
                        "height_path": str(path),
                        "spacing_um": EXP_NATIVE_SPACING_UM,
                        "shape_0": int(raw.shape[0]),
                        "shape_1": int(raw.shape[1]),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No experimental maps were cached.")
    frame = frame.sort_values(
        ["load_mpa", "sample_type", "sample_id", "time_h"]
    ).reset_index(drop=True)
    groups = frame.groupby(["load_mpa", "sample_type", "sample_id"])["time_h"]
    frame["is_initial"] = frame["time_h"].eq(groups.transform("min"))
    frame["is_endpoint"] = frame["time_h"].eq(groups.transform("max"))
    _write_dataframe_cache(frame, EXP_DF_CACHE)
    _write_cache_manifest()
    return frame, np.load(EXP_HEIGHTS_CACHE, allow_pickle=False)


def build_simulation_cache() -> tuple[pd.DataFrame, Mapping[str, np.ndarray]]:
    """Read each required SimResults run once and cache every face/time field."""
    rows: list[dict[str, object]] = []
    with StreamingNpzWriter(SIM_HEIGHTS_CACHE) as writer:
        for run in MICRO_RUNS:
            micro_id = str(run["micro_id"])
            for load_mpa, sample_type in EXPERIMENTS:
                run_dir = Path(run["sim_root"]) / f"{load_mpa}mpa_{sample_type}"
                try:
                    result = SimResults.load(
                        run_dir,
                        microstructure=Path(run["microstructure"]),
                    )
                except Exception as exc:
                    warnings.warn(f"Could not load simulation {run_dir}: {exc}")
                    continue
                height = np.asarray(result.height, dtype=float)
                if height.ndim != 4:
                    warnings.warn(
                        f"Expected (face,time,z,width) heights in {run_dir}; "
                        f"found {height.shape}."
                    )
                    continue
                vtk_time = np.asarray(result.vtk_time, dtype=float)
                sim_time = np.asarray(result.sim_time, dtype=float)
                bulk_strain = np.asarray(getattr(result, STRAIN_COLUMN), dtype=float)
                valid = np.isfinite(sim_time) & np.isfinite(bulk_strain)
                if not np.any(valid):
                    warnings.warn(f"No finite strain history in {run_dir}.")
                    continue
                order = np.argsort(sim_time[valid])
                strain_time = sim_time[valid][order]
                strain_values = bulk_strain[valid][order]
                n_times = min(height.shape[1], vtk_time.size)
                face_names = getattr(result, "samples", None)
                for face_index in range(height.shape[0]):
                    face_name = (
                        str(face_names[face_index])
                        if face_names is not None and face_index < len(face_names)
                        else f"face_{face_index}"
                    )
                    for time_index in range(n_times):
                        time_s = float(vtk_time[time_index])
                        strain = _interpolate_series(
                            strain_time,
                            strain_values,
                            time_s,
                        )
                        array = np.asarray(height[face_index, time_index], dtype=np.float64)
                        height_key = _array_key(
                            "sim",
                            micro_id,
                            load_mpa,
                            sample_type,
                            face_index,
                            time_index,
                        )
                        writer.write(height_key, array)
                        rows.append(
                            {
                                "cache_schema_version": CACHE_SCHEMA_VERSION,
                                "cache_fingerprint": _cache_fingerprint(),
                                "source": "sim",
                                "micro_id": micro_id,
                                "load_mpa": int(load_mpa),
                                "sample_type": sample_type,
                                "sample_id": micro_id,
                                "face_index": int(face_index),
                                "face_name": face_name,
                                "time_index": int(time_index),
                                "time_s": time_s,
                                "time_h": time_s / 3600.0,
                                "bulk_z_strain": float(strain),
                                "bulk_z_strain_percent": 100.0 * float(strain),
                                "height_key": height_key,
                                "spacing_um": SIM_SPACING_UM,
                                "shape_0": int(array.shape[0]),
                                "shape_1": int(array.shape[1]),
                                "run_dir": str(run_dir),
                                "microstructure": str(run["microstructure"]),
                            }
                        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No simulation maps were cached.")
    frame = frame.sort_values(
        ["micro_id", "load_mpa", "sample_type", "face_index", "time_index"]
    ).reset_index(drop=True)
    groups = frame.groupby(
        ["micro_id", "load_mpa", "sample_type", "face_index"]
    )["time_index"]
    frame["is_initial"] = frame["time_index"].eq(groups.transform("min"))
    frame["is_endpoint"] = frame["time_index"].eq(groups.transform("max"))
    _write_dataframe_cache(frame, SIM_DF_CACHE)
    _write_cache_manifest()
    return frame, np.load(SIM_HEIGHTS_CACHE, allow_pickle=False)


def load_experimental_cache(
    *, rebuild: bool = False
) -> tuple[pd.DataFrame, Mapping[str, np.ndarray]]:
    frame = None if rebuild else _read_current_dataframe_cache(EXP_DF_CACHE)
    if frame is None or not EXP_HEIGHTS_CACHE.exists():
        return build_experimental_cache()
    return frame, np.load(EXP_HEIGHTS_CACHE, allow_pickle=False)


def load_simulation_cache(
    *, rebuild: bool = False
) -> tuple[pd.DataFrame, Mapping[str, np.ndarray]]:
    frame = None if rebuild else _read_current_dataframe_cache(SIM_DF_CACHE)
    if frame is None or not SIM_HEIGHTS_CACHE.exists():
        return build_simulation_cache()
    return frame, np.load(SIM_HEIGHTS_CACHE, allow_pickle=False)


exp_df: pd.DataFrame | None = None
sim_df: pd.DataFrame | None = None
exp_heights: Mapping[str, np.ndarray] | None = None
sim_heights: Mapping[str, np.ndarray] | None = None


def initialize_caches(
    *, rebuild: bool = False
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    Mapping[str, np.ndarray],
    Mapping[str, np.ndarray],
]:
    """Populate the four requested cache objects once per Python process."""
    global exp_df, sim_df, exp_heights, sim_heights
    if rebuild or any(
        value is None for value in (exp_df, sim_df, exp_heights, sim_heights)
    ):
        for heights in (exp_heights, sim_heights):
            close = getattr(heights, "close", None)
            if callable(close):
                close()
        manifest_current = not rebuild and _cache_manifest_is_current()
        cached_exp_frame = (
            _read_current_dataframe_cache(
                EXP_DF_CACHE,
                manifest_current=manifest_current,
            )
            if not rebuild
            else None
        )
        cached_sim_frame = (
            _read_current_dataframe_cache(
                SIM_DF_CACHE,
                manifest_current=manifest_current,
            )
            if not rebuild
            else None
        )
        all_disk_caches_current = (
            cached_exp_frame is not None
            and cached_sim_frame is not None
            and EXP_HEIGHTS_CACHE.exists()
            and SIM_HEIGHTS_CACHE.exists()
        )
        if all_disk_caches_current:
            exp_df = cached_exp_frame
            sim_df = cached_sim_frame
            exp_heights = np.load(EXP_HEIGHTS_CACHE, allow_pickle=False)
            sim_heights = np.load(SIM_HEIGHTS_CACHE, allow_pickle=False)
        else:
            exp_df, exp_heights = build_experimental_cache()
            sim_df, sim_heights = build_simulation_cache()
    return exp_df, sim_df, exp_heights, sim_heights


# %% Shared crop-first height, PSD, and ACF calculations
@lru_cache(maxsize=256)
def _detrend_geometry(
    shape: tuple[int, int],
    spacing_0_um: float,
    spacing_1_um: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    row, column = np.indices(shape, dtype=np.float64)
    x = row.ravel() * float(spacing_0_um)
    y = column.ravel() * float(spacing_1_um)
    terms = [np.ones(row.size), x, y]
    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("Detrend order must be 1 or 2.")
    design = np.column_stack(terms)
    return design, np.linalg.pinv(design)


def detrend_surface(
    values: np.ndarray,
    spacing_um: float | tuple[float, float],
    *,
    order: int = 1,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("detrend_surface requires finite values.")
    if np.isscalar(spacing_um):
        d0 = d1 = float(spacing_um)
    else:
        d0, d1 = (float(value) for value in spacing_um)
    design, inverse = _detrend_geometry(values.shape, d0, d1, int(order))
    trend = (design @ (inverse @ values.ravel())).reshape(values.shape)
    return values - trend


def experimental_crop(raw_height_um: np.ndarray) -> np.ndarray:
    """The sole crop used by every experimental height analysis."""
    raw_height_um = np.asarray(raw_height_um, dtype=float)
    cropped = raw_height_um[EXP_ANALYSIS_CROP]
    if cropped.ndim != 2 or min(cropped.shape) < 3:
        raise ValueError(f"Invalid experimental crop: {cropped.shape}.")
    return cropped


def _cropped_shape(shape_0: int, shape_1: int) -> tuple[int, int]:
    return tuple(
        len(range(*selection.indices(int(size))))
        for selection, size in zip(EXP_ANALYSIS_CROP, (shape_0, shape_1))
    )


def eligible_experimental_metadata(
    metadata: pd.DataFrame,
    *,
    require_zero_time: bool,
    require_finite_strain: bool = True,
) -> pd.DataFrame:
    """Restore the source blocks' modal-shape and repeated-scan cohorts."""
    selected = metadata.copy()
    selected["_crop_shape"] = [
        _cropped_shape(shape_0, shape_1)
        for shape_0, shape_1 in zip(selected["shape_0"], selected["shape_1"])
    ]
    keep_indices: list[int] = []
    for _case, group in selected.groupby(["load_mpa", "sample_type"], sort=False):
        counts = group["_crop_shape"].value_counts(sort=False)
        if counts.empty:
            continue
        reference_shape = counts.idxmax()
        shape_matches = group["_crop_shape"].map(
            lambda value: value == reference_shape
        )
        keep_indices.extend(group.index[shape_matches])
    selected = selected.loc[keep_indices].drop(columns="_crop_shape")

    specimen_keys = ["load_mpa", "sample_type", "sample_id"]
    groups = selected.groupby(specimen_keys)["time_h"]
    selected = selected[groups.transform("nunique") >= 2].copy()
    if require_zero_time:
        groups = selected.groupby(specimen_keys)["time_h"]
        selected = selected[
            groups.transform("min") <= INITIAL_TIME_TOL_H
        ].copy()
    if require_finite_strain:
        selected = selected[
            np.isfinite(selected["bulk_z_strain_percent"])
        ].copy()
    return selected.sort_values(specimen_keys + ["time_h"]).reset_index(drop=True)


def prepare_experimental_crop(raw_height_um: np.ndarray) -> np.ndarray:
    """Crop first, then plane-level and mean-remove the complete crop."""
    height = detrend_surface(
        experimental_crop(raw_height_um),
        EXP_NATIVE_SPACING_UM,
        order=1,
    )
    return height - np.mean(height)


def surface_sa(height_um: np.ndarray) -> float:
    height = np.asarray(height_um, dtype=float)
    centered = height - np.nanmean(height)
    return float(np.nanmean(np.abs(centered)))


def hann2d_rms(shape: tuple[int, int]) -> np.ndarray:
    window = np.hanning(shape[0])[:, None] * np.hanning(shape[1])[None, :]
    rms = float(np.sqrt(np.mean(window**2)))
    return window / rms if rms > 0 else window


def periodogram_2d(
    height_um: np.ndarray,
    spacing_um: float | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Shared RMS-Hann physical 2-D PSD used by every spectral figure."""
    height = np.asarray(height_um, dtype=float)
    if np.isscalar(spacing_um):
        d0 = d1 = float(spacing_um)
    else:
        d0, d1 = (float(value) for value in spacing_um)
    centered = height - np.mean(height)
    tapered = centered * hann2d_rms(height.shape)
    transformed = np.fft.fft2(tapered)
    n0, n1 = height.shape
    psd = d0 * d1 / (n0 * n1) * np.abs(transformed) ** 2
    f0 = np.fft.fftfreq(n0, d=d0)
    f1 = np.fft.fftfreq(n1, d=d1)
    return f0, f1, psd, 1.0 / (n0 * d0), 1.0 / (n1 * d1)


def radial_psd_wavelength_binned(
    f0: np.ndarray,
    f1: np.ndarray,
    psd2d: np.ndarray,
    wavelength_edges_um: np.ndarray,
    *,
    min_modes: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius, wavelength = radial_frequency_wavelength_grid(f0, f1)
    centers = np.sqrt(wavelength_edges_um[:-1] * wavelength_edges_um[1:])
    radial = np.full(centers.size, np.nan)
    modes = np.zeros(centers.size, dtype=int)
    for index, (low, high) in enumerate(
        zip(wavelength_edges_um[:-1], wavelength_edges_um[1:])
    ):
        selected = (
            np.isfinite(wavelength)
            & np.isfinite(psd2d)
            & (radius > 0)
            & (wavelength >= low)
            & (wavelength < high)
        )
        modes[index] = int(np.count_nonzero(selected))
        if modes[index] >= min_modes:
            radial[index] = float(np.mean(psd2d[selected]))
    return centers, radial, modes


def radial_frequency_wavelength_grid(
    f0: np.ndarray,
    f1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the common radial-frequency and wavelength grids."""
    F1, F0 = np.meshgrid(f1, f0)
    radius = np.sqrt(F0**2 + F1**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / radius
    return radius, wavelength


def wavelength_bin_mode_counts(
    shape: tuple[int, int],
    spacing_um: float | tuple[float, float],
    wavelength_edges_um: np.ndarray,
) -> np.ndarray:
    """Count 2-D Fourier modes in each wavelength bin."""
    if np.isscalar(spacing_um):
        d0 = d1 = float(spacing_um)
    else:
        d0, d1 = (float(value) for value in spacing_um)
    f0 = np.fft.fftfreq(int(shape[0]), d=d0)
    f1 = np.fft.fftfreq(int(shape[1]), d=d1)
    radius, wavelength = radial_frequency_wavelength_grid(f0, f1)
    counts = np.zeros(len(wavelength_edges_um) - 1, dtype=int)
    for index, (low, high) in enumerate(
        zip(wavelength_edges_um[:-1], wavelength_edges_um[1:])
    ):
        counts[index] = int(
            np.count_nonzero(
                np.isfinite(wavelength)
                & (radius > 0)
                & (wavelength >= low)
                & (wavelength < high)
            )
        )
    return counts


def endpoint_wavelength_edges() -> np.ndarray:
    """Build contiguous, mode-supported log bins for the matched field.

    ``ENDPOINT_WAVELENGTH_BINS`` is the requested maximum.  If configurable
    wavelength limits make that grid too fine, the bin count is reduced until
    every bin contains at least ``ENDPOINT_MIN_MODES`` reference-field modes.
    """
    if not 0 < ENDPOINT_WAVELENGTH_MIN_UM < ENDPOINT_WAVELENGTH_MAX_UM:
        raise ValueError("Endpoint wavelength limits must satisfy 0 < min < max.")
    if ENDPOINT_WAVELENGTH_BINS < 1 or ENDPOINT_MIN_MODES < 1:
        raise ValueError("Endpoint bin and mode counts must be positive integers.")
    reference_shape = tuple(
        int(round(length / MATCHED_TARGET_SPACING_UM))
        for length in MATCHED_FIELD_SHAPE_UM
    )
    for n_bins in range(int(ENDPOINT_WAVELENGTH_BINS), 0, -1):
        edges = np.geomspace(
            ENDPOINT_WAVELENGTH_MIN_UM,
            ENDPOINT_WAVELENGTH_MAX_UM,
            n_bins + 1,
        )
        counts = wavelength_bin_mode_counts(
            reference_shape,
            MATCHED_TARGET_SPACING_UM,
            edges,
        )
        if counts.size and np.all(counts >= ENDPOINT_MIN_MODES):
            if n_bins != ENDPOINT_WAVELENGTH_BINS:
                warnings.warn(
                    "Reduced endpoint wavelength bins from "
                    f"{ENDPOINT_WAVELENGTH_BINS} to {n_bins} so every bin has "
                    f"at least {ENDPOINT_MIN_MODES} Fourier modes.",
                    stacklevel=2,
                )
            return edges
    raise ValueError(
        "The configured endpoint wavelength range contains fewer than "
        f"{ENDPOINT_MIN_MODES} supported Fourier modes."
    )


def band_powers_from_periodogram(
    f0: np.ndarray,
    f1: np.ndarray,
    psd2d: np.ndarray,
    df0: float,
    df1: float,
    wavelength_edges_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    radius, wavelength = radial_frequency_wavelength_grid(f0, f1)
    power = np.full(wavelength_edges_um.size - 1, np.nan)
    modes = np.zeros(wavelength_edges_um.size - 1, dtype=int)
    for index, (low, high) in enumerate(
        zip(wavelength_edges_um[:-1], wavelength_edges_um[1:])
    ):
        selected = (
            np.isfinite(wavelength)
            & np.isfinite(psd2d)
            & (radius > 0)
            & (wavelength >= low)
            & (wavelength < high)
        )
        modes[index] = int(np.count_nonzero(selected))
        power[index] = float(np.sum(psd2d[selected]) * df0 * df1)
    return power, modes


def overlap_corrected_acf_2d(height_um: np.ndarray) -> np.ndarray:
    height = np.asarray(height_um, dtype=float)
    finite = np.isfinite(height)
    if np.count_nonzero(finite) < 10:
        raise ValueError("Too few finite pixels for ACF.")
    centered = np.zeros_like(height)
    centered[finite] = height[finite] - np.mean(height[finite])
    mask = finite.astype(float)
    numerator = fftconvolve(centered, centered[::-1, ::-1], mode="full")
    overlap = np.rint(fftconvolve(mask, mask[::-1, ::-1], mode="full"))
    covariance = np.full_like(numerator, np.nan)
    valid = overlap >= 1
    covariance[valid] = numerator[valid] / overlap[valid]
    center = tuple(np.asarray(covariance.shape) // 2)
    zero_lag = covariance[center]
    if not np.isfinite(zero_lag) or zero_lag <= 0:
        raise ValueError("ACF zero-lag covariance is not positive.")
    return covariance / zero_lag


def radial_acf_overlap_corrected(
    height_um: np.ndarray,
    spacing_um: float | tuple[float, float],
    *,
    bin_width_um: float,
    max_lag_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.isscalar(spacing_um):
        d0 = d1 = float(spacing_um)
    else:
        d0, d1 = (float(value) for value in spacing_um)
    rho = overlap_corrected_acf_2d(height_um)
    lag0 = np.arange(-(height_um.shape[0] - 1), height_um.shape[0]) * d0
    lag1 = np.arange(-(height_um.shape[1] - 1), height_um.shape[1]) * d1
    L1, L0 = np.meshgrid(lag1, lag0)
    radius = np.sqrt(L0**2 + L1**2)
    edges = np.arange(0.0, max_lag_um + bin_width_um, bin_width_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radial = np.full(centers.size, np.nan)
    counts = np.zeros(centers.size, dtype=int)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (
            np.isfinite(radius)
            & np.isfinite(rho)
            & (radius >= low)
            & (radius < high)
        )
        counts[index] = int(np.count_nonzero(selected))
        if counts[index]:
            radial[index] = float(np.mean(rho[selected]))
    keep = centers < max_lag_um
    return centers[keep], radial[keep], counts[keep]


def _finite_column_median(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape[1], np.nan)
    for column in range(values.shape[1]):
        finite = values[:, column]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            result[column] = float(np.median(finite))
    return result


# %% Shared derived tables for Figures 1--5
def build_delta_sa_tables(
    exp_metadata: pd.DataFrame,
    sim_metadata: pd.DataFrame,
    experimental_heights: Mapping[str, np.ndarray],
    simulation_heights: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the original crop Sa and face-mean simulation Sa exactly once."""
    exp_rows = []
    eligible_exp = eligible_experimental_metadata(
        exp_metadata,
        require_zero_time=False,
        require_finite_strain=False,
    )
    for record in eligible_exp.itertuples(index=False):
        height = prepare_experimental_crop(
            np.asarray(experimental_heights[record.height_key], dtype=float)
        )
        exp_rows.append({**record._asdict(), "sa_um": surface_sa(height)})
    exp_sa = pd.DataFrame(exp_rows).sort_values(
        ["load_mpa", "sample_type", "sample_id", "time_h"]
    )
    exp_group = exp_sa.groupby(["load_mpa", "sample_type", "sample_id"])
    exp_sa["sa_initial_um"] = exp_group["sa_um"].transform("first")
    exp_sa["delta_sa_um"] = exp_sa["sa_um"] - exp_sa["sa_initial_um"]
    exp_sa = exp_sa[
        np.isfinite(exp_sa["bulk_z_strain_percent"])
    ].copy()

    sim_face_rows = []
    for record in sim_metadata.itertuples(index=False):
        height = np.asarray(simulation_heights[record.height_key], dtype=float)
        sim_face_rows.append({**record._asdict(), "sa_face_um": surface_sa(height)})
    sim_face = pd.DataFrame(sim_face_rows)
    sim_sa = (
        sim_face.groupby(
            [
                "micro_id",
                "load_mpa",
                "sample_type",
                "time_index",
                "time_s",
                "time_h",
            ],
            as_index=False,
        )
        .agg(
            bulk_z_strain=("bulk_z_strain", "mean"),
            bulk_z_strain_percent=("bulk_z_strain_percent", "mean"),
            sa_mean_um=("sa_face_um", "mean"),
            sa_std_um=("sa_face_um", "std"),
            n_faces=("face_index", "nunique"),
        )
        .sort_values(["micro_id", "load_mpa", "sample_type", "time_index"])
    )
    sim_group = sim_sa.groupby(["micro_id", "load_mpa", "sample_type"])
    sim_sa["sa_initial_um"] = sim_group["sa_mean_um"].transform("first")
    sim_sa["delta_sa_um"] = sim_sa["sa_mean_um"] - sim_sa["sa_initial_um"]
    sim_sa = sim_sa[
        np.isfinite(sim_sa["bulk_z_strain_percent"])
    ].copy()
    return exp_sa.reset_index(drop=True), sim_sa.reset_index(drop=True)


def _bracket_edges() -> np.ndarray:
    edges = [float(BRACKET_WAVELENGTH_MIN_UM)]
    while edges[-1] < BRACKET_WAVELENGTH_MAX_UM:
        edges.append(min(edges[-1] + BRACKET_WIDTH_UM, BRACKET_WAVELENGTH_MAX_UM))
    return np.asarray(edges, dtype=float)


def build_experimental_spatial_table(
    exp_metadata: pd.DataFrame,
    experimental_heights: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Compute each full-crop periodogram and ACF once for Figures 3--5."""
    radial_edges = np.logspace(
        np.log10(RADIAL_WAVELENGTH_MIN_UM),
        np.log10(RADIAL_WAVELENGTH_MAX_UM),
        RADIAL_WAVELENGTH_BINS + 1,
    )
    bracket_edges = _bracket_edges()
    rows = []
    selected = eligible_experimental_metadata(
        exp_metadata,
        require_zero_time=True,
    )
    for record in selected.itertuples(index=False):
        height = prepare_experimental_crop(
            np.asarray(experimental_heights[record.height_key], dtype=float)
        )
        f0, f1, psd2d, df0, df1 = periodogram_2d(height, EXP_NATIVE_SPACING_UM)
        wavelength, radial_psd_values, radial_modes = radial_psd_wavelength_binned(
            f0,
            f1,
            psd2d,
            radial_edges,
            min_modes=RADIAL_MIN_MODES,
        )
        bracket_power, bracket_modes = band_powers_from_periodogram(
            f0,
            f1,
            psd2d,
            df0,
            df1,
            bracket_edges,
        )
        acf_lag, acf_values, acf_counts = radial_acf_overlap_corrected(
            height,
            EXP_NATIVE_SPACING_UM,
            bin_width_um=ACF_CHANGE_BIN_WIDTH_UM,
            max_lag_um=ACF_CHANGE_MAX_LAG_UM,
        )
        rows.append(
            {
                **record._asdict(),
                "radial_wavelength_um": wavelength,
                "radial_psd_um4": radial_psd_values,
                "radial_modes": radial_modes,
                "bracket_edges_um": bracket_edges,
                "bracket_power_um2": bracket_power,
                "bracket_modes": bracket_modes,
                "acf_lag_um": acf_lag,
                "acf": acf_values,
                "acf_counts": acf_counts,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["sample_type", "load_mpa", "sample_id", "time_h"]
    ).reset_index(drop=True)


def derive_strain_group_definitions(
    frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Derive three disjoint, load-pair strain levels for Figures 3--5.

    Membership is defined only by load, so an observation can never appear in
    two groups even when the observed strain histories overlap numerically.
    The displayed level is the mean strain of the same observations used for
    the corresponding grouped PSD and ACF calculations.
    """
    required = {"load_mpa", "bulk_z_strain_percent"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Cannot derive strain groups without columns: "
            + ", ".join(sorted(missing_columns))
        )

    flat_loads = [load for pair in STRAIN_LOAD_GROUPS for load in pair]
    if len(flat_loads) != len(set(flat_loads)):
        raise ValueError("STRAIN_LOAD_GROUPS must contain disjoint load pairs.")

    positive = frame[
        frame["bulk_z_strain_percent"] > INITIAL_STRAIN_TOL_PERCENT
    ].copy()
    if positive.empty:
        raise ValueError("No positive-strain observations are available.")

    palette = strain_group_colors(len(STRAIN_LOAD_GROUPS))
    definitions: list[dict[str, object]] = []
    for index, load_pair in enumerate(STRAIN_LOAD_GROUPS, start=1):
        pair = positive[positive["load_mpa"].isin(load_pair)]
        load_mean_strains: list[float] = []
        for load in load_pair:
            values = pair.loc[
                pair["load_mpa"] == load, "bulk_z_strain_percent"
            ].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(
                    f"No positive-strain observations are available for {load} MPa."
                )
            load_mean_strains.append(float(np.mean(values)))

        observed = pair["bulk_z_strain_percent"].to_numpy(dtype=float)
        observed = observed[np.isfinite(observed)]
        definitions.append(
            {
                "strain_group": f"group_{index}",
                "loads_mpa": tuple(int(load) for load in load_pair),
                "strain_level_percent": float(np.mean(observed)),
                "load_mean_strains_percent": tuple(load_mean_strains),
                "observed_strain_min_percent": float(np.min(observed)),
                "observed_strain_max_percent": float(np.max(observed)),
                "n_observations": int(len(pair)),
                "color": palette[index - 1],
            }
        )

    levels = np.asarray(
        [definition["strain_level_percent"] for definition in definitions],
        dtype=float,
    )
    if not np.all(np.diff(levels) > 0.0):
        details = ", ".join(
            f"{definition['loads_mpa']}: "
            f"{float(definition['strain_level_percent']):.6g}%"
            for definition in definitions
        )
        raise ValueError(
            "The requested load pairs do not produce strictly increasing strain "
            f"levels ({details})."
        )

    precision = 2
    while (
        len({f"{level:.{precision}f}" for level in levels}) != len(levels)
        and precision < 6
    ):
        precision += 1
    for definition in definitions:
        level = float(definition["strain_level_percent"])
        definition["display_label"] = (
            rf"$\overline{{\varepsilon}}_{{zz}}={level:.{precision}f}\%$"
        )
    return definitions


def assign_positive_strain_groups(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Assign positive-strain observations to one exclusive load-pair group."""
    selected = frame[
        frame["bulk_z_strain_percent"] > INITIAL_STRAIN_TOL_PERCENT
    ].copy()
    if selected.empty:
        raise ValueError("No positive-strain observations are available.")

    load_to_group: dict[int, Mapping[str, object]] = {}
    for definition in definitions:
        for load in definition["loads_mpa"]:
            load = int(load)
            if load in load_to_group:
                raise ValueError(f"Load {load} MPa belongs to more than one group.")
            load_to_group[load] = definition

    numeric_loads = pd.to_numeric(selected["load_mpa"], errors="coerce")
    unmatched = sorted(
        {
            int(load)
            for load in numeric_loads.dropna().unique()
            if int(load) not in load_to_group
        }
    )
    if unmatched:
        raise ValueError(
            "Positive-strain observations have no load-pair group: "
            + ", ".join(f"{load} MPa" for load in unmatched)
        )

    selected["strain_group"] = numeric_loads.map(
        lambda load: load_to_group[int(load)]["strain_group"]
        if np.isfinite(load)
        else pd.NA
    )
    if selected["strain_group"].isna().any():
        raise ValueError("Some positive-strain observations could not be grouped.")

    categories = [str(definition["strain_group"]) for definition in definitions]
    selected["strain_group"] = pd.Categorical(
        selected["strain_group"], categories=categories, ordered=True
    )
    selected["strain_level_percent"] = numeric_loads.map(
        lambda load: float(load_to_group[int(load)]["strain_level_percent"])
    )
    selected["strain_group_label"] = numeric_loads.map(
        lambda load: str(load_to_group[int(load)]["display_label"])
    )
    selected["strain_group_loads_mpa"] = numeric_loads.map(
        lambda load: "/".join(
            str(value) for value in load_to_group[int(load)]["loads_mpa"]
        )
    )
    return selected


def error_from_values(values: np.ndarray, mode: str = "sem") -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    std = float(np.std(values, ddof=1))
    if mode == "std":
        return std
    if mode == "sem":
        return std / np.sqrt(values.size)
    if mode == "ci95":
        return 1.96 * std / np.sqrt(values.size)
    raise ValueError("mode must be 'std', 'sem', or 'ci95'.")


# %% Figure 1: all experimental/simulation Delta Sa scatter and linear fits
def pooled_delta_sa_fit(
    exp_sa: pd.DataFrame,
    sim_sa: pd.DataFrame,
) -> dict[str, float | np.ndarray]:
    exp_valid = exp_sa[
        np.isfinite(exp_sa["bulk_z_strain_percent"])
        & np.isfinite(exp_sa["delta_sa_um"])
    ]
    sim_valid = sim_sa[
        np.isfinite(sim_sa["bulk_z_strain_percent"])
        & np.isfinite(sim_sa["delta_sa_um"])
    ]
    x_exp = exp_valid["bulk_z_strain_percent"].to_numpy(dtype=float)
    y_exp = exp_valid["delta_sa_um"].to_numpy(dtype=float)
    x_sim = sim_valid["bulk_z_strain_percent"].to_numpy(dtype=float)
    y_sim = sim_valid["delta_sa_um"].to_numpy(dtype=float)
    m_exp = float(np.sum(x_exp * y_exp) / np.sum(x_exp**2))
    m_sim = float(np.sum(x_sim * y_sim) / np.sum(x_sim**2))
    scale = m_exp / m_sim
    return {
        "x_exp": x_exp,
        "y_exp": y_exp,
        "x_sim": x_sim,
        "y_sim": y_sim,
        "m_exp": m_exp,
        "m_sim": m_sim,
        "scale": scale,
    }


def plot_all_delta_sa_linear_fit(
    exp_sa: pd.DataFrame,
    sim_sa: pd.DataFrame,
    *,
    save: bool = True,
) -> tuple[plt.Figure, dict[str, float | np.ndarray]]:
    """Port of the original pooled scatter/fit presentation at figures.py:8401."""
    fit = pooled_delta_sa_fit(exp_sa, sim_sa)
    fig, ax = plt.subplots(dpi=150)
    ax.axhline(0, color="0.5", lw=0.8)

    # Match Figure 2 source markers: simulation squares, experiment circles.
    for load, _sample_type in EXPERIMENTS:
        sim_group = sim_sa[sim_sa["load_mpa"] == load]
        exp_group = exp_sa[exp_sa["load_mpa"] == load]
        ax.scatter(
            sim_group["bulk_z_strain_percent"],
            sim_group["delta_sa_um"] * float(fit["scale"]),
            s=55,
            alpha=0.5,
            marker="s",
            color=load_color(load),
            edgecolor="k",
            linewidth=0.5,
        )
        ax.scatter(
            exp_group["bulk_z_strain_percent"],
            exp_group["delta_sa_um"],
            s=16,
            marker="o",
            color=load_color(load),
        )

    x_plot = np.linspace(0.0, 19.0, 100)
    ax.plot(
        x_plot,
        float(fit["m_sim"]) * x_plot * float(fit["scale"]),
        color="tab:blue",
    )
    ax.plot(
        x_plot,
        float(fit["m_exp"]) * x_plot,
        color="tab:red",
        linestyle="--",
    )
    ax.set_xlabel(r"$\varepsilon_{zz}$ [%]")
    ax.set_ylabel(r"$\Delta S_a$ [$\mu$m]")
    ax.grid(True, alpha=0.25)
    source_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            label="Exp.",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker="s",
            markeredgecolor="black",
            markerfacecolor="black",
            markersize=6,
            alpha=0.5,
            linestyle="None",
            label="Sim.",
        ),
    ]
    ax.legend(
        handles=source_handles,
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    fig.set_size_inches(fig.get_size_inches() * 1.35, forward=True)
    if save:
        fig.savefig(OUTPUT_DIR / "01_all_delta_sa_linear_fit.png", bbox_inches="tight")
    return fig, fit


# %% Figure 2: original six-panel experimental/simulation Delta Sa figure
def plot_six_panel_delta_sa_vs_strain(
    exp_sa: pd.DataFrame,
    sim_sa: pd.DataFrame,
    *,
    scale: float,
    save: bool = True,
) -> plt.Figure:
    """Port of the final A--F publication block at figures.py:9024."""
    panel_labels = (("A", "B", "C"), ("D", "E", "F"))
    fig, axes = plt.subplots(2, 3, dpi=150, sharex=False, sharey=False)
    for row in range(2):
        for column in range(3):
            axis = axes[row, column]
            load, sample_type = PLOT_GRID[row][column]
            color = load_color(load)
            exp_group = exp_sa[
                (exp_sa["load_mpa"] == load)
                & (exp_sa["sample_type"] == sample_type)
            ]
            sim_group = sim_sa[
                (sim_sa["load_mpa"] == load)
                & (sim_sa["sample_type"] == sample_type)
            ]

            if not exp_group.empty:
                exp_summary = (
                    exp_group.groupby("time_h", as_index=False)
                    .agg(
                        mean_delta_sa_um=("delta_sa_um", "mean"),
                        std_delta_sa_um=("delta_sa_um", "std"),
                        mean_bulk_z_strain_percent=(
                            "bulk_z_strain_percent",
                            "mean",
                        ),
                        std_bulk_z_strain_percent=(
                            "bulk_z_strain_percent",
                            "std",
                        ),
                        n=("delta_sa_um", "count"),
                    )
                    .sort_values("mean_bulk_z_strain_percent")
                )
                exp_summary["std_delta_sa_um"] = exp_summary[
                    "std_delta_sa_um"
                ].fillna(0.0)
                x = exp_summary["mean_bulk_z_strain_percent"].to_numpy(dtype=float)
                y = exp_summary["mean_delta_sa_um"].to_numpy(dtype=float)
                spread = exp_summary["std_delta_sa_um"].to_numpy(dtype=float)
                axis.plot(
                    x,
                    y,
                    color=color,
                    lw=2.0,
                    marker="o",
                    markersize=4,
                )
                axis.fill_between(
                    x,
                    y - spread,
                    y + spread,
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                )

            if not sim_group.empty:
                sim_summary = (
                    sim_group.groupby("time_s", as_index=False)
                    .agg(
                        mean_bulk_z_strain_percent=(
                            "bulk_z_strain_percent",
                            "mean",
                        ),
                        mean_delta_sa_um=("delta_sa_um", "mean"),
                    )
                    .sort_values("mean_bulk_z_strain_percent")
                )
                axis.scatter(
                    sim_summary["mean_bulk_z_strain_percent"],
                    sim_summary["mean_delta_sa_um"] * scale,
                    s=55,
                    alpha=0.5,
                    color=color,
                    marker="s",
                    edgecolor="k",
                    linewidth=0.5,
                )

            axis.axhline(0, color="0.5", lw=0.8)
            axis.set_title(f"{load} MPa")
            axis.grid(True, alpha=0.25)
            add_panel_label(axis, panel_labels[row][column])

    for axis in axes[1, :]:
        axis.set_xlabel(r"$\overline{\varepsilon}_{zz}$ [%]")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\overline{\Delta S_a}$ [$\mu$m]")

    exp_handle = Line2D(
        [0],
        [0],
        color="black",
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=5,
        label="Exp.",
    )
    sim_handle = Line2D(
        [0],
        [0],
        color="black",
        marker="s",
        linestyle="None",
        markeredgecolor="black",
        markerfacecolor="black",
        markersize=6,
        label="Sim.",
    )
    fig.legend(
        handles=[exp_handle, sim_handle],
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        frameon=False,
        fontsize=9,
    )
    fig.set_size_inches(fig.get_size_inches() * np.array([1.6, 1.35]), forward=True)
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
    if save:
        fig.savefig(
            OUTPUT_DIR / "02_six_panel_exp_sim_delta_sa_vs_strain.png",
            bbox_inches="tight",
        )
    return fig


# %% Figure 3: original 3--400 um radial PSD and PSD gain by strain group
def plot_radial_psd_and_gain_by_strain(
    spatial: pd.DataFrame,
    *,
    strain_group_definitions: Sequence[Mapping[str, object]] | None = None,
    show_individual_traces: bool = False,
    save: bool = True,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    initial = spatial[
        np.abs(spatial["bulk_z_strain_percent"]) <= INITIAL_STRAIN_TOL_PERCENT
    ].copy()
    if strain_group_definitions is None:
        strain_group_definitions = derive_strain_group_definitions(spatial)
    positive = assign_positive_strain_groups(spatial, strain_group_definitions)
    groups: list[tuple[str, str, pd.DataFrame, object]] = []
    if not initial.empty:
        groups.append(
            ("initial", r"initial, $\varepsilon_{zz} \approx 0$", initial, "black")
        )
    for definition in strain_group_definitions:
        group_key = str(definition["strain_group"])
        group = positive[positive["strain_group"] == group_key]
        if group.empty:
            raise ValueError(f"No observations were assigned to {group_key}.")
        groups.append(
            (
                group_key,
                str(definition["display_label"]),
                group,
                definition["color"],
            )
        )
    if not groups or groups[0][0] != "initial":
        raise ValueError("No initial PSD group was found.")

    results = []
    for name, label, group, color in groups:
        curves = np.vstack(group["radial_psd_um4"].to_numpy())
        results.append(
            {
                "name": name,
                "label": label,
                "color": color,
                "n": curves.shape[0],
                "wavelength_um": np.asarray(group.iloc[0]["radial_wavelength_um"]),
                "curves": curves,
                "mean_psd_um4": np.nanmean(curves, axis=0),
                "std_psd_um4": (
                    np.nanstd(curves, axis=0, ddof=1)
                    if curves.shape[0] > 1
                    else np.zeros(curves.shape[1])
                ),
                "metadata": group,
            }
        )
    baseline_psd = results[0]["mean_psd_um4"]

    summary_rows = []
    trace_rows = []
    for result in results:
        wavelength = result["wavelength_um"]
        mean_psd = result["mean_psd_um4"]
        std_psd = result["std_psd_um4"]
        mean_gain = mean_psd / baseline_psd
        for index in range(len(wavelength)):
            summary_rows.append(
                {
                    "group": result["name"],
                    "label": result["label"],
                    "n": result["n"],
                    "frequency_um_inv": 1.0 / wavelength[index],
                    "wavelength_um": wavelength[index],
                    "mean_psd_um4": mean_psd[index],
                    "std_psd_um4": std_psd[index],
                    "gain_vs_initial": mean_gain[index],
                }
            )
        metadata = result["metadata"].reset_index(drop=True)
        for curve_index, curve in enumerate(result["curves"]):
            record = metadata.iloc[curve_index]
            gain_curve = curve / baseline_psd
            for index in range(len(wavelength)):
                trace_rows.append(
                    {
                        "group": result["name"],
                        "label": result["label"],
                        "curve_index": curve_index,
                        "load_mpa": record["load_mpa"],
                        "sample_type": record["sample_type"],
                        "sample": record["sample_id"],
                        "time_h": record["time_h"],
                        "bulk_z_strain_percent": record[
                            "bulk_z_strain_percent"
                        ],
                        "frequency_um_inv": 1.0 / wavelength[index],
                        "wavelength_um": wavelength[index],
                        "psd_um4": curve[index],
                        "gain_vs_initial": gain_curve[index],
                    }
                )
    summary = pd.DataFrame(summary_rows)
    traces = pd.DataFrame(trace_rows)
    if save:
        summary.to_csv(
            OUTPUT_DIR / "height_radial_psd_group_means_by_strain_to_400um.csv",
            index=False,
        )
        traces.to_csv(
            OUTPUT_DIR / "height_radial_psd_individual_traces_by_strain_to_400um.csv",
            index=False,
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    ax_psd, ax_gain = axes
    for result in results:
        wavelength = result["wavelength_um"]
        color = result["color"]
        valid_base = (
            np.isfinite(wavelength)
            & (wavelength >= RADIAL_WAVELENGTH_MIN_UM)
            & (wavelength <= RADIAL_WAVELENGTH_MAX_UM)
        )
        if show_individual_traces:
            for curve in result["curves"]:
                valid = valid_base & np.isfinite(curve) & (curve > 0)
                if np.any(valid):
                    order = np.argsort(wavelength[valid])
                    ax_psd.plot(
                        wavelength[valid][order],
                        curve[valid][order],
                        color=color,
                        alpha=0.14 if result["name"] != "initial" else 0.10,
                        lw=0.7,
                    )
        mean_curve = result["mean_psd_um4"]
        valid = valid_base & np.isfinite(mean_curve) & (mean_curve > 0)
        if np.any(valid):
            order = np.argsort(wavelength[valid])
            ax_psd.plot(
                wavelength[valid][order],
                mean_curve[valid][order],
                color=color,
                lw=2.3,
                label=f"{result['label']}, n={result['n']}",
            )
    ax_psd.set_xlim(RADIAL_WAVELENGTH_MIN_UM, RADIAL_WAVELENGTH_MAX_UM)
    ax_psd.set_yscale("log")
    ax_psd.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax_psd.set_ylabel(r"$\overline{C_h}(\lambda)$ [$\mu$m$^4$]")
    ax_psd.grid(True, which="both", alpha=0.25)
    ax_psd.legend(loc="lower right", frameon=False, fontsize=6)

    for result in results:
        if result["name"] == "initial":
            continue
        wavelength = result["wavelength_um"]
        color = result["color"]
        valid_base = (
            np.isfinite(wavelength)
            & (wavelength >= RADIAL_WAVELENGTH_MIN_UM)
            & (wavelength <= RADIAL_WAVELENGTH_MAX_UM)
            & np.isfinite(baseline_psd)
            & (baseline_psd > 0)
        )
        if show_individual_traces:
            for curve in result["curves"]:
                gain = curve / baseline_psd
                valid = valid_base & np.isfinite(gain) & (gain > 0)
                if np.any(valid):
                    order = np.argsort(wavelength[valid])
                    ax_gain.plot(
                        wavelength[valid][order],
                        gain[valid][order],
                        color=color,
                        alpha=0.12,
                        lw=0.7,
                    )
        mean_gain = result["mean_psd_um4"] / baseline_psd
        valid = valid_base & np.isfinite(mean_gain) & (mean_gain > 0)
        if np.any(valid):
            order = np.argsort(wavelength[valid])
            ax_gain.plot(
                wavelength[valid][order],
                mean_gain[valid][order],
                color=color,
                lw=2.3,
                label=result["label"],
            )
    ax_gain.axhline(1.0, color="0.45", lw=0.9, ls="--")
    ax_gain.set_xlim(RADIAL_WAVELENGTH_MIN_UM, RADIAL_WAVELENGTH_MAX_UM)
    ax_gain.set_yscale("log")
    ax_gain.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax_gain.set_ylabel(
        r"$\overline{C_h}(\lambda,\varepsilon_{zz})/"
        r"\overline{C_h}(\lambda,0)$"
    )
    ax_gain.grid(True, which="both", alpha=0.25)
    ax_gain.legend(loc="lower right", frameon=False, fontsize=6)

    for axis, panel_label in zip((ax_psd, ax_gain), ("A", "B")):
        configure_log_wavelength_axis(
            axis,
            RADIAL_WAVELENGTH_MIN_UM,
            RADIAL_WAVELENGTH_MAX_UM,
        )
        add_panel_label(axis, panel_label)
    fig.tight_layout()
    if save:
        fig.savefig(
            OUTPUT_DIR / "two_panel_height_psd_and_gain_by_strain_to_400um.png",
            bbox_inches="tight",
        )
    return fig, summary, traces


# %% Figures 4/5: original corrected bracketed PSD gain and radial ACF change
def paired_psd_gain_and_acf_change(
    spatial: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bracket_edges = _bracket_edges()
    band_centers = 0.5 * (bracket_edges[:-1] + bracket_edges[1:])
    psd_rows = []
    acf_rows = []
    for (load, sample_type, sample_id), group in spatial.groupby(
        ["load_mpa", "sample_type", "sample_id"]
    ):
        group = group.sort_values("time_h")
        if len(group) < 2 or float(group.iloc[0]["time_h"]) > INITIAL_TIME_TOL_H:
            continue
        initial = group.iloc[0]
        initial_power = np.asarray(initial["bracket_power_um2"], dtype=float)
        initial_acf = np.asarray(initial["acf"], dtype=float)
        for record in group.itertuples(index=False):
            current_power = np.asarray(record.bracket_power_um2, dtype=float)
            valid = (
                np.isfinite(current_power)
                & np.isfinite(initial_power)
                & (current_power > 0)
                & (initial_power > 0)
            )
            log_gain = np.full_like(current_power, np.nan)
            log_gain[valid] = np.log10(current_power[valid] / initial_power[valid])
            specimen_id = f"{sample_type}_{int(load)}MPa_{sample_id}"
            for index, center in enumerate(band_centers):
                psd_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "load_mpa": int(load),
                        "sample_type": sample_type,
                        "sample": sample_id,
                        "time_h": float(record.time_h),
                        "bulk_z_strain_percent": float(
                            record.bulk_z_strain_percent
                        ),
                        "band_index": index,
                        "lambda_min_um": bracket_edges[index],
                        "lambda_max_um": bracket_edges[index + 1],
                        "lambda_center_um": center,
                        "log10_gain": log_gain[index],
                        "gain": (
                            10.0 ** log_gain[index]
                            if np.isfinite(log_gain[index])
                            else np.nan
                        ),
                    }
                )
            current_acf = np.asarray(record.acf, dtype=float)
            delta_acf = current_acf - initial_acf
            lag = np.asarray(record.acf_lag_um, dtype=float)
            for index in range(len(lag)):
                acf_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "load_mpa": int(load),
                        "sample_type": sample_type,
                        "sample": sample_id,
                        "time_h": float(record.time_h),
                        "bulk_z_strain_percent": float(
                            record.bulk_z_strain_percent
                        ),
                        "r_um": lag[index],
                        "delta_acf": delta_acf[index],
                        "acf_current": current_acf[index],
                        "acf_initial": initial_acf[index],
                    }
                )
    return pd.DataFrame(psd_rows), pd.DataFrame(acf_rows)


def summarize_grouped_psd_gain_and_acf_change(
    psd_long: pd.DataFrame,
    acf_long: pd.DataFrame,
    strain_group_definitions: Sequence[Mapping[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = psd_long[
        [
            "specimen_id",
            "load_mpa",
            "sample_type",
            "sample",
            "time_h",
            "bulk_z_strain_percent",
        ]
    ].drop_duplicates()
    if strain_group_definitions is None:
        strain_group_definitions = derive_strain_group_definitions(observations)
    grouped = assign_positive_strain_groups(observations, strain_group_definitions)
    keys = ["specimen_id", "load_mpa", "sample_type", "sample", "time_h"]
    group_columns = keys + [
        "strain_group",
        "strain_level_percent",
        "strain_group_label",
        "strain_group_loads_mpa",
    ]
    psd_grouped = psd_long.merge(
        grouped[group_columns], on=keys, how="inner"
    )
    acf_grouped = acf_long.merge(
        grouped[group_columns], on=keys, how="inner"
    )

    psd_rows = []
    for (strain_group, band_index), group in psd_grouped.groupby(
        ["strain_group", "band_index"], observed=False
    ):
        values = group["log10_gain"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            first = group.iloc[0]
            psd_rows.append(
                {
                    "strain_group": strain_group,
                    "strain_level_percent": first["strain_level_percent"],
                    "strain_group_label": first["strain_group_label"],
                    "strain_group_loads_mpa": first["strain_group_loads_mpa"],
                    "band_index": band_index,
                    "lambda_min_um": first["lambda_min_um"],
                    "lambda_max_um": first["lambda_max_um"],
                    "lambda_center_um": first["lambda_center_um"],
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "err": error_from_values(values, "sem"),
                    "n": int(values.size),
                }
            )
    acf_rows = []
    for (strain_group, lag), group in acf_grouped.groupby(
        ["strain_group", "r_um"], observed=False
    ):
        values = group["delta_acf"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            first = group.iloc[0]
            acf_rows.append(
                {
                    "strain_group": strain_group,
                    "strain_level_percent": first["strain_level_percent"],
                    "strain_group_label": first["strain_group_label"],
                    "strain_group_loads_mpa": first["strain_group_loads_mpa"],
                    "r_um": lag,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "err": error_from_values(values, "sem"),
                    "n": int(values.size),
                }
            )
    psd_summary = pd.DataFrame(psd_rows)
    acf_summary = pd.DataFrame(acf_rows)
    categories = [
        str(definition["strain_group"])
        for definition in strain_group_definitions
    ]
    for frame in (psd_summary, acf_summary):
        frame["strain_group"] = pd.Categorical(
            frame["strain_group"], categories=categories, ordered=True
        )
    psd_summary = psd_summary.sort_values(["strain_group", "lambda_center_um"])
    acf_summary = acf_summary.sort_values(["strain_group", "r_um"])
    return psd_summary, acf_summary


def plot_bracketed_psd_gain_and_acf_change(
    psd_summary: pd.DataFrame,
    acf_summary: pd.DataFrame,
    strain_group_definitions: Sequence[Mapping[str, object]],
    *,
    save: bool = True,
) -> plt.Figure:
    """Exact corrected A/B presentation from figures.py:23737--23807."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax_psd, ax_acf = axes
    for definition in strain_group_definitions:
        group_key = str(definition["strain_group"])
        selected = psd_summary[
            psd_summary["strain_group"] == group_key
        ].sort_values(
            "lambda_center_um"
        )
        wavelength = selected["lambda_center_um"].to_numpy(dtype=float)
        mean = selected["mean"].to_numpy(dtype=float)
        error = selected["err"].to_numpy(dtype=float)
        ax_psd.plot(
            wavelength,
            mean,
            lw=2.0,
            color=definition["color"],
            label=str(definition["display_label"]),
        )
        ax_psd.fill_between(
            wavelength,
            mean - error,
            mean + error,
            color=definition["color"],
            alpha=0.15,
            linewidth=0,
        )
    ax_psd.axhline(0.0, color="0.45", lw=1.0, ls="--")
    ax_psd.set_xlim(BRACKET_WAVELENGTH_MIN_UM, BRACKET_WAVELENGTH_MAX_UM)
    ax_psd.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax_psd.set_ylabel(
        r"$\overline{\log_{10}\!\left["
        r"C_h(\lambda,\varepsilon_{zz})/C_h(\lambda,0)\right]}$"
    )
    ax_psd.grid(True, alpha=0.25)
    ax_psd.legend(loc="upper right", frameon=False, fontsize=7)

    for definition in strain_group_definitions:
        group_key = str(definition["strain_group"])
        selected = acf_summary[
            acf_summary["strain_group"] == group_key
        ].sort_values("r_um")
        ax_acf.plot(
            selected["r_um"],
            selected["mean"],
            lw=2.0,
            color=definition["color"],
            label=str(definition["display_label"]),
        )
        ax_acf.fill_between(
            selected["r_um"],
            selected["mean"] - selected["err"],
            selected["mean"] + selected["err"],
            color=definition["color"],
            alpha=0.15,
            linewidth=0,
        )
    ax_acf.axhline(0.0, color="0.45", lw=1.0, ls="--")
    ax_acf.set_xlim(0.0, ACF_CHANGE_MAX_LAG_UM)
    ax_acf.set_xlabel(r"$r$ [$\mu$m]")
    ax_acf.set_ylabel(r"$\overline{\Delta C}(r)$")
    ax_acf.grid(True, alpha=0.25)
    ax_acf.legend(loc="upper right", frameon=False, fontsize=7)
    configure_log_wavelength_axis(
        ax_psd,
        BRACKET_WAVELENGTH_MIN_UM,
        BRACKET_WAVELENGTH_MAX_UM,
    )
    add_panel_label(ax_psd, "A")
    add_panel_label(ax_acf, "B")
    fig.tight_layout()
    if save:
        fig.savefig(
            OUTPUT_DIR / "two_panel_corrected_psd_gain_and_acf_change_by_strain.png",
            bbox_inches="tight",
        )
    return fig


# %% Figures 6/7: matched-field endpoint and full-crop sensitivity
def resample_pixel_centers(
    values: np.ndarray,
    source_spacing_um: float | tuple[float, float],
    target_spacing_um: float = MATCHED_TARGET_SPACING_UM,
) -> np.ndarray:
    """Interpolate pixel-center coordinates; this does not add optical bandwidth."""
    values = np.asarray(values, dtype=float)
    if np.isscalar(source_spacing_um):
        source = (float(source_spacing_um), float(source_spacing_um))
    else:
        source = tuple(float(value) for value in source_spacing_um)
    target = (float(target_spacing_um), float(target_spacing_um))
    sigma = [
        0.5 * (target_d / source_d - 1.0) if target_d > source_d else 0.0
        for source_d, target_d in zip(source, target)
    ]
    filtered = (
        gaussian_filter(values, sigma=sigma, mode="nearest")
        if any(value > 0 for value in sigma)
        else values
    )
    source_axes = [
        (np.arange(size, dtype=float) + 0.5) * spacing
        for size, spacing in zip(values.shape, source)
    ]
    target_axes = []
    for source_axis, target_d in zip(source_axes, target):
        span = float(source_axis[-1] - source_axis[0])
        count = int(np.floor(span / target_d)) + 1
        used_span = (count - 1) * target_d
        offset = 0.5 * (span - used_span)
        target_axes.append(
            source_axis[0] + offset + np.arange(count, dtype=float) * target_d
        )
    interpolator = RegularGridInterpolator(
        tuple(source_axes), filtered, method="linear", bounds_error=True
    )
    grid = np.meshgrid(*target_axes, indexing="ij")
    points = np.column_stack([axis.ravel() for axis in grid])
    return interpolator(points).reshape(tuple(len(axis) for axis in target_axes))


def _centered_matched_fields(values: np.ndarray) -> list[np.ndarray]:
    field_shape = tuple(
        int(round(length / MATCHED_TARGET_SPACING_UM))
        for length in MATCHED_FIELD_SHAPE_UM
    )
    values = np.asarray(values, dtype=float)
    if values.shape == field_shape[::-1]:
        values = values.T
    n0 = values.shape[0] // field_shape[0]
    n1 = values.shape[1] // field_shape[1]
    if n0 < 1 or n1 < 1:
        raise ValueError(
            f"Field {values.shape} cannot contain a matched {field_shape} section."
        )
    used0 = n0 * field_shape[0]
    used1 = n1 * field_shape[1]
    start0 = (values.shape[0] - used0) // 2
    start1 = (values.shape[1] - used1) // 2
    return [
        values[
            start0 + i * field_shape[0] : start0 + (i + 1) * field_shape[0],
            start1 + j * field_shape[1] : start1 + (j + 1) * field_shape[1],
        ]
        for i in range(n0)
        for j in range(n1)
    ]


def manuscript_matched_fields(
    raw_height_um: np.ndarray,
    source_spacing_um: float,
    *,
    source: str,
) -> list[np.ndarray]:
    """The one sectioning operator, used only by the endpoint four-panel figure."""
    raw_height_um = np.asarray(raw_height_um, dtype=float)
    raw_shape = raw_height_um.shape
    if source == "exp":
        height = experimental_crop(fill_missing_nearest(raw_height_um))
    elif source == "sim":
        height = raw_height_um.copy()
        missing = ~np.isfinite(height)
        if np.any(missing):
            finite_mean = np.nanmean(height)
            if not np.isfinite(finite_mean):
                raise ValueError("Simulation height field contains no finite values.")
            height[missing] = finite_mean
    else:
        raise ValueError(f"Unknown source {source!r}.")
    if not np.isclose(source_spacing_um, MATCHED_TARGET_SPACING_UM):
        height = resample_pixel_centers(height, source_spacing_um)
    fields = _centered_matched_fields(height)
    if (
        source == "exp"
        and raw_shape == EXPECTED_EXP_RAW_SHAPE
        and len(fields) != EXPECTED_CROP_FIRST_MATCHED_FIELDS
    ):
        raise AssertionError(
            "Standard crop-first experimental map produced "
            f"{len(fields)} matched fields; expected "
            f"{EXPECTED_CROP_FIRST_MATCHED_FIELDS}."
        )
    leveled_fields = []
    for field in fields:
        leveled = detrend_surface(field, MATCHED_TARGET_SPACING_UM, order=1)
        leveled_fields.append(leveled - np.mean(leveled))
    return leveled_fields


def full_resampled_experimental_field(
    raw_height_um: np.ndarray,
    source_spacing_um: float,
) -> np.ndarray:
    """Crop, resample to 1 um, and plane-level once without sectioning."""
    height = experimental_crop(fill_missing_nearest(raw_height_um))
    if not np.isclose(source_spacing_um, MATCHED_TARGET_SPACING_UM):
        height = resample_pixel_centers(height, source_spacing_um)
    leveled = detrend_surface(height, MATCHED_TARGET_SPACING_UM, order=1)
    return leveled - np.mean(leveled)


def endpoint_curve_from_fields(
    fields: Sequence[np.ndarray],
    wavelength_edges_um: np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Calculate raw PSD and overlap-corrected ACF for prepared 1 um fields."""
    if not fields:
        raise ValueError("At least one endpoint height field is required.")
    psd_curves = []
    acf_curves = []
    wavelength = None
    lag = None
    psd_modes = None
    acf_counts = None
    for field in fields:
        f0, f1, psd2d, _df0, _df1 = periodogram_2d(
            field, MATCHED_TARGET_SPACING_UM
        )
        wavelength, psd, modes = radial_psd_wavelength_binned(
            f0,
            f1,
            psd2d,
            wavelength_edges_um,
            min_modes=ENDPOINT_MIN_MODES,
        )
        if np.any(~np.isfinite(psd)):
            raise ValueError(
                "Endpoint wavelength bins are not supported by every field."
            )
        psd_curves.append(psd)
        lag, acf, counts = radial_acf_overlap_corrected(
            field,
            MATCHED_TARGET_SPACING_UM,
            bin_width_um=ENDPOINT_ACF_BIN_WIDTH_UM,
            max_lag_um=ENDPOINT_ACF_MAX_LAG_UM,
        )
        acf_curves.append(acf)
        if psd_modes is None:
            psd_modes = modes
        if acf_counts is None:
            acf_counts = counts
    return {
        "wavelength_um": np.asarray(wavelength),
        "psd": _finite_column_median(np.vstack(psd_curves)),
        "psd_support_count": np.asarray(psd_modes),
        "lag_um": np.asarray(lag),
        "acf": _finite_column_median(np.vstack(acf_curves)),
        "acf_support_count": np.asarray(acf_counts),
        "n_fields": len(fields),
    }


def endpoint_curve_from_matched_fields(
    raw_height_um: np.ndarray,
    source_spacing_um: float,
    *,
    source: str,
    wavelength_edges_um: np.ndarray | None = None,
) -> dict[str, np.ndarray | int]:
    fields = manuscript_matched_fields(
        raw_height_um,
        source_spacing_um,
        source=source,
    )
    edges = (
        endpoint_wavelength_edges()
        if wavelength_edges_um is None
        else wavelength_edges_um
    )
    return endpoint_curve_from_fields(fields, edges)


def endpoint_curve_from_full_resampled_experiment(
    raw_height_um: np.ndarray,
    source_spacing_um: float,
    *,
    wavelength_edges_um: np.ndarray | None = None,
) -> dict[str, np.ndarray | int]:
    field = full_resampled_experimental_field(raw_height_um, source_spacing_um)
    edges = (
        endpoint_wavelength_edges()
        if wavelength_edges_um is None
        else wavelength_edges_um
    )
    return endpoint_curve_from_fields([field], edges)


def _endpoint_curve_rows(
    curve: Mapping[str, object],
    common: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = []
    for curve_type, x_key, y_key, support_key in (
        ("psd", "wavelength_um", "psd", "psd_support_count"),
        ("acf", "lag_um", "acf", "acf_support_count"),
    ):
        for x_value, y_value, support_count in zip(
            np.asarray(curve[x_key]),
            np.asarray(curve[y_key]),
            np.asarray(curve[support_key]),
        ):
            rows.append(
                {
                    **common,
                    "curve_type": curve_type,
                    "x_um": float(x_value),
                    "y": float(y_value),
                    "support_count": int(support_count),
                }
            )
    return rows


def summarize_endpoint_curves(curves: pd.DataFrame) -> pd.DataFrame:
    return (
        curves.groupby(
            [
                "analysis_variant",
                "source",
                "load_mpa",
                "sample_type",
                "curve_type",
                "x_um",
            ],
            as_index=False,
        )["y"]
        .agg(mean="mean", median="median", std="std", n="count")
    )


def build_endpoint_spatial_curves(
    exp_metadata: pd.DataFrame,
    sim_metadata: pd.DataFrame,
    experimental_heights: Mapping[str, np.ndarray],
    simulation_heights: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    wavelength_edges = endpoint_wavelength_edges()
    eligible_exp = eligible_experimental_metadata(
        exp_metadata,
        require_zero_time=True,
    )
    endpoint_time = eligible_exp.groupby(
        ["load_mpa", "sample_type", "sample_id"]
    )["time_h"].transform("max")
    exp_endpoint = eligible_exp[eligible_exp["time_h"].eq(endpoint_time)]
    for record in exp_endpoint.itertuples(index=False):
        curve = endpoint_curve_from_matched_fields(
            np.asarray(experimental_heights[record.height_key], dtype=float),
            float(record.spacing_um),
            source="exp",
            wavelength_edges_um=wavelength_edges,
        )
        common = {
            "analysis_variant": "matched_256x128",
            "source": "exp",
            "load_mpa": int(record.load_mpa),
            "sample_type": record.sample_type,
            "sample_id": record.sample_id,
            "replicate_id": record.sample_id,
            "n_fields": int(curve["n_fields"]),
        }
        rows.extend(_endpoint_curve_rows(curve, common))

    sim_endpoint = sim_metadata[sim_metadata["is_endpoint"]]
    for record in sim_endpoint.itertuples(index=False):
        curve = endpoint_curve_from_matched_fields(
            np.asarray(simulation_heights[record.height_key], dtype=float),
            float(record.spacing_um),
            source="sim",
            wavelength_edges_um=wavelength_edges,
        )
        common = {
            "analysis_variant": "matched_256x128",
            "source": "sim",
            "load_mpa": int(record.load_mpa),
            "sample_type": record.sample_type,
            "sample_id": record.micro_id,
            "replicate_id": f"{record.micro_id}_face{record.face_index}",
            "n_fields": int(curve["n_fields"]),
        }
        rows.extend(_endpoint_curve_rows(curve, common))
    curves = pd.DataFrame(rows)
    return curves, summarize_endpoint_curves(curves)


def build_full_crop_endpoint_sensitivity(
    exp_metadata: pd.DataFrame,
    experimental_heights: Mapping[str, np.ndarray],
    matched_endpoint_curves: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Change only the experiment to one cropped, 1 um-resampled field."""
    wavelength_edges = endpoint_wavelength_edges()
    rows = []
    eligible_exp = eligible_experimental_metadata(
        exp_metadata,
        require_zero_time=True,
    )
    endpoint_time = eligible_exp.groupby(
        ["load_mpa", "sample_type", "sample_id"]
    )["time_h"].transform("max")
    exp_endpoint = eligible_exp[eligible_exp["time_h"].eq(endpoint_time)]
    for record in exp_endpoint.itertuples(index=False):
        curve = endpoint_curve_from_full_resampled_experiment(
            np.asarray(experimental_heights[record.height_key], dtype=float),
            float(record.spacing_um),
            wavelength_edges_um=wavelength_edges,
        )
        common = {
            "analysis_variant": "exp_full_crop_1um",
            "source": "exp",
            "load_mpa": int(record.load_mpa),
            "sample_type": record.sample_type,
            "sample_id": record.sample_id,
            "replicate_id": record.sample_id,
            "n_fields": 1,
        }
        rows.extend(_endpoint_curve_rows(curve, common))

    simulation_rows = matched_endpoint_curves[
        matched_endpoint_curves["source"].eq("sim")
    ].copy()
    simulation_rows["analysis_variant"] = "exp_full_crop_1um"
    curves = pd.concat([pd.DataFrame(rows), simulation_rows], ignore_index=True)
    return curves, summarize_endpoint_curves(curves)


def plot_endpoint_spatial_configuration(
    summary: pd.DataFrame,
    *,
    output_filename: str = "four_panel_endpoint_exp_vs_sim_psd_acf.png",
    save: bool = True,
) -> plt.Figure:
    """Original four-panel presentation with endpoint curves as input."""
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=False)
    ax_psd_int, ax_psd_unint = axes[0]
    ax_acf_int, ax_acf_unint = axes[1]

    def plot_group(axis, cases, curve_type, ylabel, legend_location):
        for load, sample_type in cases:
            for source, linestyle, alpha, suffix in (
                ("exp", "-", 1.0, "Exp."),
                ("sim", "--", 0.95, "Sim."),
            ):
                selected = summary[
                    (summary["source"] == source)
                    & (summary["load_mpa"] == load)
                    & (summary["sample_type"] == sample_type)
                    & (summary["curve_type"] == curve_type)
                ].sort_values("x_um")
                if selected.empty:
                    continue
                axis.plot(
                    selected["x_um"],
                    selected["mean"],
                    color=load_color(load),
                    linestyle=linestyle,
                    lw=2.0 if source == "exp" else 1.8,
                    alpha=alpha,
                    label=f"{load} MPa {suffix}",
                )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(
            loc=legend_location,
            frameon=False,
            fontsize=6,
            ncol=2,
        )

    plot_group(
        ax_psd_int,
        INTERRUPTED_CASES,
        "psd",
        r"$\overline{C_h}(\lambda)$ [$\mu$m$^4$]",
        "lower right",
    )
    plot_group(
        ax_psd_unint,
        UNINTERRUPTED_CASES,
        "psd",
        r"$\overline{C_h}(\lambda)$ [$\mu$m$^4$]",
        "lower right",
    )
    plot_group(
        ax_acf_int,
        INTERRUPTED_CASES,
        "acf",
        r"$\overline{C}(r)$",
        "upper right",
    )
    plot_group(
        ax_acf_unint,
        UNINTERRUPTED_CASES,
        "acf",
        r"$\overline{C}(r)$",
        "upper right",
    )

    for axis in (ax_psd_int, ax_psd_unint):
        axis.set_yscale("log")
        axis.set_xlabel(r"$\lambda$ [$\mu$m]")
        configure_log_wavelength_axis(
            axis,
            ENDPOINT_WAVELENGTH_MIN_UM,
            ENDPOINT_WAVELENGTH_MAX_UM,
        )
    for axis in (ax_acf_int, ax_acf_unint):
        axis.set_xlim(0.0, ENDPOINT_ACF_MAX_LAG_UM)
        axis.set_ylim(-0.35, 1.05)
        axis.set_xlabel(r"$r$ [$\mu$m]")
        axis.axhline(0.0, color="0.5", lw=0.8)
        axis.axhline(np.exp(-1.0), color="0.5", lw=0.8, ls=":")

    for axis, panel_label in zip(axes.ravel(), ("A", "B", "C", "D")):
        add_panel_label(axis, panel_label)
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / output_filename, bbox_inches="tight")
    return fig


# %% Run the original layouts plus the endpoint support sensitivity figure
def run_publication_figures(
    *,
    rebuild_caches: bool = False,
    show_individual_psd_traces: bool = False,
    save: bool = True,
    show: bool = True,
) -> dict[str, object]:
    cached_exp_df, cached_sim_df, cached_exp_heights, cached_sim_heights = (
        initialize_caches(rebuild=rebuild_caches)
    )

    exp_sa, sim_sa = build_delta_sa_tables(
        cached_exp_df,
        cached_sim_df,
        cached_exp_heights,
        cached_sim_heights,
    )
    spatial = build_experimental_spatial_table(cached_exp_df, cached_exp_heights)
    strain_group_definitions = derive_strain_group_definitions(spatial)

    figure_1, fit = plot_all_delta_sa_linear_fit(exp_sa, sim_sa, save=save)
    figure_2 = plot_six_panel_delta_sa_vs_strain(
        exp_sa,
        sim_sa,
        scale=float(fit["scale"]),
        save=save,
    )
    figure_3, psd_summary, psd_traces = plot_radial_psd_and_gain_by_strain(
        spatial,
        strain_group_definitions=strain_group_definitions,
        show_individual_traces=show_individual_psd_traces,
        save=save,
    )

    psd_gain_long, acf_change_long = paired_psd_gain_and_acf_change(spatial)
    psd_gain_summary, acf_change_summary = (
        summarize_grouped_psd_gain_and_acf_change(
            psd_gain_long,
            acf_change_long,
            strain_group_definitions,
        )
    )
    if save:
        psd_gain_long.to_csv(
            OUTPUT_DIR / "corrected_psd_bracket_gain_long.csv", index=False
        )
        acf_change_long.to_csv(
            OUTPUT_DIR / "corrected_radial_acf_change_long.csv", index=False
        )
        psd_gain_summary.to_csv(
            OUTPUT_DIR / "corrected_psd_bracket_gain_summary.csv", index=False
        )
        acf_change_summary.to_csv(
            OUTPUT_DIR / "corrected_radial_acf_change_summary.csv", index=False
        )
    figure_4_5 = plot_bracketed_psd_gain_and_acf_change(
        psd_gain_summary,
        acf_change_summary,
        strain_group_definitions,
        save=save,
    )

    endpoint_curves, endpoint_summary = build_endpoint_spatial_curves(
        cached_exp_df,
        cached_sim_df,
        cached_exp_heights,
        cached_sim_heights,
    )
    if save:
        endpoint_curves.to_csv(
            OUTPUT_DIR / "endpoint_exp_sim_psd_acf_curves_long.csv", index=False
        )
        endpoint_summary.to_csv(
            OUTPUT_DIR / "endpoint_exp_sim_psd_acf_curves_summary.csv", index=False
        )
    figure_6 = plot_endpoint_spatial_configuration(endpoint_summary, save=save)

    full_crop_endpoint_curves, full_crop_endpoint_summary = (
        build_full_crop_endpoint_sensitivity(
            cached_exp_df,
            cached_exp_heights,
            endpoint_curves,
        )
    )
    if save:
        full_crop_endpoint_curves.to_csv(
            OUTPUT_DIR / "endpoint_full_crop_1um_psd_acf_curves_long.csv",
            index=False,
        )
        full_crop_endpoint_summary.to_csv(
            OUTPUT_DIR / "endpoint_full_crop_1um_psd_acf_curves_summary.csv",
            index=False,
        )
    figure_7 = plot_endpoint_spatial_configuration(
        full_crop_endpoint_summary,
        output_filename="four_panel_endpoint_exp_full_crop_1um_vs_sim_psd_acf.png",
        save=save,
    )

    figures = [figure_1, figure_2, figure_3, figure_4_5, figure_6, figure_7]
    if show:
        plt.show()
    return {
        "figures": figures,
        "fit": fit,
        "exp_sa": exp_sa,
        "sim_sa": sim_sa,
        "experimental_spatial": spatial,
        "strain_group_definitions": strain_group_definitions,
        "psd_group_summary": psd_summary,
        "psd_group_traces": psd_traces,
        "psd_gain_long": psd_gain_long,
        "acf_change_long": acf_change_long,
        "psd_gain_summary": psd_gain_summary,
        "acf_change_summary": acf_change_summary,
        "endpoint_curves": endpoint_curves,
        "endpoint_summary": endpoint_summary,
        "full_crop_endpoint_curves": full_crop_endpoint_curves,
        "full_crop_endpoint_summary": full_crop_endpoint_summary,
    }


if __name__ == "__main__":
    run_publication_figures()
