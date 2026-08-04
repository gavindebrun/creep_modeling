from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label,
)
from scipy.signal import fftconvolve
from scipy.stats import bootstrap


@dataclass(frozen=True)
class RadialPSDResult:
    """Radial PSD and specimen-bootstrap results at one time."""

    frequency_um_inv: np.ndarray
    wavelength_um: np.ndarray

    # Shape: (nsamples, nbins)
    specimen_psd_um4: np.ndarray

    # Shape: (nbins,)
    mean_psd_um4: np.ndarray
    ci_low_um4: np.ndarray
    ci_high_um4: np.ndarray

    modes_per_bin: np.ndarray

    # One value per specimen. This should be near floating-point precision.
    parseval_relative_error: np.ndarray

    # Both summaries are retained; estimate_psd_um4 selects one for plotting.
    median_psd_um4: np.ndarray | None = None
    estimator: str = "mean"

    @property
    def estimate_psd_um4(self) -> np.ndarray:
        if self.estimator == "mean":
            return self.mean_psd_um4
        if self.estimator == "median" and self.median_psd_um4 is not None:
            return self.median_psd_um4
        raise ValueError(f"Unsupported or unavailable estimator: {self.estimator!r}.")


@dataclass(frozen=True)
class PSD2DResult:
    """Full two-dimensional PSDs at one time."""

    frequency_axis0_um_inv: np.ndarray
    frequency_axis1_um_inv: np.ndarray

    # Shape: (nsamples, n0, n1)
    specimen_psd_um4: np.ndarray

    # Shape: (n0, n1)
    mean_psd_um4: np.ndarray

    # One value per specimen. This should be near floating-point precision.
    parseval_relative_error: np.ndarray

    # Fractions removed by an optional spatial exclusion mask.
    excluded_fraction: np.ndarray | None = None
    taper_effective_fraction: np.ndarray | None = None

    # Both summaries are retained; estimate_psd_um4 selects one for plotting.
    median_psd_um4: np.ndarray | None = None
    estimator: str = "mean"

    @property
    def estimate_psd_um4(self) -> np.ndarray:
        if self.estimator == "mean":
            return self.mean_psd_um4
        if self.estimator == "median" and self.median_psd_um4 is not None:
            return self.median_psd_um4
        raise ValueError(f"Unsupported or unavailable estimator: {self.estimator!r}.")


@dataclass(frozen=True)
class PSDAnalysisResult:
    """Full PSD analysis at one time."""

    radial: RadialPSDResult
    full_2d: PSD2DResult
    excluded_masks: np.ndarray | None = None
    center_indent_mask: CenterIndentMaskResult | None = None


@dataclass(frozen=True)
class CenterIndentMaskResult:
    """One fixed center-indent exclusion mask per specimen."""

    # Shape: (nsamples, n0, n1). True means excluded from analysis.
    excluded_masks: np.ndarray

    # Shape: (nsamples, 2), ordered as array-axis indices (axis 0, axis 1).
    detected_center_indices: np.ndarray

    # The detected depression before the physical guard band was added.
    detected_masks: np.ndarray

    detected_area_um2: np.ndarray
    excluded_area_um2: np.ndarray
    guard_um: np.ndarray
    center_search_fraction: float
    method: str = "automatic_center_depression"
    fixed_diagonal_um: float | None = None


@dataclass(frozen=True)
class AutocorrelationLengthResult:
    """ISO 25178-style spatial parameters for every specimen at one time."""

    threshold: float
    specimen_sal_um: np.ndarray
    specimen_fastest_decay_angle_deg: np.ndarray
    specimen_slowest_decay_um: np.ndarray
    specimen_slowest_decay_angle_deg: np.ndarray
    specimen_texture_aspect_ratio: np.ndarray
    surface_type: str = "S-F leveled topography"
    l_filter_cutoff_um: float | None = None

    @property
    def mean_sal_um(self) -> float:
        return float(np.mean(self.specimen_sal_um))

    @property
    def median_sal_um(self) -> float:
        return float(np.median(self.specimen_sal_um))

    @property
    def mean_texture_aspect_ratio(self) -> float:
        return float(np.nanmean(self.specimen_texture_aspect_ratio))

    @property
    def median_texture_aspect_ratio(self) -> float:
        return float(np.nanmedian(self.specimen_texture_aspect_ratio))


@dataclass(frozen=True)
class ACFAnalysisResult:
    """Specimen ACF length metrics and the specimen-aggregate 2D ACF."""

    lengths: AutocorrelationLengthResult
    aggregate_acf: np.ndarray
    lag_axis0_um: np.ndarray
    lag_axis1_um: np.ndarray
    estimator: str = "mean"


@dataclass(frozen=True)
class DirectionalPSDResult:
    """Sector-averaged PSDs resolved by wavevector direction."""

    frequency_um_inv: np.ndarray
    wavelength_um: np.ndarray
    direction_deg: np.ndarray
    specimen_psd_um4: np.ndarray
    mean_psd_um4: np.ndarray
    median_psd_um4: np.ndarray
    modes_per_sector_bin: np.ndarray
    estimator: str = "mean"

    @property
    def estimate_psd_um4(self) -> np.ndarray:
        if self.estimator == "mean":
            return self.mean_psd_um4
        if self.estimator == "median":
            return self.median_psd_um4
        raise ValueError(f"Unsupported estimator: {self.estimator!r}.")


@dataclass(frozen=True)
class SurfaceRoughnessResult:
    """Areal amplitude, distribution, extreme, and slope parameters."""

    parameter_names: tuple[str, ...]
    specimen_values: np.ndarray


@dataclass(frozen=True)
class MagnificationEvolutionResult:
    """Time-resolved summary for one profilometry magnification."""

    label: str
    spacing_um: tuple[float, float]
    cropped_shape: tuple[int, int]
    wavelength_limits_um: tuple[float, float]
    radial_results: tuple[RadialPSDResult, ...]
    directional_results: tuple[DirectionalPSDResult, ...]
    roughness_parameter_names: tuple[str, ...]
    roughness_specimen_values: np.ndarray
    acf_metric_names: tuple[str, ...]
    acf_specimen_values: np.ndarray
    spectral_length_names: tuple[str, ...]
    spectral_length_values_um: np.ndarray
    band_names: tuple[str, ...]
    band_bounds_um: np.ndarray
    band_power_um2: np.ndarray
    rigid_shift_to_t0_pixels: np.ndarray
    registration_peak_to_mean: np.ndarray
    nested_window_fractions: np.ndarray
    nested_window_sq_um: np.ndarray
    mask_sensitivity_diagonals_um: np.ndarray
    mask_sensitivity_sq_um: np.ndarray
    mask_sensitivity_sal_um: np.ndarray
    mask_sensitivity_band_power_um2: np.ndarray
    line_psd_axis0_um3: np.ndarray
    line_psd_axis1_um3: np.ndarray
    output_dir: Path


@dataclass(frozen=True)
class SmallWavelengthCutoffResult:
    """50x high-frequency reliability cutoff from the 2D/1D PSD ratio."""

    times_hours: np.ndarray
    wavelength_um: np.ndarray
    normalized_ratio_axis0: np.ndarray
    normalized_ratio_axis1: np.ndarray
    cutoff_axis0_um: np.ndarray
    cutoff_axis1_um: np.ndarray
    cutoff_um: np.ndarray
    ci_low_um: np.ndarray
    ci_high_um: np.ndarray
    detected: np.ndarray
    bootstrap_detection_fraction: np.ndarray
    status: np.ndarray


@dataclass(frozen=True)
class LargeWavelengthCutoffResult:
    """10x low-frequency roll-off estimates and field-size diagnostics."""

    times_hours: np.ndarray
    window_fractions: np.ndarray
    window_short_side_um: np.ndarray
    cutoff_um: np.ndarray
    ci_low_um: np.ndarray
    ci_high_um: np.ndarray
    delta_bic: np.ndarray
    scaling_slope: np.ndarray
    detected: np.ndarray
    bootstrap_detection_fraction: np.ndarray
    full_window_stable: np.ndarray


@dataclass(frozen=True)
class SpectralChangeResult:
    """Paired PSD changes with a simultaneous confidence band."""

    label: str
    wavelength_um: np.ndarray
    log10_psd_ratio: np.ndarray
    estimate: np.ndarray
    simultaneous_ci_low: np.ndarray
    simultaneous_ci_high: np.ndarray
    significant: np.ndarray
    valid_morphology_band: np.ndarray


@dataclass(frozen=True)
class PowerRelevanceResult:
    """Central wavelength interval containing a specified height variance."""

    fraction: float
    specimen_bounds_um: np.ndarray
    estimate_bounds_um: np.ndarray
    ci_low_bounds_um: np.ndarray
    ci_high_bounds_um: np.ndarray


@dataclass(frozen=True)
class WavelengthSelectionResult:
    """Data-derived morphology cutoffs and evolution-relevant wavelengths."""

    small: SmallWavelengthCutoffResult
    large: LargeWavelengthCutoffResult
    fine_change: SpectralChangeResult
    coarse_change: SpectralChangeResult
    power: PowerRelevanceResult
    morphology_lambda_min_um: float
    morphology_lambda_max_um: float
    power_lambda_min_um: float
    power_lambda_max_um: float
    evolution_lambda_min_um: float
    evolution_lambda_max_um: float
    output_dir: Path


@dataclass(frozen=True)
class CompleteAnalysisResult:
    """Outputs from the complete two-magnification longitudinal analysis."""

    times_hours: np.ndarray
    fine: MagnificationEvolutionResult
    coarse: MagnificationEvolutionResult
    overlap_wavelength_um: tuple[float, float]
    overlap_log10_psd_ratio: np.ndarray
    wavelength_selection: WavelengthSelectionResult
    output_dir: Path


def _as_spacing_tuple(spacing_um: float | tuple[float, float]) -> tuple[float, float]:
    if np.isscalar(spacing_um):
        d0 = d1 = float(spacing_um)
    else:
        d0, d1 = map(float, spacing_um)

    if d0 <= 0.0 or d1 <= 0.0:
        raise ValueError("Pixel spacings must be positive.")

    return d0, d1


def vickers_indent_diagonal_um(
    load_kgf: float,
    hardness_hv: float,
) -> float:
    """Return the mean Vickers impression diagonal from load and hardness."""
    load = float(load_kgf)
    hardness = float(hardness_hv)
    if load <= 0.0:
        raise ValueError("load_kgf must be positive.")
    if hardness <= 0.0:
        raise ValueError("hardness_hv must be positive.")

    # HV = 1.8544 * load_kgf / diagonal_mm**2.
    return 1_000.0 * np.sqrt(1.8544 * load / hardness)


def center_offset_after_crop_um(
    original_shape: tuple[int, int],
    crop_slices: tuple[slice, slice],
    *,
    spacing_um: float | tuple[float, float],
) -> tuple[float, float]:
    """Locate the original array center relative to an interior crop center."""
    if len(original_shape) != 2 or len(crop_slices) != 2:
        raise ValueError("original_shape and crop_slices must each have two items.")

    spacing = _as_spacing_tuple(spacing_um)
    offsets = []
    for axis_size, crop_slice, axis_spacing in zip(
        original_shape,
        crop_slices,
        spacing,
    ):
        axis_size = int(axis_size)
        if axis_size < 1:
            raise ValueError("original_shape entries must be positive.")
        if not isinstance(crop_slice, slice):
            raise TypeError("crop_slices must contain slice objects.")

        start, stop, step = crop_slice.indices(axis_size)
        if step != 1:
            raise ValueError("Center-mask crop slices must have unit step.")
        if stop <= start:
            raise ValueError("Center-mask crop slices must retain at least one pixel.")

        original_center = 0.5 * (axis_size - 1)
        crop_center_in_original = 0.5 * (start + stop - 1)
        offsets.append(
            (original_center - crop_center_in_original) * axis_spacing
        )

    return float(offsets[0]), float(offsets[1])


def _normalized_crop_slices(
    original_shape: tuple[int, int],
    crop_slices: tuple[slice, slice],
) -> tuple[tuple[slice, slice], tuple[int, int]]:
    if len(original_shape) != 2 or len(crop_slices) != 2:
        raise ValueError("original_shape and crop_slices must each have two items.")

    normalized = []
    starts = []
    for axis_size, crop_slice in zip(original_shape, crop_slices):
        if not isinstance(crop_slice, slice):
            raise TypeError("crop_slices must contain slice objects.")
        start, stop, step = crop_slice.indices(int(axis_size))
        if step != 1:
            raise ValueError("Analysis crop slices must have unit step.")
        if stop <= start:
            raise ValueError("Analysis crop slices must retain at least one pixel.")
        normalized.append(slice(start, stop))
        starts.append(start)

    return (normalized[0], normalized[1]), (starts[0], starts[1])


def _prepare_uncropped_height_maps(
    height_maps_um: np.ndarray,
    *,
    spacing_um: float | tuple[float, float],
    crop_slices: tuple[slice, slice],
    center_mask_diagonal_um: float,
) -> tuple[np.ndarray, CenterIndentMaskResult]:
    """Build the original-coordinate diamond mask, then crop maps and mask."""
    maps = np.asarray(height_maps_um, dtype=np.float64)
    if maps.ndim != 4:
        raise ValueError(
            "height_maps_um must have shape (ntimes, nsamples, n0, n1)."
        )
    if maps.shape[0] < 1 or maps.shape[1] < 2:
        raise ValueError("At least one time and two specimens are required.")
    if not np.all(np.isfinite(maps)):
        raise ValueError("height_maps_um contains NaN or infinite values.")

    original_shape = maps.shape[-2:]
    normalized_crop, crop_starts = _normalized_crop_slices(
        original_shape,
        crop_slices,
    )
    full_mask = fixed_center_diamond_exclusion_masks(
        maps,
        spacing_um=spacing_um,
        diagonal_um=center_mask_diagonal_um,
    )

    slice0, slice1 = normalized_crop
    cropped_maps = np.asarray(maps[..., slice0, slice1], dtype=np.float64)
    cropped_excluded = full_mask.excluded_masks[:, slice0, slice1].copy()
    cropped_detected = full_mask.detected_masks[:, slice0, slice1].copy()
    if np.any(
        np.count_nonzero(cropped_detected, axis=(1, 2))
        != np.count_nonzero(full_mask.detected_masks, axis=(1, 2))
    ):
        raise ValueError("The requested crop truncates the center-indent mask.")

    d0, d1 = _as_spacing_tuple(spacing_um)
    pixel_area = d0 * d1
    detected_area = (
        np.count_nonzero(cropped_detected, axis=(1, 2)) * pixel_area
    )
    excluded_area = (
        np.count_nonzero(cropped_excluded, axis=(1, 2)) * pixel_area
    )
    cropped_centers = full_mask.detected_center_indices - np.asarray(
        crop_starts,
        dtype=np.float64,
    )

    cropped_mask = CenterIndentMaskResult(
        excluded_masks=cropped_excluded,
        detected_center_indices=cropped_centers,
        detected_masks=cropped_detected,
        detected_area_um2=detected_area,
        excluded_area_um2=excluded_area,
        guard_um=full_mask.guard_um.copy(),
        center_search_fraction=0.0,
        method="fixed_physical_center_diamond_then_crop",
        fixed_diagonal_um=float(center_mask_diagonal_um),
    )
    return cropped_maps, cropped_mask


def _specimen_estimator(
    use_median: bool,
) -> tuple[str, Callable[..., np.ndarray]]:
    if use_median:
        return "median", np.median
    return "mean", np.mean


def _selected_time_maps(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    minimum_samples: int = 2,
) -> np.ndarray:
    maps = np.asarray(height_maps_um)

    if maps.ndim != 4:
        raise ValueError(
            "height_maps_um must have shape (ntimes, nsamples, n0, n1)."
        )

    if not 0 <= time_index < maps.shape[0]:
        raise IndexError(
            f"time_index={time_index} is outside [0, {maps.shape[0] - 1}]."
        )

    maps_t = np.asarray(maps[time_index], dtype=np.float64)

    if minimum_samples < 1:
        raise ValueError("minimum_samples must be at least one.")
    if maps_t.shape[0] < minimum_samples:
        raise ValueError(
            f"At least {minimum_samples} independent specimen(s) are required."
        )

    if not np.all(np.isfinite(maps_t)):
        raise ValueError(
            "The selected maps contain NaN or infinite values. "
            "Do not replace missing pixels with zero before an FFT."
        )

    return maps_t


def _selected_exclusion_masks(
    exclusion_masks: np.ndarray | None,
    *,
    time_index: int,
    selected_shape: tuple[int, int, int],
) -> np.ndarray:
    """Return boolean masks shaped like one selected time slice."""
    if exclusion_masks is None:
        return np.zeros(selected_shape, dtype=bool)

    masks = np.asarray(exclusion_masks, dtype=bool)
    if masks.ndim == 4:
        if not 0 <= time_index < masks.shape[0]:
            raise IndexError("time_index is outside the exclusion-mask stack.")
        masks = masks[time_index]
    elif masks.ndim != 3:
        raise ValueError(
            "exclusion_masks must have shape (nsamples, n0, n1) or "
            "(ntimes, nsamples, n0, n1)."
        )

    if masks.shape != selected_shape:
        raise ValueError(
            f"exclusion_masks has shape {masks.shape}, expected {selected_shape}."
        )

    valid_count = np.count_nonzero(~masks, axis=(1, 2))
    if np.any(valid_count < 4):
        raise ValueError("Each specimen must retain at least four unmasked pixels.")

    return masks


def _center_search_mask(
    shape: tuple[int, int],
    fraction: float,
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("center_search_fraction must be in (0, 1].")

    n0, n1 = shape
    width0 = max(3, min(n0, int(round(fraction * n0))))
    width1 = max(3, min(n1, int(round(fraction * n1))))
    start0 = (n0 - width0) // 2
    start1 = (n1 - width1) // 2

    search_mask = np.zeros(shape, dtype=bool)
    search_mask[start0 : start0 + width0, start1 : start1 + width1] = True
    return search_mask


def fixed_center_diamond_exclusion_masks(
    height_maps_um: np.ndarray,
    *,
    spacing_um: float | tuple[float, float],
    diagonal_um: float,
    center_offset_um: tuple[float, float] = (0.0, 0.0),
) -> CenterIndentMaskResult:
    """Create one centered Vickers-diamond mask for every specimen and time."""
    maps = np.asarray(height_maps_um)
    if maps.ndim != 4:
        raise ValueError(
            "height_maps_um must have shape (ntimes, nsamples, n0, n1)."
        )

    d0, d1 = _as_spacing_tuple(spacing_um)
    diagonal = float(diagonal_um)
    if diagonal <= 0.0:
        raise ValueError("diagonal_um must be positive.")

    try:
        offset0, offset1 = map(float, center_offset_um)
    except (TypeError, ValueError) as error:
        raise ValueError("center_offset_um must contain two numbers.") from error

    _, nsamples, n0, n1 = maps.shape
    coordinate0 = (np.arange(n0) - 0.5 * (n0 - 1)) * d0
    coordinate1 = (np.arange(n1) - 0.5 * (n1 - 1)) * d1
    distance_to_diamond_centerline = (
        np.abs(coordinate0[:, None] - offset0)
        + np.abs(coordinate1[None, :] - offset1)
    )
    mask_2d = distance_to_diamond_centerline <= 0.5 * diagonal

    if not np.any(mask_2d):
        raise ValueError("The requested center mask contains no pixels.")
    if np.mean(mask_2d) > 0.35:
        raise ValueError("The requested center mask excludes more than 35% of a map.")

    masks = np.broadcast_to(mask_2d, (nsamples, n0, n1)).copy()
    center_index = np.array(
        [
            0.5 * (n0 - 1) + offset0 / d0,
            0.5 * (n1 - 1) + offset1 / d1,
        ],
        dtype=np.float64,
    )
    centers = np.broadcast_to(center_index, (nsamples, 2)).copy()
    area = np.full(
        nsamples,
        np.count_nonzero(mask_2d) * d0 * d1,
        dtype=np.float64,
    )

    return CenterIndentMaskResult(
        excluded_masks=masks,
        detected_center_indices=centers,
        detected_masks=masks.copy(),
        detected_area_um2=area.copy(),
        excluded_area_um2=area,
        guard_um=np.zeros(nsamples, dtype=np.float64),
        center_search_fraction=0.0,
        method="fixed_physical_center_diamond",
        fixed_diagonal_um=diagonal,
    )


def _central_depression_component(
    smoothed_height_um: np.ndarray,
    search_mask: np.ndarray,
    *,
    threshold_sigma: float,
    minimum_component_pixels: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Segment the low connected component around the deepest center seed."""
    if threshold_sigma <= 0.0:
        raise ValueError("indent_threshold_sigma must be positive.")
    if minimum_component_pixels < 1:
        raise ValueError("minimum_component_pixels must be positive.")

    image = np.asarray(smoothed_height_um, dtype=np.float64)
    if image.shape != search_mask.shape:
        raise ValueError("The center search mask does not match the height map.")

    background = image[~search_mask]
    if background.size < 16:
        background = image.ravel()

    baseline = float(np.median(background))
    robust_sigma = 1.4826 * float(np.median(np.abs(background - baseline)))

    search_values = np.where(search_mask, image, np.inf)
    seed_flat = int(np.argmin(search_values))
    seed = tuple(map(int, np.unravel_index(seed_flat, image.shape)))
    seed_height = float(image[seed])
    depression_depth = baseline - seed_height

    numerical_floor = np.finfo(float).eps * max(1.0, abs(baseline), abs(seed_height))
    if not np.isfinite(depression_depth) or depression_depth <= numerical_floor:
        raise ValueError(
            "No central depression was detected. Check the center location or "
            "disable mask_center_indent."
        )

    threshold_drop = np.clip(
        threshold_sigma * robust_sigma,
        0.10 * depression_depth,
        0.60 * depression_depth,
    )
    threshold = baseline - threshold_drop
    candidate = search_mask & (image <= threshold)

    connectivity = generate_binary_structure(2, 2)
    labels, _ = label(candidate, structure=connectivity)
    component_label = int(labels[seed])

    if component_label == 0:
        raise ValueError("The center-indent seed is not in the segmented depression.")

    component = labels == component_label
    if np.count_nonzero(component) < minimum_component_pixels:
        relaxed_threshold = baseline - 0.05 * depression_depth
        relaxed_candidate = search_mask & (image <= relaxed_threshold)
        labels, _ = label(relaxed_candidate, structure=connectivity)
        component_label = int(labels[seed])
        component = labels == component_label

    if np.count_nonzero(component) < minimum_component_pixels:
        raise ValueError(
            "The detected center depression is too small to define a reliable "
            "indent mask. Inspect the mask diagnostic."
        )

    component = binary_closing(component, structure=connectivity, iterations=2)
    component = binary_fill_holes(component)

    if np.count_nonzero(component & ~search_mask):
        raise RuntimeError("Center-indent segmentation escaped its search region.")

    boundary = search_mask & ~binary_erosion(
        search_mask,
        structure=connectivity,
        iterations=1,
        border_value=0,
    )
    if np.any(component & boundary):
        warnings.warn(
            "The detected indent reaches the center search boundary. Increase "
            "center_search_fraction and inspect the saved mask overlay.",
            stacklevel=2,
        )

    return component, seed


def detect_center_indent_masks(
    height_maps_um: np.ndarray,
    *,
    spacing_um: float | tuple[float, float],
    center_search_fraction: float = 0.50,
    detection_smoothing_um: float | None = None,
    indent_threshold_sigma: float = 4.0,
    indent_guard_um: float | None = None,
    guard_fraction_of_equivalent_radius: float = 0.25,
    minimum_component_pixels: int = 16,
    common_mask_across_specimens: bool = True,
) -> CenterIndentMaskResult:
    """
    Attempt to detect a visible central Vickers depression in the height stack.

    A connected low region is detected separately at every time. Its temporal
    union forms one fixed mask for that specimen, so the mask cannot change with
    time. By default, the specimen masks are also unioned into one common mask
    to keep the spatial weighting identical across specimens.

    ``indent_guard_um`` adds a physical guard around the detected depression.
    When omitted, the guard is 25% of its equivalent radius, with a minimum of
    three pixels. This method is not valid when the indent is not visibly
    distinguishable in the height channel. Always inspect the saved overlays;
    a synthetic test cannot validate detection on experimental maps.
    """
    maps = np.asarray(height_maps_um, dtype=np.float64)
    if maps.ndim != 4:
        raise ValueError(
            "height_maps_um must have shape (ntimes, nsamples, n0, n1)."
        )
    if not np.all(np.isfinite(maps)):
        raise ValueError("height_maps_um contains NaN or infinite values.")
    if guard_fraction_of_equivalent_radius < 0.0:
        raise ValueError("guard_fraction_of_equivalent_radius cannot be negative.")
    if indent_guard_um is not None and indent_guard_um < 0.0:
        raise ValueError("indent_guard_um cannot be negative.")

    d0, d1 = _as_spacing_tuple(spacing_um)
    ntimes, nsamples, n0, n1 = maps.shape
    search_mask = _center_search_mask((n0, n1), center_search_fraction)

    if detection_smoothing_um is None:
        detection_smoothing_um = 2.0 * max(d0, d1)
    detection_smoothing_um = float(detection_smoothing_um)
    if detection_smoothing_um <= 0.0:
        raise ValueError("detection_smoothing_um must be positive.")
    smoothing_sigma = (detection_smoothing_um / d0, detection_smoothing_um / d1)

    detected_masks = np.zeros((nsamples, n0, n1), dtype=bool)
    detected_centers = np.empty((nsamples, 2), dtype=np.float64)

    for specimen_index in range(nsamples):
        specimen_union = np.zeros((n0, n1), dtype=bool)
        specimen_seeds: list[tuple[int, int]] = []

        detection_maps = []
        for time_index in range(ntimes):
            leveled = _level_height_map(
                maps[time_index, specimen_index],
                (d0, d1),
                remove_plane=True,
            )
            detection_maps.append(gaussian_filter(leveled, sigma=smoothing_sigma))

        detection_maps.append(np.median(np.stack(detection_maps, axis=0), axis=0))

        for detection_map in detection_maps:
            try:
                component, seed = _central_depression_component(
                    detection_map,
                    search_mask,
                    threshold_sigma=indent_threshold_sigma,
                    minimum_component_pixels=minimum_component_pixels,
                )
            except ValueError:
                continue
            specimen_union |= component
            specimen_seeds.append(seed)

        if not specimen_seeds:
            raise ValueError(
                f"No center indent could be detected for specimen {specimen_index}. "
                "Inspect the map center and detection settings."
            )

        specimen_union = binary_closing(
            specimen_union,
            structure=generate_binary_structure(2, 2),
            iterations=2,
        )
        specimen_union = binary_fill_holes(specimen_union)
        detected_masks[specimen_index] = specimen_union
        detected_centers[specimen_index] = np.median(
            np.asarray(specimen_seeds, dtype=np.float64),
            axis=0,
        )

    if common_mask_across_specimens:
        common_detected = np.any(detected_masks, axis=0)
        common_detected = binary_closing(
            common_detected,
            structure=generate_binary_structure(2, 2),
            iterations=2,
        )
        common_detected = binary_fill_holes(common_detected)
        detected_masks[:] = common_detected

    pixel_area = d0 * d1
    detected_area = np.count_nonzero(detected_masks, axis=(1, 2)) * pixel_area
    map_area = n0 * n1 * pixel_area
    if np.any(detected_area > 0.25 * map_area):
        raise ValueError(
            "Automatic indent detection covers more than 25% of a map. "
            "Reduce center_search_fraction or inspect the source maps."
        )

    excluded_masks = np.empty_like(detected_masks)
    guard_values = np.empty(nsamples, dtype=np.float64)
    for specimen_index, detected_mask in enumerate(detected_masks):
        equivalent_radius_um = np.sqrt(detected_area[specimen_index] / np.pi)
        automatic_guard_um = max(
            3.0 * max(d0, d1),
            guard_fraction_of_equivalent_radius * equivalent_radius_um,
        )
        guard = automatic_guard_um if indent_guard_um is None else indent_guard_um
        guard_values[specimen_index] = guard

        distance_from_detected = distance_transform_edt(
            ~detected_mask,
            sampling=(d0, d1),
        )
        excluded_masks[specimen_index] = distance_from_detected <= guard

    excluded_area = np.count_nonzero(excluded_masks, axis=(1, 2)) * pixel_area
    if np.any(excluded_area > 0.35 * map_area):
        raise ValueError(
            "The guarded indent mask excludes more than 35% of a map. "
            "Reduce indent_guard_um or inspect the saved mask overlay."
        )

    return CenterIndentMaskResult(
        excluded_masks=excluded_masks,
        detected_center_indices=detected_centers,
        detected_masks=detected_masks,
        detected_area_um2=detected_area,
        excluded_area_um2=excluded_area,
        guard_um=guard_values,
        center_search_fraction=float(center_search_fraction),
    )


def _level_height_map(
    height_map_um: np.ndarray,
    spacing_um: tuple[float, float],
    *,
    remove_plane: bool,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Remove a valid-pixel mean or least-squares plane."""
    height_map = np.asarray(height_map_um, dtype=np.float64)

    if height_map.ndim != 2:
        raise ValueError("height_map_um must be a two-dimensional array.")
    if min(height_map.shape) < 2:
        raise ValueError("Each height-map axis must contain at least two pixels.")
    if not np.all(np.isfinite(height_map)):
        raise ValueError("The height map contains NaN or infinite values.")

    if valid_mask is None:
        valid = np.ones(height_map.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != height_map.shape:
            raise ValueError("valid_mask must have the same shape as height_map_um.")
        if np.count_nonzero(valid) < 4:
            raise ValueError("At least four valid pixels are required.")

    centered_map = height_map - np.mean(height_map[valid])
    if not remove_plane:
        return np.where(valid, centered_map, 0.0)

    n0, n1 = height_map.shape
    d0, d1 = spacing_um
    x0 = (np.arange(n0) - 0.5 * (n0 - 1)) * d0
    x1 = (np.arange(n1) - 0.5 * (n1 - 1)) * d1

    coordinate0 = np.broadcast_to(x0[:, None], height_map.shape)[valid]
    coordinate1 = np.broadcast_to(x1[None, :], height_map.shape)[valid]
    design = np.column_stack(
        [
            np.ones(np.count_nonzero(valid), dtype=np.float64),
            coordinate0,
            coordinate1,
        ]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, height_map[valid], rcond=None)
    if rank < 3:
        raise ValueError("Valid pixels do not support a unique best-fit plane.")

    plane = (
        coefficients[0]
        + coefficients[1] * x0[:, None]
        + coefficients[2] * x1[None, :]
    )
    return np.where(valid, height_map - plane, 0.0)


def normalized_autocorrelation_2d(
    height_map_um: np.ndarray,
    *,
    spacing_um: float | tuple[float, float] = 1.3,
    remove_plane: bool = False,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the normalized, overlap-corrected linear 2D autocorrelation.

    No tapering window is applied because it would alter the correlation
    length. Zero padding through linear convolution avoids circular wraparound.
    The autocovariance is divided by the valid-pixel pair count at every lag,
    which handles both finite-map overlap and an optional exclusion mask.

    Returns
    -------
    normalized_acf
        Shape ``(2*n0 - 1, 2*n1 - 1)`` with zero lag at its center.
    lag_axis0_um, lag_axis1_um
        Physical lag coordinates for the two array axes.
    """
    d0, d1 = _as_spacing_tuple(spacing_um)
    height_map = np.asarray(height_map_um, dtype=np.float64)
    if exclusion_mask is None:
        valid = np.ones(height_map.shape, dtype=bool)
    else:
        mask = np.asarray(exclusion_mask, dtype=bool)
        if mask.shape != height_map.shape:
            raise ValueError(
                "exclusion_mask must have the same shape as height_map_um."
            )
        valid = ~mask

    leveled_map = _level_height_map(
        height_map,
        (d0, d1),
        remove_plane=remove_plane,
        valid_mask=valid,
    )
    n0, n1 = leveled_map.shape
    reference_variance = float(np.mean(leveled_map[valid] ** 2))

    correlation_sum = fftconvolve(
        leveled_map,
        leveled_map[::-1, ::-1],
        mode="full",
    )
    valid_float = valid.astype(np.float64)
    valid_pair_count = fftconvolve(
        valid_float,
        valid_float[::-1, ::-1],
        mode="full",
    )
    valid_pair_count = np.rint(np.maximum(valid_pair_count, 0.0))

    lag_index0 = np.arange(-(n0 - 1), n0)
    lag_index1 = np.arange(-(n1 - 1), n1)
    autocovariance = np.full(correlation_sum.shape, np.nan, dtype=np.float64)
    supported = valid_pair_count > 0.0
    autocovariance[supported] = (
        correlation_sum[supported] / valid_pair_count[supported]
    )

    center = (n0 - 1, n1 - 1)
    zero_lag = float(autocovariance[center])
    variance_floor = np.finfo(float).eps * reference_variance
    if (
        not np.isfinite(zero_lag)
        or reference_variance <= np.finfo(float).tiny
        or zero_lag <= variance_floor
    ):
        raise ValueError(
            "The leveled height map has zero variance, so its autocorrelation "
            "length is undefined."
        )

    normalized_acf = autocovariance / zero_lag
    normalized_acf = 0.5 * (normalized_acf + normalized_acf[::-1, ::-1])
    normalized_acf[center] = 1.0

    return normalized_acf, lag_index0 * d0, lag_index1 * d1


def gaussian_areal_roughness_surface(
    sf_height_map_um: np.ndarray,
    *,
    spacing_um: float | tuple[float, float] = 1.3,
    l_filter_cutoff_um: float,
    remove_plane: bool = False,
    exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construct an S-L roughness surface using an areal Gaussian L-filter.

    The input must already be an outlier-corrected S-F surface. The Gaussian
    low-pass result is the waviness surface, which is subtracted from the S-F
    surface. One cutoff wavelength is removed from every edge after filtering,
    matching the evaluation-area treatment used by NPL's areal reference
    software.

    This function enforces the reference-software sampling conditions: at least
    50 points per cutoff, a field span of at least three cutoffs along each axis,
    and a cutoff that is an integer multiple of each sampling interval.
    """
    d0, d1 = _as_spacing_tuple(spacing_um)
    cutoff = float(l_filter_cutoff_um)
    if cutoff <= 0.0:
        raise ValueError("l_filter_cutoff_um must be positive.")

    points_per_cutoff0 = cutoff / d0
    points_per_cutoff1 = cutoff / d1
    integer_points0 = int(round(points_per_cutoff0))
    integer_points1 = int(round(points_per_cutoff1))

    if not np.isclose(points_per_cutoff0, integer_points0, rtol=0.0, atol=1e-9):
        raise ValueError(
            "l_filter_cutoff_um must be an integer multiple of axis-0 spacing."
        )
    if not np.isclose(points_per_cutoff1, integer_points1, rtol=0.0, atol=1e-9):
        raise ValueError(
            "l_filter_cutoff_um must be an integer multiple of axis-1 spacing."
        )
    if min(integer_points0, integer_points1) < 50:
        raise ValueError(
            "The L-filter cutoff must contain at least 50 sampled points along "
            "each axis."
        )

    height_map = np.asarray(sf_height_map_um, dtype=np.float64)
    if exclusion_mask is None:
        valid = np.ones(height_map.shape, dtype=bool)
    else:
        mask = np.asarray(exclusion_mask, dtype=bool)
        if mask.shape != height_map.shape:
            raise ValueError(
                "exclusion_mask must have the same shape as sf_height_map_um."
            )
        valid = ~mask

    leveled_map = _level_height_map(
        height_map,
        (d0, d1),
        remove_plane=remove_plane,
        valid_mask=valid,
    )
    n0, n1 = leveled_map.shape
    span0 = (n0 - 1) * d0
    span1 = (n1 - 1) * d1
    if span0 < 3.0 * cutoff or span1 < 3.0 * cutoff:
        raise ValueError(
            "Each map span must be at least three times l_filter_cutoff_um."
        )

    alpha = np.sqrt(np.log(2.0) / np.pi)
    sigma_um = alpha * cutoff / np.sqrt(2.0 * np.pi)
    sigma_pixels = (sigma_um / d0, sigma_um / d1)
    truncate = np.sqrt(2.0 * np.pi) / alpha

    if exclusion_mask is None:
        waviness_surface = gaussian_filter(
            leveled_map,
            sigma=sigma_pixels,
            mode="nearest",
            truncate=truncate,
        )
    else:
        valid_float = valid.astype(np.float64)
        weighted_waviness = gaussian_filter(
            leveled_map,
            sigma=sigma_pixels,
            mode="nearest",
            truncate=truncate,
        )
        local_weight = gaussian_filter(
            valid_float,
            sigma=sigma_pixels,
            mode="nearest",
            truncate=truncate,
        )
        waviness_surface = np.zeros_like(leveled_map)
        supported = local_weight > np.finfo(float).eps
        waviness_surface[supported] = (
            weighted_waviness[supported] / local_weight[supported]
        )

    roughness_surface = np.where(valid, leveled_map - waviness_surface, 0.0)

    return roughness_surface[
        integer_points0 : n0 - integer_points0,
        integer_points1 : n1 - integer_points1,
    ]


def _bilinear_sample(
    values: np.ndarray,
    row: np.ndarray,
    column: np.ndarray,
) -> np.ndarray:
    row = np.clip(row, 0.0, values.shape[0] - 1.0)
    column = np.clip(column, 0.0, values.shape[1] - 1.0)

    row0 = np.floor(row).astype(int)
    column0 = np.floor(column).astype(int)
    row1 = np.minimum(row0 + 1, values.shape[0] - 1)
    column1 = np.minimum(column0 + 1, values.shape[1] - 1)

    row_fraction = row - row0
    column_fraction = column - column0

    return (
        values[row0, column0]
        * (1.0 - row_fraction)
        * (1.0 - column_fraction)
        + values[row1, column0] * row_fraction * (1.0 - column_fraction)
        + values[row0, column1] * (1.0 - row_fraction) * column_fraction
        + values[row1, column1] * row_fraction * column_fraction
    )


def _acf_decay_metrics(
    normalized_acf: np.ndarray,
    spacing_um: tuple[float, float],
    *,
    threshold: float,
    n_directions: int,
    radial_step_um: float,
    maximum_lag_fraction: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """Find fastest and slowest threshold crossings over line orientations."""
    d0, d1 = spacing_um
    center0 = 0.5 * (normalized_acf.shape[0] - 1)
    center1 = 0.5 * (normalized_acf.shape[1] - 1)
    maximum_lag0 = maximum_lag_fraction * center0 * d0
    maximum_lag1 = maximum_lag_fraction * center1 * d1

    angles = np.linspace(0.0, np.pi, n_directions, endpoint=False)
    crossing_lengths = np.full(n_directions, np.nan, dtype=np.float64)

    for direction_index, angle in enumerate(angles):
        axis0_component = abs(np.sin(angle))
        axis1_component = abs(np.cos(angle))
        radial_limit0 = (
            maximum_lag0 / axis0_component
            if axis0_component > np.finfo(float).eps
            else np.inf
        )
        radial_limit1 = (
            maximum_lag1 / axis1_component
            if axis1_component > np.finfo(float).eps
            else np.inf
        )
        radial_limit = min(radial_limit0, radial_limit1)
        number_steps = max(1, int(np.ceil(radial_limit / radial_step_um)))
        radius = np.linspace(0.0, radial_limit, number_steps + 1)

        row = center0 + radius * np.sin(angle) / d0
        column = center1 + radius * np.cos(angle) / d1
        directional_acf = _bilinear_sample(normalized_acf, row, column)

        crossing_candidates = np.flatnonzero(directional_acf <= threshold)
        if crossing_candidates.size == 0:
            continue

        upper_index = int(crossing_candidates[0])
        if upper_index == 0:
            crossing_lengths[direction_index] = 0.0
            continue

        lower_index = upper_index - 1
        lower_value = directional_acf[lower_index]
        upper_value = directional_acf[upper_index]
        value_change = upper_value - lower_value

        if value_change == 0.0:
            crossing_lengths[direction_index] = radius[upper_index]
        else:
            fraction = (threshold - lower_value) / value_change
            crossing_lengths[direction_index] = (
                radius[lower_index]
                + fraction * (radius[upper_index] - radius[lower_index])
            )

    finite_direction = np.isfinite(crossing_lengths)
    if not np.any(finite_direction):
        raise ValueError(
            "The ACF does not fall to the requested threshold within the map. "
            "A larger field of view or a larger maximum_lag_fraction is required "
            "to resolve Sal."
        )

    fastest_index = int(np.nanargmin(crossing_lengths))
    sal_um = float(crossing_lengths[fastest_index])
    fastest_angle_deg = float(np.degrees(angles[fastest_index]))

    if np.all(finite_direction):
        slowest_index = int(np.nanargmax(crossing_lengths))
        slowest_decay_um = float(crossing_lengths[slowest_index])
        slowest_angle_deg = float(np.degrees(angles[slowest_index]))
        texture_aspect_ratio = sal_um / slowest_decay_um
    else:
        slowest_decay_um = np.nan
        slowest_angle_deg = np.nan
        texture_aspect_ratio = np.nan

    return (
        sal_um,
        fastest_angle_deg,
        slowest_decay_um,
        slowest_angle_deg,
        texture_aspect_ratio,
    )


def autocorrelation_lengths_at_time(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    spacing_um: float | tuple[float, float] = 1.3,
    threshold: float = 0.2,
    remove_plane: bool = False,
    n_directions: int = 720,
    radial_step_um: float | None = None,
    maximum_lag_fraction: float = 0.5,
    exclusion_masks: np.ndarray | None = None,
) -> AutocorrelationLengthResult:
    """
    Calculate ISO 25178-style Sal and Str for every specimen at one time.

    Sal is the shortest physical lag at which the normalized 2D ACF reaches
    ``threshold``. The default threshold of 0.2 is the conventional ISO value.
    Str is the ratio of fastest to slowest threshold-decay lengths. Angles are
    line orientations modulo 180 degrees, measured from array axis 1 toward
    array axis 0.

    The input should be the same outlier-corrected, scale-limited surface for
    every time and specimen. Set ``remove_plane=True`` only when a best-fit
    plane has not already been removed upstream. The default search is limited
    to half the map span on each axis to reject poorly supported edge lags.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one.")
    if n_directions < 4:
        raise ValueError("n_directions must be at least four.")
    if not 0.0 < maximum_lag_fraction <= 1.0:
        raise ValueError("maximum_lag_fraction must be in (0, 1].")

    d0, d1 = _as_spacing_tuple(spacing_um)
    if radial_step_um is None:
        radial_step_um = 0.25 * min(d0, d1)
    radial_step_um = float(radial_step_um)
    if radial_step_um <= 0.0:
        raise ValueError("radial_step_um must be positive.")

    maps_t = _selected_time_maps(
        height_maps_um,
        time_index,
        minimum_samples=1,
    )
    masks_t = _selected_exclusion_masks(
        exclusion_masks,
        time_index=time_index,
        selected_shape=maps_t.shape,
    )
    nsamples = maps_t.shape[0]
    sal_um = np.empty(nsamples, dtype=np.float64)
    fastest_angle_deg = np.empty(nsamples, dtype=np.float64)
    slowest_decay_um = np.empty(nsamples, dtype=np.float64)
    slowest_angle_deg = np.empty(nsamples, dtype=np.float64)
    texture_aspect_ratio = np.empty(nsamples, dtype=np.float64)

    for specimen_index, height_map in enumerate(maps_t):
        normalized_acf, _, _ = normalized_autocorrelation_2d(
            height_map,
            spacing_um=(d0, d1),
            remove_plane=remove_plane,
            exclusion_mask=masks_t[specimen_index],
        )
        metrics = _acf_decay_metrics(
            normalized_acf,
            (d0, d1),
            threshold=threshold,
            n_directions=n_directions,
            radial_step_um=radial_step_um,
            maximum_lag_fraction=maximum_lag_fraction,
        )
        (
            sal_um[specimen_index],
            fastest_angle_deg[specimen_index],
            slowest_decay_um[specimen_index],
            slowest_angle_deg[specimen_index],
            texture_aspect_ratio[specimen_index],
        ) = metrics

    return AutocorrelationLengthResult(
        threshold=threshold,
        specimen_sal_um=sal_um,
        specimen_fastest_decay_angle_deg=fastest_angle_deg,
        specimen_slowest_decay_um=slowest_decay_um,
        specimen_slowest_decay_angle_deg=slowest_angle_deg,
        specimen_texture_aspect_ratio=texture_aspect_ratio,
    )


def roughness_autocorrelation_lengths_at_time(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    spacing_um: float | tuple[float, float] = 1.3,
    l_filter_cutoff_um: float,
    threshold: float = 0.2,
    remove_plane: bool = False,
    n_directions: int = 720,
    radial_step_um: float | None = None,
    maximum_lag_fraction: float = 0.5,
    exclusion_masks: np.ndarray | None = None,
) -> AutocorrelationLengthResult:
    """Calculate Sal and Str on Gaussian-filtered S-L roughness surfaces."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one.")
    if n_directions < 4:
        raise ValueError("n_directions must be at least four.")
    if not 0.0 < maximum_lag_fraction <= 1.0:
        raise ValueError("maximum_lag_fraction must be in (0, 1].")

    d0, d1 = _as_spacing_tuple(spacing_um)
    if radial_step_um is None:
        radial_step_um = 0.25 * min(d0, d1)
    radial_step_um = float(radial_step_um)
    if radial_step_um <= 0.0:
        raise ValueError("radial_step_um must be positive.")

    maps_t = _selected_time_maps(
        height_maps_um,
        time_index,
        minimum_samples=1,
    )
    masks_t = _selected_exclusion_masks(
        exclusion_masks,
        time_index=time_index,
        selected_shape=maps_t.shape,
    )
    crop0 = int(round(float(l_filter_cutoff_um) / d0))
    crop1 = int(round(float(l_filter_cutoff_um) / d1))
    nsamples = maps_t.shape[0]
    sal_um = np.empty(nsamples, dtype=np.float64)
    fastest_angle_deg = np.empty(nsamples, dtype=np.float64)
    slowest_decay_um = np.empty(nsamples, dtype=np.float64)
    slowest_angle_deg = np.empty(nsamples, dtype=np.float64)
    texture_aspect_ratio = np.empty(nsamples, dtype=np.float64)

    for specimen_index, height_map in enumerate(maps_t):
        roughness_surface = gaussian_areal_roughness_surface(
            height_map,
            spacing_um=(d0, d1),
            l_filter_cutoff_um=l_filter_cutoff_um,
            remove_plane=remove_plane,
            exclusion_mask=masks_t[specimen_index],
        )
        cropped_mask = masks_t[specimen_index][
            crop0 : maps_t.shape[1] - crop0,
            crop1 : maps_t.shape[2] - crop1,
        ]
        normalized_acf, _, _ = normalized_autocorrelation_2d(
            roughness_surface,
            spacing_um=(d0, d1),
            exclusion_mask=cropped_mask,
        )
        metrics = _acf_decay_metrics(
            normalized_acf,
            (d0, d1),
            threshold=threshold,
            n_directions=n_directions,
            radial_step_um=radial_step_um,
            maximum_lag_fraction=maximum_lag_fraction,
        )
        (
            sal_um[specimen_index],
            fastest_angle_deg[specimen_index],
            slowest_decay_um[specimen_index],
            slowest_angle_deg[specimen_index],
            texture_aspect_ratio[specimen_index],
        ) = metrics

    return AutocorrelationLengthResult(
        threshold=threshold,
        specimen_sal_um=sal_um,
        specimen_fastest_decay_angle_deg=fastest_angle_deg,
        specimen_slowest_decay_um=slowest_decay_um,
        specimen_slowest_decay_angle_deg=slowest_angle_deg,
        specimen_texture_aspect_ratio=texture_aspect_ratio,
        surface_type="S-L roughness",
        l_filter_cutoff_um=float(l_filter_cutoff_um),
    )


def calculate_acf_analysis_at_time(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    spacing_um: float | tuple[float, float],
    threshold: float = 0.2,
    n_directions: int = 720,
    radial_step_um: float | None = None,
    maximum_lag_fraction: float = 0.5,
    exclusion_masks: np.ndarray | None = None,
    use_median: bool = False,
) -> ACFAnalysisResult:
    """Calculate specimen metrics and an aggregate overlap-corrected 2D ACF."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one.")
    if n_directions < 4:
        raise ValueError("n_directions must be at least four.")
    if not 0.0 < maximum_lag_fraction <= 1.0:
        raise ValueError("maximum_lag_fraction must be in (0, 1].")

    d0, d1 = _as_spacing_tuple(spacing_um)
    if radial_step_um is None:
        radial_step_um = 0.25 * min(d0, d1)
    radial_step_um = float(radial_step_um)
    if radial_step_um <= 0.0:
        raise ValueError("radial_step_um must be positive.")

    maps_t = _selected_time_maps(
        height_maps_um,
        time_index,
        minimum_samples=1,
    )
    masks_t = _selected_exclusion_masks(
        exclusion_masks,
        time_index=time_index,
        selected_shape=maps_t.shape,
    )
    nsamples = maps_t.shape[0]
    metric_values = np.full((nsamples, 5), np.nan, dtype=np.float64)
    specimen_acfs = []
    lag0 = lag1 = None

    for specimen_index, height_map in enumerate(maps_t):
        normalized_acf, lag0, lag1 = normalized_autocorrelation_2d(
            height_map,
            spacing_um=(d0, d1),
            remove_plane=False,
            exclusion_mask=masks_t[specimen_index],
        )
        specimen_acfs.append(normalized_acf)
        try:
            metric_values[specimen_index] = _acf_decay_metrics(
                normalized_acf,
                (d0, d1),
                threshold=threshold,
                n_directions=n_directions,
                radial_step_um=radial_step_um,
                maximum_lag_fraction=maximum_lag_fraction,
            )
        except ValueError as error:
            warnings.warn(
                f"Specimen {specimen_index} ACF length is unresolved: {error}",
                stacklevel=2,
            )

    estimator_name, estimator = _specimen_estimator(use_median)
    aggregate_acf = estimator(np.stack(specimen_acfs, axis=0), axis=0)
    lengths = AutocorrelationLengthResult(
        threshold=threshold,
        specimen_sal_um=metric_values[:, 0],
        specimen_fastest_decay_angle_deg=metric_values[:, 1],
        specimen_slowest_decay_um=metric_values[:, 2],
        specimen_slowest_decay_angle_deg=metric_values[:, 3],
        specimen_texture_aspect_ratio=metric_values[:, 4],
    )
    if lag0 is None or lag1 is None:
        raise RuntimeError("No ACF was calculated.")

    return ACFAnalysisResult(
        lengths=lengths,
        aggregate_acf=aggregate_acf,
        lag_axis0_um=lag0,
        lag_axis1_um=lag1,
        estimator=estimator_name,
    )


def surface_roughness_parameters_at_time(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    spacing_um: float | tuple[float, float],
    exclusion_masks: np.ndarray | None = None,
) -> SurfaceRoughnessResult:
    """Calculate common areal amplitude, distribution, and slope parameters."""
    parameter_names = (
        "Sa_um",
        "Sq_um",
        "Ssk",
        "Sku",
        "Sp_um",
        "Sv_um",
        "Sz_um",
        "Sdq",
        "Sdr_percent",
    )
    d0, d1 = _as_spacing_tuple(spacing_um)
    maps_t = _selected_time_maps(
        height_maps_um,
        time_index,
        minimum_samples=1,
    )
    masks_t = _selected_exclusion_masks(
        exclusion_masks,
        time_index=time_index,
        selected_shape=maps_t.shape,
    )
    values = np.empty((maps_t.shape[0], len(parameter_names)), dtype=np.float64)

    for specimen_index, height_map in enumerate(maps_t):
        valid = ~masks_t[specimen_index]
        centered = height_map - np.mean(height_map[valid])
        valid_heights = centered[valid]
        sq = float(np.sqrt(np.mean(valid_heights**2)))
        if sq <= np.finfo(float).tiny:
            raise ValueError(
                f"Specimen {specimen_index} has zero valid-pixel variance."
            )

        gradient0, gradient1 = np.gradient(
            centered,
            d0,
            d1,
            edge_order=2,
        )
        gradient_valid = binary_erosion(
            valid,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
        if np.count_nonzero(gradient_valid) < 4:
            raise ValueError("Too few valid pixels remain for slope parameters.")
        gradient_squared = (
            gradient0[gradient_valid] ** 2
            + gradient1[gradient_valid] ** 2
        )

        sp = float(np.max(valid_heights))
        sv = float(-np.min(valid_heights))
        values[specimen_index] = (
            np.mean(np.abs(valid_heights)),
            sq,
            np.mean(valid_heights**3) / sq**3,
            np.mean(valid_heights**4) / sq**4,
            sp,
            sv,
            sp + sv,
            np.sqrt(np.mean(gradient_squared)),
            100.0 * np.mean(np.sqrt(1.0 + gradient_squared) - 1.0),
        )

    return SurfaceRoughnessResult(
        parameter_names=parameter_names,
        specimen_values=values,
    )


def _rms_normalized_radial_hann(
    shape: tuple[int, int],
    spacing_um: tuple[float, float],
) -> np.ndarray:
    """
    Construct an RMS-normalized circular Hann window.

    The circular support is the largest physical circle that fits inside
    the rectangular map. The corners are therefore zero.
    """
    n0, n1 = shape
    d0, d1 = spacing_um

    x0 = (np.arange(n0) + 0.5 - n0 / 2.0) * d0
    x1 = (np.arange(n1) + 0.5 - n1 / 2.0) * d1

    radius = np.hypot(x0[:, None], x1[None, :])
    support_radius = 0.5 * min(n0 * d0, n1 * d1)

    window = np.zeros((n0, n1), dtype=np.float64)
    inside = radius < support_radius

    window[inside] = 0.5 * (
        1.0 + np.cos(np.pi * radius[inside] / support_radius)
    )

    window /= np.sqrt(np.mean(window**2))

    return window


def _masked_psd_window(
    base_window: np.ndarray,
    exclusion_mask: np.ndarray,
    spacing_um: tuple[float, float],
    *,
    mask_taper_um: float,
) -> tuple[np.ndarray, float]:
    """Add a raised-cosine taper around an excluded region."""
    mask = np.asarray(exclusion_mask, dtype=bool)
    if mask.shape != base_window.shape:
        raise ValueError("exclusion_mask does not match the PSD window shape.")
    if mask_taper_um < 0.0:
        raise ValueError("mask_taper_um cannot be negative.")

    if not np.any(mask):
        return np.array(base_window, copy=True), 1.0

    valid = ~mask
    if mask_taper_um == 0.0:
        exclusion_taper = valid.astype(np.float64)
    else:
        distance_from_mask = distance_transform_edt(
            valid,
            sampling=spacing_um,
        )
        phase = np.clip(distance_from_mask / mask_taper_um, 0.0, 1.0)
        exclusion_taper = np.sin(0.5 * np.pi * phase) ** 2
        exclusion_taper[mask] = 0.0

    combined_window = base_window * exclusion_taper
    rms = float(np.sqrt(np.mean(combined_window**2)))
    if not np.isfinite(rms) or rms <= np.finfo(float).tiny:
        raise ValueError("The exclusion mask leaves no usable PSD window support.")

    combined_window /= rms
    taper_effective_fraction = float(np.mean(exclusion_taper**2))
    return combined_window, taper_effective_fraction


def calculate_2d_psd_at_time(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    spacing_um: float | tuple[float, float] = 1.3,
    use_median: bool = False,
    exclusion_masks: np.ndarray | None = None,
    mask_taper_um: float | None = None,
) -> PSD2DResult:
    """
    Calculate the full two-dimensional PSD for every specimen at one time.

    Parameters
    ----------
    height_maps_um
        Array with shape:

            (ntimes, nsamples, n0, n1)

        Heights must be in micrometers for the returned PSD to have units
        of micrometers^4.

    time_index
        Time index to analyze.

    spacing_um
        Pixel spacing along the two spatial array axes.

        A scalar means equal spacing on both axes. A tuple is ordered as:

            (spacing along axis 2, spacing along axis 3)

        of the original four-dimensional array.

    use_median
        Use the pointwise specimen median for the aggregate 2D PSD. Both the
        mean and median are retained in the returned result.

    exclusion_masks
        Boolean center-indent masks. True pixels are excluded. A fixed mask
        with shape ``(nsamples, n0, n1)`` is preferred for longitudinal data.

    mask_taper_um
        Width of the smooth FFT taper outside an exclusion mask. The default is
        five pixels. A hard-edged mask is allowed with zero but is not advised.
    """
    maps_t = _selected_time_maps(height_maps_um, time_index)
    d0, d1 = _as_spacing_tuple(spacing_um)

    nsamples, n0, n1 = maps_t.shape
    masks_t = _selected_exclusion_masks(
        exclusion_masks,
        time_index=time_index,
        selected_shape=maps_t.shape,
    )
    base_window = _rms_normalized_radial_hann((n0, n1), (d0, d1))
    if mask_taper_um is None:
        mask_taper_um = 5.0 * max(d0, d1)
    mask_taper_um = float(mask_taper_um)

    f0 = np.fft.fftfreq(n0, d=d0)
    f1 = np.fft.fftfreq(n1, d=d1)
    df0 = 1.0 / (n0 * d0)
    df1 = 1.0 / (n1 * d1)

    pixel_area = d0 * d1
    n_pixels = n0 * n1

    specimen_psd = np.empty((nsamples, n0, n1), dtype=np.float64)
    parseval_error = np.empty(nsamples, dtype=np.float64)
    excluded_fraction = np.mean(masks_t, axis=(1, 2))
    taper_effective_fraction = np.empty(nsamples, dtype=np.float64)

    for specimen_index, height_map in enumerate(maps_t):
        window, taper_effective_fraction[specimen_index] = _masked_psd_window(
            base_window,
            masks_t[specimen_index],
            (d0, d1),
            mask_taper_um=mask_taper_um,
        )
        squared_window = window**2
        weighted_mean = np.sum(squared_window * height_map) / np.sum(squared_window)
        centered_map = height_map - weighted_mean
        windowed_map = window * centered_map
        fft_height = np.fft.fft2(windowed_map)

        psd_2d = pixel_area / n_pixels * np.abs(fft_height) ** 2
        specimen_psd[specimen_index] = psd_2d

        mean_square_real = np.mean(windowed_map**2)
        mean_square_from_psd = np.sum(psd_2d) * df0 * df1
        denominator = max(mean_square_real, np.finfo(float).tiny)

        parseval_error[specimen_index] = (
            abs(mean_square_from_psd - mean_square_real) / denominator
        )

    estimator_name, _ = _specimen_estimator(use_median)

    return PSD2DResult(
        frequency_axis0_um_inv=f0,
        frequency_axis1_um_inv=f1,
        specimen_psd_um4=specimen_psd,
        mean_psd_um4=np.mean(specimen_psd, axis=0),
        parseval_relative_error=parseval_error,
        excluded_fraction=excluded_fraction,
        taper_effective_fraction=taper_effective_fraction,
        median_psd_um4=np.median(specimen_psd, axis=0),
        estimator=estimator_name,
    )


def _bootstrap_keyword_args(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    signature = inspect.signature(bootstrap)

    if "rng" in signature.parameters:
        return {"rng": rng}

    return {"random_state": rng}


def _radial_average_from_2d_psd(
    full_2d: PSD2DResult,
    *,
    spacing_um: float | tuple[float, float],
    min_modes_per_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency, specimen radial PSDs, and mode counts."""
    f0 = full_2d.frequency_axis0_um_inv
    f1 = full_2d.frequency_axis1_um_inv
    nsamples, n0, n1 = full_2d.specimen_psd_um4.shape
    d0, d1 = _as_spacing_tuple(spacing_um)

    radial_frequency = np.hypot(f0[:, None], f1[None, :])
    df0 = 1.0 / (n0 * d0)
    df1 = 1.0 / (n1 * d1)
    bin_width = max(df0, df1)
    full_annulus_limit = min(0.5 / d0, 0.5 / d1)
    edges = np.arange(
        0.5 * bin_width,
        full_annulus_limit + np.finfo(float).eps * full_annulus_limit,
        bin_width,
    )
    if edges.size < 2:
        raise ValueError("The map is too small to construct radial-frequency bins.")

    number_bins = edges.size - 1
    radial_flat = radial_frequency.ravel()
    bin_index = np.searchsorted(edges, radial_flat, side="right") - 1
    valid_mode = (bin_index >= 0) & (bin_index < number_bins)
    valid_bin_index = bin_index[valid_mode]
    modes_per_bin = np.bincount(valid_bin_index, minlength=number_bins)
    frequency_sum = np.bincount(
        valid_bin_index,
        weights=radial_flat[valid_mode],
        minlength=number_bins,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        bin_frequency = frequency_sum / modes_per_bin

    retained = (
        (modes_per_bin >= int(min_modes_per_bin))
        & np.isfinite(bin_frequency)
        & (bin_frequency > 0.0)
    )
    if not np.any(retained):
        raise ValueError("No radial bins satisfy min_modes_per_bin.")

    specimen_psd = np.empty(
        (nsamples, int(np.count_nonzero(retained))),
        dtype=np.float64,
    )
    for specimen_index, psd_2d in enumerate(full_2d.specimen_psd_um4):
        annular_sum = np.bincount(
            valid_bin_index,
            weights=psd_2d.ravel()[valid_mode],
            minlength=number_bins,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            radial_psd = annular_sum / modes_per_bin
        specimen_psd[specimen_index] = radial_psd[retained]

    return (
        bin_frequency[retained],
        specimen_psd,
        modes_per_bin[retained],
    )


def radial_psd_at_time(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    spacing_um: float | tuple[float, float] = 1.3,
    confidence_level: float = 0.95,
    n_resamples: int = 20_000,
    bootstrap_method: str = "BCa",
    min_modes_per_bin: int = 8,
    seed: int = 12345,
    use_median: bool = False,
    exclusion_masks: np.ndarray | None = None,
    mask_taper_um: float | None = None,
    _full_2d: PSD2DResult | None = None,
) -> RadialPSDResult:
    """
    Calculate specimen-level radial PSDs and a bootstrap confidence interval.

    Parameters
    ----------
    height_maps_um
        Array with shape:

            (ntimes, nsamples, n0, n1)

        Heights must be in micrometers for the returned PSD to have units
        of micrometers^4.

    time_index
        Time index to analyze.

    spacing_um
        Pixel spacing along the two spatial array axes.

        A scalar means equal spacing on both axes. A tuple is ordered as:

            (spacing along axis 2, spacing along axis 3)

        of the original four-dimensional array.

    confidence_level
        Bootstrap confidence level.

    n_resamples
        Number of specimen-level bootstrap resamples.

    bootstrap_method
        Method accepted by scipy.stats.bootstrap. "BCa" is the default.

    min_modes_per_bin
        Discard radial bins containing fewer Fourier modes than this.

    seed
        Random seed for reproducibility.

    use_median
        Bootstrap and plot the specimen median instead of the specimen mean.
        Individual specimen PSD calculations are unchanged.

    exclusion_masks, mask_taper_um
        Optional fixed center-indent masks and their smooth FFT taper width.
    """
    if _full_2d is None:
        full_2d = calculate_2d_psd_at_time(
            height_maps_um,
            time_index,
            spacing_um=spacing_um,
            use_median=use_median,
            exclusion_masks=exclusion_masks,
            mask_taper_um=mask_taper_um,
        )
    else:
        full_2d = _full_2d

    retained_frequency, specimen_psd, retained_modes = (
        _radial_average_from_2d_psd(
            full_2d,
            spacing_um=spacing_um,
            min_modes_per_bin=min_modes_per_bin,
        )
    )

    estimator_name, estimator = _specimen_estimator(use_median)

    def specimen_estimate(sample: np.ndarray, axis: int = 0) -> np.ndarray:
        return estimator(sample, axis=axis)

    bootstrap_result = bootstrap(
        (specimen_psd,),
        specimen_estimate,
        axis=0,
        vectorized=True,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        batch=min(1000, n_resamples),
        method=bootstrap_method,
        **_bootstrap_keyword_args(seed),
    )

    return RadialPSDResult(
        frequency_um_inv=retained_frequency,
        wavelength_um=1.0 / retained_frequency,
        specimen_psd_um4=specimen_psd,
        mean_psd_um4=np.mean(specimen_psd, axis=0),
        ci_low_um4=np.asarray(bootstrap_result.confidence_interval.low),
        ci_high_um4=np.asarray(bootstrap_result.confidence_interval.high),
        modes_per_bin=retained_modes,
        parseval_relative_error=full_2d.parseval_relative_error,
        median_psd_um4=np.median(specimen_psd, axis=0),
        estimator=estimator_name,
    )


def _integrated_line_psds_on_radial_grid(
    full_2d: PSD2DResult,
    radial_frequency_um_inv: np.ndarray,
    *,
    spacing_um: float | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the 2D PSD transversely and interpolate both 1D PSDs."""
    d0, d1 = _as_spacing_tuple(spacing_um)
    f0 = full_2d.frequency_axis0_um_inv
    f1 = full_2d.frequency_axis1_um_inv
    df0 = 1.0 / (f0.size * d0)
    df1 = 1.0 / (f1.size * d1)
    psd_2d = full_2d.specimen_psd_um4

    # C1D(f0) = integral C2D(f0, f1) df1, and conversely for f1.
    line_axis0 = np.sum(psd_2d, axis=2) * df1
    line_axis1 = np.sum(psd_2d, axis=1) * df0

    def symmetrize_and_interpolate(
        frequency: np.ndarray,
        line_psd: np.ndarray,
    ) -> np.ndarray:
        absolute_frequency = np.abs(frequency)
        unique_frequency, inverse = np.unique(
            absolute_frequency,
            return_inverse=True,
        )
        counts = np.bincount(inverse).astype(np.float64)
        symmetric = np.empty(
            (line_psd.shape[0], unique_frequency.size),
            dtype=np.float64,
        )
        for specimen_index, curve in enumerate(line_psd):
            symmetric[specimen_index] = (
                np.bincount(inverse, weights=curve) / counts
            )

        interpolated = np.empty(
            (line_psd.shape[0], radial_frequency_um_inv.size),
            dtype=np.float64,
        )
        for specimen_index, curve in enumerate(symmetric):
            interpolated[specimen_index] = np.interp(
                radial_frequency_um_inv,
                unique_frequency,
                curve,
            )
        return interpolated

    return (
        symmetrize_and_interpolate(f0, line_axis0),
        symmetrize_and_interpolate(f1, line_axis1),
    )


def calculate_psd_analysis_at_time(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    spacing_um: float | tuple[float, float] = 1.3,
    confidence_level: float = 0.95,
    n_resamples: int = 20_000,
    bootstrap_method: str = "BCa",
    min_modes_per_bin: int = 8,
    seed: int = 12345,
    use_median: bool = False,
    exclusion_masks: np.ndarray | None = None,
    mask_taper_um: float | None = None,
) -> PSDAnalysisResult:
    """Calculate both full 2D PSDs and radial PSD results."""
    full_2d = calculate_2d_psd_at_time(
        height_maps_um,
        time_index,
        spacing_um=spacing_um,
        use_median=use_median,
        exclusion_masks=exclusion_masks,
        mask_taper_um=mask_taper_um,
    )
    radial = radial_psd_at_time(
        height_maps_um,
        time_index,
        spacing_um=spacing_um,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        min_modes_per_bin=min_modes_per_bin,
        seed=seed,
        use_median=use_median,
        exclusion_masks=exclusion_masks,
        mask_taper_um=mask_taper_um,
        _full_2d=full_2d,
    )

    selected_masks = None
    if exclusion_masks is not None:
        selected_masks = _selected_exclusion_masks(
            exclusion_masks,
            time_index=time_index,
            selected_shape=full_2d.specimen_psd_um4.shape,
        )

    return PSDAnalysisResult(
        radial=radial,
        full_2d=full_2d,
        excluded_masks=selected_masks,
    )


def directional_psd_from_2d(
    full_2d: PSD2DResult,
    *,
    spacing_um: float | tuple[float, float],
    sector_width_deg: float = 15.0,
    min_modes_per_bin: int = 8,
    min_modes_per_sector_bin: int = 2,
    use_median: bool = False,
) -> DirectionalPSDResult:
    """Average a 2D PSD in wavevector sectors modulo 180 degrees."""
    d0, d1 = _as_spacing_tuple(spacing_um)
    sector_width = float(sector_width_deg)
    if sector_width <= 0.0 or sector_width > 90.0:
        raise ValueError("sector_width_deg must be in (0, 90].")
    number_sectors = round(180.0 / sector_width)
    if not np.isclose(number_sectors * sector_width, 180.0):
        raise ValueError("sector_width_deg must divide 180 degrees exactly.")

    f0 = full_2d.frequency_axis0_um_inv
    f1 = full_2d.frequency_axis1_um_inv
    radial_frequency = np.hypot(f0[:, None], f1[None, :])
    direction = np.mod(np.arctan2(f0[:, None], f1[None, :]), np.pi)
    df0 = 1.0 / (f0.size * d0)
    df1 = 1.0 / (f1.size * d1)
    bin_width = max(df0, df1)
    full_annulus_limit = min(0.5 / d0, 0.5 / d1)
    edges = np.arange(
        0.5 * bin_width,
        full_annulus_limit + np.finfo(float).eps * full_annulus_limit,
        bin_width,
    )
    if edges.size < 2:
        raise ValueError("The map is too small for directional PSD bins.")

    radial_flat = radial_frequency.ravel()
    direction_flat = direction.ravel()
    bin_index = np.searchsorted(edges, radial_flat, side="right") - 1
    number_bins = edges.size - 1
    valid_mode = (bin_index >= 0) & (bin_index < number_bins)
    total_modes = np.bincount(
        bin_index[valid_mode],
        minlength=number_bins,
    )
    frequency_sum = np.bincount(
        bin_index[valid_mode],
        weights=radial_flat[valid_mode],
        minlength=number_bins,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        frequency = frequency_sum / total_modes
    retained = (total_modes >= min_modes_per_bin) & np.isfinite(frequency)
    if not np.any(retained):
        raise ValueError("No directional PSD bins satisfy min_modes_per_bin.")

    sector_centers_deg = np.arange(number_sectors) * sector_width
    sector_centers_rad = np.deg2rad(sector_centers_deg)
    modes = np.zeros((number_sectors, number_bins), dtype=np.int64)
    specimen_psd = np.full(
        (
            full_2d.specimen_psd_um4.shape[0],
            number_sectors,
            int(np.count_nonzero(retained)),
        ),
        np.nan,
        dtype=np.float64,
    )

    for sector_index, center_angle in enumerate(sector_centers_rad):
        angular_distance = np.abs(
            np.mod(direction_flat - center_angle + 0.5 * np.pi, np.pi)
            - 0.5 * np.pi
        )
        sector_mode = valid_mode & (
            angular_distance <= 0.5 * np.deg2rad(sector_width)
        )
        sector_bins = bin_index[sector_mode]
        modes[sector_index] = np.bincount(
            sector_bins,
            minlength=number_bins,
        )

        for specimen_index, psd_2d in enumerate(full_2d.specimen_psd_um4):
            sector_sum = np.bincount(
                sector_bins,
                weights=psd_2d.ravel()[sector_mode],
                minlength=number_bins,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                sector_average = sector_sum / modes[sector_index]
            sector_average[modes[sector_index] < min_modes_per_sector_bin] = np.nan
            specimen_psd[specimen_index, sector_index] = sector_average[retained]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_psd = np.nanmean(specimen_psd, axis=0)
        median_psd = np.nanmedian(specimen_psd, axis=0)

    estimator_name, _ = _specimen_estimator(use_median)
    retained_frequency = frequency[retained]
    return DirectionalPSDResult(
        frequency_um_inv=retained_frequency,
        wavelength_um=1.0 / retained_frequency,
        direction_deg=sector_centers_deg,
        specimen_psd_um4=specimen_psd,
        mean_psd_um4=mean_psd,
        median_psd_um4=median_psd,
        modes_per_sector_bin=modes[:, retained],
        estimator=estimator_name,
    )


def band_power_from_2d(
    full_2d: PSD2DResult,
    *,
    spacing_um: float | tuple[float, float],
    band_bounds_um: np.ndarray,
) -> np.ndarray:
    """Integrate two-sided 2D PSD power over wavelength bands."""
    bounds = np.asarray(band_bounds_um, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("band_bounds_um must have shape (nbands, 2).")
    if np.any(bounds <= 0.0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Every wavelength band must have positive low < high.")

    d0, d1 = _as_spacing_tuple(spacing_um)
    f0 = full_2d.frequency_axis0_um_inv
    f1 = full_2d.frequency_axis1_um_inv
    radial_frequency = np.hypot(f0[:, None], f1[None, :])
    df0 = 1.0 / (f0.size * d0)
    df1 = 1.0 / (f1.size * d1)
    power = np.empty(
        (full_2d.specimen_psd_um4.shape[0], bounds.shape[0]),
        dtype=np.float64,
    )

    for band_index, (short_wavelength, long_wavelength) in enumerate(bounds):
        in_band = (
            (radial_frequency >= 1.0 / long_wavelength)
            & (radial_frequency <= 1.0 / short_wavelength)
            & (radial_frequency > 0.0)
        )
        if not np.any(in_band):
            raise ValueError(
                f"No Fourier modes fall in wavelength band "
                f"[{short_wavelength}, {long_wavelength}] um."
            )
        power[:, band_index] = (
            np.sum(full_2d.specimen_psd_um4[:, in_band], axis=1)
            * df0
            * df1
        )

    return power


def spectral_length_scales_from_radial_psd(
    radial: RadialPSDResult,
    *,
    wavelength_limits_um: tuple[float, float],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return peak and cumulative-power wavelength scales per specimen."""
    lower, upper = map(float, wavelength_limits_um)
    if lower <= 0.0 or upper <= lower:
        raise ValueError("wavelength_limits_um must satisfy 0 < lower < upper.")

    names = (
        "lambda_peak_um",
        "lambda_10_um",
        "lambda_50_um",
        "lambda_90_um",
    )
    values = np.full(
        (radial.specimen_psd_um4.shape[0], len(names)),
        np.nan,
        dtype=np.float64,
    )
    retained = (
        (radial.wavelength_um >= lower)
        & (radial.wavelength_um <= upper)
        & np.isfinite(radial.wavelength_um)
    )
    if np.count_nonzero(retained) < 3:
        raise ValueError("Too few radial PSD bins fall in the wavelength limits.")

    wavelength = radial.wavelength_um[retained]
    frequency = radial.frequency_um_inv[retained]
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    log_wavelength = np.log(wavelength)

    for specimen_index, psd in enumerate(radial.specimen_psd_um4):
        contribution = (
            2.0
            * np.pi
            * frequency**2
            * psd[retained]
        )[order]
        finite = np.isfinite(contribution) & (contribution >= 0.0)
        if np.count_nonzero(finite) < 3:
            continue
        local_wavelength = wavelength[finite]
        local_log_wavelength = log_wavelength[finite]
        local_contribution = contribution[finite]
        segment_power = (
            0.5
            * (local_contribution[:-1] + local_contribution[1:])
            * np.diff(local_log_wavelength)
        )
        cumulative = np.concatenate(([0.0], np.cumsum(segment_power)))
        total = cumulative[-1]
        if not np.isfinite(total) or total <= np.finfo(float).tiny:
            continue

        values[specimen_index, 0] = local_wavelength[
            int(np.nanargmax(local_contribution))
        ]
        normalized_cumulative = cumulative / total
        values[specimen_index, 1:] = np.interp(
            (0.10, 0.50, 0.90),
            normalized_cumulative,
            local_wavelength,
        )

    return names, values


def _log_bin_specimen_curves(
    frequency_um_inv: np.ndarray,
    specimen_curves: np.ndarray,
    modes_per_bin: np.ndarray,
    *,
    bins_per_decade: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebin linear-frequency curves into equal-width log-frequency bins."""
    frequency = np.asarray(frequency_um_inv, dtype=np.float64)
    curves = np.asarray(specimen_curves, dtype=np.float64)
    modes = np.asarray(modes_per_bin, dtype=np.float64)
    if frequency.ndim != 1 or modes.shape != frequency.shape:
        raise ValueError("frequency and modes_per_bin must be matching 1D arrays.")
    if curves.shape[-1] != frequency.size:
        raise ValueError("The last specimen_curves axis must match frequency.")
    if bins_per_decade < 4:
        raise ValueError("bins_per_decade must be at least four.")

    valid_frequency = (
        np.isfinite(frequency)
        & (frequency > 0.0)
        & np.isfinite(modes)
        & (modes > 0.0)
    )
    frequency = frequency[valid_frequency]
    modes = modes[valid_frequency]
    curves = curves[..., valid_frequency]
    order = np.argsort(frequency)
    frequency = frequency[order]
    modes = modes[order]
    curves = curves[..., order]

    decades = np.log10(frequency[-1] / frequency[0])
    number_bins = max(8, int(np.ceil(decades * bins_per_decade)))
    edges = np.geomspace(frequency[0], frequency[-1], number_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bin_index = np.searchsorted(edges, frequency, side="right") - 1

    output_curves = []
    output_frequency = []
    output_modes = []
    for local_bin in range(number_bins):
        selected = bin_index == local_bin
        if not np.any(selected):
            continue
        local_modes = modes[selected]
        local_curves = curves[..., selected]
        finite = np.isfinite(local_curves) & (local_curves > 0.0)
        weighted_values = np.where(
            finite,
            local_curves * local_modes,
            0.0,
        )
        effective_weight = np.sum(
            np.where(finite, local_modes, 0.0),
            axis=-1,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            binned_curve = np.sum(weighted_values, axis=-1) / effective_weight
        output_curves.append(binned_curve)
        output_frequency.append(
            np.exp(np.average(np.log(frequency[selected]), weights=local_modes))
        )
        output_modes.append(np.sum(local_modes))

    binned = np.stack(output_curves, axis=-1)
    retained = np.all(np.isfinite(binned), axis=tuple(range(binned.ndim - 1)))
    if np.count_nonzero(retained) < 6:
        raise ValueError("Too few finite logarithmic PSD bins remain.")
    return (
        np.asarray(output_frequency, dtype=np.float64)[retained],
        binned[..., retained],
        np.asarray(output_modes, dtype=np.float64)[retained],
    )


def _isotonic_decreasing(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted pool-adjacent-violators fit constrained to be decreasing."""
    y = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if y.ndim != 1 or w.shape != y.shape:
        raise ValueError("values and weights must be matching 1D arrays.")

    block_values: list[float] = []
    block_weights: list[float] = []
    block_lengths: list[int] = []
    for value, weight in zip(y, w):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_lengths.append(1)
        while (
            len(block_values) >= 2
            and block_values[-2] < block_values[-1]
        ):
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / merged_weight
            merged_length = block_lengths[-2] + block_lengths[-1]
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_lengths[-2:] = [merged_length]

    return np.concatenate(
        [
            np.full(length, value, dtype=np.float64)
            for value, length in zip(block_values, block_lengths)
        ]
    )


def _noise_ratio_crossing_um(
    frequency_um_inv: np.ndarray,
    normalized_ratio: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Locate the 2D/1D PSD noise-floor crossing after monotone smoothing."""
    frequency = np.asarray(frequency_um_inv, dtype=np.float64)
    ratio = np.asarray(normalized_ratio, dtype=np.float64)
    valid = np.isfinite(ratio) & (ratio > 0.0)
    if np.count_nonzero(valid) < 4:
        return np.nan, np.full_like(ratio, np.nan)

    fitted_log_ratio = np.full_like(ratio, np.nan)
    local_fit = _isotonic_decreasing(
        np.log10(ratio[valid]),
        np.asarray(weights, dtype=np.float64)[valid],
    )
    fitted_log_ratio[valid] = local_fit
    local_frequency = frequency[valid]
    crossing = np.flatnonzero(local_fit <= 0.0)
    if crossing.size == 0 or crossing[0] == 0:
        return np.nan, 10.0**fitted_log_ratio

    upper_index = int(crossing[0])
    lower_index = upper_index - 1
    y0 = local_fit[lower_index]
    y1 = local_fit[upper_index]
    x0 = np.log10(local_frequency[lower_index])
    x1 = np.log10(local_frequency[upper_index])
    if np.isclose(y0, y1):
        crossing_log_frequency = x1
    else:
        crossing_log_frequency = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
    return float(1.0 / 10.0**crossing_log_frequency), 10.0**fitted_log_ratio


def _small_wavelength_noise_cutoff(
    evolution: MagnificationEvolutionResult,
    times_hours: np.ndarray,
    *,
    use_median: bool,
    confidence_level: float,
    n_resamples: int,
    bins_per_decade: int,
    seed: int,
) -> SmallWavelengthCutoffResult:
    """Apply the Jacobs et al. white-noise criterion to both map axes."""
    radial = np.stack(
        [result.specimen_psd_um4 for result in evolution.radial_results],
        axis=0,
    )
    frequency = evolution.radial_results[0].frequency_um_inv
    modes = evolution.radial_results[0].modes_per_bin
    ntimes, nsamples, number_raw_bins = radial.shape
    binned_frequency, binned_radial_flat, binned_modes = (
        _log_bin_specimen_curves(
            frequency,
            radial.reshape(ntimes * nsamples, number_raw_bins),
            modes,
            bins_per_decade=bins_per_decade,
        )
    )
    binned_radial = binned_radial_flat.reshape(ntimes, nsamples, -1)

    binned_lines = []
    for line_psd in (evolution.line_psd_axis0_um3, evolution.line_psd_axis1_um3):
        _, binned_line_flat, _ = _log_bin_specimen_curves(
            frequency,
            line_psd.reshape(ntimes * nsamples, number_raw_bins),
            modes,
            bins_per_decade=bins_per_decade,
        )
        binned_lines.append(binned_line_flat.reshape(ntimes, nsamples, -1))

    d0, d1 = evolution.spacing_um
    expected_ratios = (d1, d0)
    normalized_specimen_ratios = [
        binned_radial / np.maximum(line, np.finfo(float).tiny) / expected
        for line, expected in zip(binned_lines, expected_ratios)
    ]
    estimator = np.nanmedian if use_median else np.nanmean
    axis_cutoffs = np.full((2, ntimes), np.nan, dtype=np.float64)
    axis_status = np.full((2, ntimes), "ambiguous", dtype="U32")
    combined_status = np.full(ntimes, "ambiguous", dtype="U32")
    combined_cutoff = np.full(ntimes, np.nan, dtype=np.float64)
    bootstrap_cutoff = np.full(
        (ntimes, n_resamples),
        np.nan,
        dtype=np.float64,
    )

    for time_index in range(ntimes):
        local_axis_cutoffs = []
        for axis_index, (line, expected) in enumerate(
            zip(binned_lines, expected_ratios)
        ):
            aggregate_ratio = (
                estimator(binned_radial[time_index], axis=0)
                / np.maximum(
                    estimator(line[time_index], axis=0),
                    np.finfo(float).tiny,
                )
                / expected
            )
            cutoff, fitted_ratio = _noise_ratio_crossing_um(
                binned_frequency,
                aggregate_ratio,
                binned_modes,
            )
            axis_cutoffs[axis_index, time_index] = cutoff
            finite_fit = fitted_ratio[np.isfinite(fitted_ratio)]
            if np.isfinite(cutoff):
                axis_status[axis_index, time_index] = "crossing_detected"
            elif finite_fit.size and finite_fit[-1] > 1.0:
                axis_status[axis_index, time_index] = "below_measured_range"
            elif finite_fit.size and finite_fit[0] <= 1.0:
                axis_status[axis_index, time_index] = (
                    "no_signal_dominated_range"
                )
            local_axis_cutoffs.append(cutoff)
        finite_cutoffs = np.asarray(local_axis_cutoffs)[
            np.isfinite(local_axis_cutoffs)
        ]
        if finite_cutoffs.size:
            combined_cutoff[time_index] = float(np.max(finite_cutoffs))
            combined_status[time_index] = "crossing_detected"
        elif np.all(axis_status[:, time_index] == "below_measured_range"):
            combined_status[time_index] = "below_measured_range"
        elif np.any(
            axis_status[:, time_index] == "no_signal_dominated_range"
        ):
            combined_status[time_index] = "no_signal_dominated_range"

        rng = np.random.default_rng(seed + time_index)
        completed = 0
        while completed < n_resamples:
            batch_size = min(250, n_resamples - completed)
            sample_indices = rng.integers(
                0,
                nsamples,
                size=(batch_size, nsamples),
            )
            sampled_radial = binned_radial[time_index][sample_indices]
            aggregate_radial = estimator(sampled_radial, axis=1)
            aggregate_lines = [
                estimator(line[time_index][sample_indices], axis=1)
                for line in binned_lines
            ]
            for bootstrap_index in range(batch_size):
                replicate_cutoffs = []
                for aggregate_line, expected in zip(
                    aggregate_lines,
                    expected_ratios,
                ):
                    ratio = (
                        aggregate_radial[bootstrap_index]
                        / np.maximum(
                            aggregate_line[bootstrap_index],
                            np.finfo(float).tiny,
                        )
                        / expected
                    )
                    cutoff, _ = _noise_ratio_crossing_um(
                        binned_frequency,
                        ratio,
                        binned_modes,
                    )
                    if np.isfinite(cutoff):
                        replicate_cutoffs.append(cutoff)
                if replicate_cutoffs:
                    bootstrap_cutoff[
                        time_index,
                        completed + bootstrap_index,
                    ] = np.max(replicate_cutoffs)
            completed += batch_size

    alpha = 1.0 - confidence_level
    ci_low = np.full(ntimes, np.nan, dtype=np.float64)
    ci_high = np.full(ntimes, np.nan, dtype=np.float64)
    detection_fraction = np.mean(np.isfinite(bootstrap_cutoff), axis=1)
    for time_index, values in enumerate(bootstrap_cutoff):
        finite = values[np.isfinite(values)]
        if finite.size:
            ci_low[time_index], ci_high[time_index] = np.quantile(
                finite,
                [0.5 * alpha, 1.0 - 0.5 * alpha],
            )
    detected = (
        np.isfinite(combined_cutoff)
        & (detection_fraction >= confidence_level)
    )
    combined_status[
        np.isfinite(combined_cutoff) & ~detected
    ] = "bootstrap_unstable"
    combined_status[detected] = "accepted"

    return SmallWavelengthCutoffResult(
        times_hours=np.asarray(times_hours, dtype=np.float64),
        wavelength_um=1.0 / binned_frequency,
        normalized_ratio_axis0=normalized_specimen_ratios[0],
        normalized_ratio_axis1=normalized_specimen_ratios[1],
        cutoff_axis0_um=axis_cutoffs[0],
        cutoff_axis1_um=axis_cutoffs[1],
        cutoff_um=combined_cutoff,
        ci_low_um=ci_low,
        ci_high_um=ci_high,
        detected=detected,
        bootstrap_detection_fraction=detection_fraction,
        status=combined_status,
    )


def _fit_low_frequency_rolloff(
    frequency_um_inv: np.ndarray,
    psd_um4: np.ndarray,
    *,
    min_segment_bins: int,
) -> tuple[float, float, float, np.ndarray]:
    """Fit a flat low-frequency plateau joined to a log-log scaling line."""
    frequency = np.asarray(frequency_um_inv, dtype=np.float64)
    psd = np.asarray(psd_um4, dtype=np.float64)
    valid = np.isfinite(frequency) & (frequency > 0.0) & np.isfinite(psd) & (psd > 0.0)
    x = np.log10(frequency[valid])
    y = np.log10(psd[valid])
    number_points = x.size
    fitted_full = np.full_like(psd, np.nan)
    if number_points < 2 * min_segment_bins + 1:
        return np.nan, np.nan, np.nan, fitted_full

    null_design = np.column_stack((np.ones(number_points), x - np.mean(x)))
    null_coefficients = np.linalg.lstsq(null_design, y, rcond=None)[0]
    null_residual = y - null_design @ null_coefficients
    null_sse = max(float(np.sum(null_residual**2)), np.finfo(float).tiny)
    null_bic = number_points * np.log(null_sse / number_points) + 2 * np.log(
        number_points
    )

    best_sse = np.inf
    best_break = np.nan
    best_slope = np.nan
    best_fit = None
    for break_index in range(
        min_segment_bins - 1,
        number_points - min_segment_bins,
    ):
        breakpoint = x[break_index]
        design = np.column_stack(
            (np.ones(number_points), np.maximum(x - breakpoint, 0.0))
        )
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = design @ coefficients
        sse = max(float(np.sum((y - fitted) ** 2)), np.finfo(float).tiny)
        if sse < best_sse:
            best_sse = sse
            best_break = breakpoint
            best_slope = float(coefficients[1])
            best_fit = fitted

    broken_bic = number_points * np.log(best_sse / number_points) + 3 * np.log(
        number_points
    )
    if best_fit is not None:
        fitted_full[valid] = 10.0**best_fit
    return (
        float(1.0 / 10.0**best_break),
        float(null_bic - broken_bic),
        best_slope,
        fitted_full,
    )


def plot_radial_psd_vs_wavelength(
    result: RadialPSDResult,
    *,
    ax: plt.Axes | None = None,
    show_specimens: bool = True,
    wavelength_range_um: tuple[float, float] | None = None,
) -> plt.Axes:
    """
    Plot the radial 2D PSD against wavelength.

    The confidence interval is pointwise, not a simultaneous confidence
    band for the entire spectrum.
    """
    if ax is None:
        _, ax = plt.subplots()

    wavelength = result.wavelength_um
    estimate_psd = result.estimate_psd_um4
    ci_low = result.ci_low_um4
    ci_high = result.ci_high_um4

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(estimate_psd)
        & np.isfinite(ci_low)
        & np.isfinite(ci_high)
        & (wavelength > 0.0)
        & (estimate_psd > 0.0)
        & (ci_low > 0.0)
        & (ci_high > 0.0)
    )

    if wavelength_range_um is not None:
        wavelength_min, wavelength_max = wavelength_range_um

        if wavelength_min <= 0.0 or wavelength_max <= wavelength_min:
            raise ValueError(
                "wavelength_range_um must be (positive_minimum, larger_maximum)."
            )

        valid &= (wavelength >= wavelength_min) & (wavelength <= wavelength_max)

    if not np.any(valid):
        raise ValueError("No positive finite PSD values remain after filtering.")

    order = np.argsort(wavelength[valid])
    wavelength_plot = wavelength[valid][order]
    estimate_plot = estimate_psd[valid][order]
    low_plot = ci_low[valid][order]
    high_plot = ci_high[valid][order]

    if show_specimens:
        for specimen_curve in result.specimen_psd_um4:
            ax.loglog(
                wavelength_plot,
                specimen_curve[valid][order],
                linewidth=0.7,
                alpha=0.18,
            )

    ax.loglog(
        wavelength_plot,
        estimate_plot,
        linewidth=1.8,
        label=f"Specimen {result.estimator}",
    )

    ax.fill_between(
        wavelength_plot,
        low_plot,
        high_plot,
        alpha=0.25,
        linewidth=0,
        label="95% pointwise BCa bootstrap CI",
    )

    ax.set_xlabel(r"Wavelength, $\lambda$ ($\mu$m)")
    ax.set_ylabel(
        r"Radially averaged 2D PSD, $C_{\mathrm{iso}}$ ($\mu$m$^4$)"
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()

    return ax


def plot_variance_contribution_vs_wavelength(
    result: RadialPSDResult,
    *,
    ax: plt.Axes | None = None,
    wavelength_range_um: tuple[float, float] | None = None,
) -> plt.Axes:
    """
    Plot 2*pi*f^2*C_iso(f), the variance contribution per log-wavelength.
    """
    if ax is None:
        _, ax = plt.subplots()

    frequency = result.frequency_um_inv
    wavelength = result.wavelength_um

    estimate = 2.0 * np.pi * frequency**2 * result.estimate_psd_um4
    low = 2.0 * np.pi * frequency**2 * result.ci_low_um4
    high = 2.0 * np.pi * frequency**2 * result.ci_high_um4

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(estimate)
        & np.isfinite(low)
        & np.isfinite(high)
        & (wavelength > 0.0)
        & (estimate > 0.0)
        & (low > 0.0)
        & (high > 0.0)
    )

    if wavelength_range_um is not None:
        wavelength_min, wavelength_max = wavelength_range_um

        if wavelength_min <= 0.0 or wavelength_max <= wavelength_min:
            raise ValueError(
                "wavelength_range_um must be (positive_minimum, larger_maximum)."
            )

        valid &= (wavelength >= wavelength_min) & (wavelength <= wavelength_max)

    if not np.any(valid):
        raise ValueError("No positive finite variance-contribution values remain.")

    order = np.argsort(wavelength[valid])
    wavelength_plot = wavelength[valid][order]

    ax.loglog(
        wavelength_plot,
        estimate[valid][order],
        linewidth=1.8,
        label=f"Specimen {result.estimator}",
    )
    ax.fill_between(
        wavelength_plot,
        low[valid][order],
        high[valid][order],
        alpha=0.25,
        linewidth=0,
        label="95% pointwise BCa bootstrap CI",
    )

    ax.set_xlabel(r"Wavelength, $\lambda$ ($\mu$m)")
    ax.set_ylabel(r"$2\pi f^2 C_{\mathrm{iso}}(f)$ ($\mu$m$^2$)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()

    return ax


def _positive_log_norm(
    arrays: list[np.ndarray],
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
) -> LogNorm:
    positive_values = np.concatenate(
        [np.asarray(array)[np.asarray(array) > 0.0].ravel() for array in arrays]
    )

    if positive_values.size == 0:
        raise ValueError("Cannot create a logarithmic color scale with no positives.")

    vmin, vmax = np.percentile(
        positive_values,
        [lower_percentile, upper_percentile],
    )

    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = positive_values.min()

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = positive_values.max()

    if vmax <= vmin:
        vmax = np.nextafter(vmin, np.inf)

    return LogNorm(vmin=vmin, vmax=vmax)


def plot_2d_psd(
    result: PSD2DResult,
    *,
    specimen_index: int | None = None,
    ax: plt.Axes | None = None,
    norm: LogNorm | None = None,
    title: str | None = None,
    cmap: str = "magma",
) -> plt.Axes:
    """
    Plot the selected aggregate full 2D PSD, or one specimen's full 2D PSD.

    Frequencies are shown after fftshift, so zero frequency is centered.
    """
    if ax is None:
        _, ax = plt.subplots()

    if specimen_index is None:
        psd = result.estimate_psd_um4
        default_title = f"{result.estimator.title()} full 2D PSD"
    else:
        psd = result.specimen_psd_um4[specimen_index]
        default_title = f"Specimen {specimen_index} full 2D PSD"

    if norm is None:
        norm = _positive_log_norm([psd])

    f0_shifted = np.fft.fftshift(result.frequency_axis0_um_inv)
    f1_shifted = np.fft.fftshift(result.frequency_axis1_um_inv)
    psd_shifted = np.fft.fftshift(psd)

    image = ax.imshow(
        psd_shifted,
        origin="lower",
        aspect="auto",
        extent=[
            f1_shifted[0],
            f1_shifted[-1],
            f0_shifted[0],
            f0_shifted[-1],
        ],
        norm=norm,
        cmap=cmap,
    )

    ax.figure.colorbar(
        image,
        ax=ax,
        label=r"2D PSD, $C_{\mathrm{2D}}$ ($\mu$m$^4$)",
    )
    ax.set_xlabel(r"$f_1$ (cycles/$\mu$m)")
    ax.set_ylabel(r"$f_0$ (cycles/$\mu$m)")
    ax.set_title(default_title if title is None else title)

    return ax


def _specimen_summary_with_ci(
    specimen_values: np.ndarray,
    *,
    use_median: bool,
    confidence_level: float,
    n_resamples: int,
    seed: int,
    bootstrap_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate arrays shaped (ntimes, nsamples, ...) over specimens."""
    values = np.asarray(specimen_values, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("specimen_values must have time and specimen axes.")
    if values.shape[1] < 2:
        raise ValueError("At least two specimens are required for a CI.")

    sample_first = np.moveaxis(values, 1, 0)
    estimator = np.nanmedian if use_median else np.nanmean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        estimate = estimator(sample_first, axis=0)

    def statistic(sample: np.ndarray, axis: int = 0) -> np.ndarray:
        return estimator(sample, axis=axis)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = bootstrap(
                (sample_first,),
                statistic,
                axis=0,
                vectorized=True,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                batch=min(1000, n_resamples),
                method=bootstrap_method,
                **_bootstrap_keyword_args(seed),
            )
        low = np.asarray(result.confidence_interval.low)
        high = np.asarray(result.confidence_interval.high)
    except ValueError:
        low = np.full_like(estimate, np.nan, dtype=np.float64)
        high = np.full_like(estimate, np.nan, dtype=np.float64)

    failed = ~np.isfinite(low) | ~np.isfinite(high)
    if np.any(failed):
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            0,
            sample_first.shape[0],
            size=(n_resamples, sample_first.shape[0]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bootstrap_estimates = estimator(sample_first[indices], axis=1)
        alpha = 100.0 * (1.0 - confidence_level) / 2.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            percentile_low = np.nanpercentile(
                bootstrap_estimates,
                alpha,
                axis=0,
            )
            percentile_high = np.nanpercentile(
                bootstrap_estimates,
                100.0 - alpha,
                axis=0,
            )
        low = np.where(failed, percentile_low, low)
        high = np.where(failed, percentile_high, high)

    return estimate, low, high


def _paired_change_summary_with_ci(
    specimen_values: np.ndarray,
    *,
    use_median: bool,
    confidence_level: float,
    n_resamples: int,
    seed: int,
    bootstrap_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(specimen_values, dtype=np.float64)
    change = values - values[[0], ...]
    estimate, low, high = _specimen_summary_with_ci(
        change,
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
        bootstrap_method=bootstrap_method,
    )
    estimate[0] = 0.0
    low[0] = 0.0
    high[0] = 0.0
    return change, estimate, low, high


def _save_metric_time_plots(
    times_hours: np.ndarray,
    names: tuple[str, ...],
    estimate: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    *,
    output_dir: Path,
    prefix: str,
    change: bool = False,
) -> None:
    for metric_index, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        ax.plot(times_hours, estimate[:, metric_index], marker="o", color="black")
        ax.fill_between(
            times_hours,
            ci_low[:, metric_index],
            ci_high[:, metric_index],
            color="0.75",
            alpha=0.7,
        )
        ax.axhline(0.0, color="0.6", linewidth=0.8) if change else None
        ax.set_xlabel("Hold time (h)")
        ax.set_ylabel(("Change in " if change else "") + name)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        suffix = "change" if change else "absolute"
        fig.savefig(
            output_dir / f"{prefix}_{name}_{suffix}.png",
            dpi=250,
        )
        plt.close(fig)


def _save_acf_plot(
    result: ACFAnalysisResult,
    *,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(
        result.aggregate_acf,
        origin="lower",
        extent=[
            result.lag_axis1_um[0],
            result.lag_axis1_um[-1],
            result.lag_axis0_um[0],
            result.lag_axis0_um[-1],
        ],
        cmap="coolwarm",
        vmin=-0.25,
        vmax=1.0,
        aspect="equal",
    )
    ax.contour(
        result.lag_axis1_um,
        result.lag_axis0_um,
        result.aggregate_acf,
        levels=[result.lengths.threshold],
        colors="black",
        linewidths=0.8,
    )
    ax.set_xlim(
        0.25 * result.lag_axis1_um[0],
        0.25 * result.lag_axis1_um[-1],
    )
    ax.set_ylim(
        0.25 * result.lag_axis0_um[0],
        0.25 * result.lag_axis0_um[-1],
    )
    ax.set_xlabel("Axis-1 lag (um)")
    ax.set_ylabel("Axis-0 lag (um)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Normalized autocorrelation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _save_directional_psd_plot(
    result: DirectionalPSDResult,
    *,
    wavelength_limits_um: tuple[float, float],
    output_path: Path,
    title: str,
) -> None:
    wavelength = result.wavelength_um
    order = np.argsort(wavelength)
    estimate = result.estimate_psd_um4[:, order]
    positive = estimate[np.isfinite(estimate) & (estimate > 0.0)]
    if positive.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    image = ax.pcolormesh(
        wavelength[order],
        result.direction_deg,
        np.log10(np.maximum(estimate, np.min(positive))),
        shading="auto",
        cmap="viridis",
    )
    ax.set_xscale("log")
    ax.set_xlim(wavelength_limits_um)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Wavevector direction (degrees)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="log10 directional PSD (um^4)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _save_spectral_redistribution_plot(
    radial_results: tuple[RadialPSDResult, ...],
    times_hours: np.ndarray,
    *,
    wavelength_limits_um: tuple[float, float],
    use_median: bool,
    output_path: Path,
    title: str,
) -> np.ndarray:
    specimen_psd = np.stack(
        [result.specimen_psd_um4 for result in radial_results],
        axis=0,
    )
    floor = np.finfo(float).tiny
    log_ratio = np.log10(
        np.maximum(specimen_psd, floor)
        / np.maximum(specimen_psd[[0], ...], floor)
    )
    estimator = np.nanmedian if use_median else np.nanmean
    estimate = estimator(log_ratio, axis=1)
    wavelength = radial_results[0].wavelength_um
    retained = (
        (wavelength >= wavelength_limits_um[0])
        & (wavelength <= wavelength_limits_um[1])
    )
    order = np.argsort(wavelength[retained])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.pcolormesh(
        wavelength[retained][order],
        times_hours,
        estimate[:, retained][:, order],
        shading="auto",
        cmap="coolwarm",
        vmin=-np.nanmax(np.abs(estimate[:, retained])),
        vmax=np.nanmax(np.abs(estimate[:, retained])),
    )
    ax.set_xscale("log")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Hold time (h)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="log10[PSD(t) / PSD(0)]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return log_ratio


def _nested_window_sq(
    height_maps_um: np.ndarray,
    exclusion_masks: np.ndarray,
    *,
    spacing_um: float | tuple[float, float],
    fractions: tuple[float, ...],
) -> np.ndarray:
    maps = np.asarray(height_maps_um, dtype=np.float64)
    masks = np.asarray(exclusion_masks, dtype=bool)
    if masks.shape != maps.shape[1:]:
        raise ValueError("exclusion_masks do not match the height-map stack.")

    output = np.empty(
        (len(fractions), maps.shape[0], maps.shape[1]),
        dtype=np.float64,
    )
    for fraction_index, fraction in enumerate(fractions):
        fraction = float(fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Nested-window fractions must be in (0, 1].")
        size0 = max(8, round(fraction * maps.shape[2]))
        size1 = max(8, round(fraction * maps.shape[3]))
        start0 = (maps.shape[2] - size0) // 2
        start1 = (maps.shape[3] - size1) // 2
        window_maps = maps[
            :,
            :,
            start0 : start0 + size0,
            start1 : start1 + size1,
        ]
        window_masks = masks[
            :,
            start0 : start0 + size0,
            start1 : start1 + size1,
        ]
        for time_index in range(maps.shape[0]):
            parameters = surface_roughness_parameters_at_time(
                window_maps,
                time_index,
                spacing_um=spacing_um,
                exclusion_masks=window_masks,
            )
            sq_index = parameters.parameter_names.index("Sq_um")
            output[fraction_index, time_index] = (
                parameters.specimen_values[:, sq_index]
            )
    return output


def estimate_rigid_shifts_to_t0(
    height_maps_um: np.ndarray,
    *,
    exclusion_masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate integer rigid shifts to t=0 without resampling the analysis maps.

    The diagnostic uses phase correlation after valid-pixel centering and a
    Hann window. PSD, ACF, and scalar roughness calculations retain the original
    samples because interpolation would alter short-wavelength amplitudes.
    """
    maps = np.asarray(height_maps_um, dtype=np.float64)
    if maps.ndim != 4:
        raise ValueError(
            "height_maps_um must have shape (ntimes, nsamples, n0, n1)."
        )
    masks = _selected_exclusion_masks(
        exclusion_masks,
        time_index=0,
        selected_shape=maps.shape[1:],
    )
    window = np.outer(
        np.hanning(maps.shape[2]),
        np.hanning(maps.shape[3]),
    )
    shifts = np.zeros((maps.shape[0], maps.shape[1], 2), dtype=np.float64)
    peak_to_mean = np.empty((maps.shape[0], maps.shape[1]), dtype=np.float64)

    for specimen_index in range(maps.shape[1]):
        valid = ~masks[specimen_index]
        reference = maps[0, specimen_index]
        reference_centered = np.where(
            valid,
            reference - np.mean(reference[valid]),
            0.0,
        )
        reference_fft = np.fft.fft2(reference_centered * window)

        for time_index in range(maps.shape[0]):
            moving = maps[time_index, specimen_index]
            moving_centered = np.where(
                valid,
                moving - np.mean(moving[valid]),
                0.0,
            )
            moving_fft = np.fft.fft2(moving_centered * window)
            cross_power = reference_fft * np.conj(moving_fft)
            magnitude = np.abs(cross_power)
            supported = magnitude > np.finfo(float).eps * np.max(magnitude)
            normalized_cross_power = np.zeros_like(cross_power)
            normalized_cross_power[supported] = (
                cross_power[supported] / magnitude[supported]
            )
            correlation = np.abs(np.fft.ifft2(normalized_cross_power))
            peak_index = np.unravel_index(
                int(np.argmax(correlation)),
                correlation.shape,
            )
            signed_shift = np.asarray(peak_index, dtype=np.float64)
            for axis, axis_size in enumerate(correlation.shape):
                if signed_shift[axis] > axis_size // 2:
                    signed_shift[axis] -= axis_size
            shifts[time_index, specimen_index] = signed_shift
            peak_to_mean[time_index, specimen_index] = (
                correlation[peak_index]
                / max(float(np.mean(correlation)), np.finfo(float).tiny)
            )

    return shifts, peak_to_mean


def save_center_indent_mask_diagnostics(
    height_maps_um: np.ndarray,
    mask_result: CenterIndentMaskResult,
    *,
    output_dir: str | Path,
    prefix: str = "profilometry_psd",
) -> Path:
    """Save fixed masks and one temporal-median overlay per specimen."""
    maps = np.asarray(height_maps_um, dtype=np.float64)
    if maps.ndim != 4:
        raise ValueError(
            "height_maps_um must have shape (ntimes, nsamples, n0, n1)."
        )

    nsamples = maps.shape[1]
    if mask_result.excluded_masks.shape != maps.shape[1:]:
        raise ValueError("The center-indent masks do not match height_maps_um.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path / f"{prefix}_center_indent_masks.npz",
        excluded_masks=mask_result.excluded_masks,
        detected_masks=mask_result.detected_masks,
        detected_center_indices=mask_result.detected_center_indices,
        detected_area_um2=mask_result.detected_area_um2,
        excluded_area_um2=mask_result.excluded_area_um2,
        guard_um=mask_result.guard_um,
        center_search_fraction=np.asarray(mask_result.center_search_fraction),
        method=np.asarray(mask_result.method),
        fixed_diagonal_um=np.asarray(
            np.nan
            if mask_result.fixed_diagonal_um is None
            else mask_result.fixed_diagonal_um
        ),
    )

    for specimen_index in range(nsamples):
        temporal_median = np.median(maps[:, specimen_index], axis=0)
        lower, upper = np.percentile(temporal_median, [1.0, 99.0])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower = float(np.min(temporal_median))
            upper = float(np.max(temporal_median))

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
        image = axes[0].imshow(
            temporal_median,
            origin="lower",
            cmap="viridis",
            vmin=lower,
            vmax=upper,
        )
        axes[0].contour(
            mask_result.detected_masks[specimen_index],
            levels=[0.5],
            colors=["white"],
            linewidths=1.0,
        )
        axes[0].contour(
            mask_result.excluded_masks[specimen_index],
            levels=[0.5],
            colors=["red"],
            linewidths=1.2,
        )
        center0, center1 = mask_result.detected_center_indices[specimen_index]
        axes[0].plot(center1, center0, marker="+", color="cyan", markersize=9)
        axes[0].set_title("Temporal median with fixed mask")
        axes[0].set_xlabel("Array axis 1 (pixel)")
        axes[0].set_ylabel("Array axis 0 (pixel)")
        fig.colorbar(image, ax=axes[0], label="Height (um)")

        axes[1].imshow(
            mask_result.excluded_masks[specimen_index],
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
        )
        axes[1].contour(
            mask_result.detected_masks[specimen_index],
            levels=[0.5],
            colors=["red"],
            linewidths=1.0,
        )
        axes[1].set_title(
            "Excluded mask\n"
            f"area={mask_result.excluded_area_um2[specimen_index]:.1f} um^2, "
            f"guard={mask_result.guard_um[specimen_index]:.2f} um"
        )
        axes[1].set_xlabel("Array axis 1 (pixel)")
        axes[1].set_ylabel("Array axis 0 (pixel)")

        fig.tight_layout()
        fig.savefig(
            output_path
            / f"{prefix}_specimen_{specimen_index:03d}_center_indent_mask.png",
            dpi=250,
        )
        plt.close(fig)

    return output_path


def save_results(
    analysis: PSDAnalysisResult,
    *,
    output_dir: str | Path = "psd_output",
    prefix: str = "profilometry_psd",
    plot_specimen_2d: bool = True,
    show_specimens_radial: bool = True,
) -> Path:
    """Save numerical arrays and figures for the PSD analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    radial = analysis.radial
    full_2d = analysis.full_2d

    saved_arrays: dict[str, np.ndarray] = {
        "radial_estimator": np.asarray(radial.estimator),
        "radial_frequency_um_inv": radial.frequency_um_inv,
        "radial_wavelength_um": radial.wavelength_um,
        "radial_specimen_psd_um4": radial.specimen_psd_um4,
        "radial_estimate_psd_um4": radial.estimate_psd_um4,
        "radial_mean_psd_um4": radial.mean_psd_um4,
        "radial_ci_low_um4": radial.ci_low_um4,
        "radial_ci_high_um4": radial.ci_high_um4,
        "radial_modes_per_bin": radial.modes_per_bin,
        "full_2d_estimator": np.asarray(full_2d.estimator),
        "full_2d_frequency_axis0_um_inv": full_2d.frequency_axis0_um_inv,
        "full_2d_frequency_axis1_um_inv": full_2d.frequency_axis1_um_inv,
        "full_2d_specimen_psd_um4": full_2d.specimen_psd_um4,
        "full_2d_estimate_psd_um4": full_2d.estimate_psd_um4,
        "full_2d_mean_psd_um4": full_2d.mean_psd_um4,
        "parseval_relative_error": radial.parseval_relative_error,
    }

    if radial.median_psd_um4 is not None:
        saved_arrays["radial_median_psd_um4"] = radial.median_psd_um4
    if full_2d.median_psd_um4 is not None:
        saved_arrays["full_2d_median_psd_um4"] = full_2d.median_psd_um4
    if full_2d.excluded_fraction is not None:
        saved_arrays["excluded_fraction"] = full_2d.excluded_fraction
    if full_2d.taper_effective_fraction is not None:
        saved_arrays["mask_taper_effective_fraction"] = (
            full_2d.taper_effective_fraction
        )
    if analysis.excluded_masks is not None:
        saved_arrays["excluded_masks"] = analysis.excluded_masks
    if analysis.center_indent_mask is not None:
        indent_mask = analysis.center_indent_mask
        saved_arrays.update(
            {
                "indent_mask_method": np.asarray(indent_mask.method),
                "indent_fixed_diagonal_um": np.asarray(
                    np.nan
                    if indent_mask.fixed_diagonal_um is None
                    else indent_mask.fixed_diagonal_um
                ),
                "indent_detected_masks": indent_mask.detected_masks,
                "indent_detected_center_indices": (
                    indent_mask.detected_center_indices
                ),
                "indent_detected_area_um2": indent_mask.detected_area_um2,
                "indent_excluded_area_um2": indent_mask.excluded_area_um2,
                "indent_guard_um": indent_mask.guard_um,
                "indent_center_search_fraction": np.asarray(
                    indent_mask.center_search_fraction
                ),
            }
        )

    np.savez_compressed(output_path / f"{prefix}_results.npz", **saved_arrays)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    plot_radial_psd_vs_wavelength(
        radial,
        ax=ax,
        show_specimens=show_specimens_radial,
    )
    fig.tight_layout()
    fig.savefig(output_path / f"{prefix}_radial_psd_vs_wavelength.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    plot_variance_contribution_vs_wavelength(radial, ax=ax)
    fig.tight_layout()
    fig.savefig(
        output_path / f"{prefix}_variance_contribution_vs_wavelength.png",
        dpi=300,
    )
    plt.close(fig)

    all_2d_arrays = [full_2d.estimate_psd_um4]
    if plot_specimen_2d:
        all_2d_arrays.extend(full_2d.specimen_psd_um4)
    shared_norm = _positive_log_norm(all_2d_arrays)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    plot_2d_psd(
        full_2d,
        ax=ax,
        norm=shared_norm,
        title=f"{full_2d.estimator.title()} full 2D PSD",
    )
    fig.tight_layout()
    fig.savefig(
        output_path / f"{prefix}_{full_2d.estimator}_full_2d_psd.png",
        dpi=300,
    )
    plt.close(fig)

    if plot_specimen_2d:
        for specimen_index in range(full_2d.specimen_psd_um4.shape[0]):
            fig, ax = plt.subplots(figsize=(6.0, 5.0))
            plot_2d_psd(
                full_2d,
                specimen_index=specimen_index,
                ax=ax,
                norm=shared_norm,
                title=f"Specimen {specimen_index} full 2D PSD",
            )
            fig.tight_layout()
            fig.savefig(
                output_path / f"{prefix}_specimen_{specimen_index:03d}_full_2d_psd.png",
                dpi=300,
            )
            plt.close(fig)

    return output_path


def _analysis_wavelength_limits(
    shape: tuple[int, int],
    spacing_um: float | tuple[float, float],
    *,
    points_per_shortest_wavelength: float,
    cycles_per_longest_wavelength: float,
) -> tuple[float, float]:
    d0, d1 = _as_spacing_tuple(spacing_um)
    if points_per_shortest_wavelength <= 2.0:
        raise ValueError("points_per_shortest_wavelength must exceed two.")
    if cycles_per_longest_wavelength < 2.0:
        raise ValueError("cycles_per_longest_wavelength must be at least two.")
    shortest = points_per_shortest_wavelength * max(d0, d1)
    longest = min(shape[0] * d0, shape[1] * d1) / cycles_per_longest_wavelength
    if longest <= shortest:
        raise ValueError(
            "The cropped field is too small for the requested wavelength limits."
        )
    return float(shortest), float(longest)


def _mask_sensitivity_analysis(
    uncropped_maps_um: np.ndarray,
    cropped_maps_um: np.ndarray,
    nominal_mask: CenterIndentMaskResult,
    *,
    spacing_um: float | tuple[float, float],
    crop_slices: tuple[slice, slice],
    nominal_diagonal_um: float,
    sensitivity_diagonals_um: tuple[float | None, ...],
    band_bounds_um: np.ndarray,
    mask_taper_um: float,
    acf_threshold: float,
    acf_n_directions: int,
    nominal_roughness: np.ndarray,
    roughness_names: tuple[str, ...],
    nominal_acf: np.ndarray,
    nominal_band_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ntimes, nsamples = cropped_maps_um.shape[:2]
    ndiagonals = len(sensitivity_diagonals_um)
    diagonal_values = np.asarray(
        [
            np.nan if value is None else float(value)
            for value in sensitivity_diagonals_um
        ],
        dtype=np.float64,
    )
    sq_values = np.empty((ndiagonals, ntimes, nsamples), dtype=np.float64)
    sal_values = np.empty_like(sq_values)
    band_values = np.empty(
        (ndiagonals, ntimes, nsamples, band_bounds_um.shape[0]),
        dtype=np.float64,
    )
    sq_index = roughness_names.index("Sq_um")

    for sensitivity_index, diagonal in enumerate(sensitivity_diagonals_um):
        is_nominal = diagonal is not None and np.isclose(
            float(diagonal),
            nominal_diagonal_um,
        )
        if is_nominal:
            sq_values[sensitivity_index] = nominal_roughness[:, :, sq_index]
            sal_values[sensitivity_index] = nominal_acf[:, :, 0]
            band_values[sensitivity_index] = nominal_band_power
            continue

        if diagonal is None:
            sensitivity_masks = np.zeros(
                cropped_maps_um.shape[1:],
                dtype=bool,
            )
        else:
            _, sensitivity_mask_result = _prepare_uncropped_height_maps(
                uncropped_maps_um,
                spacing_um=spacing_um,
                crop_slices=crop_slices,
                center_mask_diagonal_um=float(diagonal),
            )
            sensitivity_masks = sensitivity_mask_result.excluded_masks

        for time_index in range(ntimes):
            roughness = surface_roughness_parameters_at_time(
                cropped_maps_um,
                time_index,
                spacing_um=spacing_um,
                exclusion_masks=sensitivity_masks,
            )
            sq_values[sensitivity_index, time_index] = (
                roughness.specimen_values[:, sq_index]
            )
            acf = calculate_acf_analysis_at_time(
                cropped_maps_um,
                time_index,
                spacing_um=spacing_um,
                threshold=acf_threshold,
                n_directions=acf_n_directions,
                exclusion_masks=sensitivity_masks,
            )
            sal_values[sensitivity_index, time_index] = (
                acf.lengths.specimen_sal_um
            )
            full_2d = calculate_2d_psd_at_time(
                cropped_maps_um,
                time_index,
                spacing_um=spacing_um,
                exclusion_masks=sensitivity_masks,
                mask_taper_um=mask_taper_um,
            )
            band_values[sensitivity_index, time_index] = band_power_from_2d(
                full_2d,
                spacing_um=spacing_um,
                band_bounds_um=band_bounds_um,
            )

    return diagonal_values, sq_values, sal_values, band_values


def _run_magnification_evolution(
    label: str,
    uncropped_maps_um: np.ndarray,
    spacing_um: float | tuple[float, float],
    times_hours: np.ndarray,
    *,
    crop_slices: tuple[slice, slice],
    center_mask_diagonal_um: float,
    mask_taper_um: float,
    wavelength_limits_um: tuple[float, float],
    band_names: tuple[str, ...],
    band_bounds_um: np.ndarray,
    output_dir: Path,
    use_median: bool,
    confidence_level: float,
    n_resamples: int,
    bootstrap_method: str,
    min_modes_per_bin: int,
    seed: int,
    plot_specimen_2d: bool,
    directional_sector_width_deg: float,
    acf_threshold: float,
    acf_n_directions: int,
    nested_window_fractions: tuple[float, ...],
    mask_sensitivity_diagonals_um: tuple[float | None, ...],
) -> MagnificationEvolutionResult:
    maps, center_mask = _prepare_uncropped_height_maps(
        uncropped_maps_um,
        spacing_um=spacing_um,
        crop_slices=crop_slices,
        center_mask_diagonal_um=center_mask_diagonal_um,
    )
    spacing = _as_spacing_tuple(spacing_um)
    magnitude_dir = output_dir / label
    magnitude_dir.mkdir(parents=True, exist_ok=True)
    save_center_indent_mask_diagnostics(
        maps,
        center_mask,
        output_dir=magnitude_dir / "mask_diagnostics",
        prefix=label,
    )
    rigid_shifts, registration_peak_to_mean = estimate_rigid_shifts_to_t0(
        maps,
        exclusion_masks=center_mask.excluded_masks,
    )

    radial_results = []
    directional_results = []
    roughness_values = []
    acf_values = []
    spectral_length_values = []
    band_power_values = []
    line_psd_axis0_values = []
    line_psd_axis1_values = []
    roughness_names = None
    spectral_length_names = None
    acf_metric_names = (
        "Sal_um",
        "fastest_decay_angle_deg",
        "slowest_decay_um",
        "slowest_decay_angle_deg",
        "Str",
    )

    for time_index, time_hours in enumerate(times_hours):
        print(f"[{label}] analyzing time {time_hours:g} h")
        time_dir = magnitude_dir / f"time_{time_index:03d}_{time_hours:g}h"
        time_dir.mkdir(parents=True, exist_ok=True)
        base_analysis = calculate_psd_analysis_at_time(
            maps,
            time_index,
            spacing_um=spacing,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_method=bootstrap_method,
            min_modes_per_bin=min_modes_per_bin,
            seed=seed + time_index,
            use_median=use_median,
            exclusion_masks=center_mask.excluded_masks,
            mask_taper_um=mask_taper_um,
        )
        analysis = PSDAnalysisResult(
            radial=base_analysis.radial,
            full_2d=base_analysis.full_2d,
            excluded_masks=base_analysis.excluded_masks,
            center_indent_mask=center_mask,
        )
        prefix = f"{label}_time_{time_index:03d}"
        save_results(
            analysis,
            output_dir=time_dir,
            prefix=prefix,
            plot_specimen_2d=plot_specimen_2d,
        )
        radial_results.append(analysis.radial)
        line_axis0, line_axis1 = _integrated_line_psds_on_radial_grid(
            analysis.full_2d,
            analysis.radial.frequency_um_inv,
            spacing_um=spacing,
        )
        line_psd_axis0_values.append(line_axis0)
        line_psd_axis1_values.append(line_axis1)

        directional = directional_psd_from_2d(
            analysis.full_2d,
            spacing_um=spacing,
            sector_width_deg=directional_sector_width_deg,
            min_modes_per_bin=min_modes_per_bin,
            use_median=use_median,
        )
        directional_results.append(directional)
        np.savez_compressed(
            time_dir / f"{prefix}_directional_psd.npz",
            frequency_um_inv=directional.frequency_um_inv,
            wavelength_um=directional.wavelength_um,
            direction_deg=directional.direction_deg,
            specimen_psd_um4=directional.specimen_psd_um4,
            mean_psd_um4=directional.mean_psd_um4,
            median_psd_um4=directional.median_psd_um4,
            modes_per_sector_bin=directional.modes_per_sector_bin,
            estimator=np.asarray(directional.estimator),
        )
        _save_directional_psd_plot(
            directional,
            wavelength_limits_um=wavelength_limits_um,
            output_path=time_dir / f"{prefix}_directional_psd.png",
            title=f"{label} directional PSD, {time_hours:g} h",
        )

        acf = calculate_acf_analysis_at_time(
            maps,
            time_index,
            spacing_um=spacing,
            threshold=acf_threshold,
            n_directions=acf_n_directions,
            exclusion_masks=center_mask.excluded_masks,
            use_median=use_median,
        )
        acf_matrix = np.column_stack(
            [
                acf.lengths.specimen_sal_um,
                acf.lengths.specimen_fastest_decay_angle_deg,
                acf.lengths.specimen_slowest_decay_um,
                acf.lengths.specimen_slowest_decay_angle_deg,
                acf.lengths.specimen_texture_aspect_ratio,
            ]
        )
        acf_values.append(acf_matrix)
        np.savez_compressed(
            time_dir / f"{prefix}_acf.npz",
            aggregate_acf=acf.aggregate_acf,
            lag_axis0_um=acf.lag_axis0_um,
            lag_axis1_um=acf.lag_axis1_um,
            metric_names=np.asarray(acf_metric_names),
            specimen_metrics=acf_matrix,
            threshold=np.asarray(acf_threshold),
            estimator=np.asarray(acf.estimator),
        )
        _save_acf_plot(
            acf,
            output_path=time_dir / f"{prefix}_acf.png",
            title=f"{label} 2D ACF, {time_hours:g} h",
        )

        roughness = surface_roughness_parameters_at_time(
            maps,
            time_index,
            spacing_um=spacing,
            exclusion_masks=center_mask.excluded_masks,
        )
        roughness_names = roughness.parameter_names
        roughness_values.append(roughness.specimen_values)

        spectral_length_names, length_values = (
            spectral_length_scales_from_radial_psd(
                analysis.radial,
                wavelength_limits_um=wavelength_limits_um,
            )
        )
        spectral_length_values.append(length_values)
        band_power_values.append(
            band_power_from_2d(
                analysis.full_2d,
                spacing_um=spacing,
                band_bounds_um=band_bounds_um,
            )
        )

    if roughness_names is None or spectral_length_names is None:
        raise RuntimeError("No time points were analyzed.")
    radial_tuple = tuple(radial_results)
    directional_tuple = tuple(directional_results)
    roughness_array = np.stack(roughness_values, axis=0)
    acf_array = np.stack(acf_values, axis=0)
    spectral_array = np.stack(spectral_length_values, axis=0)
    band_array = np.stack(band_power_values, axis=0)
    line_psd_axis0_array = np.stack(line_psd_axis0_values, axis=0)
    line_psd_axis1_array = np.stack(line_psd_axis1_values, axis=0)
    nested_sq = _nested_window_sq(
        maps,
        center_mask.excluded_masks,
        spacing_um=spacing,
        fractions=nested_window_fractions,
    )
    (
        sensitivity_diagonals,
        sensitivity_sq,
        sensitivity_sal,
        sensitivity_band,
    ) = _mask_sensitivity_analysis(
        uncropped_maps_um,
        maps,
        center_mask,
        spacing_um=spacing,
        crop_slices=crop_slices,
        nominal_diagonal_um=center_mask_diagonal_um,
        sensitivity_diagonals_um=mask_sensitivity_diagonals_um,
        band_bounds_um=band_bounds_um,
        mask_taper_um=mask_taper_um,
        acf_threshold=acf_threshold,
        acf_n_directions=acf_n_directions,
        nominal_roughness=roughness_array,
        roughness_names=roughness_names,
        nominal_acf=acf_array,
        nominal_band_power=band_array,
    )

    summary_dir = magnitude_dir / "evolution"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_inputs = (
        ("roughness", roughness_names, roughness_array),
        (
            "acf",
            ("Sal_um", "slowest_decay_um", "Str"),
            acf_array[:, :, [0, 2, 4]],
        ),
        ("spectral_length", spectral_length_names, spectral_array),
        ("band_power", band_names, band_array),
    )
    saved_summary: dict[str, np.ndarray] = {}
    for summary_index, (summary_name, names, values) in enumerate(summary_inputs):
        estimate, low, high = _specimen_summary_with_ci(
            values,
            use_median=use_median,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed + 1000 + summary_index,
            bootstrap_method=bootstrap_method,
        )
        change_values, change_estimate, change_low, change_high = (
            _paired_change_summary_with_ci(
                values,
                use_median=use_median,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                seed=seed + 2000 + summary_index,
                bootstrap_method=bootstrap_method,
            )
        )
        saved_summary.update(
            {
                f"{summary_name}_summary_names": np.asarray(names),
                f"{summary_name}_estimate": estimate,
                f"{summary_name}_ci_low": low,
                f"{summary_name}_ci_high": high,
                f"{summary_name}_specimen_change": change_values,
                f"{summary_name}_change_estimate": change_estimate,
                f"{summary_name}_change_ci_low": change_low,
                f"{summary_name}_change_ci_high": change_high,
            }
        )
        _save_metric_time_plots(
            times_hours,
            names,
            estimate,
            low,
            high,
            output_dir=summary_dir,
            prefix=f"{label}_{summary_name}",
        )
        _save_metric_time_plots(
            times_hours,
            names,
            change_estimate,
            change_low,
            change_high,
            output_dir=summary_dir,
            prefix=f"{label}_{summary_name}",
            change=True,
        )

    log_psd_ratio = _save_spectral_redistribution_plot(
        radial_tuple,
        times_hours,
        wavelength_limits_um=wavelength_limits_um,
        use_median=use_median,
        output_path=summary_dir / f"{label}_spectral_redistribution.png",
        title=f"{label} spectral redistribution",
    )

    estimator = np.nanmedian if use_median else np.nanmean
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for fraction_index, fraction in enumerate(nested_window_fractions):
        ax.plot(
            times_hours,
            estimator(nested_sq[fraction_index], axis=1),
            marker="o",
            label=f"{fraction:.2f} field",
        )
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Sq (um)")
    ax.set_title(f"{label} nested-window convergence")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(summary_dir / f"{label}_nested_window_sq.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for sensitivity_index, diagonal in enumerate(sensitivity_diagonals):
        mask_label = "no mask" if np.isnan(diagonal) else f"{diagonal:g} um"
        axes[0].plot(
            times_hours,
            estimator(sensitivity_sq[sensitivity_index], axis=1),
            marker="o",
            label=mask_label,
        )
        axes[1].plot(
            times_hours,
            estimator(sensitivity_sal[sensitivity_index], axis=1),
            marker="o",
            label=mask_label,
        )
    axes[0].set_ylabel("Sq (um)")
    axes[1].set_ylabel("Sal (um)")
    for ax in axes:
        ax.set_xlabel("Hold time (h)")
        ax.grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle(f"{label} center-mask sensitivity")
    fig.tight_layout()
    fig.savefig(summary_dir / f"{label}_mask_sensitivity.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis_index, ax in enumerate(axes):
        for specimen_index in range(rigid_shifts.shape[1]):
            ax.plot(
                times_hours,
                rigid_shifts[:, specimen_index, axis_index],
                marker="o",
                alpha=0.45,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Hold time (h)")
        ax.set_ylabel(f"Shift to t=0, axis {axis_index} (pixels)")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"{label} rigid-registration diagnostic (not applied to PSD/ACF)"
    )
    fig.tight_layout()
    fig.savefig(summary_dir / f"{label}_registration_diagnostic.png", dpi=250)
    plt.close(fig)

    radial_specimen_stack = np.stack(
        [result.specimen_psd_um4 for result in radial_tuple],
        axis=0,
    )
    np.savez_compressed(
        magnitude_dir / f"{label}_evolution_results.npz",
        times_hours=times_hours,
        spacing_um=np.asarray(spacing),
        cropped_shape=np.asarray(maps.shape[-2:]),
        wavelength_limits_um=np.asarray(wavelength_limits_um),
        radial_frequency_um_inv=radial_tuple[0].frequency_um_inv,
        radial_wavelength_um=radial_tuple[0].wavelength_um,
        radial_specimen_psd_um4=radial_specimen_stack,
        radial_log10_ratio_to_t0=log_psd_ratio,
        roughness_parameter_names=np.asarray(roughness_names),
        roughness_specimen_values=roughness_array,
        acf_metric_names=np.asarray(acf_metric_names),
        acf_specimen_values=acf_array,
        spectral_length_names=np.asarray(spectral_length_names),
        spectral_length_values_um=spectral_array,
        band_names=np.asarray(band_names),
        band_bounds_um=band_bounds_um,
        band_power_um2=band_array,
        rigid_shift_to_t0_pixels=rigid_shifts,
        registration_peak_to_mean=registration_peak_to_mean,
        nested_window_fractions=np.asarray(nested_window_fractions),
        nested_window_sq_um=nested_sq,
        mask_sensitivity_diagonals_um=sensitivity_diagonals,
        mask_sensitivity_sq_um=sensitivity_sq,
        mask_sensitivity_sal_um=sensitivity_sal,
        mask_sensitivity_band_power_um2=sensitivity_band,
        line_psd_axis0_um3=line_psd_axis0_array,
        line_psd_axis1_um3=line_psd_axis1_array,
        **saved_summary,
    )

    return MagnificationEvolutionResult(
        label=label,
        spacing_um=spacing,
        cropped_shape=maps.shape[-2:],
        wavelength_limits_um=wavelength_limits_um,
        radial_results=radial_tuple,
        directional_results=directional_tuple,
        roughness_parameter_names=roughness_names,
        roughness_specimen_values=roughness_array,
        acf_metric_names=acf_metric_names,
        acf_specimen_values=acf_array,
        spectral_length_names=spectral_length_names,
        spectral_length_values_um=spectral_array,
        band_names=band_names,
        band_bounds_um=band_bounds_um,
        band_power_um2=band_array,
        rigid_shift_to_t0_pixels=rigid_shifts,
        registration_peak_to_mean=registration_peak_to_mean,
        nested_window_fractions=np.asarray(nested_window_fractions),
        nested_window_sq_um=nested_sq,
        mask_sensitivity_diagonals_um=sensitivity_diagonals,
        mask_sensitivity_sq_um=sensitivity_sq,
        mask_sensitivity_sal_um=sensitivity_sal,
        mask_sensitivity_band_power_um2=sensitivity_band,
        line_psd_axis0_um3=line_psd_axis0_array,
        line_psd_axis1_um3=line_psd_axis1_array,
        output_dir=magnitude_dir,
    )


def _subfield_radial_spectra(
    evolution: MagnificationEvolutionResult,
    cropped_maps_um: np.ndarray,
    exclusion_masks: np.ndarray,
    *,
    spacing_um: float | tuple[float, float],
    window_fractions: tuple[float, ...],
    mask_taper_um: float,
    min_modes_per_bin: int,
    use_median: bool,
) -> list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]:
    """Return specimen PSDs from full and translated smaller subfields."""
    maps = np.asarray(cropped_maps_um, dtype=np.float64)
    masks = np.asarray(exclusion_masks, dtype=bool)
    d0, d1 = _as_spacing_tuple(spacing_um)
    n0, n1 = maps.shape[-2:]
    estimator = np.nanmedian if use_median else np.nanmean
    output = []

    fractions = tuple(sorted({float(value) for value in window_fractions}))
    if not fractions or not np.isclose(fractions[-1], 1.0):
        raise ValueError("rolloff_window_fractions must include 1.0.")
    if any(value <= 0.0 or value > 1.0 for value in fractions):
        raise ValueError("rolloff_window_fractions must lie in (0, 1].")

    for fraction in fractions:
        if np.isclose(fraction, 1.0):
            frequency = evolution.radial_results[0].frequency_um_inv
            modes = evolution.radial_results[0].modes_per_bin
            specimen_psd = np.stack(
                [result.specimen_psd_um4 for result in evolution.radial_results],
                axis=0,
            )
            short_side = min(n0 * d0, n1 * d1)
            output.append(
                (fraction, short_side, frequency, modes, specimen_psd)
            )
            continue

        size0 = max(16, int(round(fraction * n0)))
        size1 = max(16, int(round(fraction * n1)))
        starts0 = sorted({0, n0 - size0})
        starts1 = sorted({0, n1 - size1})
        time_psd = []
        frequency = None
        modes = None
        for time_index in range(maps.shape[0]):
            window_psd = []
            for start0 in starts0:
                for start1 in starts1:
                    stop0 = start0 + size0
                    stop1 = start1 + size1
                    local_maps = maps[:, :, start0:stop0, start1:stop1]
                    local_masks = masks[:, start0:stop0, start1:stop1]
                    full_2d = calculate_2d_psd_at_time(
                        local_maps,
                        time_index,
                        spacing_um=(d0, d1),
                        use_median=use_median,
                        exclusion_masks=local_masks,
                        mask_taper_um=mask_taper_um,
                    )
                    local_frequency, local_psd, local_modes = (
                        _radial_average_from_2d_psd(
                            full_2d,
                            spacing_um=(d0, d1),
                            min_modes_per_bin=min_modes_per_bin,
                        )
                    )
                    if frequency is None:
                        frequency = local_frequency
                        modes = local_modes
                    elif not np.allclose(frequency, local_frequency):
                        raise RuntimeError("Subfields produced inconsistent PSD grids.")
                    window_psd.append(local_psd)
            time_psd.append(estimator(np.stack(window_psd, axis=0), axis=0))

        if frequency is None or modes is None:
            raise RuntimeError("No subfield PSD was calculated.")
        short_side = min(size0 * d0, size1 * d1)
        output.append(
            (
                fraction,
                short_side,
                frequency,
                modes,
                np.stack(time_psd, axis=0),
            )
        )
    return output


def _large_wavelength_rolloff(
    subfield_spectra: list[
        tuple[float, float, np.ndarray, np.ndarray, np.ndarray]
    ],
    times_hours: np.ndarray,
    *,
    use_median: bool,
    confidence_level: float,
    n_resamples: int,
    bins_per_decade: int,
    min_segment_bins: int,
    minimum_fit_wavelength_um: float,
    required_delta_bic: float,
    seed: int,
) -> LargeWavelengthCutoffResult:
    """Estimate and bootstrap the 10x low-frequency PSD roll-off."""
    number_fractions = len(subfield_spectra)
    ntimes = len(times_hours)
    fractions = np.empty(number_fractions, dtype=np.float64)
    short_sides = np.empty(number_fractions, dtype=np.float64)
    cutoff = np.full((number_fractions, ntimes), np.nan, dtype=np.float64)
    ci_low = np.full_like(cutoff, np.nan)
    ci_high = np.full_like(cutoff, np.nan)
    delta_bic = np.full_like(cutoff, np.nan)
    slope = np.full_like(cutoff, np.nan)
    detected = np.zeros_like(cutoff, dtype=bool)
    detection_fraction = np.zeros_like(cutoff)
    estimator = np.nanmedian if use_median else np.nanmean
    alpha = 1.0 - confidence_level

    for fraction_index, (
        fraction,
        short_side,
        frequency,
        modes,
        specimen_psd,
    ) in enumerate(subfield_spectra):
        fractions[fraction_index] = fraction
        short_sides[fraction_index] = short_side
        ntimes_local, nsamples, number_raw_bins = specimen_psd.shape
        if ntimes_local != ntimes:
            raise ValueError("Subfield spectra do not match times_hours.")
        fit_selected = frequency <= 1.0 / minimum_fit_wavelength_um
        binned_frequency, binned_flat, _ = _log_bin_specimen_curves(
            frequency[fit_selected],
            specimen_psd.reshape(ntimes * nsamples, number_raw_bins)[
                :, fit_selected
            ],
            modes[fit_selected],
            bins_per_decade=bins_per_decade,
        )
        binned = binned_flat.reshape(ntimes, nsamples, -1)

        for time_index in range(ntimes):
            aggregate = estimator(binned[time_index], axis=0)
            local_cutoff, local_delta, local_slope, _ = (
                _fit_low_frequency_rolloff(
                    binned_frequency,
                    aggregate,
                    min_segment_bins=min_segment_bins,
                )
            )
            cutoff[fraction_index, time_index] = local_cutoff
            delta_bic[fraction_index, time_index] = local_delta
            slope[fraction_index, time_index] = local_slope
            detected[fraction_index, time_index] = (
                np.isfinite(local_cutoff)
                and local_delta >= required_delta_bic
                and local_slope < 0.0
            )

            rng = np.random.default_rng(
                seed + 1000 * fraction_index + time_index
            )
            bootstrap_values = np.full(n_resamples, np.nan, dtype=np.float64)
            completed = 0
            while completed < n_resamples:
                batch_size = min(250, n_resamples - completed)
                sample_indices = rng.integers(
                    0,
                    nsamples,
                    size=(batch_size, nsamples),
                )
                sampled = binned[time_index][sample_indices]
                aggregates = estimator(sampled, axis=1)
                for bootstrap_index, bootstrap_curve in enumerate(aggregates):
                    value, evidence, bootstrap_slope, _ = (
                        _fit_low_frequency_rolloff(
                            binned_frequency,
                            bootstrap_curve,
                            min_segment_bins=min_segment_bins,
                        )
                    )
                    if (
                        np.isfinite(value)
                        and evidence >= required_delta_bic
                        and bootstrap_slope < 0.0
                    ):
                        bootstrap_values[completed + bootstrap_index] = value
                completed += batch_size

            finite = bootstrap_values[np.isfinite(bootstrap_values)]
            detection_fraction[fraction_index, time_index] = (
                finite.size / n_resamples
            )
            if finite.size:
                ci_low[fraction_index, time_index], ci_high[
                    fraction_index, time_index
                ] = np.quantile(
                    finite,
                    [0.5 * alpha, 1.0 - 0.5 * alpha],
                )

    full_index = int(np.flatnonzero(np.isclose(fractions, 1.0))[0])
    stable = np.array(
        detected[full_index]
        & (detection_fraction[full_index] >= confidence_level),
        copy=True,
    )
    for time_index in range(ntimes):
        if not stable[time_index]:
            continue
        full_interval = (
            ci_low[full_index, time_index],
            ci_high[full_index, time_index],
        )
        if not np.all(np.isfinite(full_interval)):
            stable[time_index] = False
            continue
        for fraction_index in range(number_fractions):
            if fraction_index == full_index:
                continue
            local_interval = (
                ci_low[fraction_index, time_index],
                ci_high[fraction_index, time_index],
            )
            intervals_overlap = (
                detected[fraction_index, time_index]
                and detection_fraction[fraction_index, time_index]
                >= confidence_level
                and np.all(np.isfinite(local_interval))
                and max(full_interval[0], local_interval[0])
                <= min(full_interval[1], local_interval[1])
            )
            if not intervals_overlap:
                stable[time_index] = False
                break

    return LargeWavelengthCutoffResult(
        times_hours=np.asarray(times_hours, dtype=np.float64),
        window_fractions=fractions,
        window_short_side_um=short_sides,
        cutoff_um=cutoff,
        ci_low_um=ci_low,
        ci_high_um=ci_high,
        delta_bic=delta_bic,
        scaling_slope=slope,
        detected=detected,
        bootstrap_detection_fraction=detection_fraction,
        full_window_stable=stable,
    )


def _interpolate_radial_log_psd(
    radial_results: tuple[RadialPSDResult, ...],
    wavelength_grid_um: np.ndarray,
) -> np.ndarray:
    output = np.empty(
        (
            len(radial_results),
            radial_results[0].specimen_psd_um4.shape[0],
            wavelength_grid_um.size,
        ),
        dtype=np.float64,
    )
    log_grid = np.log(wavelength_grid_um)
    floor = np.finfo(float).tiny
    for time_index, result in enumerate(radial_results):
        order = np.argsort(result.wavelength_um)
        log_wavelength = np.log(result.wavelength_um[order])
        for specimen_index, psd in enumerate(result.specimen_psd_um4):
            output[time_index, specimen_index] = np.interp(
                log_grid,
                log_wavelength,
                np.log10(np.maximum(psd[order], floor)),
            )
    return output


def _log_binned_radial_stack(
    evolution: MagnificationEvolutionResult,
    *,
    bins_per_decade: int,
) -> tuple[np.ndarray, np.ndarray]:
    radial = np.stack(
        [result.specimen_psd_um4 for result in evolution.radial_results],
        axis=0,
    )
    ntimes, nsamples, number_raw_bins = radial.shape
    frequency, binned_flat, _ = _log_bin_specimen_curves(
        evolution.radial_results[0].frequency_um_inv,
        radial.reshape(ntimes * nsamples, number_raw_bins),
        evolution.radial_results[0].modes_per_bin,
        bins_per_decade=bins_per_decade,
    )
    binned = binned_flat.reshape(ntimes, nsamples, -1)
    order = np.argsort(1.0 / frequency)
    return 1.0 / frequency[order], binned[..., order]


def _joint_paired_spectral_changes(
    fine: MagnificationEvolutionResult,
    coarse: MagnificationEvolutionResult,
    *,
    fine_valid_band_um: tuple[float, float],
    coarse_valid_band_um: tuple[float, float],
    use_median: bool,
    confidence_level: float,
    n_resamples: int,
    bins_per_decade: int,
    seed: int,
) -> tuple[SpectralChangeResult, SpectralChangeResult]:
    """Build one simultaneous band over both objectives, times, and wavelengths."""
    fine_wavelength, fine_psd = _log_binned_radial_stack(
        fine,
        bins_per_decade=bins_per_decade,
    )
    coarse_wavelength, coarse_psd = _log_binned_radial_stack(
        coarse,
        bins_per_decade=bins_per_decade,
    )
    if fine_psd.shape[:2] != coarse_psd.shape[:2]:
        raise ValueError("Fine and coarse PSD stacks must have matching pairs.")

    floor = np.finfo(float).tiny
    fine_change = np.log10(
        np.maximum(fine_psd, floor) / np.maximum(fine_psd[[0]], floor)
    )
    coarse_change = np.log10(
        np.maximum(coarse_psd, floor) / np.maximum(coarse_psd[[0]], floor)
    )
    estimator = np.nanmedian if use_median else np.nanmean
    fine_estimate = estimator(fine_change, axis=1)
    coarse_estimate = estimator(coarse_change, axis=1)
    ntimes, nsamples = fine_change.shape[:2]

    fine_valid = (
        (fine_wavelength >= fine_valid_band_um[0])
        & (fine_wavelength <= fine_valid_band_um[1])
    )
    coarse_valid = (
        (coarse_wavelength >= coarse_valid_band_um[0])
        & (coarse_wavelength <= coarse_valid_band_um[1])
    )
    if not np.any(fine_valid) or not np.any(coarse_valid):
        raise ValueError("No log-binned PSD values fall in a morphology band.")

    bootstrap_fine = np.empty(
        (n_resamples, ntimes - 1, fine_wavelength.size),
        dtype=np.float64,
    )
    bootstrap_coarse = np.empty(
        (n_resamples, ntimes - 1, coarse_wavelength.size),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < n_resamples:
        batch_size = min(250, n_resamples - completed)
        sample_indices = rng.integers(
            0,
            nsamples,
            size=(batch_size, nsamples),
        )
        fine_sampled = fine_change[1:, sample_indices, :]
        coarse_sampled = coarse_change[1:, sample_indices, :]
        fine_batch = estimator(fine_sampled, axis=2).transpose(1, 0, 2)
        coarse_batch = estimator(coarse_sampled, axis=2).transpose(1, 0, 2)
        bootstrap_fine[completed : completed + batch_size] = fine_batch
        bootstrap_coarse[completed : completed + batch_size] = coarse_batch
        completed += batch_size

    fine_se = np.std(bootstrap_fine, axis=0, ddof=1)
    coarse_se = np.std(bootstrap_coarse, axis=0, ddof=1)
    fine_scale = np.maximum(fine_se, np.finfo(float).eps)
    coarse_scale = np.maximum(coarse_se, np.finfo(float).eps)
    fine_standardized = np.abs(
        (bootstrap_fine - fine_estimate[None, 1:, :]) / fine_scale[None, ...]
    )[..., fine_valid]
    coarse_standardized = np.abs(
        (bootstrap_coarse - coarse_estimate[None, 1:, :])
        / coarse_scale[None, ...]
    )[..., coarse_valid]
    maximum_statistic = np.maximum(
        np.max(fine_standardized, axis=(1, 2)),
        np.max(coarse_standardized, axis=(1, 2)),
    )
    critical_value = float(np.quantile(maximum_statistic, confidence_level))

    def make_result(
        label: str,
        wavelength: np.ndarray,
        specimen_change: np.ndarray,
        estimate: np.ndarray,
        standard_error: np.ndarray,
        valid: np.ndarray,
    ) -> SpectralChangeResult:
        low = np.zeros_like(estimate)
        high = np.zeros_like(estimate)
        low[1:] = estimate[1:] - critical_value * standard_error
        high[1:] = estimate[1:] + critical_value * standard_error
        significant = np.zeros_like(estimate, dtype=bool)
        significant[1:] = (
            ((low[1:] > 0.0) | (high[1:] < 0.0)) & valid[None, :]
        )
        return SpectralChangeResult(
            label=label,
            wavelength_um=wavelength,
            log10_psd_ratio=specimen_change,
            estimate=estimate,
            simultaneous_ci_low=low,
            simultaneous_ci_high=high,
            significant=significant,
            valid_morphology_band=valid,
        )

    return (
        make_result(
            "50x",
            fine_wavelength,
            fine_change,
            fine_estimate,
            fine_se,
            fine_valid,
        ),
        make_result(
            "10x",
            coarse_wavelength,
            coarse_change,
            coarse_estimate,
            coarse_se,
            coarse_valid,
        ),
    )


def _stitched_power_relevance(
    fine: MagnificationEvolutionResult,
    coarse: MagnificationEvolutionResult,
    *,
    fine_valid_band_um: tuple[float, float],
    coarse_valid_band_um: tuple[float, float],
    fraction: float,
    use_median: bool,
    confidence_level: float,
    n_resamples: int,
    bootstrap_method: str,
    bins_per_decade: int,
    seed: int,
) -> PowerRelevanceResult:
    """Find the central height-variance interval in a stitched two-scale PSD."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("pertinent_power_fraction must lie in (0, 1).")
    fine_wavelength, fine_psd = _log_binned_radial_stack(
        fine,
        bins_per_decade=bins_per_decade,
    )
    coarse_wavelength, coarse_psd = _log_binned_radial_stack(
        coarse,
        bins_per_decade=bins_per_decade,
    )
    fine_selected = (
        (fine_wavelength >= fine_valid_band_um[0])
        & (fine_wavelength <= fine_valid_band_um[1])
    )
    coarse_selected = (
        (coarse_wavelength >= coarse_valid_band_um[0])
        & (coarse_wavelength <= coarse_valid_band_um[1])
    )
    wavelength = np.concatenate(
        (fine_wavelength[fine_selected], coarse_wavelength[coarse_selected])
    )
    psd = np.concatenate(
        (fine_psd[..., fine_selected], coarse_psd[..., coarse_selected]),
        axis=-1,
    )
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    psd = psd[..., order]
    if wavelength.size < 4:
        raise ValueError("Too few wavelengths remain for stitched PSD power.")

    log_wavelength = np.log(wavelength)
    frequency = 1.0 / wavelength
    contribution = 2.0 * np.pi * frequency**2 * psd
    segment_power = (
        0.5
        * (contribution[..., :-1] + contribution[..., 1:])
        * np.diff(log_wavelength)
    )
    cumulative = np.concatenate(
        (
            np.zeros((*segment_power.shape[:-1], 1), dtype=np.float64),
            np.cumsum(segment_power, axis=-1),
        ),
        axis=-1,
    )
    total = cumulative[..., [-1]]
    cumulative_fraction = cumulative / np.maximum(total, np.finfo(float).tiny)
    tail = 0.5 * (1.0 - fraction)
    specimen_bounds = np.full(
        (*psd.shape[:2], 2),
        np.nan,
        dtype=np.float64,
    )
    for time_index in range(psd.shape[0]):
        for specimen_index in range(psd.shape[1]):
            local_cumulative = cumulative_fraction[time_index, specimen_index]
            if not np.all(np.isfinite(local_cumulative)):
                continue
            specimen_bounds[time_index, specimen_index] = np.interp(
                (tail, 1.0 - tail),
                local_cumulative,
                wavelength,
            )

    estimate, low, high = _specimen_summary_with_ci(
        specimen_bounds,
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
        bootstrap_method=bootstrap_method,
    )
    return PowerRelevanceResult(
        fraction=float(fraction),
        specimen_bounds_um=specimen_bounds,
        estimate_bounds_um=estimate,
        ci_low_bounds_um=low,
        ci_high_bounds_um=high,
    )


def _cross_magnification_analysis(
    fine: MagnificationEvolutionResult,
    coarse: MagnificationEvolutionResult,
    times_hours: np.ndarray,
    *,
    overlap_wavelength_um: tuple[float, float],
    output_dir: Path,
    use_median: bool,
    confidence_level: float,
    n_resamples: int,
    bootstrap_method: str,
    seed: int,
) -> np.ndarray:
    cross_dir = output_dir / "cross_magnification"
    cross_dir.mkdir(parents=True, exist_ok=True)
    wavelength_grid = np.geomspace(
        overlap_wavelength_um[0],
        overlap_wavelength_um[1],
        200,
    )
    fine_log_psd = _interpolate_radial_log_psd(
        fine.radial_results,
        wavelength_grid,
    )
    coarse_log_psd = _interpolate_radial_log_psd(
        coarse.radial_results,
        wavelength_grid,
    )
    log_ratio = fine_log_psd - coarse_log_psd
    ratio_estimate, ratio_low, ratio_high = _specimen_summary_with_ci(
        log_ratio,
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
        bootstrap_method=bootstrap_method,
    )

    color_limit = float(np.nanmax(np.abs(ratio_estimate)))
    color_limit = max(color_limit, np.finfo(float).eps)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.pcolormesh(
        wavelength_grid,
        times_hours,
        ratio_estimate,
        shading="auto",
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Hold time (h)")
    ax.set_title("50x / 10x PSD agreement in the overlap band")
    fig.colorbar(image, ax=ax, label="log10(PSD50x / PSD10x)")
    fig.tight_layout()
    fig.savefig(cross_dir / "overlap_psd_log10_ratio.png", dpi=250)
    plt.close(fig)

    for time_index, time_hours in enumerate(times_hours):
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for result, curve_label, color in (
            (fine.radial_results[time_index], "50x", "black"),
            (coarse.radial_results[time_index], "10x", "tab:red"),
        ):
            retained = (
                (result.wavelength_um >= overlap_wavelength_um[0])
                & (result.wavelength_um <= overlap_wavelength_um[1])
            )
            order = np.argsort(result.wavelength_um[retained])
            wavelength = result.wavelength_um[retained][order]
            estimate = result.estimate_psd_um4[retained][order]
            low = result.ci_low_um4[retained][order]
            high = result.ci_high_um4[retained][order]
            ax.plot(wavelength, estimate, color=color, label=curve_label)
            ax.fill_between(wavelength, low, high, color=color, alpha=0.18)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("Radial 2D PSD (um^4)")
        ax.set_title(f"Cross-magnification overlap, {time_hours:g} h")
        ax.legend()
        ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            cross_dir / f"overlap_psd_time_{time_index:03d}.png",
            dpi=250,
        )
        plt.close(fig)

    fine_overlap_index = fine.band_names.index("overlap")
    coarse_overlap_index = coarse.band_names.index("overlap")
    integrated_log_ratio = np.log10(
        np.maximum(
            fine.band_power_um2[:, :, fine_overlap_index],
            np.finfo(float).tiny,
        )
        / np.maximum(
            coarse.band_power_um2[:, :, coarse_overlap_index],
            np.finfo(float).tiny,
        )
    )
    integrated_estimate, integrated_low, integrated_high = (
        _specimen_summary_with_ci(
            integrated_log_ratio[:, :, None],
            use_median=use_median,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed + 1,
            bootstrap_method=bootstrap_method,
        )
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(times_hours, integrated_estimate[:, 0], marker="o", color="black")
    ax.fill_between(
        times_hours,
        integrated_low[:, 0],
        integrated_high[:, 0],
        color="0.75",
    )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("log10 overlap-power ratio (50x / 10x)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(cross_dir / "overlap_integrated_power_ratio.png", dpi=250)
    plt.close(fig)

    np.savez_compressed(
        cross_dir / "cross_magnification_results.npz",
        times_hours=times_hours,
        overlap_wavelength_um=np.asarray(overlap_wavelength_um),
        wavelength_grid_um=wavelength_grid,
        specimen_log10_psd_ratio=log_ratio,
        log10_psd_ratio_estimate=ratio_estimate,
        log10_psd_ratio_ci_low=ratio_low,
        log10_psd_ratio_ci_high=ratio_high,
        specimen_integrated_log10_power_ratio=integrated_log_ratio,
        integrated_log10_power_ratio_estimate=integrated_estimate[:, 0],
        integrated_log10_power_ratio_ci_low=integrated_low[:, 0],
        integrated_log10_power_ratio_ci_high=integrated_high[:, 0],
    )
    return log_ratio


def _spectral_change_intervals(
    result: SpectralChangeResult,
    times_hours: np.ndarray,
) -> list[tuple[str, float, float, float, str]]:
    intervals = []
    wavelength = result.wavelength_um
    for time_index in range(1, len(times_hours)):
        direction = np.sign(result.estimate[time_index]).astype(np.int8)
        active = result.significant[time_index]
        start = None
        active_direction = 0
        for index in range(wavelength.size + 1):
            current_active = index < wavelength.size and active[index]
            current_direction = direction[index] if current_active else 0
            begins_new = (
                current_active
                and (start is None or current_direction != active_direction)
            )
            if start is not None and (
                not current_active or current_direction != active_direction
            ):
                stop = index - 1
                intervals.append(
                    (
                        result.label,
                        float(times_hours[time_index]),
                        float(wavelength[start]),
                        float(wavelength[stop]),
                        "increase" if active_direction > 0 else "decrease",
                    )
                )
                start = None
            if begins_new:
                start = index
                active_direction = int(current_direction)
    return intervals


def _save_wavelength_selection_results(
    selection: WavelengthSelectionResult,
    *,
    fine: MagnificationEvolutionResult,
    coarse: MagnificationEvolutionResult,
    confidence_level: float,
    required_delta_bic: float,
    use_median: bool,
) -> None:
    output_dir = selection.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    small = selection.small
    large = selection.large
    full_index = int(
        np.flatnonzero(np.isclose(large.window_fractions, 1.0))[0]
    )

    np.savez_compressed(
        output_dir / "wavelength_selection_results.npz",
        times_hours=small.times_hours,
        small_wavelength_um=small.wavelength_um,
        small_normalized_ratio_axis0=small.normalized_ratio_axis0,
        small_normalized_ratio_axis1=small.normalized_ratio_axis1,
        small_cutoff_axis0_um=small.cutoff_axis0_um,
        small_cutoff_axis1_um=small.cutoff_axis1_um,
        small_cutoff_um=small.cutoff_um,
        small_cutoff_ci_low_um=small.ci_low_um,
        small_cutoff_ci_high_um=small.ci_high_um,
        small_cutoff_detected=small.detected,
        small_bootstrap_detection_fraction=(
            small.bootstrap_detection_fraction
        ),
        small_status=small.status,
        large_window_fractions=large.window_fractions,
        large_window_short_side_um=large.window_short_side_um,
        large_cutoff_um=large.cutoff_um,
        large_cutoff_ci_low_um=large.ci_low_um,
        large_cutoff_ci_high_um=large.ci_high_um,
        large_delta_bic=large.delta_bic,
        large_scaling_slope=large.scaling_slope,
        large_cutoff_detected=large.detected,
        large_bootstrap_detection_fraction=(
            large.bootstrap_detection_fraction
        ),
        large_full_window_stable=large.full_window_stable,
        fine_change_wavelength_um=selection.fine_change.wavelength_um,
        fine_specimen_log10_psd_ratio=(
            selection.fine_change.log10_psd_ratio
        ),
        fine_change_estimate=selection.fine_change.estimate,
        fine_change_simultaneous_ci_low=(
            selection.fine_change.simultaneous_ci_low
        ),
        fine_change_simultaneous_ci_high=(
            selection.fine_change.simultaneous_ci_high
        ),
        fine_change_significant=selection.fine_change.significant,
        fine_change_valid_band=(
            selection.fine_change.valid_morphology_band
        ),
        coarse_change_wavelength_um=selection.coarse_change.wavelength_um,
        coarse_specimen_log10_psd_ratio=(
            selection.coarse_change.log10_psd_ratio
        ),
        coarse_change_estimate=selection.coarse_change.estimate,
        coarse_change_simultaneous_ci_low=(
            selection.coarse_change.simultaneous_ci_low
        ),
        coarse_change_simultaneous_ci_high=(
            selection.coarse_change.simultaneous_ci_high
        ),
        coarse_change_significant=selection.coarse_change.significant,
        coarse_change_valid_band=(
            selection.coarse_change.valid_morphology_band
        ),
        power_fraction=np.asarray(selection.power.fraction),
        power_specimen_bounds_um=selection.power.specimen_bounds_um,
        power_estimate_bounds_um=selection.power.estimate_bounds_um,
        power_ci_low_bounds_um=selection.power.ci_low_bounds_um,
        power_ci_high_bounds_um=selection.power.ci_high_bounds_um,
        morphology_lambda_min_um=np.asarray(
            selection.morphology_lambda_min_um
        ),
        morphology_lambda_max_um=np.asarray(
            selection.morphology_lambda_max_um
        ),
        power_lambda_min_um=np.asarray(selection.power_lambda_min_um),
        power_lambda_max_um=np.asarray(selection.power_lambda_max_um),
        evolution_lambda_min_um=np.asarray(selection.evolution_lambda_min_um),
        evolution_lambda_max_um=np.asarray(selection.evolution_lambda_max_um),
    )

    rows = [
        (
            "time_hours,lambda_min_50x_um,lambda_min_ci_low_um,"
            "lambda_min_ci_high_um,lambda_min_detected,"
            "lambda_min_bootstrap_support,lambda_min_status,lambda_max_10x_um,"
            "lambda_max_ci_low_um,lambda_max_ci_high_um,"
            "lambda_max_delta_bic,lambda_max_detected,"
            "lambda_max_window_stable"
        )
    ]
    for time_index, time_hours in enumerate(small.times_hours):
        rows.append(
            ",".join(
                [
                    f"{time_hours:.12g}",
                    f"{small.cutoff_um[time_index]:.12g}",
                    f"{small.ci_low_um[time_index]:.12g}",
                    f"{small.ci_high_um[time_index]:.12g}",
                    str(bool(small.detected[time_index])).lower(),
                    (
                        f"{small.bootstrap_detection_fraction[time_index]:.12g}"
                    ),
                    str(small.status[time_index]),
                    f"{large.cutoff_um[full_index, time_index]:.12g}",
                    f"{large.ci_low_um[full_index, time_index]:.12g}",
                    f"{large.ci_high_um[full_index, time_index]:.12g}",
                    f"{large.delta_bic[full_index, time_index]:.12g}",
                    str(bool(large.detected[full_index, time_index])).lower(),
                    str(bool(large.full_window_stable[time_index])).lower(),
                ]
            )
        )
    (output_dir / "wavelength_cutoffs_by_time.csv").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    power_rows = [
        (
            "time_hours,power_fraction,lambda_low_um,lambda_low_ci_low_um,"
            "lambda_low_ci_high_um,lambda_high_um,lambda_high_ci_low_um,"
            "lambda_high_ci_high_um"
        )
    ]
    for time_index, time_hours in enumerate(small.times_hours):
        power_rows.append(
            ",".join(
                [
                    f"{time_hours:.12g}",
                    f"{selection.power.fraction:.12g}",
                    f"{selection.power.estimate_bounds_um[time_index, 0]:.12g}",
                    f"{selection.power.ci_low_bounds_um[time_index, 0]:.12g}",
                    f"{selection.power.ci_high_bounds_um[time_index, 0]:.12g}",
                    f"{selection.power.estimate_bounds_um[time_index, 1]:.12g}",
                    f"{selection.power.ci_low_bounds_um[time_index, 1]:.12g}",
                    f"{selection.power.ci_high_bounds_um[time_index, 1]:.12g}",
                ]
            )
        )
    (output_dir / "power_relevant_wavelengths_by_time.csv").write_text(
        "\n".join(power_rows) + "\n",
        encoding="utf-8",
    )

    intervals = _spectral_change_intervals(
        selection.fine_change,
        small.times_hours,
    ) + _spectral_change_intervals(
        selection.coarse_change,
        small.times_hours,
    )
    interval_rows = [
        "objective,time_hours,wavelength_start_um,wavelength_stop_um,direction"
    ]
    interval_rows.extend(
        f"{label},{time:.12g},{start:.12g},{stop:.12g},{direction}"
        for label, time, start, stop, direction in intervals
    )
    (output_dir / "evolution_relevant_wavelength_intervals.csv").write_text(
        "\n".join(interval_rows) + "\n",
        encoding="utf-8",
    )

    estimator = np.nanmedian if use_median else np.nanmean
    order = np.argsort(small.wavelength_um)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for axis_index, (ax, ratios, axis_name) in enumerate(
        (
            (axes[0], small.normalized_ratio_axis0, "axis 0 line PSD"),
            (axes[1], small.normalized_ratio_axis1, "axis 1 line PSD"),
        )
    ):
        for time_index, time_hours in enumerate(small.times_hours):
            ax.plot(
                small.wavelength_um[order],
                estimator(ratios[time_index], axis=0)[order],
                label=f"{time_hours:g} h",
            )
            cutoff_value = (
                small.cutoff_axis0_um[time_index]
                if axis_index == 0
                else small.cutoff_axis1_um[time_index]
            )
            if np.isfinite(cutoff_value):
                ax.axvline(cutoff_value, color="0.5", linewidth=0.5, alpha=0.5)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wavelength (um)")
        ax.set_title(axis_name)
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("(C2D / C1D) / transverse pixel spacing")
    axes[1].legend(fontsize="small")
    fig.suptitle("50x data-derived small-wavelength noise cutoff")
    fig.tight_layout()
    fig.savefig(output_dir / "50x_small_wavelength_cutoff.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for fraction_index, fraction in enumerate(large.window_fractions):
        values = large.cutoff_um[fraction_index]
        low = large.ci_low_um[fraction_index]
        high = large.ci_high_um[fraction_index]
        ax.plot(
            large.times_hours,
            values,
            marker="o",
            label=(
                f"{fraction:g} field "
                f"({large.window_short_side_um[fraction_index]:.0f} um)"
            ),
        )
        ax.fill_between(large.times_hours, low, high, alpha=0.15)
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Low-frequency roll-off wavelength (um)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    ax.set_title("10x roll-off and subfield stability")
    fig.tight_layout()
    fig.savefig(output_dir / "10x_large_wavelength_rolloff.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    power = selection.power
    ax.plot(
        small.times_hours,
        power.estimate_bounds_um[:, 0],
        marker="o",
        label="lower bound",
    )
    ax.fill_between(
        small.times_hours,
        power.ci_low_bounds_um[:, 0],
        power.ci_high_bounds_um[:, 0],
        alpha=0.18,
    )
    ax.plot(
        small.times_hours,
        power.estimate_bounds_um[:, 1],
        marker="o",
        label="upper bound",
    )
    ax.fill_between(
        small.times_hours,
        power.ci_low_bounds_um[:, 1],
        power.ci_high_bounds_um[:, 1],
        alpha=0.18,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Wavelength (um)")
    ax.set_title(
        f"Central {100 * power.fraction:g}% of measured height variance"
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "power_relevant_wavelength_interval.png", dpi=250)
    plt.close(fig)

    for change in (selection.fine_change, selection.coarse_change):
        valid_estimate = np.where(
            change.valid_morphology_band[None, :],
            change.estimate,
            np.nan,
        )
        limit = float(np.nanmax(np.abs(valid_estimate)))
        limit = max(limit, np.finfo(float).eps)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        image = ax.pcolormesh(
            change.wavelength_um,
            small.times_hours,
            valid_estimate,
            shading="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        if np.any(change.significant) and not np.all(change.significant):
            ax.contour(
                change.wavelength_um,
                small.times_hours,
                change.significant.astype(float),
                levels=[0.5],
                colors="black",
                linewidths=0.8,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("Hold time (h)")
        ax.set_title(
            f"{change.label} paired PSD change; black = simultaneous significance"
        )
        fig.colorbar(image, ax=ax, label="log10[PSD(t) / PSD(0)]")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{change.label}_paired_spectral_change.png",
            dpi=250,
        )
        plt.close(fig)

    def formatted(value: float) -> str:
        return "unresolved" if not np.isfinite(value) else f"{value:.6g} um"

    report_lines = [
        "# Data-derived wavelength selection",
        "",
        "## Results used for simulation design",
        "",
        (
            "- Conservative morphology lambda_min: "
            f"{formatted(selection.morphology_lambda_min_um)}"
        ),
        (
            "- Conservative morphology lambda_max: "
            f"{formatted(selection.morphology_lambda_max_um)}"
        ),
        (
            f"- Campaign interval containing the central "
            f"{100 * selection.power.fraction:g}% of measured height variance: "
            f"{formatted(selection.power_lambda_min_um)} to "
            f"{formatted(selection.power_lambda_max_um)}"
        ),
        (
            "- Smallest wavelength with a statistically supported PSD change: "
            f"{formatted(selection.evolution_lambda_min_um)}"
        ),
        (
            "- Largest wavelength with a statistically supported PSD change: "
            f"{formatted(selection.evolution_lambda_max_um)}"
        ),
        "",
        "These are not the 50x/10x overlap endpoints. The overlap is used only "
        "as the objective handoff and consistency check.",
        "",
        "## Exact decision rules",
        "",
        (
            "1. lambda_min is obtained from the 50x crossing where C2D/C1D "
            "equals the transverse pixel spacing, evaluated independently along "
            "both axes. The larger axis cutoff is retained. The value above is "
            f"the largest upper {100 * confidence_level:g}% specimen-bootstrap "
            "limit across time. A crossing is accepted only when at least the "
            f"same {100 * confidence_level:g}% fraction of bootstrap resamples "
            "also yields a crossing."
        ),
        (
            "2. lambda_max is the 10x breakpoint of a constant low-frequency PSD "
            "plateau joined continuously to a log-log scaling line. The roll-off "
            "fit excludes wavelengths below the independent 10x sampling "
            "guardrail. The "
            f"model must improve BIC by at least {required_delta_bic:g}, have a "
            "negative scaling slope, and have overlapping bootstrap intervals in "
            "every requested subfield size; every fit must also recur in at least "
            f"{100 * confidence_level:g}% of bootstrap resamples. If any time "
            "fails, the campaign-wide "
            "lambda_max is reported as unresolved."
        ),
        (
            "3. Evolution-relevant intervals use paired specimen curves "
            "log10[PSD(t)/PSD(0)]. One specimen-resampling bootstrap constructs a "
            f"simultaneous {100 * confidence_level:g}% band over both objectives, "
            "all nonzero times, and all tested wavelengths. A wavelength is "
            "called changed only where that simultaneous band excludes zero."
        ),
        (
            f"4. Power relevance is the central "
            f"{100 * selection.power.fraction:g}% of Sq-squared obtained by "
            "integrating 2*pi*f^2*Ciso over log wavelength. The 50x spectrum "
            "supplies the short side and the 10x spectrum the long side; they "
            "are joined inside the overlap without amplitude rescaling."
        ),
        "",
        "## Censoring and interpretation",
        "",
        (
            f"- Shortest 50x radial wavelength actually tested: "
            f"{np.min(small.wavelength_um):.6g} um. If lambda_min is unresolved, "
            "consult `lambda_min_status` in the cutoff table. "
            "`below_measured_range` means noise dominance was not reached; "
            "`no_signal_dominated_range` means the opposite; and "
            "`bootstrap_unstable` means the apparent crossing was not reproducible."
        ),
        (
            f"- Longest 10x radial wavelength actually tested: "
            f"{np.max(coarse.radial_results[0].wavelength_um):.6g} um. If "
            "lambda_max is unresolved, the data do not establish a finite upper "
            "correlation wavelength."
        ),
        "- Sal remains an ISO-style correlation *distance* at ACF = 0.2; it is "
        "reported elsewhere and is not silently converted into a wavelength.",
        "",
        "## Method sources",
        "",
        "- Jacobs, Junge, and Pastewka (2017), Quantitative characterization of "
        "surface topography using spectral analysis: "
        "https://doi.org/10.1088/2051-672X/aa51f8",
        "- ISO 25178-2:2021 areal surface-texture parameter definitions: "
        "https://www.iso.org/standard/74591.html",
        "- Singh, Paliwal, and Kasamias (2024), representative-area convergence: "
        "https://doi.org/10.1038/s41598-024-52329-4",
    ]
    (output_dir / "wavelength_selection_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def _run_wavelength_selection(
    fine: MagnificationEvolutionResult,
    coarse: MagnificationEvolutionResult,
    height_maps_10x: np.ndarray,
    spacing_10x: float | tuple[float, float],
    times_hours: np.ndarray,
    *,
    crop_slices: tuple[slice, slice],
    center_mask_diagonal_um: float,
    mask_taper_um: float,
    overlap_wavelength_um: tuple[float, float],
    output_dir: Path,
    use_median: bool,
    confidence_level: float,
    n_resamples: int,
    bootstrap_method: str,
    pertinent_power_fraction: float,
    bins_per_decade: int,
    min_segment_bins: int,
    required_delta_bic: float,
    rolloff_window_fractions: tuple[float, ...],
    min_modes_per_bin: int,
    seed: int,
) -> WavelengthSelectionResult:
    selection_dir = output_dir / "wavelength_selection"
    small = _small_wavelength_noise_cutoff(
        fine,
        times_hours,
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bins_per_decade=bins_per_decade,
        seed=seed,
    )

    coarse_maps, coarse_mask = _prepare_uncropped_height_maps(
        height_maps_10x,
        spacing_um=spacing_10x,
        crop_slices=crop_slices,
        center_mask_diagonal_um=center_mask_diagonal_um,
    )
    subfield_spectra = _subfield_radial_spectra(
        coarse,
        coarse_maps,
        coarse_mask.excluded_masks,
        spacing_um=spacing_10x,
        window_fractions=rolloff_window_fractions,
        mask_taper_um=mask_taper_um,
        min_modes_per_bin=min_modes_per_bin,
        use_median=use_median,
    )
    large = _large_wavelength_rolloff(
        subfield_spectra,
        times_hours,
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bins_per_decade=bins_per_decade,
        min_segment_bins=min_segment_bins,
        minimum_fit_wavelength_um=coarse.wavelength_limits_um[0],
        required_delta_bic=required_delta_bic,
        seed=seed + 100_000,
    )

    supported_small = small.detected & np.isfinite(small.ci_high_um)
    morphology_min = (
        float(np.max(small.ci_high_um[supported_small]))
        if np.any(supported_small)
        else np.nan
    )
    full_index = int(
        np.flatnonzero(np.isclose(large.window_fractions, 1.0))[0]
    )
    all_large_supported = np.all(large.full_window_stable) and np.all(
        np.isfinite(large.ci_high_um[full_index])
    )
    morphology_max = (
        float(np.max(large.ci_high_um[full_index]))
        if all_large_supported
        else np.nan
    )

    handoff = float(np.sqrt(np.prod(overlap_wavelength_um)))
    fine_lower = (
        morphology_min
        if np.isfinite(morphology_min)
        else float(np.min(fine.radial_results[0].wavelength_um))
    )
    coarse_upper = (
        morphology_max
        if np.isfinite(morphology_max)
        else float(np.max(coarse.radial_results[0].wavelength_um))
    )
    fine_change, coarse_change = _joint_paired_spectral_changes(
        fine,
        coarse,
        fine_valid_band_um=(fine_lower, handoff),
        coarse_valid_band_um=(handoff, coarse_upper),
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bins_per_decade=bins_per_decade,
        seed=seed + 200_000,
    )
    power = _stitched_power_relevance(
        fine,
        coarse,
        fine_valid_band_um=(fine_lower, handoff),
        coarse_valid_band_um=(handoff, coarse_upper),
        fraction=pertinent_power_fraction,
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        bins_per_decade=bins_per_decade,
        seed=seed + 300_000,
    )
    finite_power_low = np.where(
        np.isfinite(power.ci_low_bounds_um[:, 0]),
        power.ci_low_bounds_um[:, 0],
        power.estimate_bounds_um[:, 0],
    )
    finite_power_high = np.where(
        np.isfinite(power.ci_high_bounds_um[:, 1]),
        power.ci_high_bounds_um[:, 1],
        power.estimate_bounds_um[:, 1],
    )
    power_min = float(np.nanmin(finite_power_low))
    power_max = float(np.nanmax(finite_power_high))
    relevant_wavelengths = []
    for change in (fine_change, coarse_change):
        relevant = np.any(change.significant, axis=0)
        relevant_wavelengths.extend(change.wavelength_um[relevant].tolist())
    evolution_min = (
        float(np.min(relevant_wavelengths)) if relevant_wavelengths else np.nan
    )
    evolution_max = (
        float(np.max(relevant_wavelengths)) if relevant_wavelengths else np.nan
    )

    selection = WavelengthSelectionResult(
        small=small,
        large=large,
        fine_change=fine_change,
        coarse_change=coarse_change,
        power=power,
        morphology_lambda_min_um=morphology_min,
        morphology_lambda_max_um=morphology_max,
        power_lambda_min_um=power_min,
        power_lambda_max_um=power_max,
        evolution_lambda_min_um=evolution_min,
        evolution_lambda_max_um=evolution_max,
        output_dir=selection_dir,
    )
    _save_wavelength_selection_results(
        selection,
        fine=fine,
        coarse=coarse,
        confidence_level=confidence_level,
        required_delta_bic=required_delta_bic,
        use_median=use_median,
    )
    return selection


def run_complete_analysis(
    height_maps_50x: np.ndarray,
    spacing_50x: float | tuple[float, float],
    height_maps_10x: np.ndarray,
    spacing_10x: float | tuple[float, float],
    times_hours: np.ndarray,
    *,
    output_dir: str | Path = "profilometry_complete_analysis",
    crop_slices: tuple[slice, slice] = (slice(50, -50), slice(50, 750)),
    center_mask_diagonal_um: float = 50.0,
    mask_taper_um: float = 5.0,
    use_median: bool = False,
    confidence_level: float = 0.95,
    n_resamples: int = 20_000,
    bootstrap_method: str = "BCa",
    min_modes_per_bin: int = 8,
    seed: int = 12345,
    plot_specimen_2d: bool = True,
    directional_sector_width_deg: float = 15.0,
    acf_threshold: float = 0.2,
    acf_n_directions: int = 720,
    points_per_shortest_wavelength: float = 5.5,
    cycles_per_longest_wavelength: float = 5.0,
    nested_window_fractions: tuple[float, ...] = (0.50, 0.75, 1.00),
    wavelength_n_resamples: int = 2_000,
    wavelength_log_bins_per_decade: int = 12,
    pertinent_power_fraction: float = 0.98,
    rolloff_min_segment_bins: int = 4,
    rolloff_required_delta_bic: float = 10.0,
    rolloff_window_fractions: tuple[float, ...] = (0.50, 0.75, 1.00),
    mask_sensitivity_diagonals_um: tuple[float | None, ...] = (
        None,
        40.0,
        50.0,
        60.0,
    ),
) -> CompleteAnalysisResult:
    """
    Run the complete longitudinal roughness analysis at 50x and 10x.

    Both height arrays must be uncropped, plane-leveled maps in micrometers,
    ordered identically on the time and specimen axes. The default processing
    first constructs a 50 um tip-to-tip diamond at the original array center,
    then crops both maps and masks with ``[..., 50:-50, 50:750]``.

    The analysis covers every time: full, radial, and directional PSD; radial
    specimen-bootstrap intervals; integrated wavelength-band power; spectral
    peak and cumulative-power length scales; 2D ACF with Sal and Str; common
    areal amplitude/distribution/slope parameters; paired changes from t=0;
    nested-window convergence; center-mask sensitivity; a 50x data-derived
    small-wavelength noise cutoff; a 10x PSD roll-off with translated-subfield
    validation; simultaneous inference for wavelength-specific temporal PSD
    changes; and a quantitative 50x/10x overlap check. Integer phase-correlation
    shifts to t=0 are saved as a same-site diagnostic but are not applied to
    PSD/ACF maps, avoiding interpolation-induced short-wavelength attenuation.

    Results are returned and written into separate 50x, 10x, evolution,
    per-time, mask-diagnostic, and cross-magnification directories.
    """
    maps_50x = np.asarray(height_maps_50x, dtype=np.float64)
    maps_10x = np.asarray(height_maps_10x, dtype=np.float64)
    if maps_50x.ndim != 4 or maps_10x.ndim != 4:
        raise ValueError(
            "Both height-map inputs must have shape "
            "(ntimes, nsamples, n0, n1)."
        )
    if maps_50x.shape[:2] != maps_10x.shape[:2]:
        raise ValueError(
            "50x and 10x inputs must have matching time and specimen axes."
        )
    if maps_50x.shape[-2:] != maps_10x.shape[-2:]:
        raise ValueError("50x and 10x original map shapes must match.")
    if not np.all(np.isfinite(maps_50x)) or not np.all(np.isfinite(maps_10x)):
        raise ValueError("Height maps contain NaN or infinite values.")

    times = np.asarray(times_hours, dtype=np.float64)
    if times.ndim != 1 or times.size != maps_50x.shape[0]:
        raise ValueError("times_hours must have one value per time index.")
    if times.size < 2:
        raise ValueError("At least two times are required for evolution analysis.")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_hours must be finite and strictly increasing.")
    if wavelength_n_resamples < 100:
        raise ValueError("wavelength_n_resamples must be at least 100.")

    spacing_fine = _as_spacing_tuple(spacing_50x)
    spacing_coarse = _as_spacing_tuple(spacing_10x)
    if max(spacing_fine) >= min(spacing_coarse):
        raise ValueError("spacing_50x must be finer than spacing_10x.")
    normalized_crop, _ = _normalized_crop_slices(
        maps_50x.shape[-2:],
        crop_slices,
    )
    cropped_shape = (
        normalized_crop[0].stop - normalized_crop[0].start,
        normalized_crop[1].stop - normalized_crop[1].start,
    )
    fine_limits = _analysis_wavelength_limits(
        cropped_shape,
        spacing_fine,
        points_per_shortest_wavelength=points_per_shortest_wavelength,
        cycles_per_longest_wavelength=cycles_per_longest_wavelength,
    )
    coarse_limits = _analysis_wavelength_limits(
        cropped_shape,
        spacing_coarse,
        points_per_shortest_wavelength=points_per_shortest_wavelength,
        cycles_per_longest_wavelength=cycles_per_longest_wavelength,
    )
    overlap = (
        max(fine_limits[0], coarse_limits[0]),
        min(fine_limits[1], coarse_limits[1]),
    )
    if overlap[1] <= overlap[0]:
        raise ValueError("The 50x and 10x acquisition guardrails do not overlap.")

    fine_band_names = []
    fine_band_bounds = []
    if fine_limits[0] < overlap[0]:
        fine_band_names.append("fine_short")
        fine_band_bounds.append((fine_limits[0], overlap[0]))
    fine_band_names.append("overlap")
    fine_band_bounds.append(overlap)

    coarse_band_names = ["overlap"]
    coarse_band_bounds = [overlap]
    if overlap[1] < coarse_limits[1]:
        coarse_band_names.append("coarse_long")
        coarse_band_bounds.append((overlap[1], coarse_limits[1]))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(
        "Acquisition guardrails for plotting/integration (um):",
        {"50x": fine_limits, "10x": coarse_limits, "overlap": overlap},
    )
    fine = _run_magnification_evolution(
        "50x",
        maps_50x,
        spacing_fine,
        times,
        crop_slices=crop_slices,
        center_mask_diagonal_um=center_mask_diagonal_um,
        mask_taper_um=mask_taper_um,
        wavelength_limits_um=fine_limits,
        band_names=tuple(fine_band_names),
        band_bounds_um=np.asarray(fine_band_bounds, dtype=np.float64),
        output_dir=output_path,
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        min_modes_per_bin=min_modes_per_bin,
        seed=seed,
        plot_specimen_2d=plot_specimen_2d,
        directional_sector_width_deg=directional_sector_width_deg,
        acf_threshold=acf_threshold,
        acf_n_directions=acf_n_directions,
        nested_window_fractions=nested_window_fractions,
        mask_sensitivity_diagonals_um=mask_sensitivity_diagonals_um,
    )
    coarse = _run_magnification_evolution(
        "10x",
        maps_10x,
        spacing_coarse,
        times,
        crop_slices=crop_slices,
        center_mask_diagonal_um=center_mask_diagonal_um,
        mask_taper_um=mask_taper_um,
        wavelength_limits_um=coarse_limits,
        band_names=tuple(coarse_band_names),
        band_bounds_um=np.asarray(coarse_band_bounds, dtype=np.float64),
        output_dir=output_path,
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        min_modes_per_bin=min_modes_per_bin,
        seed=seed + 10_000,
        plot_specimen_2d=plot_specimen_2d,
        directional_sector_width_deg=directional_sector_width_deg,
        acf_threshold=acf_threshold,
        acf_n_directions=acf_n_directions,
        nested_window_fractions=nested_window_fractions,
        mask_sensitivity_diagonals_um=mask_sensitivity_diagonals_um,
    )
    overlap_log_ratio = _cross_magnification_analysis(
        fine,
        coarse,
        times,
        overlap_wavelength_um=overlap,
        output_dir=output_path,
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        seed=seed + 20_000,
    )
    wavelength_selection = _run_wavelength_selection(
        fine,
        coarse,
        maps_10x,
        spacing_coarse,
        times,
        crop_slices=crop_slices,
        center_mask_diagonal_um=center_mask_diagonal_um,
        mask_taper_um=mask_taper_um,
        overlap_wavelength_um=overlap,
        output_dir=output_path,
        use_median=use_median,
        confidence_level=confidence_level,
        n_resamples=wavelength_n_resamples,
        bootstrap_method=bootstrap_method,
        pertinent_power_fraction=pertinent_power_fraction,
        bins_per_decade=wavelength_log_bins_per_decade,
        min_segment_bins=rolloff_min_segment_bins,
        required_delta_bic=rolloff_required_delta_bic,
        rolloff_window_fractions=rolloff_window_fractions,
        min_modes_per_bin=min_modes_per_bin,
        seed=seed + 30_000,
    )
    print(
        "Data-derived wavelength results (um):",
        {
            "morphology_lambda_min": (
                wavelength_selection.morphology_lambda_min_um
            ),
            "morphology_lambda_max": (
                wavelength_selection.morphology_lambda_max_um
            ),
            "power_lambda_min": wavelength_selection.power_lambda_min_um,
            "power_lambda_max": wavelength_selection.power_lambda_max_um,
            "evolution_lambda_min": (
                wavelength_selection.evolution_lambda_min_um
            ),
            "evolution_lambda_max": (
                wavelength_selection.evolution_lambda_max_um
            ),
        },
    )
    print(
        "Wavelength report:",
        (wavelength_selection.output_dir / "wavelength_selection_report.md").resolve(),
    )
    print(
        "Cutoff table:",
        (wavelength_selection.output_dir / "wavelength_cutoffs_by_time.csv").resolve(),
    )

    np.savez_compressed(
        output_path / "complete_analysis_manifest.npz",
        times_hours=times,
        original_shape=np.asarray(maps_50x.shape[-2:]),
        cropped_shape=np.asarray(cropped_shape),
        spacing_50x_um=np.asarray(spacing_fine),
        spacing_10x_um=np.asarray(spacing_coarse),
        wavelength_limits_50x_um=np.asarray(fine_limits),
        wavelength_limits_10x_um=np.asarray(coarse_limits),
        overlap_wavelength_um=np.asarray(overlap),
        morphology_lambda_min_um=np.asarray(
            wavelength_selection.morphology_lambda_min_um
        ),
        morphology_lambda_max_um=np.asarray(
            wavelength_selection.morphology_lambda_max_um
        ),
        power_lambda_min_um=np.asarray(
            wavelength_selection.power_lambda_min_um
        ),
        power_lambda_max_um=np.asarray(
            wavelength_selection.power_lambda_max_um
        ),
        evolution_lambda_min_um=np.asarray(
            wavelength_selection.evolution_lambda_min_um
        ),
        evolution_lambda_max_um=np.asarray(
            wavelength_selection.evolution_lambda_max_um
        ),
        center_mask_diagonal_um=np.asarray(center_mask_diagonal_um),
        mask_taper_um=np.asarray(mask_taper_um),
        use_median=np.asarray(use_median),
        crop_axis0_start=np.asarray(normalized_crop[0].start),
        crop_axis0_stop=np.asarray(normalized_crop[0].stop),
        crop_axis1_start=np.asarray(normalized_crop[1].start),
        crop_axis1_stop=np.asarray(normalized_crop[1].stop),
    )
    return CompleteAnalysisResult(
        times_hours=times,
        fine=fine,
        coarse=coarse,
        overlap_wavelength_um=overlap,
        overlap_log10_psd_ratio=overlap_log_ratio,
        wavelength_selection=wavelength_selection,
        output_dir=output_path,
    )


def run_analysis(
    height_maps_um: np.ndarray,
    *,
    spacing_um: float | tuple[float, float],
    time_index: int,
    output_dir: str | Path = "psd_output",
    confidence_level: float = 0.95,
    n_resamples: int = 20_000,
    bootstrap_method: str = "BCa",
    min_modes_per_bin: int = 8,
    seed: int = 12345,
    plot_specimen_2d: bool = True,
    use_median: bool = False,
    center_mask_diagonal_um: float | None = None,
    center_mask_offset_um: tuple[float, float] | None = None,
    center_mask_original_shape: tuple[int, int] | None = None,
    center_mask_crop_slices: tuple[slice, slice] | None = None,
    mask_center_indent: bool = False,
    center_search_fraction: float = 0.50,
    detection_smoothing_um: float | None = None,
    indent_threshold_sigma: float = 4.0,
    indent_guard_um: float | None = None,
    guard_fraction_of_equivalent_radius: float = 0.25,
    common_mask_across_specimens: bool = True,
    mask_taper_um: float | None = None,
) -> PSDAnalysisResult:
    """Run PSD analysis with an optional fixed physical center mask."""
    indent_mask_result = None
    exclusion_masks = None
    crop_geometry_supplied = (
        center_mask_original_shape is not None
        or center_mask_crop_slices is not None
    )
    if crop_geometry_supplied:
        if (
            center_mask_original_shape is None
            or center_mask_crop_slices is None
        ):
            raise ValueError(
                "center_mask_original_shape and center_mask_crop_slices "
                "must be supplied together."
            )
        if center_mask_offset_um is not None:
            raise ValueError(
                "Choose either explicit center_mask_offset_um or the original "
                "shape and crop slices, not both."
            )
        center_mask_offset_um = center_offset_after_crop_um(
            center_mask_original_shape,
            center_mask_crop_slices,
            spacing_um=spacing_um,
        )
    elif center_mask_offset_um is None:
        center_mask_offset_um = (0.0, 0.0)

    if center_mask_diagonal_um is not None and mask_center_indent:
        raise ValueError(
            "Choose either center_mask_diagonal_um or automatic "
            "mask_center_indent detection, not both."
        )

    if center_mask_diagonal_um is not None:
        indent_mask_result = fixed_center_diamond_exclusion_masks(
            height_maps_um,
            spacing_um=spacing_um,
            diagonal_um=center_mask_diagonal_um,
            center_offset_um=center_mask_offset_um,
        )
        exclusion_masks = indent_mask_result.excluded_masks
    elif mask_center_indent:
        warnings.warn(
            "Automatic center-indent detection is only valid when the indent "
            "is visibly identifiable in the height maps and every saved mask "
            "overlay is verified. It cannot locate an invisible fiducial.",
            stacklevel=2,
        )
        indent_mask_result = detect_center_indent_masks(
            height_maps_um,
            spacing_um=spacing_um,
            center_search_fraction=center_search_fraction,
            detection_smoothing_um=detection_smoothing_um,
            indent_threshold_sigma=indent_threshold_sigma,
            indent_guard_um=indent_guard_um,
            guard_fraction_of_equivalent_radius=(
                guard_fraction_of_equivalent_radius
            ),
            common_mask_across_specimens=common_mask_across_specimens,
        )
        exclusion_masks = indent_mask_result.excluded_masks

    base_analysis = calculate_psd_analysis_at_time(
        height_maps_um,
        time_index,
        spacing_um=spacing_um,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        min_modes_per_bin=min_modes_per_bin,
        seed=seed,
        use_median=use_median,
        exclusion_masks=exclusion_masks,
        mask_taper_um=mask_taper_um,
    )
    analysis = PSDAnalysisResult(
        radial=base_analysis.radial,
        full_2d=base_analysis.full_2d,
        excluded_masks=base_analysis.excluded_masks,
        center_indent_mask=indent_mask_result,
    )

    if indent_mask_result is not None:
        save_center_indent_mask_diagnostics(
            height_maps_um,
            indent_mask_result,
            output_dir=output_dir,
        )

    save_results(
        analysis,
        output_dir=output_dir,
        plot_specimen_2d=plot_specimen_2d,
    )

    print(
        "Maximum Parseval relative error:",
        analysis.radial.parseval_relative_error.max(),
    )
    if indent_mask_result is not None:
        print(
            "Center-indent excluded fractions:",
            np.mean(indent_mask_result.excluded_masks, axis=(1, 2)),
        )
        print("Center-indent guard widths (um):", indent_mask_result.guard_um)

    return analysis


if __name__ == "__main__":
    # Both inputs are uncropped, plane-leveled height stacks with shape
    # (ntimes, nsamples, 1024, 768). Heights must be in micrometers.
    HEIGHT_MAPS_50X = None
    SPACING_50X = 0.278489
    HEIGHT_MAPS_10X = None
    SPACING_10X = 1.379951

    # One strictly increasing hold time per entry on the time axis.
    TIMES_HOURS = None
    OUTPUT_DIR = "profilometry_complete_analysis"
    USE_MEDIAN = False

    if (
        HEIGHT_MAPS_50X is None
        or HEIGHT_MAPS_10X is None
        or TIMES_HOURS is None
    ):
        raise RuntimeError(
            "Set HEIGHT_MAPS_50X, HEIGHT_MAPS_10X, and TIMES_HOURS in "
            "__main__ before running this file."
        )

    run_complete_analysis(
        HEIGHT_MAPS_50X,
        SPACING_50X,
        HEIGHT_MAPS_10X,
        SPACING_10X,
        TIMES_HOURS,
        output_dir=OUTPUT_DIR,
        use_median=USE_MEDIAN,
    )
