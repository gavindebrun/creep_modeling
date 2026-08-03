from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import inspect

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
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


@dataclass(frozen=True)
class PSDAnalysisResult:
    """Full PSD analysis at one time."""

    radial: RadialPSDResult
    full_2d: PSD2DResult


def _as_spacing_tuple(spacing_um: float | tuple[float, float]) -> tuple[float, float]:
    if np.isscalar(spacing_um):
        d0 = d1 = float(spacing_um)
    else:
        d0, d1 = map(float, spacing_um)

    if d0 <= 0.0 or d1 <= 0.0:
        raise ValueError("Pixel spacings must be positive.")

    return d0, d1


def _selected_time_maps(height_maps_um: np.ndarray, time_index: int) -> np.ndarray:
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

    if maps_t.shape[0] < 2:
        raise ValueError("At least two independent specimens are required.")

    if not np.all(np.isfinite(maps_t)):
        raise ValueError(
            "The selected maps contain NaN or infinite values. "
            "Do not replace missing pixels with zero before an FFT."
        )

    return maps_t


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


def calculate_2d_psd_at_time(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    spacing_um: float | tuple[float, float] = 1.3,
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
    """
    maps_t = _selected_time_maps(height_maps_um, time_index)
    d0, d1 = _as_spacing_tuple(spacing_um)

    nsamples, n0, n1 = maps_t.shape
    window = _rms_normalized_radial_hann((n0, n1), (d0, d1))

    f0 = np.fft.fftfreq(n0, d=d0)
    f1 = np.fft.fftfreq(n1, d=d1)
    df0 = 1.0 / (n0 * d0)
    df1 = 1.0 / (n1 * d1)

    pixel_area = d0 * d1
    n_pixels = n0 * n1

    specimen_psd = np.empty((nsamples, n0, n1), dtype=np.float64)
    parseval_error = np.empty(nsamples, dtype=np.float64)

    for specimen_index, height_map in enumerate(maps_t):
        centered_map = height_map - np.mean(height_map)
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

    return PSD2DResult(
        frequency_axis0_um_inv=f0,
        frequency_axis1_um_inv=f1,
        specimen_psd_um4=specimen_psd,
        mean_psd_um4=np.mean(specimen_psd, axis=0),
        parseval_relative_error=parseval_error,
    )


def _bootstrap_keyword_args(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    signature = inspect.signature(bootstrap)

    if "rng" in signature.parameters:
        return {"rng": rng}

    return {"random_state": rng}


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
    """
    full_2d = calculate_2d_psd_at_time(
        height_maps_um,
        time_index,
        spacing_um=spacing_um,
    )

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

    nbins = edges.size - 1
    radial_frequency_flat = radial_frequency.ravel()

    bin_index = np.searchsorted(
        edges,
        radial_frequency_flat,
        side="right",
    ) - 1

    valid_mode = (bin_index >= 0) & (bin_index < nbins)
    valid_bin_index = bin_index[valid_mode]

    modes_per_bin = np.bincount(valid_bin_index, minlength=nbins)
    frequency_sum = np.bincount(
        valid_bin_index,
        weights=radial_frequency_flat[valid_mode],
        minlength=nbins,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        bin_frequency = frequency_sum / modes_per_bin

    retained_bin = (modes_per_bin >= min_modes_per_bin) & np.isfinite(bin_frequency)

    if not np.any(retained_bin):
        raise ValueError("No radial bins satisfy min_modes_per_bin.")

    number_retained = int(np.count_nonzero(retained_bin))
    specimen_psd = np.empty((nsamples, number_retained), dtype=np.float64)

    for specimen_index, psd_2d in enumerate(full_2d.specimen_psd_um4):
        annular_sum = np.bincount(
            valid_bin_index,
            weights=psd_2d.ravel()[valid_mode],
            minlength=nbins,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            radial_psd = annular_sum / modes_per_bin

        specimen_psd[specimen_index] = radial_psd[retained_bin]

    def specimen_mean(sample: np.ndarray, axis: int = 0) -> np.ndarray:
        return np.mean(sample, axis=axis)

    bootstrap_result = bootstrap(
        (specimen_psd,),
        specimen_mean,
        axis=0,
        vectorized=True,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        batch=min(1000, n_resamples),
        method=bootstrap_method,
        **_bootstrap_keyword_args(seed),
    )

    retained_frequency = bin_frequency[retained_bin]

    return RadialPSDResult(
        frequency_um_inv=retained_frequency,
        wavelength_um=1.0 / retained_frequency,
        specimen_psd_um4=specimen_psd,
        mean_psd_um4=np.mean(specimen_psd, axis=0),
        ci_low_um4=np.asarray(bootstrap_result.confidence_interval.low),
        ci_high_um4=np.asarray(bootstrap_result.confidence_interval.high),
        modes_per_bin=modes_per_bin[retained_bin],
        parseval_relative_error=full_2d.parseval_relative_error,
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
) -> PSDAnalysisResult:
    """Calculate both full 2D PSDs and radial PSD results."""
    radial = radial_psd_at_time(
        height_maps_um,
        time_index,
        spacing_um=spacing_um,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        min_modes_per_bin=min_modes_per_bin,
        seed=seed,
    )

    full_2d = calculate_2d_psd_at_time(
        height_maps_um,
        time_index,
        spacing_um=spacing_um,
    )

    return PSDAnalysisResult(radial=radial, full_2d=full_2d)


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
    mean_psd = result.mean_psd_um4
    ci_low = result.ci_low_um4
    ci_high = result.ci_high_um4

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(mean_psd)
        & np.isfinite(ci_low)
        & np.isfinite(ci_high)
        & (wavelength > 0.0)
        & (mean_psd > 0.0)
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
    mean_plot = mean_psd[valid][order]
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
        mean_plot,
        linewidth=1.8,
        label="Specimen mean",
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

    mean = 2.0 * np.pi * frequency**2 * result.mean_psd_um4
    low = 2.0 * np.pi * frequency**2 * result.ci_low_um4
    high = 2.0 * np.pi * frequency**2 * result.ci_high_um4

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(mean)
        & np.isfinite(low)
        & np.isfinite(high)
        & (wavelength > 0.0)
        & (mean > 0.0)
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
        mean[valid][order],
        linewidth=1.8,
        label="Specimen mean",
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
    Plot the mean full 2D PSD, or one specimen's full 2D PSD.

    Frequencies are shown after fftshift, so zero frequency is centered.
    """
    if ax is None:
        _, ax = plt.subplots()

    if specimen_index is None:
        psd = result.mean_psd_um4
        default_title = "Mean full 2D PSD"
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

    np.savez_compressed(
        output_path / f"{prefix}_results.npz",
        radial_frequency_um_inv=radial.frequency_um_inv,
        radial_wavelength_um=radial.wavelength_um,
        radial_specimen_psd_um4=radial.specimen_psd_um4,
        radial_mean_psd_um4=radial.mean_psd_um4,
        radial_ci_low_um4=radial.ci_low_um4,
        radial_ci_high_um4=radial.ci_high_um4,
        radial_modes_per_bin=radial.modes_per_bin,
        full_2d_frequency_axis0_um_inv=full_2d.frequency_axis0_um_inv,
        full_2d_frequency_axis1_um_inv=full_2d.frequency_axis1_um_inv,
        full_2d_specimen_psd_um4=full_2d.specimen_psd_um4,
        full_2d_mean_psd_um4=full_2d.mean_psd_um4,
        parseval_relative_error=radial.parseval_relative_error,
    )

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

    all_2d_arrays = [full_2d.mean_psd_um4]
    if plot_specimen_2d:
        all_2d_arrays.extend(full_2d.specimen_psd_um4)
    shared_norm = _positive_log_norm(all_2d_arrays)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    plot_2d_psd(
        full_2d,
        ax=ax,
        norm=shared_norm,
        title="Mean full 2D PSD",
    )
    fig.tight_layout()
    fig.savefig(output_path / f"{prefix}_mean_full_2d_psd.png", dpi=300)
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
) -> PSDAnalysisResult:
    """Run the PSD analysis and save the standard output files."""
    analysis = calculate_psd_analysis_at_time(
        height_maps_um,
        time_index,
        spacing_um=spacing_um,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        min_modes_per_bin=min_modes_per_bin,
        seed=seed,
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

    return analysis


if __name__ == "__main__":
    # Put your real data here before running. HEIGHT_MAPS must have shape:
    # (ntimes, nsamples, nx, ny). Heights should be in micrometers.
    HEIGHT_MAPS = None

    # SPACING is the pixel pitch in micrometers. Use a scalar for equal
    # spacing, or a tuple like (axis_0_spacing_um, axis_1_spacing_um).
    SPACING = 1.3

    TIME_INDEX = 0
    OUTPUT_DIR = "psd_output"

    if HEIGHT_MAPS is None:
        raise RuntimeError(
            "Set HEIGHT_MAPS in __main__ before running this file. "
            "Expected shape is (ntimes, nsamples, nx, ny)."
        )

    run_analysis(
        HEIGHT_MAPS,
        spacing_um=SPACING,
        time_index=TIME_INDEX,
        output_dir=OUTPUT_DIR,
    )
