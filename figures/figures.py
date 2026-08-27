# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from utils.plotting.common import new_fig_ax, style_axes, save_fig
from utils.data_utils import CreepData, SimResults
from utils.config import (
    RESULTS_DIR,
    CALIBRATIONS_DIR,
    DATA_DIR,
    MICROSTRUCTURE_DIR,
    MICROSTRUCTURE_PATH,
    SIM_COLOR,
    EXP_COLOR,
    RC_PARAMS,
)
import importlib
from pathlib import Path
%matplotlib inline

PARAMS_PATH = Path("~/mimosa/params/best_row.csv")
params = pd.read_csv(PARAMS_PATH)
LOADS = [500, 530, 588]


def reload():
    importlib.reload(utils)

# %%
params

# %%
reload()
nrsx = []
gamd0x = []
for load in LOADS:
    nrsx.append(params[f"nrsx_{load}"])
    gamd0x.append(params[f"gamd0x_{load}"])

nrsx = np.array(nrsx)
gamd0x = np.array(gamd0x)
loads = np.array(LOADS)
loads[1] = 530
fig, ax = new_fig_ax()

ax.plot(loads, 1 / nrsx, marker="o")
ax.set_xlabel("Load [MPa]")
ax.set_ylabel(r"$m$")
style_axes(ax)
plt.show()

fig, ax = new_fig_ax()
ax.semilogy(loads, gamd0x, marker="o")
ax.set_xlabel("Load [MPa]")
ax.set_ylabel(r"$\dot{\gamma_{0}}$")
style_axes(ax)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from utils.plotting.common import new_fig_ax, style_axes, save_fig
from utils.data_utils import CreepData, SimResults
from utils.config import (
    RESULTS_DIR,
    CALIBRATIONS_DIR,
    DATA_DIR,
    MICROSTRUCTURE_DIR,
    MICROSTRUCTURE_PATH,
    SIM_COLOR,
    EXP_COLOR,
    RC_PARAMS,
)
import importlib
from pathlib import Path
%matplotlib inline

PARAMS_PATH = Path("/Users/gtdebru/mimosa/params/production_uninterrupted.csv")
params = pd.read_csv(PARAMS_PATH)
LOADS = [500, 530, 588]


def reload():
    importlib.reload(utils)

reload()
nrsx = []
gamd0x = []
for load in LOADS:
    nrsx.append(params[f"nrsx_{load}"])
    gamd0x.append(params[f"gamd0x_{load}"])

nrsx = np.array(nrsx)
gamd0x = np.array(gamd0x)
loads = np.array(LOADS)
fig, ax = new_fig_ax()

ax.plot(loads, 1 / nrsx, marker="o")
ax.set_xlabel("Load [MPa]")
ax.set_ylabel(r"$m$")
style_axes(ax)
plt.show()

fig, ax = new_fig_ax()
ax.semilogy(loads, gamd0x, marker="o")
ax.set_xlabel("Load [MPa]")
ax.set_ylabel(r"$\dot{\gamma_{0}}$")
style_axes(ax)
plt.show()

# %%
crp_525_int = CreepData.load(load_mpa=575, type="int", polish="polished")

# %%
strain_time = crp_525_int.strain_time
height_time = crp_525_int.time10[0]
strain = crp_525_int.mean_strain

x = []
for time in height_time:
    imin = np.argmin(np.abs(strain_time - time))
    x.append(strain[imin])

x = np.array(x)

# %%
fig, ax = new_fig_ax()

y = crp_525_int.sa10
ax.plot(x * 100, (y - y.min()) / (y.max() - y.min()))

# %%
sim_paths = {
    500: Path("/Users/gtdebru/mimosa/results/creep_fast/500mpa_rve1_final"),
    # 530: Path("/Users/gtdebru/mimosa/results/creep_fast/530mpa_rve1_final"),
    # 588: Path("/Users/gtdebru/mimosa/results/creep_fast/588mpa_rve1_final"),
}

colors = {
    500: "tab:blue",
    530: "tab:orange",
    588: "tab:green",
}

fig, ax = new_fig_ax()

for load, sim_path in sim_paths.items():
    sim = SimResults.load(sim_path, skip_spatial=True)
    exp = CreepData.load(
        load_mpa=load,
        type="unint",
        polish="polished",
        skip_spatial=True,
    )

    ax.plot(
        exp.strain_time,
        exp.mean_strain * 100,
        color=colors[load],
        linestyle="--",
        linewidth=1.5,
        label=f"",
    )

    # if load == 530:
    #     load = 530

    ax.plot(
        sim.sim_time / 3600,
        sim.epav33 * 100,
        color=colors[load],
        linestyle="-",
        linewidth=1.5,
        label=f"{load} MPa",
    )

ax.set_xlabel("Time (h)")
ax.set_ylabel("Axial strain (%)")
ax.legend()
fig.tight_layout()
plt.show()

# %%
image = axes[row, column].imshow(
    field,
    origin="lower",
    extent=(0.0, 128.0, 0.0, 256.0),
    cmap="RdBu_r",
    vmin=-color_limit,
    vmax=color_limit,
    interpolation="nearest",
    aspect="equal",
)

image = axis.imshow(
    leveled,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
    interpolation="none",
    rasterized=True,
)

image = axis.imshow(
    slip[state_index],
    origin="lower",
    extent=(0.0, 128.0, 0.0, 256.0),
    cmap="magma",
    vmin=0.0,
    vmax=color_limit,
    interpolation="nearest",
    aspect="equal",
)

# %%
# @lru_cache(maxsize=32)
def _detrend_geometry(
    shape: tuple[int, int], spacing_um: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    row, column = np.indices(shape, dtype=np.float64)
    x = row.ravel() * spacing_um
    y = column.ravel() * spacing_um
    terms = [np.ones(row.size), x, y]
    if order == 2:
        terms.extend((x * x, x * y, y * y))
    design = np.column_stack(terms)
    return design, np.linalg.pinv(design)


def detrend_surface(
    values: np.ndarray, spacing_um: float, order: int = 1
) -> np.ndarray:
    design, inverse = _detrend_geometry(values.shape, float(spacing_um), order)
    coefficients = inverse @ values.ravel()
    return values - (design @ coefficients).reshape(values.shape)


def piecewise_level(
    height: np.ndarray,
    spacing_um: float,
    grid: tuple[int, int],
) -> np.ndarray:
    leveled = np.empty_like(height, dtype=float)
    for row_index in np.array_split(np.arange(height.shape[0]), grid[0]):
        for column_index in np.array_split(np.arange(height.shape[1]), grid[1]):
            leveled[np.ix_(row_index, column_index)] = detrend_surface(
                height[np.ix_(row_index, column_index)],
                spacing_um,
            )
    return leveled

# %%
spacing_um = 1.3799510000000001

# %%
def load_height(path, level=False):

    height = (
        pd.read_csv(
            path,
            skiprows=19,
            header=None,
            # encoding="latin1",
        )
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )
    spacing_um = xy_spacing_um(path)
    missing = ~np.isfinite(height)
    if np.any(missing):
        nearest = distance_transform_edt(
            missing, return_distances=False, return_indices=True
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um)
    return height

# %%
path = Path("/Users/gtdebru/tmp/creep_int_polished_575/profilometry/10x/35c/0.csv")


height = load_height(path, level=False)
mad = np.abs(height - height.mean())
sa = mad.mean()

vmin = float(np.nanpercentile(height, 1.0))
vmax = float(np.nanpercentile(height, 99.0))
extent = (
    0.0,
    height.shape[1] * spacing_um,
    0.0,
    height.shape[0] * spacing_um,
)

fig, ax = plt.subplots(figsize=(5.7, 4.2), constrained_layout=True, dpi=200)
image = ax.imshow(
    height,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
    rasterized=True,
)
ax.set(
    xlabel=r"$x$ ($\mu$m)",
    ylabel=r"$y$ ($\mu$m)",
    title=f"{path.parent.name}, {path.stem} h, Sa = {sa:.2f}",
)
colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label(r"Plane-leveled height ($\mu$m)")

# %%
height = load_height(path, level=False)
height = height[50:-50, 50:750]
height = detrend_surface(height, spacing_um)
vmin = float(np.nanpercentile(height, 1.0))
vmax = float(np.nanpercentile(height, 99.0))

mad = np.abs(height - height.mean())
sa = mad.mean()

fig, ax = plt.subplots(figsize=(5.7, 4.2), constrained_layout=True, dpi=200)
image = ax.imshow(
    height,
    origin="lower",
    # extent=extent,
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
    rasterized=True,
)
ax.set(
    xlabel=r"$x$ ($\mu$m)",
    ylabel=r"$y$ ($\mu$m)",
    title=f"{path.parent.name}, {path.stem} h, Sa = {sa:.2f}",
)
colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label(r"Plane-leveled height ($\mu$m)")
plt.show()

# %%
height = load_height(path, level=False)
dx = 750 // 2
dy = height.shape[1] // 2

fig, axs = plt.subplots(2, 2, figsize=(5.7, 4.2), constrained_layout=True, dpi=200)

for i in [1, 2]:
    for j in [1, 2]:
        ax = axs[i - 1, j - 1]

        height = height[dx * (i - 1) : dx * i, dy * (j - 1) : dy * j]
        mad = np.abs(height - height.mean())
        sa = mad.mean()

        print(dx * (i - 1), dx * (i + 0), dy * (i - 1), dy * (i + 0))
        height = detrend_surface(height, spacing_um)
        vmin = float(np.nanpercentile(height, 1.0))
        vmax = float(np.nanpercentile(height, 99.0))

        image = ax.imshow(
            height,
            origin="lower",
            # extent=extent,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            rasterized=True,
        )
        ax.set(
            xlabel=r"$x$ ($\mu$m)",
            ylabel=r"$y$ ($\mu$m)",
            title=f"{path.parent.name}, {path.stem} h, Sa = {sa:.2f}",
        )
        colorbar = fig.colorbar(image, ax=ax)
        # colorbar.set_label(r"Plane-leveled height ($\mu$m)")
        # plt.show()

# %%
%matplotlib inline
import os

def vickers_mask(height, spacing, diagonal = 60.0):
    import numpy as np

    nx, ny = height.shape

    x = (np.arange(nx) - (nx-1) / 2) * spacing
    y = (np.arange(ny) - (ny -1) / 2) * spacing
    mask = np.abs(x[:, None]) + np.abs(y[None, :]) <= diagonal / 2

    return mask

def write_map_img(type, load, polish, mag):
    from utils.config import DATA_DIR, PROFILOMETRY_SPACING_UM, RC_PARAMS
    from utils.profilometry_psd import read_height
    from utils.plotting.common import style_axes
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(RC_PARAMS)

    root = DATA_DIR / f"creep_{type}_{polish}_{load}/profilometry/{mag}"
    spacing = PROFILOMETRY_SPACING_UM[mag]
    save_paths = []
    height_paths = sorted(root.rglob("*.csv"), key=lambda p: p.parent)
    for file in height_paths:
        img_root = file.parent / "img"
        time = str(file.stem)
        img_name = time + ".png"
        img_path = img_root / img_name
        save_paths.append(img_path)


    for hpath, savepath in zip(height_paths, save_paths):
        height = read_height(hpath, level=True, spacing_um=spacing)
        mask = vickers_mask(height, spacing)

    
            
        fig, ax = plt.subplots(dpi=200)
        # fig.set_dpi(300)

        height = height[50:-50, 50:750]
        mask = mask[50:-50, 50:750]

        vmin = float(np.nanpercentile(height[~mask], 1.0))
        vmax = float(np.nanpercentile(height[~mask], 99.0))
        mad = np.abs(height[~mask] - height[~mask].mean())
        sa = mad.mean()
        height[mask] = 0.0
        extent = (
            0.0,
            height.shape[1] * spacing,
            0.0,
            height.shape[0] * spacing,
        )

        image = ax.imshow(
            height,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            rasterized=True,
        )
        ax.set(
            xlabel=r"$x$ ($\mu$m)",
            ylabel=r"$y$ ($\mu$m)",
            title=f"{hpath.parent.name}, {hpath.stem} h, Sa = {sa:.2f}",
        )
        style_axes(ax)
        colorbar = fig.colorbar(image, ax=ax, label=r"$\text{Height} \ [\mu m]$")
        colorbar.ax.yaxis.get_label().set_rotation(270)
        colorbar.ax.yaxis.labelpad = 10 
        fig.tight_layout()
        fig.savefig(savepath)
        plt.close(fig)



# %%
import matplotlib.pyplot as plt

type = "unint"
polish = "polished"
for mag in ["10x", "50x"]:

    for load in [500, 530, 588]:

        write_map_img(type, load, polish, mag)

# %%
for path, img_dest in zip(files, save_paths):
    fig, axs = plt.subplots(3, 2, figsize=(5.7, 6.3), constrained_layout=True, dpi=200)

    ############## FULl HEIGHT #########
    ax = axs[0, 0]
    height = load_height(path, level=False)
    mad = np.abs(height - height.mean())
    sa = mad.mean()

    vmin = float(np.nanpercentile(height, 1.0))
    vmax = float(np.nanpercentile(height, 99.0))
    extent = (
        0.0,
        height.shape[1] * spacing_um,
        0.0,
        height.shape[0] * spacing_um,
    )

    image = ax.imshow(
        height,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set(
        xlabel=r"$x$ ($\mu$m)",
        ylabel=r"$y$ ($\mu$m)",
        title=f"{path.parent.name}, {path.stem} h, Sa = {sa:.2f}",
    )
    colorbar = fig.colorbar(image, ax=ax)

    ############ CROPPED HEIGHT ###############
    # height = load_height(path, level=False)
    ax = axs[0, 1]
    height = height[50:-50, 50:750]
    height = detrend_surface(height, spacing_um)
    vmin = float(np.nanpercentile(height, 1.0))
    vmax = float(np.nanpercentile(height, 99.0))

    mad = np.abs(height - height.mean())
    sa = mad.mean()

    extent = (
        0.0,
        height.shape[1] * spacing_um,
        0.0,
        height.shape[0] * spacing_um,
    )

    image = ax.imshow(
        height,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set(
        xlabel=r"$x$ ($\mu$m)",
        ylabel=r"$y$ ($\mu$m)",
        title=f"{path.parent.name}, {path.stem} h, Sa = {sa:.2f}",
    )
    colorbar = fig.colorbar(image, ax=ax)

    ############ SECTIONS ################

    dx = 750 // 2
    dy = height.shape[1] // 2

    for i in [2, 1]:
        for j in [1, 2]:
            ax = axs[i, j - 1]

            height = height[dx * (i - 1) : dx * i, dy * (j - 1) : dy * j]
            mad = np.abs(height - height.mean())
            sa = mad.mean()

            height = detrend_surface(height, spacing_um)
            vmin = float(np.nanpercentile(height, 1.0))
            vmax = float(np.nanpercentile(height, 99.0))
            extent = (
                0.0,
                height.shape[1] * spacing_um,
                0.0,
                height.shape[0] * spacing_um,
            )

            image = ax.imshow(
                height,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                rasterized=True,
            )
            ax.set(
                xlabel=r"$x$ ($\mu$m)",
                ylabel=r"$y$ ($\mu$m)",
                title=f"{path.parent.name}, {path.stem} h, Sa = {sa:.2f}",
            )
            colorbar = fig.colorbar(image, ax=ax)
            # colorbar.set_label(r"Plane-leveled height ($\mu$m)")
            # plt.show()

    fig.savefig(img_dest, dpi=200)
    plt.close(fig)

# %%
from utils.config import PROFILOMETRY_SPACING_UM

spacing_um = PROFILOMETRY_SPACING_UM["10x"]

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import inspect

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import bootstrap

from scipy.ndimage import distance_transform_edt
from functools import lru_cache
from utils.config import PROFILOMETRY_SPACING_UM


@lru_cache(maxsize=256)
def _detrend_geometry(
    shape: tuple[int, int], spacing_um: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    row, column = np.indices(shape, dtype=np.float64)
    x = row.ravel() * spacing_um
    y = column.ravel() * spacing_um
    terms = [np.ones(row.size), x, y]
    if order == 2:
        terms.extend((x * x, x * y, y * y))
    design = np.column_stack(terms)
    return design, np.linalg.pinv(design)


def detrend_surface(
    values: np.ndarray, spacing_um: float, order: int = 1
) -> np.ndarray:
    design, inverse = _detrend_geometry(values.shape, float(spacing_um), order)
    coefficients = inverse @ values.ravel()
    return values - (design @ coefficients).reshape(values.shape)


def raw_height(path: Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: Path, level: bool = False, spacing_um=PROFILOMETRY_SPACING_UM["10x"]
) -> np.ndarray:
    height = raw_height(path)
    missing = ~np.isfinite(height)
    if np.any(missing):
        nearest = distance_transform_edt(
            missing, return_distances=False, return_indices=True
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um)
    return height


import numpy as np
from scipy.stats import t

from utils.config import DATA_DIR

# import os
# from glob import glob

type = "unint"
polish = "polished"
load = 530
mag = "10x"

path = Path(DATA_DIR / f"creep_{type}_{polish}_{load}/profilometry/{mag}")
files = sorted(path.rglob("*.csv"), key=lambda p: float(p.stem))
exclude = ["37e"]
heights = {}

for file in files:
    sample = file.parent.stem
    if sample in exclude:
        continue
    time = float(file.stem)
    height = read_height(file, level=True)[50:-50, 50:750]
    if sample not in heights.keys():
        heights[sample] = {}
        heights[sample]["time"] = []
        heights[sample]["height"] = []

    heights[sample]["time"].append(time)
    heights[sample]["height"].append(height)

samples = sorted(heights.keys())

# sort each sample by time
for sample in samples:
    order = np.argsort(heights[sample]["time"])
    heights[sample]["time"] = np.array(heights[sample]["time"])[order]
    heights[sample]["height"] = [heights[sample]["height"][i] for i in order]

# reference times
times = heights[samples[0]]["time"]

# verify all samples have same times
for sample in samples:
    if not np.array_equal(heights[sample]["time"], times):
        raise ValueError(f"{sample} has different times")

# verify all height maps have same shape
shapes = {h.shape for sample in samples for h in heights[sample]["height"]}


if len(shapes) != 1:
    raise ValueError("Not all height maps have same shape")

# stack to (ntimes, nsamples, nx, ny)
height_array = np.stack(
    [
        np.stack(
            [heights[sample]["height"][it] for sample in samples],
            axis=0,
        )
        for it in range(len(times))
    ],
    axis=0,
)

samples = np.array(samples)
times = np.array(times)

print(height_array.shape)


def mean_radial_psd_ci(
    H: np.ndarray,
    spacing_um: float,
    n_bins: int | None = None,
    confidence: float = 0.95,
):
    """
    H shape: (ntimes, nsamples, nx, ny)

    Returns
    -------
    freq : (n_bins,) radial spatial frequency, cycles/um
    psd_mean : (ntimes, n_bins)
    psd_ci_low : (ntimes, n_bins)
    psd_ci_high : (ntimes, n_bins)
    psd_samples : (ntimes, nsamples, n_bins)
    """

    nt, ns, nx, ny = H.shape
    dx = dy = spacing_um

    if n_bins is None:
        n_bins = min(nx, ny) // 2

    # 2D Hann window
    wx = np.hanning(nx)
    wy = np.hanning(ny)
    window = wx[:, None] * wy[None, :]
    window_power = np.sum(window**2)

    # frequency grids
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fy, fx)  # shapes match image: (nx, ny)
    FR = np.sqrt(FX**2 + FY**2)

    fmax = FR.max()
    f_min = np.min(FR[FR > 0])
    bin_edges = np.linspace(f_min, fmax, n_bins + 1)
    freq = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_index = np.digitize(FR.ravel(), bin_edges) - 1
    valid = (bin_index >= 0) & (bin_index < n_bins)

    psd_samples = np.empty((nt, ns, n_bins), dtype=float)

    for it in range(nt):
        for isamp in range(ns):
            z = H[it, isamp]
            z = z - np.nanmean(z)

            Z = np.fft.fft2(z * window)

            # 2D PSD density, units: height^2 * um^2
            psd2 = dx * dy * np.abs(Z) ** 2 / window_power

            flat = psd2.ravel()

            radial = np.full(n_bins, np.nan)

            for ibin in range(n_bins):
                mask = valid & (bin_index == ibin)
                if np.any(mask):
                    radial[ibin] = np.mean(flat[mask])

            psd_samples[it, isamp] = radial

    psd_mean = np.nanmean(psd_samples, axis=1)

    return freq, psd_mean, psd_ci_low, psd_ci_high, psd_samples


f, psd_mean, psd_ci_low, psd_ci_high, psd = mean_radial_psd_ci(height_array, spacing_um)


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
        raise ValueError("height_maps_um must have shape (ntimes, nsamples, n0, n1).")

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

    window[inside] = 0.5 * (1.0 + np.cos(np.pi * radius[inside] / support_radius))

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

    bin_index = (
        np.searchsorted(
            edges,
            radial_frequency_flat,
            side="right",
        )
        - 1
    )

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
        _, ax = plt.subplots(dpi=200)

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
        i = 0
        for specimen_curve in result.specimen_psd_um4:
            ax.loglog(
                wavelength_plot,
                specimen_curve[valid][order],
                linewidth=0.7,
                alpha=0.18,
                label=str(i),
            )
            i += 1
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
    ax.set_ylabel(r"Radially averaged 2D PSD, $C_{\mathrm{iso}}$ ($\mu$m$^4$)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

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


HEIGHT_MAPS = height_array

# SPACING is the pixel pitch in micrometers. Use a scalar for equal
# spacing, or a tuple like (axis_0_spacing_um, axis_1_spacing_um).
SPACING = spacing_um

TIME_INDEX = 1
OUTPUT_DIR = "psd_output1"


run_analysis(
    HEIGHT_MAPS,
    spacing_um=SPACING,
    time_index=TIME_INDEX,
    output_dir=OUTPUT_DIR,
)

# %%
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence
import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.ndimage import distance_transform_edt
from scipy.stats import bootstrap, t

from utils.config import DATA_DIR, PROFILOMETRY_SPACING_UM

# =============================================================================
# Height-map reading and leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(
    shape: tuple[int, int],
    spacing_um: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    row, column = np.indices(shape, dtype=np.float64)
    x = row.ravel() * spacing_um
    y = column.ravel() * spacing_um

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))

    design = np.column_stack(terms)
    return design, np.linalg.pinv(design)


def detrend_surface(
    values: np.ndarray,
    spacing_um: float,
    order: int = 1,
) -> np.ndarray:
    design, inverse = _detrend_geometry(values.shape, float(spacing_um), order)
    coefficients = inverse @ values.ravel()
    return values - (design @ coefficients).reshape(values.shape)


def raw_height(path: Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: Path,
    *,
    level: bool = False,
    spacing_um: float = PROFILOMETRY_SPACING_UM["10x"],
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um, order=detrend_order)

    return height


def load_height_maps(
    *,
    sample_type: str,
    polish: str,
    load: int | float,
    mag: str,
    exclude: Sequence[str] = (),
    crop: tuple[slice, slice] = (slice(50, -50), slice(50, 750)),
    level: bool = True,
    detrend_order: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load profilometry height maps.

    Parameters
    ----------
    sample_type
        Example: "unint".

    polish
        Example: "polished".

    load
        Example: 530.

    mag
        Example: "10x".

    exclude
        Specimen folder names to exclude, for example ["37e"].

    crop
        Crop applied to every height map. Default is [50:-50, 50:750].

    level
        If True, detrend each raw height map before cropping.

    detrend_order
        1 for planar leveling, 2 for quadratic leveling.

    Returns
    -------
    height_array
        Array with shape:

            (ntimes, nsamples, n0, n1)

    times
        Time values from CSV stems.

    samples
        Specimen names.
    """
    spacing_um = PROFILOMETRY_SPACING_UM[mag]

    path = Path(DATA_DIR / f"creep_{sample_type}_{polish}_{load}/profilometry/{mag}")

    if not path.exists():
        raise FileNotFoundError(f"Profilometry path does not exist: {path}")

    files = sorted(path.rglob("*.csv"), key=lambda p: float(p.stem))
    exclude = set(exclude)

    heights: dict[str, dict[str, list]] = {}

    for file in files:
        sample = file.parent.stem

        if sample in exclude:
            continue

        time_value = float(file.stem)

        height = read_height(
            file,
            level=level,
            spacing_um=spacing_um,
            detrend_order=detrend_order,
        )

        height = height[crop]

        if sample not in heights:
            heights[sample] = {
                "time": [],
                "height": [],
            }

        heights[sample]["time"].append(time_value)
        heights[sample]["height"].append(height)

    if not heights:
        raise ValueError("No height maps were loaded. Check path and exclude list.")

    samples = np.array(sorted(heights.keys()))

    # Sort each sample by time.
    for sample in samples:
        order = np.argsort(heights[sample]["time"])
        heights[sample]["time"] = np.asarray(heights[sample]["time"])[order]
        heights[sample]["height"] = [heights[sample]["height"][i] for i in order]

    # Reference times.
    times = np.asarray(heights[samples[0]]["time"], dtype=float)

    # Verify all samples have identical times.
    for sample in samples:
        sample_times = np.asarray(heights[sample]["time"], dtype=float)

        if not np.array_equal(sample_times, times):
            raise ValueError(
                f"{sample} has different times.\n"
                f"Reference times: {times}\n"
                f"{sample} times: {sample_times}"
            )

    # Verify all height maps have same shape.
    shapes = {h.shape for sample in samples for h in heights[sample]["height"]}

    if len(shapes) != 1:
        raise ValueError(f"Not all height maps have same shape: {shapes}")

    # Stack to shape: (ntimes, nsamples, n0, n1)
    height_array = np.stack(
        [
            np.stack(
                [heights[sample]["height"][time_index] for sample in samples],
                axis=0,
            )
            for time_index in range(len(times))
        ],
        axis=0,
    )

    return height_array, times, samples


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class SaResult:
    """
    Sa summary at each time.

    Sa is computed specimen-by-specimen as:

        mean(abs(z - mean(z)))

    where z is a leveled, cropped height map.
    """

    times: np.ndarray

    # Shape: (ntimes, nsamples)
    specimen_sa_um: np.ndarray

    # Shape: (ntimes,)
    mean_sa_um: np.ndarray
    std_sa_um: np.ndarray
    ci_low_sa_um: np.ndarray
    ci_high_sa_um: np.ndarray
    confidence_level: float


@dataclass(frozen=True)
class RadialPSDResult:
    """Radial PSD and specimen-bootstrap result at one time."""

    frequency_um_inv: np.ndarray
    wavelength_um: np.ndarray

    # Shape: (nsamples, nbins)
    specimen_psd_um4: np.ndarray

    # Shape: (nbins,)
    mean_psd_um4: np.ndarray
    ci_low_um4: np.ndarray
    ci_high_um4: np.ndarray

    modes_per_bin: np.ndarray

    # One value per specimen.
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

    # One value per specimen.
    parseval_relative_error: np.ndarray


@dataclass(frozen=True)
class PSDTimeResult:
    """PSD analysis at one time."""

    time: float
    radial: RadialPSDResult
    full_2d: PSD2DResult


@dataclass(frozen=True)
class FullAnalysisResult:
    """Full analysis over all times."""

    sample_type: str
    polish: str
    load: int | float
    mag: str
    samples: np.ndarray
    times: np.ndarray
    height_shape: tuple[int, int]
    spacing_um: float | tuple[float, float]
    sa: SaResult
    psd_by_time: list[PSDTimeResult]


# =============================================================================
# Shared utilities
# =============================================================================


def _as_spacing_tuple(
    spacing_um: float | tuple[float, float],
) -> tuple[float, float]:
    if np.isscalar(spacing_um):
        d0 = d1 = float(spacing_um)
    else:
        d0, d1 = map(float, spacing_um)

    if d0 <= 0.0 or d1 <= 0.0:
        raise ValueError("Pixel spacings must be positive.")

    return d0, d1


def _selected_time_maps(
    height_maps_um: np.ndarray,
    time_index: int,
) -> np.ndarray:
    maps = np.asarray(height_maps_um)

    if maps.ndim != 4:
        raise ValueError("height_maps_um must have shape (ntimes, nsamples, n0, n1).")

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
    the rectangular map.
    """
    n0, n1 = shape
    d0, d1 = spacing_um

    x0 = (np.arange(n0) + 0.5 - n0 / 2.0) * d0
    x1 = (np.arange(n1) + 0.5 - n1 / 2.0) * d1

    radius = np.hypot(x0[:, None], x1[None, :])
    support_radius = 0.5 * min(n0 * d0, n1 * d1)

    window = np.zeros((n0, n1), dtype=np.float64)
    inside = radius < support_radius

    window[inside] = 0.5 * (1.0 + np.cos(np.pi * radius[inside] / support_radius))

    window /= np.sqrt(np.mean(window**2))

    return window


def _bootstrap_keyword_args(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    signature = inspect.signature(bootstrap)

    if "rng" in signature.parameters:
        return {"rng": rng}

    return {"random_state": rng}


# =============================================================================
# Sa calculation
# =============================================================================


def calculate_sa_by_time(
    height_maps_um: np.ndarray,
    times: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> SaResult:
    """
    Calculate specimen-level Sa and summary statistics at every time.

    height_maps_um shape:

        (ntimes, nsamples, n0, n1)
    """
    height_maps_um = np.asarray(height_maps_um, dtype=np.float64)

    if height_maps_um.ndim != 4:
        raise ValueError("height_maps_um must have shape (ntimes, nsamples, n0, n1).")

    ntimes, nsamples, _, _ = height_maps_um.shape

    specimen_sa = np.empty((ntimes, nsamples), dtype=np.float64)

    for time_index in range(ntimes):
        for sample_index in range(nsamples):
            z = height_maps_um[time_index, sample_index]
            z_centered = z - np.mean(z)
            specimen_sa[time_index, sample_index] = np.mean(np.abs(z_centered))

    mean_sa = np.mean(specimen_sa, axis=1)

    if nsamples > 1:
        std_sa = np.std(specimen_sa, axis=1, ddof=1)
        sem_sa = std_sa / np.sqrt(nsamples)
        alpha = 1.0 - confidence_level
        tcrit = t.ppf(1.0 - alpha / 2.0, df=nsamples - 1)

        ci_low = mean_sa - tcrit * sem_sa
        ci_high = mean_sa + tcrit * sem_sa
    else:
        std_sa = np.full(ntimes, np.nan)
        ci_low = np.full(ntimes, np.nan)
        ci_high = np.full(ntimes, np.nan)

    return SaResult(
        times=np.asarray(times, dtype=float),
        specimen_sa_um=specimen_sa,
        mean_sa_um=mean_sa,
        std_sa_um=std_sa,
        ci_low_sa_um=ci_low,
        ci_high_sa_um=ci_high,
        confidence_level=confidence_level,
    )


# =============================================================================
# 2D PSD calculation
# =============================================================================


def calculate_2d_psd_at_time(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    spacing_um: float | tuple[float, float],
) -> PSD2DResult:
    """
    Calculate the full two-dimensional PSD for every specimen at one time.

    height_maps_um shape:

        (ntimes, nsamples, n0, n1)

    Returned PSD units are micrometers^4 if heights and spacings are in
    micrometers.
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

        # 2D PSD density, units height^2 * length^2 = um^4
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


# =============================================================================
# Radial PSD calculation
# =============================================================================


def radial_psd_from_2d(
    full_2d: PSD2DResult,
    *,
    spacing_um: float | tuple[float, float],
    confidence_level: float = 0.95,
    n_resamples: int = 20_000,
    bootstrap_method: str = "BCa",
    min_modes_per_bin: int = 8,
    radial_binning: str = "log",
    radial_bin_width_factor: float = 1.0,
    bins_per_decade: float = 8.0,
    seed: int = 12345,
) -> RadialPSDResult:
    """
    Calculate specimen-level radial PSDs and bootstrap confidence intervals
    from precomputed full 2D PSDs.

    Parameters
    ----------
    radial_binning
        Either "log" or "linear".

        "log" is recommended when plotting PSD over multiple decades of
        wavelength.

    radial_bin_width_factor
        Used only for linear radial bins. The linear bin width is:

            radial_bin_width_factor * max(df0, df1)

    bins_per_decade
        Used only for log radial bins. Controls the number of radial PSD
        points per decade in spatial frequency.

        Smaller values give smoother curves.
        Larger values give more spectral detail but noisier curves.
    """
    f0 = full_2d.frequency_axis0_um_inv
    f1 = full_2d.frequency_axis1_um_inv

    nsamples, n0, n1 = full_2d.specimen_psd_um4.shape
    d0, d1 = _as_spacing_tuple(spacing_um)

    radial_frequency = np.hypot(f0[:, None], f1[None, :])

    df0 = 1.0 / (n0 * d0)
    df1 = 1.0 / (n1 * d1)

    full_annulus_limit = min(0.5 / d0, 0.5 / d1)

    positive_frequencies = radial_frequency[
        (radial_frequency > 0.0)
        & np.isfinite(radial_frequency)
        & (radial_frequency <= full_annulus_limit)
    ]

    if positive_frequencies.size == 0:
        raise ValueError("No positive radial frequencies are available.")

    f_min_positive = positive_frequencies.min()
    f_max = full_annulus_limit

    radial_binning = radial_binning.lower()

    if radial_binning == "linear":
        if radial_bin_width_factor <= 0.0:
            raise ValueError("radial_bin_width_factor must be positive.")

        bin_width = radial_bin_width_factor * max(df0, df1)

        edges = np.arange(
            0.5 * bin_width,
            f_max + np.finfo(float).eps * f_max,
            bin_width,
        )

    elif radial_binning == "log":
        if bins_per_decade <= 0.0:
            raise ValueError("bins_per_decade must be positive.")

        n_decades = np.log10(f_max / f_min_positive)

        if n_decades <= 0.0:
            raise ValueError("Frequency range is too small for logarithmic bins.")

        nbins = int(np.ceil(bins_per_decade * n_decades))

        if nbins < 1:
            raise ValueError("Too few logarithmic radial bins were requested.")

        edges = np.logspace(
            np.log10(f_min_positive),
            np.log10(f_max),
            nbins + 1,
        )

    else:
        raise ValueError('radial_binning must be either "log" or "linear".')

    if edges.size < 2:
        raise ValueError("The map is too small to construct radial-frequency bins.")

    nbins = edges.size - 1
    radial_frequency_flat = radial_frequency.ravel()

    bin_index = (
        np.searchsorted(
            edges,
            radial_frequency_flat,
            side="right",
        )
        - 1
    )

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


def calculate_psd_at_time(
    height_maps_um: np.ndarray,
    time_index: int,
    *,
    spacing_um: float | tuple[float, float],
    confidence_level: float = 0.95,
    n_resamples: int = 20_000,
    bootstrap_method: str = "BCa",
    min_modes_per_bin: int = 8,
    seed: int = 12345,
) -> PSDTimeResult:
    """Calculate full 2D PSD and radial PSD at one time."""
    full_2d = calculate_2d_psd_at_time(
        height_maps_um,
        time_index,
        spacing_um=spacing_um,
    )

    radial = radial_psd_from_2d(
        full_2d,
        spacing_um=spacing_um,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        min_modes_per_bin=min_modes_per_bin,
        seed=seed,
    )

    return PSDTimeResult(
        time=float(time_index),
        radial=radial,
        full_2d=full_2d,
    )


def calculate_psd_gain_final_vs_initial(
    analysis: FullAnalysisResult,
    *,
    initial_time_index: int = 0,
    final_time_index: int = -1,
    confidence_level: float = 0.95,
    n_resamples: int = 20_000,
    bootstrap_method: str = "BCa",
    seed: int = 12345,
    paired: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate mean PSD gain and a direct pointwise bootstrap CI.

    Gain is defined as:

        gain(lambda) = mean_specimen[PSD_final(lambda)]
                     / mean_specimen[PSD_initial(lambda)]

    Parameters
    ----------
    analysis
        FullAnalysisResult returned by run_full_analysis.

    initial_time_index
        Initial/reference time index.

    final_time_index
        Final/comparison time index.

    confidence_level
        Bootstrap confidence level.

    n_resamples
        Number of bootstrap resamples.

    bootstrap_method
        Method accepted by scipy.stats.bootstrap, for example "BCa",
        "percentile", or "basic".

    seed
        Random seed.

    paired
        If True, bootstrap initial and final specimen curves using the same
        specimen indices. This is appropriate when the same specimens are
        measured at initial and final times.

    Returns
    -------
    wavelength_um
        Wavelength array.

    gain_mean
        Mean PSD gain curve.

    gain_ci_low
        Lower pointwise bootstrap confidence bound.

    gain_ci_high
        Upper pointwise bootstrap confidence bound.
    """
    initial = analysis.psd_by_time[initial_time_index].radial
    final = analysis.psd_by_time[final_time_index].radial

    wavelength_initial = initial.wavelength_um
    wavelength_final = final.wavelength_um

    if not (
        wavelength_initial.shape == wavelength_final.shape
        and np.allclose(wavelength_initial, wavelength_final, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError(
            "Initial and final radial PSD wavelength grids differ. "
            "Interpolate to a common grid before calculating gain CIs."
        )

    psd_initial = np.asarray(initial.specimen_psd_um4, dtype=np.float64)
    psd_final = np.asarray(final.specimen_psd_um4, dtype=np.float64)

    if psd_initial.shape != psd_final.shape:
        raise ValueError(
            "Initial and final specimen PSD arrays must have the same shape."
        )

    if psd_initial.shape[0] < 2:
        raise ValueError("At least two specimens are required for bootstrap CIs.")

    if not np.all(np.isfinite(psd_initial)) or not np.all(np.isfinite(psd_final)):
        raise ValueError("PSD arrays contain NaN or infinite values.")

    if np.any(psd_initial <= 0.0) or np.any(psd_final <= 0.0):
        raise ValueError(
            "PSD arrays must be positive to calculate a well-defined gain."
        )

    def gain_statistic(
        initial_sample: np.ndarray,
        final_sample: np.ndarray,
        axis: int = 0,
    ) -> np.ndarray:
        return np.mean(final_sample, axis=axis) / np.mean(initial_sample, axis=axis)

    bootstrap_result = bootstrap(
        (psd_initial, psd_final),
        gain_statistic,
        axis=0,
        vectorized=True,
        paired=paired,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        batch=min(1000, n_resamples),
        method=bootstrap_method,
        **_bootstrap_keyword_args(seed),
    )

    gain_mean = final.mean_psd_um4 / initial.mean_psd_um4
    gain_ci_low = np.asarray(bootstrap_result.confidence_interval.low)
    gain_ci_high = np.asarray(bootstrap_result.confidence_interval.high)

    return (
        wavelength_initial,
        gain_mean,
        gain_ci_low,
        gain_ci_high,
    )


# =============================================================================
# Plotting
# =============================================================================


def plot_radial_psd_vs_wavelength(
    result: RadialPSDResult,
    *,
    ax: plt.Axes | None = None,
    show_specimens: bool = True,
    wavelength_range_um: tuple[float, float] | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot radial PSD against wavelength.

    The confidence interval is pointwise, not simultaneous over wavelength.
    """
    if ax is None:
        _, ax = plt.subplots(dpi=200)

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
                "wavelength_range_um must be " "(positive_minimum, larger_maximum)."
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
        for specimen_index, specimen_curve in enumerate(result.specimen_psd_um4):
            ax.loglog(
                wavelength_plot,
                specimen_curve[valid][order],
                linewidth=0.7,
                alpha=0.18,
                # label=f"Specimen {specimen_index}",
            )

    ax.loglog(
        wavelength_plot,
        mean_plot,
        color="black",
        linewidth=2.0,
        label="Specimen mean",
    )

    ax.fill_between(
        wavelength_plot,
        low_plot,
        high_plot,
        color="black",
        alpha=0.20,
        linewidth=0,
        label="Pointwise bootstrap CI",
    )

    ax.set_xlabel(r"Wavelength, $\lambda$ ($\mu$m)")
    ax.set_ylabel(r"Radially averaged 2D PSD, $C_{\mathrm{iso}}$ ($\mu$m$^4$)")
    ax.grid(True, which="both", alpha=0.2)

    if title is not None:
        ax.set_title(title)

    # ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    ax.legend(loc="upper left")

    return ax


def _positive_log_norm(
    arrays: Sequence[np.ndarray],
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


def plot_sa_vs_time(
    result: SaResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot mean Sa with confidence intervals versus time."""
    if ax is None:
        _, ax = plt.subplots(dpi=200)

    yerr = np.vstack(
        [
            result.mean_sa_um - result.ci_low_sa_um,
            result.ci_high_sa_um - result.mean_sa_um,
        ]
    )

    ax.errorbar(
        result.times,
        result.mean_sa_um,
        yerr=yerr,
        fmt="o-",
        capsize=4,
        label=f"Mean Sa with {100 * result.confidence_level:.1f}% CI",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$S_a$ ($\mu$m)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return ax


def plot_psd_gain_final_vs_initial(
    analysis: FullAnalysisResult,
    *,
    ax: plt.Axes | None = None,
    initial_time_index: int = 0,
    final_time_index: int = -1,
    wavelength_range_um: tuple[float, float] | None = None,
    use_db: bool = False,
    title: str | None = None,
    ci_alpha: float = 0.25,
    confidence_level: float = 0.95,
    n_resamples: int = 20_000,
    bootstrap_method: str = "BCa",
    seed: int = 12345,
    paired: bool = True,
) -> plt.Axes:
    """
    Plot PSD gain, final vs initial, with a direct pointwise bootstrap CI.

    The gain is computed using only the mean radial PSD curves:

        gain(lambda) = mean_PSD_final(lambda) / mean_PSD_initial(lambda)

    The CI is calculated by bootstrapping the gain statistic itself across
    specimens. Individual specimen curves are used internally for the
    bootstrap but are not plotted.

    Parameters
    ----------
    analysis
        FullAnalysisResult returned by run_full_analysis.

    ax
        Optional matplotlib axes.

    initial_time_index
        Time index used as the initial reference.

    final_time_index
        Time index used as the final state.

    wavelength_range_um
        Optional wavelength plotting range.

    use_db
        If True, plot:

            10 * log10(final / initial)

        If False, plot the linear ratio.

    title
        Optional plot title.

    ci_alpha
        Transparency for the confidence interval band.

    confidence_level
        Bootstrap confidence level.

    n_resamples
        Number of bootstrap resamples.

    bootstrap_method
        Method accepted by scipy.stats.bootstrap.

    seed
        Random seed.

    paired
        If True, use paired specimen-level bootstrap resampling.

    Returns
    -------
    ax
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(dpi=200)

    wavelength, gain, gain_low, gain_high = calculate_psd_gain_final_vs_initial(
        analysis,
        initial_time_index=initial_time_index,
        final_time_index=final_time_index,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        seed=seed,
        paired=paired,
    )

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(gain)
        & np.isfinite(gain_low)
        & np.isfinite(gain_high)
        & (wavelength > 0.0)
        & (gain > 0.0)
        & (gain_low > 0.0)
        & (gain_high > 0.0)
    )

    if wavelength_range_um is not None:
        wavelength_min, wavelength_max = wavelength_range_um

        if wavelength_min <= 0.0 or wavelength_max <= wavelength_min:
            raise ValueError(
                "wavelength_range_um must be " "(positive_minimum, larger_maximum)."
            )

        valid &= (wavelength >= wavelength_min) & (wavelength <= wavelength_max)

    if not np.any(valid):
        raise ValueError("No positive finite PSD gain values remain after filtering.")

    if use_db:
        gain_plot = 10.0 * np.log10(gain)
        gain_low_plot = 10.0 * np.log10(gain_low)
        gain_high_plot = 10.0 * np.log10(gain_high)
    else:
        gain_plot = gain
        gain_low_plot = gain_low
        gain_high_plot = gain_high

    order = np.argsort(wavelength[valid])

    wavelength_plot = wavelength[valid][order]
    gain_plot = gain_plot[valid][order]
    gain_low_plot = gain_low_plot[valid][order]
    gain_high_plot = gain_high_plot[valid][order]

    if use_db:
        ax.semilogx(
            wavelength_plot,
            gain_plot,
            color="black",
            linewidth=2.0,
            label="Mean PSD gain",
        )

        ax.fill_between(
            wavelength_plot,
            gain_low_plot,
            gain_high_plot,
            color="black",
            alpha=ci_alpha,
            linewidth=0,
            label=f"{100 * confidence_level:.1f}% pointwise paired bootstrap CI",
        )

        ax.axhline(
            0.0,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

        ax.set_ylabel(r"PSD gain, $10\log_{10}(C_f/C_i)$ (dB)")

    else:
        ax.loglog(
            wavelength_plot,
            gain_plot,
            color="black",
            linewidth=2.0,
            label="Mean PSD gain",
        )

        ax.fill_between(
            wavelength_plot,
            gain_low_plot,
            gain_high_plot,
            color="black",
            alpha=ci_alpha,
            linewidth=0,
            label=f"{100 * confidence_level:.1f}% pointwise paired bootstrap CI",
        )

        ax.axhline(
            1.0,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

        ax.set_ylabel(r"PSD gain, $C_f/C_i$")

    initial_time = analysis.times[initial_time_index]
    final_time = analysis.times[final_time_index]

    ax.set_xlabel(r"Wavelength, $\lambda$ ($\mu$m)")
    ax.grid(True, which="both", alpha=0.25)

    if title is None:
        title = (
            f"PSD gain vs wavelength: "
            f"final time {final_time:g} / initial time {initial_time:g}"
        )

    ax.set_title(title)
    ax.legend()

    return ax


# =============================================================================
# Saving
# =============================================================================


def save_time_results(
    time_result: PSDTimeResult,
    *,
    output_dir: str | Path,
    time_value: float,
    prefix: str,
    show_specimens_radial: bool = True,
    plot_specimen_2d: bool = False,
) -> None:
    """Save numerical PSD arrays and PSD figures for one time."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    radial = time_result.radial
    full_2d = time_result.full_2d

    time_label = f"t_{time_value:g}".replace(".", "p")
    file_prefix = f"{prefix}_{time_label}"

    np.savez_compressed(
        output_path / f"{file_prefix}_psd_results.npz",
        time=time_value,
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

    # Radial PSD plot.
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=200)

    plot_radial_psd_vs_wavelength(
        radial,
        ax=ax,
        show_specimens=show_specimens_radial,
        title=f"Radial PSD vs wavelength, time = {time_value:g}",
    )

    fig.tight_layout()
    fig.savefig(
        output_path / f"{file_prefix}_radial_psd_vs_wavelength.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Mean 2D PSD plot.
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=200)

    plot_2d_psd(
        full_2d,
        ax=ax,
        title=f"Mean full 2D PSD, time = {time_value:g}",
    )

    fig.tight_layout()
    fig.savefig(
        output_path / f"{file_prefix}_mean_full_2d_psd.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Optional specimen-level 2D PSD plots.
    if plot_specimen_2d:
        arrays_for_norm = [full_2d.mean_psd_um4]
        arrays_for_norm.extend(full_2d.specimen_psd_um4)
        shared_norm = _positive_log_norm(arrays_for_norm)

        for specimen_index in range(full_2d.specimen_psd_um4.shape[0]):
            fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=200)

            plot_2d_psd(
                full_2d,
                specimen_index=specimen_index,
                ax=ax,
                norm=shared_norm,
                title=(
                    f"Specimen {specimen_index} full 2D PSD, " f"time = {time_value:g}"
                ),
            )

            fig.tight_layout()
            fig.savefig(
                output_path
                / f"{file_prefix}_specimen_{specimen_index:03d}_full_2d_psd.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def save_sa_results(
    sa: SaResult,
    *,
    output_dir: str | Path,
    prefix: str,
) -> None:
    """Save Sa summary CSV, specimen Sa array, and Sa plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        {
            "time": sa.times,
            "mean_sa_um": sa.mean_sa_um,
            "std_sa_um": sa.std_sa_um,
            "ci_low_sa_um": sa.ci_low_sa_um,
            "ci_high_sa_um": sa.ci_high_sa_um,
            "confidence_level": sa.confidence_level,
        }
    )

    summary.to_csv(output_path / f"{prefix}_sa_summary.csv", index=False)

    np.savez_compressed(
        output_path / f"{prefix}_sa_results.npz",
        times=sa.times,
        specimen_sa_um=sa.specimen_sa_um,
        mean_sa_um=sa.mean_sa_um,
        std_sa_um=sa.std_sa_um,
        ci_low_sa_um=sa.ci_low_sa_um,
        ci_high_sa_um=sa.ci_high_sa_um,
        confidence_level=sa.confidence_level,
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    plot_sa_vs_time(sa, ax=ax)
    fig.tight_layout()
    fig.savefig(
        output_path / f"{prefix}_sa_vs_time.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_psd_gain_plot(
    analysis: FullAnalysisResult,
    *,
    output_dir: str | Path,
    prefix: str,
    initial_time_index: int = 0,
    final_time_index: int = -1,
    wavelength_range_um: tuple[float, float] | None = None,
    use_db: bool = False,
) -> None:
    """Save PSD gain plot, final vs initial, using mean radial PSD curves."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=200)

    plot_psd_gain_final_vs_initial(
        analysis,
        ax=ax,
        initial_time_index=initial_time_index,
        final_time_index=final_time_index,
        wavelength_range_um=wavelength_range_um,
        use_db=use_db,
    )

    fig.tight_layout()

    suffix = "db" if use_db else "linear"

    fig.savefig(
        output_path / f"{prefix}_psd_gain_final_vs_initial_{suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# Main analysis runner
# =============================================================================


def run_full_analysis(
    *,
    sample_type: str,
    polish: str,
    load: int | float,
    mag: str,
    exclude: Sequence[str] = (),
    output_dir: str | Path | None = None,
    confidence_level: float = 0.95,
    n_resamples: int = 20_000,
    bootstrap_method: str = "BCa",
    min_modes_per_bin: int = 8,
    seed: int = 12345,
    show_specimens_radial: bool = True,
    plot_specimen_2d: bool = False,
    level: bool = True,
    detrend_order: int = 1,
) -> FullAnalysisResult:
    """
    Run complete Sa and PSD analysis over all times.

    Inputs
    ------
    sample_type, polish, load, mag, exclude

    Example
    -------
    result = run_full_analysis(
        sample_type="unint",
        polish="polished",
        load=530,
        mag="10x",
        exclude=["37e"],
    )
    """
    spacing_um = PROFILOMETRY_SPACING_UM[mag]

    if output_dir is None:
        output_dir = f"psd_output_creep_{sample_type}_{polish}_{load}_{mag}"

    prefix = f"creep_{sample_type}_{polish}_{load}_{mag}"

    height_array, times, samples = load_height_maps(
        sample_type=sample_type,
        polish=polish,
        load=load,
        mag=mag,
        exclude=exclude,
        crop=(slice(50, -50), slice(50, 750)),
        level=level,
        detrend_order=detrend_order,
    )

    print("Loaded height array shape:", height_array.shape)
    print("Samples:", samples)
    print("Times:", times)
    print("Spacing um:", spacing_um)

    sa = calculate_sa_by_time(
        height_array,
        times,
        confidence_level=confidence_level,
    )

    save_sa_results(
        sa,
        output_dir=output_dir,
        prefix=prefix,
    )

    psd_by_time: list[PSDTimeResult] = []

    for time_index, time_value in enumerate(times):
        print(f"Analyzing time {time_value:g} ({time_index + 1}/{len(times)})")

        full_2d = calculate_2d_psd_at_time(
            height_array,
            time_index,
            spacing_um=spacing_um,
        )

        radial = radial_psd_from_2d(
            full_2d,
            spacing_um=spacing_um,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_method=bootstrap_method,
            min_modes_per_bin=8.0,
            radial_binning="log",
            radial_bin_width_factor=1.0,
            bins_per_decade=8.0,
            seed=seed + time_index,
        )

        time_result = PSDTimeResult(
            time=float(time_value),
            radial=radial,
            full_2d=full_2d,
        )

        psd_by_time.append(time_result)

        save_time_results(
            time_result,
            output_dir=output_dir,
            time_value=float(time_value),
            prefix=prefix,
            show_specimens_radial=show_specimens_radial,
            plot_specimen_2d=plot_specimen_2d,
        )

        print(
            f"  Max Parseval relative error: "
            f"{radial.parseval_relative_error.max():.3e}"
        )

    result = FullAnalysisResult(
        sample_type=sample_type,
        polish=polish,
        load=load,
        mag=mag,
        samples=samples,
        times=times,
        height_shape=tuple(height_array.shape[-2:]),
        spacing_um=spacing_um,
        sa=sa,
        psd_by_time=psd_by_time,
    )

    save_psd_gain_plot(
        result,
        output_dir=output_dir,
        prefix=prefix,
        initial_time_index=0,
        final_time_index=-1,
        use_db=False,
    )

    # save_psd_gain_plot(
    #     result,
    #     output_dir=output_dir,
    #     prefix=prefix,
    #     initial_time_index=0,
    #     final_time_index=-1,
    #     use_db=True,
    # )

    print(f"Saved outputs to: {Path(output_dir).resolve()}")

    return result


# =============================================================================
# Example call
# =============================================================================

result = run_full_analysis(
    sample_type="unint",
    polish="polished",
    load=500,
    mag="10x",
    exclude=["37e"],
    confidence_level=0.95,
    n_resamples=20_000,
    bootstrap_method="BCa",
    min_modes_per_bin=8,
    seed=12345,
    show_specimens_radial=True,
    plot_specimen_2d=False,
    level=True,
    detrend_order=1,
)

# %%
exclude47_10x = [
    "36a",
    "34a",
    "32a",
]
exclude47_50x = [
    "35a",
]

# %%
from PIL import Image

# 1. Open the PNG image
image = Image.open(
    "/Users/gtdebru/mimosa/tmp/psd_output_creep_unint_polished_530_10x/creep_unint_polished_530_10x_psd_gain_final_vs_initial_linear.png"
)

# 2. Convert from RGBA to RGB (PDF format requires RGB)
rgb_image = image.convert("RGB")

# 3. Save as a PDF
rgb_image.save("/Users/gtdebru/Downloads/psd_gain.pdf", "PDF")

# %%
from utils.data_utils import SimResults

micro1 = "microstructures/production/micro1_production.dat"
micro1_530 = SimResults.load(
    "hpc_downloads/gtdebru/micro1_production/530mpa_unint", microstructure=micro1
)
heights = np.transpose(micro1_530.height, (1, 0, 2, 3))

# %%
import numpy as np

heights = np.transpose(micro1_530.height, (1, 0, 2, 3))
heights.shape

# %%
np.where(heights == heights.max()), np.where(heights == heights.min())

# %%
heights.max(), heights.min()

# %%
%matplotlib inline
import matplotlib.pyplot as plt
from utils.plotting.common import new_fig_ax

fig, ax = plt.subplots(dpi = 200)

arr = heights[-1, 2, :, :]
im = ax.imshow(arr, cmap = 'coolwarm')
fig.colorbar(im, ax = ax, label = r"$\text{Height} \ (\mu m)$")
plt.show()

# %%
fig, ax = plt.subplots(dpi=200)

arr = heights[-1, 3, :, :]
im = ax.imshow(arr, cmap="coolwarm")
fig.colorbar(im, ax=ax, label=r"$\text{Height} \ (\mu m)$")
plt.show()

# %%
import pandas as pd
from io import StringIO

path = "/Users/gtdebru/Downloads/untitled-0397_4.csv"
filename = path

start_row = 1086  # 1-indexed row where first header appears
skip_before = start_row - 1

n_datasets = 4


def is_blank_line(line):
    """
    Treat truly empty lines, or CSV rows like ',,,,', as blank.
    """
    if not line.strip():
        return True

    cells = line.rstrip("\n\r").split(",")
    return all(cell.strip().strip('"') == "" for cell in cells)


with open(filename, "r", newline="") as f:
    lines = f.readlines()

# Drop everything before the first sub-dataset header
lines = lines[skip_before:]

blocks = []
current_block = []

for line in lines:
    if is_blank_line(line):
        if current_block:
            blocks.append(current_block)
            current_block = []

            if len(blocks) == n_datasets:
                break
    else:
        current_block.append(line)

# In case the final block does not end with a blank line
if current_block and len(blocks) < n_datasets:
    blocks.append(current_block)

datasets = []

for block in blocks:
    df = pd.read_csv(StringIO("".join(block)), skipinitialspace=True)

    df.columns = df.columns.str.strip().str.strip('"').str.strip()

    datasets.append(df)

print(f"Read {len(datasets)} datasets")

for i, df in enumerate(datasets):
    if i == 0:
        print(f"Columns: {df.columns.tolist()}")
    print(f"Dataset {i}: {df.shape}")

# %%
import numpy as np

df1 = datasets[0]

xy1 = df1[["x_c", "y_c"]].to_numpy()

x = df1["x_c"].to_numpy()
y = df1["y_c"].to_numpy()

x_unique = np.unique(x)
y_unique = np.unique(y)

nx = len(x_unique)
ny = len(y_unique)


X1 = x.reshape(ny, nx)
Y1 = y.reshape(ny, nx)

u = df1["u"].to_numpy()
v = df1["v"].to_numpy()

U1 = u.reshape(ny, nx)
V1 = v.reshape(ny, nx)

u_c = df1["u_c"].to_numpy()
v_c = df1["v_c"].to_numpy()

U1c = u_c.reshape(ny, nx)
V1c = v_c.reshape(ny, nx)

print(X1.shape)
print(Y1.shape)

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(x, y, s=10)
plt.axis("equal")
plt.xlabel("x_c")
plt.ylabel("y_c")
plt.title("DIC coordinate grid")
plt.show()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

m = ax.pcolormesh(
    Y1,  # y position
    X1,  # z position
    U1c,  # z displacement
    vmin=U1c.min(),
    vmax=U1c.max(),
    cmap="coolwarm",
    shading="auto",
)

ax.set_aspect("equal")

fig.colorbar(m, ax=ax, label="z displacement")

ax.set_xlabel("y")
ax.set_ylabel("z")
ax.set_title("z displacement")
fig.set_size_inches(fig.get_size_inches() * 1.5, forward=True)


plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt


def centers_to_edges(c):
    c = np.asarray(c)
    c = np.sort(np.unique(c))

    dc = np.diff(c)

    edges = np.empty(len(c) + 1)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * dc[0]
    edges[-1] = c[-1] + 0.5 * dc[-1]

    return edges


# Horizontal axis is y
y_edges = centers_to_edges(Y1)

# Vertical axis is z
z_edges = centers_to_edges(X1)

fig, ax = plt.subplots(dpi=100)

m = ax.pcolormesh(
    Y1,
    X1,
    V1c,
    vmin=V1c.min(),
    vmax=V1c.max(),
    cmap="coolwarm",
    shading="auto",
)

# Draw thin pixel/cell boundaries
ax.vlines(
    y_edges,
    z_edges.min(),
    z_edges.max(),
    colors="k",
    linewidth=0.2,
    alpha=0.35,
)

ax.hlines(
    z_edges,
    y_edges.min(),
    y_edges.max(),
    colors="k",
    linewidth=0.2,
    alpha=0.35,
)

ax.set_aspect("equal")

fig.colorbar(m, ax=ax, label="y displacement")

ax.set_xlabel("y")
ax.set_ylabel("z")

fig.set_size_inches(fig.get_size_inches() * 1.5, forward=True)

plt.show()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=100)

m = ax.pcolormesh(
    Y1,  # y position
    X1,  # z position
    V1c,  # y displacement
    vmin=V1c.min(),
    vmax=V1c.max(),
    cmap="coolwarm",
    shading="auto",
)

ax.set_aspect("equal")

fig.colorbar(m, ax=ax, label="y displacement")

ax.set_xlabel("y")
ax.set_ylabel("z")

fig.set_size_inches(fig.get_size_inches() * 1.5, forward=True)

plt.show()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=100)

sy_lo = 4
sy_hi = 4
sx_lo = 10
sx_hi = 5
m = ax.pcolormesh(
    Y1[sy_lo:-sy_hi, sx_lo:-sx_hi],
    X1[sy_lo:-sy_hi, sx_lo:-sx_hi],
    np.abs(V1c[sy_lo:-sy_hi, sx_lo:-sx_hi]),
    # vmin=V1c.min(),
    # vmax=V1c.max(),
    cmap="coolwarm",
    shading="auto",
)

ax.set_aspect("equal")

fig.colorbar(m, ax=ax, label="y displacement")

ax.set_xlabel("y")
ax.set_ylabel("z")
fig.set_size_inches(fig.get_size_inches() * 1.5, forward=True)
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Generate a smooth gradient from red to blue
# Lower values map to red, higher values map to blue
red_blue_seq = mcolors.LinearSegmentedColormap.from_list(
    "RedBlueSeq",
    ["darkblue", "darkred"],
)

fig, ax = plt.subplots(dpi=100)

sy_lo = 4
sy_hi = 4
sx_lo = 10
sx_hi = 5
m = ax.pcolormesh(
    Y1[sy_lo:-sy_hi, sx_lo:-sx_hi],
    X1[sy_lo:-sy_hi, sx_lo:-sx_hi],
    np.abs(U1c[sy_lo:-sy_hi, sx_lo:-sx_hi]),
    # vmin=V1c.min(),
    # vmax=V1c.max(),
    cmap="coolwarm",
    shading="auto",
)

ax.set_aspect("equal")

fig.colorbar(m, ax=ax, label="z displacement")

ax.set_xlabel("y")
ax.set_ylabel("z")
fig.set_size_inches(fig.get_size_inches() * 1.5, forward=True)
plt.show()

# %%
import numpy as np


def strain_from_dic_df(df):
    """
    Computes small-strain components from DIC displacement fields.

    Assumes:
        x_c -> z position
        y_c -> y position
        u_c -> z displacement
        v_c -> y displacement

    Returns dictionary containing coordinate grids, displacement grids,
    and strain fields.
    """

    # Sort into regular grid using y as rows, z as columns
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())  # x_c is z
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(
            f"Data are not a complete rectangular grid: "
            f"len(df)={len(df_grid)}, ny*nz={ny*nz}"
        )

    Z = df_grid["x_c"].to_numpy().reshape(ny, nz)
    Y = df_grid["y_c"].to_numpy().reshape(ny, nz)

    U = df_grid["u_c"].to_numpy().reshape(ny, nz)  # z displacement
    V = df_grid["v_c"].to_numpy().reshape(ny, nz)  # y displacement

    # 1D coordinate arrays for gradient spacing
    z = Z[0, :]
    y = Y[:, 0]

    # Gradients
    # U and V have shape (ny, nz)
    # axis 0 = y direction
    # axis 1 = z direction
    dU_dy, dU_dz = np.gradient(U, y, z, edge_order=2)
    dV_dy, dV_dz = np.gradient(V, y, z, edge_order=2)

    eps_zz = dU_dz
    eps_yy = dV_dy
    eps_yz = 0.5 * (dU_dy + dV_dz)
    gamma_yz = dU_dy + dV_dz

    return {
        "Y": Y,
        "Z": Z,
        "U": U,
        "V": V,
        "dU_dy": dU_dy,
        "dU_dz": dU_dz,
        "dV_dy": dV_dy,
        "dV_dz": dV_dz,
        "eps_zz": eps_zz,
        "eps_yy": eps_yy,
        "eps_yz": eps_yz,
        "gamma_yz": gamma_yz,
    }


strain1 = strain_from_dic_df(datasets[0])

Y1 = strain1["Y"]
Z1 = strain1["Z"]

eps_zz_1 = strain1["eps_zz"]
eps_yy_1 = strain1["eps_yy"]
eps_yz_1 = strain1["eps_yz"]
gamma_yz_1 = strain1["gamma_yz"]
strain_datasets = [strain_from_dic_df(df) for df in datasets]
import matplotlib.pyplot as plt

s = strain_datasets[0]

fig, ax = plt.subplots(dpi=100)

m = ax.pcolormesh(
    s["Y"],
    s["Z"],
    s["eps_zz"],
    cmap="coolwarm",
    shading="auto",
)

ax.set_aspect("equal")
fig.colorbar(m, ax=ax, label="strain")

ax.set_xlabel("y")
ax.set_ylabel("z")
ax.set_title("epsilon_zz")

plt.show()
fig, ax = plt.subplots(dpi=100)

m = ax.pcolormesh(
    s["Y"],
    s["Z"],
    s["eps_yy"],
    cmap="coolwarm",
    shading="auto",
)

ax.set_aspect("equal")
fig.colorbar(m, ax=ax, label="strain")

ax.set_xlabel("y")
ax.set_ylabel("z")
ax.set_title("epsilon_yy")

plt.show()
fig, ax = plt.subplots(dpi=100)

m = ax.pcolormesh(
    s["Y"],
    s["Z"],
    s["gamma_yz"],
    cmap="coolwarm",
    shading="auto",
)

ax.set_aspect("equal")
fig.colorbar(m, ax=ax, label="strain")

ax.set_xlabel("y")
ax.set_ylabel("z")
ax.set_title("gamma_yz")

plt.show()

# %%
import numpy as np

# Bulk z engineering strain for each DIC dataset
# Mapping:
#   x_c = z position
#   u_c = z displacement
#
# Engineering strain:
#   eps_z = Delta L / L0 = Delta u_z / Delta z
#
# This uses a least-squares line fit:
#   u_c = eps_z * x_c + intercept
# so the slope is the bulk z engineering strain.

bulk_z_strains = []

for i, df in enumerate(datasets):
    z = df["x_c"].to_numpy(dtype=float)
    u = df["u_c"].to_numpy(dtype=float)

    # Remove NaN/inf if present
    mask = np.isfinite(z) & np.isfinite(u)
    z = z[mask]
    u = u[mask]

    # Fit u = eps_z * z + b
    eps_z, intercept = np.polyfit(z, u, 1)

    bulk_z_strains.append(eps_z)

    print(f"Dataset {i}: bulk z engineering strain = {eps_z:.6e}  ({100*eps_z:.4f}%)")

bulk_z_strains = np.array(bulk_z_strains)

# %%
import numpy as np

# Dataset 1 crop values
sy_lo = 4
sy_hi = 4
sx_lo = 10
sx_hi = 5

# Dataset 1
df = datasets[0]

# Sort into the same row-major grid used for X1/Y1/U1c/V1c
df_grid = df.sort_values(["y_c", "x_c"]).copy()

z_vals = np.sort(df_grid["x_c"].unique())
y_vals = np.sort(df_grid["y_c"].unique())

nz = len(z_vals)
ny = len(y_vals)

if len(df_grid) != ny * nz:
    raise ValueError(
        f"Dataset is not a complete grid: len={len(df_grid)}, ny*nz={ny*nz}"
    )

Z = df_grid["x_c"].to_numpy(dtype=float).reshape(ny, nz)  # z position
U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement

# Apply the same crop as your plot
Z_crop = Z[sy_lo:-sy_hi, sx_lo:-sx_hi]
U_crop = U[sy_lo:-sy_hi, sx_lo:-sx_hi]

# Flatten cropped region
z = Z_crop.ravel()
u = U_crop.ravel()

# Remove NaN/inf if present
mask = np.isfinite(z) & np.isfinite(u)
z = z[mask]
u = u[mask]

# Bulk z engineering strain from least-squares fit:
# u_z = eps_z * z + intercept
eps_z, intercept = np.polyfit(z, u, 1)

print(f"Bulk z engineering strain, cropped dataset 1 = {eps_z:.6e}")
print(f"Bulk z engineering strain, cropped dataset 1 = {100 * eps_z:.4f}%")

# %%
np.sum(df1["sigma"].values == -1)

# %%
import numpy as np
import pandas as pd

# Same crop used in your plot
sy_lo = 4
sy_hi = 4
sx_lo = 10
sx_hi = 5


def crop_slice(lo, hi):
    """Handles hi=0 correctly."""
    return slice(lo, None if hi == 0 else -hi)


ys = crop_slice(sy_lo, sy_hi)
xs = crop_slice(sx_lo, sx_hi)

bulk_results = []

for i, df in enumerate(datasets):
    # Sort into grid order: rows = y, columns = z/x_c
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(
            f"Dataset {i} is not a complete grid: " f"len={len(df_grid)}, ny*nz={ny*nz}"
        )

    # Position/displacement mapping
    # x_c = z position
    # u_c = z displacement
    Z = df_grid["x_c"].to_numpy(dtype=float).reshape(ny, nz)
    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)

    # Apply same crop
    Z_crop = Z[ys, xs]
    U_crop = U[ys, xs]

    # Flatten cropped region
    z = Z_crop.ravel()
    u = U_crop.ravel()

    # Remove NaN/inf
    mask = np.isfinite(z) & np.isfinite(u)
    z = z[mask]
    u = u[mask]

    if len(z) < 2:
        raise ValueError(f"Dataset {i} crop has fewer than 2 valid points.")

    # Bulk z engineering strain from fit:
    # u_z = eps_z * z + intercept
    eps_z, intercept = np.polyfit(z, u, 1)

    bulk_results.append(
        {
            "dataset": i,
            "ny_full": ny,
            "nz_full": nz,
            "ny_crop": Z_crop.shape[0],
            "nz_crop": Z_crop.shape[1],
            "eps_z": eps_z,
            "eps_z_percent": 100 * eps_z,
            "intercept": intercept,
        }
    )

    print(
        f"Dataset {i}: "
        f"crop shape = {Z_crop.shape}, "
        f"bulk z engineering strain = {eps_z:.6e} "
        f"({100 * eps_z:.4f}%)"
    )

bulk_results = pd.DataFrame(bulk_results)
bulk_results

# %%
import numpy as np

# Bulk z engineering strain for each DIC dataset
# Mapping:
#   x_c = z position
#   u_c = z displacement
#
# Engineering strain:
#   eps_z = Delta L / L0 = Delta u_z / Delta z
#
# This uses a least-squares line fit:
#   u_c = eps_z * x_c + intercept
# so the slope is the bulk z engineering strain.
#
# Points where sigma == -1 are excluded.

bulk_z_strains = []
bulk_z_intercepts = []

for i, df in enumerate(datasets):
    z = df["x_c"].to_numpy(dtype=float)
    u = df["u_c"].to_numpy(dtype=float)
    sigma = df["sigma"].to_numpy(dtype=float)

    # Keep only valid points:
    # finite z, finite u, finite sigma, and sigma != -1
    mask = np.isfinite(z) & np.isfinite(u) & np.isfinite(sigma) & (sigma != -1)

    z_valid = z[mask]
    u_valid = u[mask]

    if len(z_valid) < 2:
        raise ValueError(
            f"Dataset {i}: fewer than 2 valid points after masking sigma == -1"
        )

    # Fit u = eps_z * z + b
    eps_z, intercept = np.polyfit(z_valid, u_valid, 1)

    bulk_z_strains.append(eps_z)
    bulk_z_intercepts.append(intercept)

    print(
        f"Dataset {i}: "
        f"valid points = {len(z_valid)}/{len(z)}, "
        f"bulk z engineering strain = {eps_z:.6e}  ({100 * eps_z:.4f}%)"
    )

bulk_z_strains = np.array(bulk_z_strains)
bulk_z_intercepts = np.array(bulk_z_intercepts)

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# df1 = first dataset
df1 = datasets[0].copy()

# Sort into grid order
df1_grid = df1.sort_values(["y_c", "x_c"]).copy()

z_vals = np.sort(df1_grid["x_c"].unique())
y_vals = np.sort(df1_grid["y_c"].unique())

nz = len(z_vals)
ny = len(y_vals)

# Reshape full fields
Z = df1_grid["x_c"].to_numpy(dtype=float).reshape(ny, nz)  # z position
Y = df1_grid["y_c"].to_numpy(dtype=float).reshape(ny, nz)  # y position
sigma = df1_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

# Categorical mask:
# 1 = sigma == -1 -> green
# 0 = otherwise   -> purple
sigma_mask = np.where(np.isclose(sigma, -1), 1, 0)

cmap = ListedColormap(["purple", "green"])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

fig, ax = plt.subplots(dpi=100)

m = ax.pcolormesh(
    Y,
    Z,
    sigma_mask,
    cmap=cmap,
    norm=norm,
    shading="auto",
)

ax.set_aspect("equal")

cbar = fig.colorbar(m, ax=ax, ticks=[0, 1])
cbar.ax.set_yticklabels(["sigma != -1", "sigma == -1"])

ax.set_xlabel("y")
ax.set_ylabel("z")
ax.set_title("sigma mask")

fig.set_size_inches(fig.get_size_inches() * 1.5, forward=True)

plt.show()

# %%
df1["sigma"].max(), df1["sigma"].min()

# %%
import numpy as np
import matplotlib.pyplot as plt

# df1 = first dataset
df1 = datasets[0].copy()

# Sort into grid order
df1_grid = df1.sort_values(["y_c", "x_c"]).copy()

z_vals = np.sort(df1_grid["x_c"].unique())
y_vals = np.sort(df1_grid["y_c"].unique())

nz = len(z_vals)
ny = len(y_vals)

# Reshape full fields
Z = df1_grid["x_c"].to_numpy(dtype=float).reshape(ny, nz)  # z position
Y = df1_grid["y_c"].to_numpy(dtype=float).reshape(ny, nz)  # y position
S = df1_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

fig, ax = plt.subplots(dpi=100)

m = ax.pcolormesh(
    Y,
    Z,
    S,
    cmap="viridis",
    shading="auto",
)

ax.set_aspect("equal")

fig.colorbar(m, ax=ax, label="sigma")

ax.set_xlabel("y")
ax.set_ylabel("z")
ax.set_title("sigma")

fig.set_size_inches(fig.get_size_inches() * 1.5, forward=True)

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Per-dataset crops: (sy_lo, sy_hi, sx_lo, sx_hi)
crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_sigma_grid(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    Z = df_grid["x_c"].to_numpy(dtype=float).reshape(ny, nz)  # z
    Y = df_grid["y_c"].to_numpy(dtype=float).reshape(ny, nz)  # y
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    return Y, Z, S


# Build fields
fields = []
vmin = 1000.0
vmax = -10000.0
for i, df in enumerate(datasets):
    Y, Z, S = df_to_sigma_grid(df)

    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]
    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)
    vmin = min(vmin, S[S > -1].min())
    vmax = max(vmax, S.max())

    fields.append(
        {
            "full": {
                "Y": Y,
                "Z": Z,
                "S": S,
            },
            "crop": {
                "Y": Y[ys, xs],
                "Z": Z[ys, xs],
                "S": S[ys, xs],
            },
        }
    )

# Use first uncropped heatmap as panel aspect ratio
Y0 = fields[0]["full"]["Y"]
Z0 = fields[0]["full"]["Z"]

panel_aspect = (Z0.max() - Z0.min()) / (Y0.max() - Y0.min())

fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120)

fig.set_size_inches(fig.get_size_inches() * np.array([2.0, 2.0]), forward=True)

dx, dz = [], []
nx, nz = [], []

for i, field in enumerate(fields):
    for j, key in enumerate(["full", "crop"]):
        ax = axes[i, j]
        Y = field[key]["Y"]
        Z = field[key]["Z"]
        S = field[key]["S"]
        if key == "crop":
            nxi, nzi = Z.shape
            dxi = 2 / nxi
            dzi = 9 / nzi
            print(dzi)
            dx.append(dxi)
            dz.append(dzi)
            nx.append(nxi)
            nz.append(nzi)

        valid = np.isfinite(S) & (S != -1)

        # if np.any(valid):
        #     vmin = S[valid].min()
        #     vmax = S[valid].max()
        # else:
        #     vmin = np.nanmin(S)
        #     vmax = np.nanmax(S)

        m = ax.pcolormesh(
            Y,
            Z,
            S,
            cmap="viridis",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        # Same height/width ratio for every heatmap box
        ax.set_box_aspect(panel_aspect)

        ax.set_title(f"D{i} {key}", fontsize=9, pad=2)
        ax.set_xlabel("y", fontsize=8, labelpad=1)
        ax.set_ylabel("z", fontsize=8, labelpad=1)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", labelsize=7, pad=1)

        for tick in ax.get_xticklabels():
            tick.set_rotation(35)
            tick.set_ha("right")
fig.set_layout_engine("none")
fig.subplots_adjust(
    left=0.0,
    right=0.7,
    bottom=0,
    top=1,
    wspace=-0.75,
    hspace=0.2,
)

plt.show()

# %%
dx, dz, nx, nz

# %%
dx = 0.25
dz = 0.2

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_strain_grid(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())  # x_c = z
    y_vals = np.sort(df_grid["y_c"].unique())  # y_c = y

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    Z = df_grid["x_c"].to_numpy(dtype=float).reshape(ny, nz)  # z position
    Y = df_grid["y_c"].to_numpy(dtype=float).reshape(ny, nz)  # y position
    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    z = Z[0, :]
    y = Y[:, 0]

    # epsilon_zz = du_z / dz
    dU_dy, dU_dz = np.gradient(U, y, z, edge_order=2)
    Ezz = dU_dz

    return Y, Z, Ezz, S


# First compute all cropped eps_zz fields so they share the same color scale
fields = []
all_eps = []

for i, df in enumerate(datasets):
    Y, Z, Ezz, S = df_to_strain_grid(df)

    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]
    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Yc = Y[ys, xs]
    Zc = Z[ys, xs]
    Ezzc = Ezz[ys, xs]
    Sc = S[ys, xs]

    # Mask invalid sigma
    Ezzc = np.where(Sc == -1, np.nan, Ezzc)

    fields.append(
        {
            "Y": Yc,
            "Z": Zc,
            "Ezz": Ezzc,
        }
    )

    valid = np.isfinite(Ezzc)
    if np.any(valid):
        all_eps.append(Ezzc[valid].ravel())

all_eps = np.concatenate(all_eps)

# Shared symmetric color scale
vabs = np.nanmax(np.abs(all_eps))
norm = mpl.colors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

cmap = plt.get_cmap("coolwarm").copy()
cmap.set_bad("lightgray")

# Separate plots
for i, field in enumerate(fields):
    fig, ax = plt.subplots(dpi=120)

    m = ax.pcolormesh(
        field["Y"],
        field["Z"],
        field["Ezz"],
        cmap=cmap,
        # norm=norm,
        shading="auto",
    )

    ax.set_aspect("equal", adjustable="box")

    fig.colorbar(m, ax=ax, label="epsilon_zz")

    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(f"Dataset {i}: epsilon_zz")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.set_size_inches(fig.get_size_inches() * 1.3, forward=True)

    plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Crops: (sy_lo, sy_hi, sx_lo, sx_hi)
crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]

dy_mm = 0.25
dz_mm = 0.20
lambda_min_mm = 0.20

# If u_c/v_c are already in mm, leave this as 1.0.
# If u_c/v_c are in pixels, set this to your mm/pixel scale.
disp_scale_mm_per_unit = 1.0


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_grids_mm(df, dy_mm, dz_mm, disp_scale_mm_per_unit=1.0):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz) * disp_scale_mm_per_unit
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz) * disp_scale_mm_per_unit
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    y = np.arange(ny) * dy_mm
    z = np.arange(nz) * dz_mm

    Y, Z = np.meshgrid(y, z, indexing="ij")

    return Y, Z, U, V, S


def strain_fields(U, V, dy_mm, dz_mm):
    # Array shape is (ny, nz)
    # axis 0 = y
    # axis 1 = z
    dU_dy, dU_dz = np.gradient(U, dy_mm, dz_mm, edge_order=2)
    dV_dy, dV_dz = np.gradient(V, dy_mm, dz_mm, edge_order=2)

    eps_zz = dU_dz
    eps_yy = dV_dy
    gamma_yz = dU_dy + dV_dz

    return eps_zz, eps_yy, gamma_yz


def characteristic_wavelength_psd(A, dy_mm, dz_mm, lambda_min_mm):
    A = np.asarray(A, dtype=float)
    valid = np.isfinite(A)

    if np.count_nonzero(valid) < 4:
        return np.nan

    A0 = A.copy()
    A0[~valid] = np.nanmean(A0[valid])
    A0 = A0 - np.nanmean(A0)

    ny, nz = A0.shape

    wy = np.hanning(ny)
    wz = np.hanning(nz)
    W = np.outer(wy, wz)

    F = np.fft.fft2(A0 * W)
    P = np.abs(F) ** 2

    fy = np.fft.fftfreq(ny, d=dy_mm)
    fz = np.fft.fftfreq(nz, d=dz_mm)

    FY, FZ = np.meshgrid(fy, fz, indexing="ij")
    FR = np.sqrt(FY**2 + FZ**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / FR

    mask = (
        np.isfinite(wavelength)
        & (FR > 0)
        & (wavelength >= lambda_min_mm)
        & np.isfinite(P)
    )

    if not np.any(mask):
        return np.nan

    idx = np.argmax(P[mask])
    return wavelength[mask][idx]


def autocorrelation_lengths(A, dy_mm, dz_mm):
    A = np.asarray(A, dtype=float)
    valid = np.isfinite(A)

    if np.count_nonzero(valid) < 4:
        return np.nan, np.nan, np.nan

    A0 = A.copy()
    A0[~valid] = np.nanmean(A0[valid])
    A0 = A0 - np.nanmean(A0)

    F = np.fft.fft2(A0)
    ac = np.fft.ifft2(np.abs(F) ** 2).real
    ac = np.fft.fftshift(ac)

    cy, cz = np.array(ac.shape) // 2

    if ac[cy, cz] == 0:
        return np.nan, np.nan, np.nan

    ac = ac / ac[cy, cz]

    ny, nz = A.shape

    y_lags = (np.arange(ny) - cy) * dy_mm
    z_lags = (np.arange(nz) - cz) * dz_mm

    ac_y = ac[:, cz]
    ac_z = ac[cy, :]

    def first_one_over_e_length(lags, vals):
        center = np.argmin(np.abs(lags))
        lags_pos = lags[center:]
        vals_pos = vals[center:]

        target = np.exp(-1)

        for k in range(1, len(vals_pos)):
            if vals_pos[k] <= target:
                x0, x1 = lags_pos[k - 1], lags_pos[k]
                y0, y1 = vals_pos[k - 1], vals_pos[k]

                if y1 == y0:
                    return x1

                return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

        return np.nan

    corr_y = first_one_over_e_length(y_lags, ac_y)
    corr_z = first_one_over_e_length(z_lags, ac_z)

    YY, ZZ = np.meshgrid(y_lags, z_lags, indexing="ij")
    R = np.sqrt(YY**2 + ZZ**2)

    r = R.ravel()
    a = ac.ravel()

    mask = np.isfinite(r) & np.isfinite(a)
    r = r[mask]
    a = a[mask]

    order = np.argsort(r)
    r = r[order]
    a = a[order]

    target = np.exp(-1)
    corr_radial = np.nan

    for k in range(1, len(r)):
        if r[k] > 0 and a[k] <= target:
            r0, r1 = r[k - 1], r[k]
            a0, a1 = a[k - 1], a[k]

            if a1 == a0:
                corr_radial = r1
            else:
                corr_radial = r0 + (target - a0) * (r1 - r0) / (a1 - a0)

            break

    return corr_y, corr_z, corr_radial


results = []

for i, df in enumerate(datasets):
    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]
    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Y, Z, U, V, S = df_to_grids_mm(
        df,
        dy_mm=dy_mm,
        dz_mm=dz_mm,
        disp_scale_mm_per_unit=disp_scale_mm_per_unit,
    )

    Yc = Y[ys, xs]
    Zc = Z[ys, xs]
    Uc = U[ys, xs]
    Vc = V[ys, xs]
    Sc = S[ys, xs]

    eps_zz, eps_yy, gamma_yz = strain_fields(Uc, Vc, dy_mm, dz_mm)

    invalid = Sc == -1
    eps_zz = np.where(invalid, np.nan, eps_zz)
    eps_yy = np.where(invalid, np.nan, eps_yy)
    gamma_yz = np.where(invalid, np.nan, gamma_yz)

    fields = [
        ("z strain", eps_zz),
        ("y strain", eps_yy),
        ("shear strain", gamma_yz),
    ]

    fig, axes = plt.subplots(1, 3, dpi=120, constrained_layout=True)
    fig.set_size_inches(fig.get_size_inches() * np.array([1.8, 1.1]), forward=True)

    for ax, (label, A) in zip(axes, fields):
        var_A = np.nanvar(A)

        char_lambda = characteristic_wavelength_psd(
            A,
            dy_mm=dy_mm,
            dz_mm=dz_mm,
            lambda_min_mm=lambda_min_mm,
        )

        corr_y, corr_z, corr_radial = autocorrelation_lengths(
            A,
            dy_mm=dy_mm,
            dz_mm=dz_mm,
        )

        results.append(
            {
                "dataset": i,
                "field": label,
                "variance": var_A,
                "characteristic_wavelength_mm": char_lambda,
                "corr_length_y_mm": corr_y,
                "corr_length_z_mm": corr_z,
                "corr_length_radial_mm": corr_radial,
            }
        )

        # Individual color scale for every plot
        vabs = np.nanmax(np.abs(A))

        if not np.isfinite(vabs) or vabs == 0:
            vabs = 1.0

        norm = mpl.colors.TwoSlopeNorm(
            vmin=-vabs,
            vcenter=0.0,
            vmax=vabs,
        )

        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("lightgray")

        m = ax.pcolormesh(
            Yc,
            Zc,
            A,
            cmap=cmap,
            norm=norm,
            shading="auto",
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("y")
        ax.set_ylabel("z")

        # Only title text is variance
        ax.set_title(f"var={var_A:.3e}")

        fig.colorbar(m, ax=ax, label=label)

    plt.show()

results = pd.DataFrame(results)
results

# %%
dx, dz

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]

dy_mm = 0.25
dz_mm = 0.20
lambda_min_mm = 0.20 * 5


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_grids(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())  # x_c = z
    y_vals = np.sort(df_grid["y_c"].unique())  # y_c = y

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz)  # y displacement
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    y = np.arange(ny) * dy_mm
    z = np.arange(nz) * dz_mm
    Y, Z = np.meshgrid(y, z, indexing="ij")

    return Y, Z, U, V, S


def strain_fields(U, V, dy_mm, dz_mm):
    # Shape: (ny, nz)
    # axis 0 = y
    # axis 1 = z
    dU_dy, dU_dz = np.gradient(U, dy_mm, dz_mm, edge_order=2)
    dV_dy, dV_dz = np.gradient(V, dy_mm, dz_mm, edge_order=2)

    eps_zz = dU_dz
    eps_yy = dV_dy
    gamma_yz = dU_dy + dV_dz

    return eps_zz, eps_yy, gamma_yz


def fill_nan_with_mean(A):
    A = np.asarray(A, dtype=float)
    valid = np.isfinite(A)

    if not np.any(valid):
        raise ValueError("Array has no finite values.")

    B = A.copy()
    B[~valid] = np.nanmean(B[valid])
    return B


def psd_2d_hann(A, dy_mm, dz_mm):
    """
    Rectangular/separable Hann-windowed 2D PSD.

    Returns:
        FY, FZ: shifted frequency grids in cycles/mm
        PSD: shifted PSD
    """
    A = np.asarray(A, dtype=float)

    A0 = fill_nan_with_mean(A)
    A0 = A0 - np.mean(A0)

    ny, nz = A0.shape

    wy = np.hanning(ny)
    wz = np.hanning(nz)
    W = np.outer(wy, wz)

    Aw = A0 * W

    F = np.fft.fft2(Aw)

    # Window-power normalization
    PSD = np.abs(F) ** 2 / np.sum(W**2)

    fy = np.fft.fftfreq(ny, d=dy_mm)
    fz = np.fft.fftfreq(nz, d=dz_mm)

    PSD = np.fft.fftshift(PSD)
    fy = np.fft.fftshift(fy)
    fz = np.fft.fftshift(fz)

    FZ, FY = np.meshgrid(fz, fy, indexing="xy")

    return FY, FZ, PSD


def radial_average_psd_equal_count(FY, FZ, PSD, lambda_min_mm=0.20, n_bins=30):
    """
    Radial PSD average using equal-count radial bins.

    This avoids empty annuli and therefore avoids discontinuous radial PSD curves.
    """
    FR = np.sqrt(FY**2 + FZ**2)

    f = FR.ravel()
    p = PSD.ravel()

    mask = np.isfinite(f) & np.isfinite(p) & (f > 0)

    # Remove wavelengths below lambda_min:
    # lambda >= lambda_min -> f <= 1/lambda_min
    if lambda_min_mm is not None:
        fmax_allowed = 1.0 / lambda_min_mm
        mask &= f <= fmax_allowed

    f = f[mask]
    p = p[mask]

    if len(f) == 0:
        return np.array([]), np.array([]), np.nan, np.nan

    order = np.argsort(f)
    f = f[order]
    p = p[order]

    # Do not use more bins than useful samples
    n_bins = min(n_bins, max(5, len(f) // 4))

    chunks = np.array_split(np.arange(len(f)), n_bins)

    f_radial = []
    psd_radial = []

    for idx in chunks:
        if len(idx) == 0:
            continue

        f_radial.append(np.mean(f[idx]))
        psd_radial.append(np.mean(p[idx]))

    f_radial = np.array(f_radial)
    psd_radial = np.array(psd_radial)

    valid = np.isfinite(f_radial) & np.isfinite(psd_radial)

    f_radial = f_radial[valid]
    psd_radial = psd_radial[valid]

    if len(f_radial) == 0:
        return np.array([]), np.array([]), np.nan, np.nan

    peak_idx = np.argmax(psd_radial)

    f_char = f_radial[peak_idx]
    psd_char = psd_radial[peak_idx]
    lambda_char = 1.0 / f_char

    return f_radial, psd_radial, lambda_char, psd_char


def masked_autocorrelation_2d(A, dy_mm, dz_mm):
    """
    Mask-aware normalized autocorrelation.
    """
    A = np.asarray(A, dtype=float)
    valid = np.isfinite(A)

    if np.count_nonzero(valid) < 4:
        raise ValueError("Not enough finite points for autocorrelation.")

    A0 = np.zeros_like(A)
    A0[valid] = A[valid] - np.mean(A[valid])

    M = valid.astype(float)

    FA = np.fft.fft2(A0)
    FM = np.fft.fft2(M)

    ac_num = np.fft.ifft2(FA * np.conj(FA)).real
    ac_den = np.fft.ifft2(FM * np.conj(FM)).real

    with np.errstate(divide="ignore", invalid="ignore"):
        ac = ac_num / ac_den

    ac = np.fft.fftshift(ac)

    ny, nz = A.shape
    cy, cz = ny // 2, nz // 2

    if not np.isfinite(ac[cy, cz]) or ac[cy, cz] == 0:
        raise ValueError("Invalid autocorrelation center.")

    ac = ac / ac[cy, cz]

    y_lags = (np.arange(ny) - cy) * dy_mm
    z_lags = (np.arange(nz) - cz) * dz_mm

    return y_lags, z_lags, ac


def first_one_over_e_length(lags, values):
    """
    Returns:
        length, crossed
    If no 1/e crossing occurs, returns max positive lag and crossed=False.
    """
    lags = np.asarray(lags, dtype=float)
    values = np.asarray(values, dtype=float)

    center = np.argmin(np.abs(lags))

    x = lags[center:]
    y = values[center:]

    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, False

    target = np.exp(-1)

    for k in range(1, len(x)):
        if y[k] <= target:
            x0, x1 = x[k - 1], x[k]
            y0, y1 = y[k - 1], y[k]

            if y1 == y0:
                return x1, True

            x_cross = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            return x_cross, True

    return x[-1], False


def radial_autocorrelation(y_lags, z_lags, ac, n_bins=30):
    ZL, YL = np.meshgrid(z_lags, y_lags, indexing="xy")
    R = np.sqrt(YL**2 + ZL**2)

    r = R.ravel()
    a = ac.ravel()

    mask = np.isfinite(r) & np.isfinite(a)

    r = r[mask]
    a = a[mask]

    order = np.argsort(r)
    r = r[order]
    a = a[order]

    n_bins = min(n_bins, max(5, len(r) // 4))
    chunks = np.array_split(np.arange(len(r)), n_bins)

    r_radial = []
    ac_radial = []

    for idx in chunks:
        if len(idx) == 0:
            continue
        r_radial.append(np.mean(r[idx]))
        ac_radial.append(np.mean(a[idx]))

    r_radial = np.array(r_radial)
    ac_radial = np.array(ac_radial)

    corr_r, crossed_r = first_one_over_e_length(r_radial, ac_radial)

    return r_radial, ac_radial, corr_r, crossed_r


results = []

for i, df in enumerate(datasets):
    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]
    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Y, Z, U, V, S = df_to_grids(df)

    Yc = Y[ys, xs]
    Zc = Z[ys, xs]
    Uc = U[ys, xs]
    Vc = V[ys, xs]
    Sc = S[ys, xs]

    eps_zz, eps_yy, gamma_yz = strain_fields(Uc, Vc, dy_mm, dz_mm)

    invalid = Sc == -1

    eps_zz = np.where(invalid, np.nan, eps_zz)
    eps_yy = np.where(invalid, np.nan, eps_yy)
    gamma_yz = np.where(invalid, np.nan, gamma_yz)

    strain_data = [
        ("z strain", eps_zz),
        ("y strain", eps_yy),
        ("shear strain", gamma_yz),
    ]

    for field_name, A in strain_data:
        var_A = np.nanvar(A)

        FY, FZ, PSD = psd_2d_hann(A, dy_mm, dz_mm)

        f_radial, psd_radial, lambda_char, psd_char = radial_average_psd_equal_count(
            FY,
            FZ,
            PSD,
            lambda_min_mm=lambda_min_mm,
            n_bins=25,
        )

        y_lags, z_lags, ac = masked_autocorrelation_2d(A, dy_mm, dz_mm)

        cy = np.argmin(np.abs(y_lags))
        cz = np.argmin(np.abs(z_lags))

        ac_y = ac[:, cz]
        ac_z = ac[cy, :]

        corr_y, corr_y_crossed = first_one_over_e_length(y_lags, ac_y)
        corr_z, corr_z_crossed = first_one_over_e_length(z_lags, ac_z)

        r_ac, ac_radial, corr_r, corr_r_crossed = radial_autocorrelation(
            y_lags,
            z_lags,
            ac,
            n_bins=25,
        )

        results.append(
            {
                "dataset": i,
                "field": field_name,
                "variance": var_A,
                "characteristic_wavelength_mm": lambda_char,
                "corr_y_mm": corr_y,
                "corr_y_crossed_1e": corr_y_crossed,
                "corr_z_mm": corr_z,
                "corr_z_crossed_1e": corr_z_crossed,
                "corr_radial_mm": corr_r,
                "corr_radial_crossed_1e": corr_r_crossed,
            }
        )

        fig, axes = plt.subplots(1, 4, dpi=120)
        fig.set_size_inches(fig.get_size_inches() * np.array([2.8, 1.15]), forward=True)

        ax0, ax1, ax2, ax3 = axes

        # 1. Strain field
        vabs = np.nanmax(np.abs(A))
        if not np.isfinite(vabs) or vabs == 0:
            vabs = 1.0

        strain_norm = mpl.colors.TwoSlopeNorm(
            vmin=-vabs,
            vcenter=0.0,
            vmax=vabs,
        )

        strain_cmap = plt.get_cmap("coolwarm").copy()
        strain_cmap.set_bad("lightgray")

        m0 = ax0.pcolormesh(
            Yc - Yc.min(),
            Zc - Zc.min(),
            A,
            cmap=strain_cmap,
            # norm=strain_norm,
            shading="auto",
        )

        ax0.set_aspect("equal", adjustable="box")
        ax0.set_xlabel("y [mm]")
        ax0.set_ylabel("z [mm]")
        ax0.set_title(f"var={var_A:.3e}")
        fig.colorbar(m0, ax=ax0, label=field_name)

        # 2. 2D PSD
        PSD_plot = np.where(PSD > 0, PSD, np.nan)

        psd_vmin = np.nanpercentile(PSD_plot, 5)
        psd_vmax = np.nanpercentile(PSD_plot, 99)

        if not np.isfinite(psd_vmin) or psd_vmin <= 0:
            psd_vmin = np.nanmin(PSD_plot[PSD_plot > 0])

        if not np.isfinite(psd_vmax) or psd_vmax <= psd_vmin:
            psd_vmax = np.nanmax(PSD_plot)

        m1 = ax1.pcolormesh(
            FZ,
            FY,
            PSD_plot,
            shading="auto",
            cmap="magma",
            norm=mpl.colors.LogNorm(vmin=psd_vmin, vmax=psd_vmax),
        )

        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlabel("fz [cycles/mm]")
        ax1.set_ylabel("fy [cycles/mm]")
        ax1.set_title("2D PSD")
        fig.colorbar(m1, ax=ax1, label="PSD")

        # 3. Radial PSD
        valid_radial = (
            np.isfinite(f_radial) & np.isfinite(psd_radial) & (psd_radial > 0)
        )

        f_plot = f_radial[valid_radial]
        p_plot = psd_radial[valid_radial]

        ax2.plot(f_plot, p_plot, "k-o", ms=3, lw=1.2)
        ax2.set_yscale("log")
        ax2.set_xlabel("frequency [cycles/mm]")
        ax2.set_ylabel("radial PSD")
        ax2.set_title("radial PSD")

        if np.isfinite(lambda_char):
            f_char = 1.0 / lambda_char

            ax2.axvline(f_char, color="r", ls="--", lw=1)
            ax2.plot(
                f_char,
                psd_char,
                "ro",
                ms=5,
                zorder=5,
            )

            ax2.text(
                0.98,
                0.95,
                f"lambda={lambda_char:.3g} mm",
                transform=ax2.transAxes,
                ha="right",
                va="top",
            )

        # 4. Radial autocorrelation
        valid_ac = np.isfinite(r_ac) & np.isfinite(ac_radial)

        ax3.plot(r_ac[valid_ac], ac_radial[valid_ac], "k-o", ms=3, lw=1.2)
        ax3.axhline(np.exp(-1), color="r", ls="--", lw=1)

        if np.isfinite(corr_r):
            ax3.axvline(corr_r, color="r", ls="--", lw=1)

        ax3.set_xlabel("lag r [mm]")
        ax3.set_ylabel("autocorrelation")
        ax3.set_ylim(-0.2, 1.05)
        ax3.set_title("radial autocorr")

        ax3.text(
            0.98,
            0.95,
            f"Lr={corr_r:.3g} mm" + ("" if corr_r_crossed else " max"),
            transform=ax3.transAxes,
            ha="right",
            va="top",
        )

        fig.suptitle(f"D{i} {field_name}", y=1.02)
        fig.tight_layout()
        plt.show()

results = pd.DataFrame(results)
results

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Crops: (sy_lo, sy_hi, sx_lo, sx_hi)
crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]

dy_mm = 0.25
dz_mm = 0.20

# Nyquist minimum wavelength = 2*dz.
# "twice the Nyquist lower limit" => minimum wavelength = 2*(2*dz) = 4*dz.
lambda_min_mm = 4.0 * dz_mm


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_grids(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())  # x_c = z
    y_vals = np.sort(df_grid["y_c"].unique())  # y_c = y

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz)  # y displacement
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    y = np.arange(ny) * dy_mm
    z = np.arange(nz) * dz_mm
    Y, Z = np.meshgrid(y, z, indexing="ij")

    return Y, Z, U, V, S


def axial_strain(U, dy_mm, dz_mm):
    # U is z displacement, shape = (ny, nz)
    # epsilon_zz = dU/dz
    dU_dy, dU_dz = np.gradient(U, dy_mm, dz_mm, edge_order=2)
    return dU_dz


def interpolate_nan_1d(x, a):
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)

    valid = np.isfinite(a)

    if np.count_nonzero(valid) < 2:
        return None

    out = a.copy()
    out[~valid] = np.interp(x[~valid], x[valid], a[valid])

    return out


def remove_mean_and_linear_trend_rows(A, z):
    """
    1. Remove global mean axial strain.
    2. Remove best-fit linear axial trend along z for each transverse row.
    """
    A = np.asarray(A, dtype=float).copy()
    z = np.asarray(z, dtype=float)

    # Remove mean axial strain over whole cropped field
    mean_axial_strain = np.nanmean(A)
    A = A - mean_axial_strain

    A_detrended = np.full_like(A, np.nan, dtype=float)
    row_slopes = np.full(A.shape[0], np.nan)

    for j in range(A.shape[0]):
        row = A[j, :]
        valid = np.isfinite(row)

        if np.count_nonzero(valid) < 2:
            continue

        # Fit row strain as a function of z:
        # eps_zz(z) = slope*z + intercept
        slope, intercept = np.polyfit(z[valid], row[valid], 1)
        trend = slope * z + intercept

        A_detrended[j, valid] = row[valid] - trend[valid]
        row_slopes[j] = slope

    return A_detrended, mean_axial_strain, row_slopes


def row_averaged_1d_psd_z(A, dz_mm, lambda_min_mm):
    """
    Calculate 1D PSD along loading direction z for every transverse row,
    then average the row spectra.

    A shape: (ny, nz)
    axis 1 is z.
    """
    A = np.asarray(A, dtype=float)
    ny, nz = A.shape

    z = np.arange(nz) * dz_mm

    freqs = np.fft.rfftfreq(nz, d=dz_mm)  # cycles/mm
    window = np.hanning(nz)
    window_power = np.sum(window**2)

    row_psds = []

    for j in range(ny):
        row = A[j, :]

        row_filled = interpolate_nan_1d(z, row)

        if row_filled is None:
            continue

        # Remove any remaining row mean after interpolation/detrending
        row_filled = row_filled - np.mean(row_filled)

        row_win = row_filled * window

        F = np.fft.rfft(row_win)

        psd = (np.abs(F) ** 2) / window_power

        row_psds.append(psd)

    if len(row_psds) == 0:
        raise ValueError("No valid transverse rows for PSD.")

    row_psds = np.vstack(row_psds)

    psd_mean = np.nanmean(row_psds, axis=0)
    psd_std = np.nanstd(row_psds, axis=0)

    # Exclude zero frequency and wavelengths shorter than lambda_min_mm
    # lambda >= lambda_min_mm -> f <= 1/lambda_min_mm
    f_max = 1.0 / lambda_min_mm

    valid_peak = (
        np.isfinite(freqs) & np.isfinite(psd_mean) & (freqs > 0) & (freqs <= f_max)
    )

    if not np.any(valid_peak):
        f_char = np.nan
        lambda_char = np.nan
        psd_char = np.nan
    else:
        idx_local = np.argmax(psd_mean[valid_peak])
        f_char = freqs[valid_peak][idx_local]
        psd_char = psd_mean[valid_peak][idx_local]
        lambda_char = 1.0 / f_char

    return freqs, psd_mean, psd_std, row_psds, f_char, lambda_char, psd_char


results = []

for i, df in enumerate(datasets):
    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]
    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Y, Z, U, V, S = df_to_grids(df)

    Yc = Y[ys, xs]
    Zc = Z[ys, xs]
    Uc = U[ys, xs]
    Sc = S[ys, xs]

    Ezz = axial_strain(Uc, dy_mm, dz_mm)

    # Mask invalid DIC points
    Ezz = np.where(Sc == -1, np.nan, Ezz)

    # Local z coordinates starting at zero
    z = Zc[0, :] - Zc[0, 0]

    # Remove mean axial strain and row-wise linear axial trend
    Ezz_resid, mean_Ezz, row_slopes = remove_mean_and_linear_trend_rows(Ezz, z)

    # Row-averaged 1D PSD along z
    freqs, psd_mean, psd_std, row_psds, f_char, lambda_char, psd_char = (
        row_averaged_1d_psd_z(
            Ezz_resid,
            dz_mm=dz_mm,
            lambda_min_mm=lambda_min_mm,
        )
    )

    results.append(
        {
            "dataset": i,
            "mean_axial_strain_removed": mean_Ezz,
            "mean_abs_row_linear_slope_removed": np.nanmean(np.abs(row_slopes)),
            "lambda_min_mm": lambda_min_mm,
            "f_max_cycles_per_mm": 1.0 / lambda_min_mm,
            "characteristic_frequency_cycles_per_mm": f_char,
            "characteristic_wavelength_mm": lambda_char,
            "psd_at_characteristic": psd_char,
            "n_rows_used": row_psds.shape[0],
        }
    )

    valid_plot = (freqs > 0) & np.isfinite(psd_mean) & (psd_mean > 0)

    fig, ax = plt.subplots(dpi=120)

    # Individual row spectra in light gray
    for row_psd in row_psds:
        ax.plot(
            freqs[valid_plot],
            row_psd[valid_plot],
            color="0.75",
            lw=0.7,
            alpha=0.5,
        )

    # Mean row spectrum
    ax.plot(
        freqs[valid_plot],
        psd_mean[valid_plot],
        color="k",
        lw=2.0,
        label="row-averaged PSD",
    )

    # Characteristic wavelength marker
    if np.isfinite(f_char):
        ax.axvline(f_char, color="r", ls="--", lw=1.2)
        ax.plot(f_char, psd_char, "ro", ms=5)
        ax.text(
            0.98,
            0.95,
            f"lambda = {lambda_char:.3g} mm",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="r",
        )

    # Cutoff from twice Nyquist lower wavelength
    ax.axvline(
        1.0 / lambda_min_mm,
        color="b",
        ls=":",
        lw=1.0,
        label=f"lambda_min = {lambda_min_mm:.3g} mm",
    )

    ax.set_yscale("log")
    ax.set_xlabel("z frequency [cycles/mm]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Dataset {i}: axial strain residual PSD")
    ax.legend()

    fig.set_size_inches(fig.get_size_inches() * 1.25, forward=True)

    plt.show()

results = pd.DataFrame(results)
results

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]

dy_mm = 0.25
dz_mm = 0.20

# Nyquist minimum wavelength = 2*dz.
# Twice that lower wavelength limit:
lambda_min_mm = 2 * (2 * dz_mm)  # = 0.8 mm
f_max = 1.0 / lambda_min_mm  # = 1.25 cycles/mm


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_grids(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz)  # y displacement
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    y = np.arange(ny) * dy_mm
    z = np.arange(nz) * dz_mm
    Y, Z = np.meshgrid(y, z, indexing="ij")

    return Y, Z, U, V, S


def axial_strain(U, dy_mm, dz_mm):
    dU_dy, dU_dz = np.gradient(U, dy_mm, dz_mm, edge_order=2)
    return dU_dz


def interpolate_nan_1d(x, a):
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)

    valid = np.isfinite(a)

    if np.count_nonzero(valid) < 2:
        return None

    out = a.copy()
    out[~valid] = np.interp(x[~valid], x[valid], a[valid])

    return out


def remove_mean_and_linear_trend_rows(A, z):
    A = np.asarray(A, dtype=float).copy()
    z = np.asarray(z, dtype=float)

    mean_axial_strain = np.nanmean(A)
    A = A - mean_axial_strain

    A_detrended = np.full_like(A, np.nan, dtype=float)
    row_slopes = np.full(A.shape[0], np.nan)

    for j in range(A.shape[0]):
        row = A[j, :]
        valid = np.isfinite(row)

        if np.count_nonzero(valid) < 2:
            continue

        slope, intercept = np.polyfit(z[valid], row[valid], 1)
        trend = slope * z + intercept

        A_detrended[j, valid] = row[valid] - trend[valid]
        row_slopes[j] = slope

    return A_detrended, mean_axial_strain, row_slopes


def row_averaged_1d_psd_z(A, dz_mm, lambda_min_mm):
    A = np.asarray(A, dtype=float)

    ny, nz = A.shape
    z = np.arange(nz) * dz_mm

    freqs = np.fft.rfftfreq(nz, d=dz_mm)

    window = np.hanning(nz)
    window_power = np.sum(window**2)

    row_psds = []

    for j in range(ny):
        row = A[j, :]

        row_filled = interpolate_nan_1d(z, row)

        if row_filled is None:
            continue

        row_filled = row_filled - np.mean(row_filled)

        F = np.fft.rfft(row_filled * window)

        psd = np.abs(F) ** 2 / window_power

        row_psds.append(psd)

    if len(row_psds) == 0:
        raise ValueError("No valid transverse rows for PSD.")

    row_psds = np.vstack(row_psds)

    psd_mean = np.nanmean(row_psds, axis=0)
    psd_std = np.nanstd(row_psds, axis=0)

    f_max = 1.0 / lambda_min_mm

    allowed = (
        np.isfinite(freqs) & np.isfinite(psd_mean) & (freqs > 0) & (freqs <= f_max)
    )

    if not np.any(allowed):
        f_char = np.nan
        lambda_char = np.nan
        psd_char = np.nan
    else:
        idx = np.argmax(psd_mean[allowed])
        f_allowed = freqs[allowed]
        psd_allowed = psd_mean[allowed]

        f_char = f_allowed[idx]
        psd_char = psd_allowed[idx]
        lambda_char = 1.0 / f_char

    return freqs, psd_mean, psd_std, row_psds, allowed, f_char, lambda_char, psd_char


results = []

for i, df in enumerate(datasets):
    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]
    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Y, Z, U, V, S = df_to_grids(df)

    Uc = U[ys, xs]
    Sc = S[ys, xs]
    Zc = Z[ys, xs]

    Ezz = axial_strain(Uc, dy_mm, dz_mm)

    Ezz = np.where(Sc == -1, np.nan, Ezz)

    z = Zc[0, :] - Zc[0, 0]

    Ezz_resid, mean_Ezz, row_slopes = remove_mean_and_linear_trend_rows(Ezz, z)

    freqs, psd_mean, psd_std, row_psds, allowed, f_char, lambda_char, psd_char = (
        row_averaged_1d_psd_z(
            Ezz_resid,
            dz_mm=dz_mm,
            lambda_min_mm=lambda_min_mm,
        )
    )

    results.append(
        {
            "dataset": i,
            "mean_axial_strain_removed": mean_Ezz,
            "mean_abs_row_linear_slope_removed": np.nanmean(np.abs(row_slopes)),
            "lambda_min_mm": lambda_min_mm,
            "f_max_cycles_per_mm": f_max,
            "characteristic_frequency_cycles_per_mm": f_char,
            "characteristic_wavelength_mm": lambda_char,
            "psd_at_characteristic": psd_char,
            "n_rows_used": row_psds.shape[0],
        }
    )

    fig, ax = plt.subplots(dpi=120)

    # Plot ONLY allowed frequencies
    for row_psd in row_psds:
        ax.plot(
            freqs[allowed],
            row_psd[allowed],
            color="0.75",
            lw=0.7,
            alpha=0.5,
        )

    ax.plot(
        freqs[allowed],
        psd_mean[allowed],
        color="k",
        lw=2.0,
        label="mean row PSD",
    )

    if np.isfinite(f_char):
        ax.axvline(f_char, color="r", ls="--", lw=1.2)
        ax.plot(f_char, psd_char, "ro", ms=5)

        ax.text(
            0.98,
            0.95,
            f"lambda = {lambda_char:.3g} mm",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="r",
        )

    ax.set_yscale("log")
    ax.set_xlim(0, f_max)
    ax.set_xlabel("z frequency [cycles/mm]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Dataset {i}: axial strain residual PSD")
    ax.legend()

    fig.set_size_inches(fig.get_size_inches() * 1.25, forward=True)

    plt.show()

results = pd.DataFrame(results)
results

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]

dy_mm = 0.25
dz_mm = 0.20

# Nyquist shortest wavelength = 2*dz.
# Twice the Nyquist lower wavelength limit:
lambda_min_mm = 2 * (2 * dz_mm)  # 0.8 mm
f_max = 1.0 / lambda_min_mm  # 1.25 cycles/mm


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_grids(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz)  # y displacement
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    y = np.arange(ny) * dy_mm
    z = np.arange(nz) * dz_mm
    Y, Z = np.meshgrid(y, z, indexing="ij")

    return Y, Z, U, V, S


def axial_strain(U, dy_mm, dz_mm):
    _, dU_dz = np.gradient(U, dy_mm, dz_mm, edge_order=2)
    return dU_dz


def interp_nan_row(z, row):
    valid = np.isfinite(row)

    if np.count_nonzero(valid) < 2:
        return None

    out = row.copy()
    out[~valid] = np.interp(z[~valid], z[valid], row[valid])
    return out


def remove_mean_and_row_linear_trend(Ezz, z):
    E = np.asarray(Ezz, dtype=float).copy()

    mean_removed = np.nanmean(E)
    E = E - mean_removed

    E_resid = np.full_like(E, np.nan, dtype=float)
    row_slopes = np.full(E.shape[0], np.nan)

    for j in range(E.shape[0]):
        row = E[j, :]
        valid = np.isfinite(row)

        if np.count_nonzero(valid) < 2:
            continue

        slope, intercept = np.polyfit(z[valid], row[valid], 1)
        trend = slope * z + intercept

        E_resid[j, valid] = row[valid] - trend[valid]
        row_slopes[j] = slope

    return E_resid, mean_removed, row_slopes


def frequency_limited_row_psd(E, z, f_max):
    """
    Directly evaluates the Hann-windowed 1D PSD only at allowed frequencies.

    No frequencies above f_max are computed.
    """
    E = np.asarray(E, dtype=float)
    z = np.asarray(z, dtype=float)

    ny, nz = E.shape
    dz = z[1] - z[0]

    # DFT frequency spacing for this row length
    df = 1.0 / (nz * dz)

    # Allowed positive frequencies only
    freqs = np.arange(df, f_max + 0.5 * df, df)

    window = np.hanning(nz)
    window_power = np.sum(window**2)

    # Direct Fourier matrix only for allowed frequencies
    # shape: (n_allowed_freqs, nz)
    Fmat = np.exp(-2j * np.pi * freqs[:, None] * z[None, :])

    row_psds = []

    for j in range(ny):
        row = E[j, :]

        row_filled = interp_nan_row(z, row)

        if row_filled is None:
            continue

        # Remove remaining row mean before PSD
        row_filled = row_filled  # - np.mean(row_filled)

        row_win = row_filled * window

        amps = Fmat @ row_win

        psd = np.abs(amps) ** 2 / window_power

        row_psds.append(psd)

    if len(row_psds) == 0:
        raise ValueError("No valid rows for PSD.")

    row_psds = np.vstack(row_psds)

    psd_mean = np.mean(row_psds, axis=0)
    psd_std = np.std(row_psds, axis=0)

    peak_idx = np.argmax(psd_mean)

    f_char = freqs[peak_idx]
    lambda_char = 1.0 / f_char
    psd_char = psd_mean[peak_idx]

    return freqs, psd_mean, psd_std, row_psds, f_char, lambda_char, psd_char


results = []

for i, df in enumerate(datasets):
    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]

    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Y, Z, U, V, S = df_to_grids(df)

    Uc = U[ys, xs]
    Sc = S[ys, xs]
    Zc = Z[ys, xs]

    Ezz = axial_strain(Uc, dy_mm, dz_mm)

    # Mask invalid DIC points
    Ezz = np.where(Sc == -1, np.nan, Ezz)

    # Local z coordinate starting at 0
    z = Zc[0, :] - Zc[0, 0]

    # Remove mean axial strain and best-fit row-wise linear axial trend
    Ezz_resid, mean_Ezz_removed, row_slopes_removed = remove_mean_and_row_linear_trend(
        Ezz,
        z,
    )

    # PSD calculated only over allowed frequencies
    freqs, psd_mean, psd_std, row_psds, f_char, lambda_char, psd_char = (
        frequency_limited_row_psd(
            Ezz_resid,
            z,
            f_max=f_max,
        )
    )

    results.append(
        {
            "dataset": i,
            "mean_axial_strain_removed": mean_Ezz_removed,
            "mean_abs_row_linear_slope_removed": np.nanmean(np.abs(row_slopes_removed)),
            "lambda_min_mm": lambda_min_mm,
            "f_max_cycles_per_mm": f_max,
            "characteristic_frequency_cycles_per_mm": f_char,
            "characteristic_wavelength_mm": lambda_char,
            "psd_at_characteristic": psd_char,
            "n_rows_used": row_psds.shape[0],
        }
    )

    fig, ax = plt.subplots(dpi=120)

    for row_psd in row_psds:
        ax.plot(
            freqs,
            row_psd,
            color="0.75",
            lw=0.7,
            alpha=0.45,
        )

    ax.plot(
        freqs,
        psd_mean,
        color="k",
        lw=2.0,
        label="row-averaged PSD",
    )

    ax.axvline(f_char, color="r", ls="--", lw=1.2)
    ax.plot(f_char, psd_char, "ro", ms=5)

    ax.text(
        0.98,
        0.95,
        f"lambda = {lambda_char:.3g} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="r",
    )

    ax.set_yscale("log")
    ax.set_xlabel("z frequency [cycles/mm]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Dataset {i}: axial strain residual PSD")
    ax.legend()

    fig.set_size_inches(fig.get_size_inches() * 1.25, forward=True)

    plt.show()

results = pd.DataFrame(results)
results

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]

dy_mm = 0.25
dz_mm = 0.20

# Nyquist shortest wavelength = 2*dz.
# Twice the Nyquist lower wavelength limit:
lambda_min_mm = 2 * (2 * dz_mm)  # 0.8 mm
f_max = 1.0 / lambda_min_mm  # 1.25 cycles/mm


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_grids(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz)  # y displacement
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    y = np.arange(ny) * dy_mm
    z = np.arange(nz) * dz_mm
    Y, Z = np.meshgrid(y, z, indexing="ij")

    return Y, Z, U, V, S


def axial_strain(U, dy_mm, dz_mm):
    _, dU_dz = np.gradient(U, dy_mm, dz_mm, edge_order=2)
    return dU_dz


def interp_nan_row(z, row):
    valid = np.isfinite(row)

    if np.count_nonzero(valid) < 2:
        return None

    out = row.copy()
    out[~valid] = np.interp(z[~valid], z[valid], row[valid])
    return out


def remove_global_mean(Ezz):
    E = np.asarray(Ezz, dtype=float).copy()
    mean_removed = np.nanmean(E)
    E_resid = E - mean_removed
    return E_resid, mean_removed


def frequency_limited_row_psd(E, z, f_max):
    """
    Directly evaluates the Hann-windowed 1D PSD only at allowed frequencies.

    No frequencies above f_max are computed.
    """
    E = np.asarray(E, dtype=float)
    z = np.asarray(z, dtype=float)

    ny, nz = E.shape
    dz = z[1] - z[0]

    # DFT frequency spacing for this row length
    df = 1.0 / (nz * dz)

    # Allowed positive frequencies only
    freqs = np.arange(df, f_max + 0.5 * df, df)

    window = np.hanning(nz)
    window_power = np.sum(window**2)

    # Direct Fourier matrix only for allowed frequencies
    Fmat = np.exp(-2j * np.pi * freqs[:, None] * z[None, :])

    row_psds = []

    for j in range(ny):
        row = E[j, :]

        row_filled = interp_nan_row(z, row)

        if row_filled is None:
            continue

        # Remove remaining row mean before PSD
        # This removes the DC component only, not a linear trend.
        row_filled = row_filled - np.mean(row_filled)

        row_win = row_filled * window

        amps = Fmat @ row_win

        psd = np.abs(amps) ** 2 / window_power

        row_psds.append(psd)

    if len(row_psds) == 0:
        raise ValueError("No valid rows for PSD.")

    row_psds = np.vstack(row_psds)

    psd_mean = np.mean(row_psds, axis=0)
    psd_std = np.std(row_psds, axis=0)

    peak_idx = np.argmax(psd_mean)

    f_char = freqs[peak_idx]
    lambda_char = 1.0 / f_char
    psd_char = psd_mean[peak_idx]

    return freqs, psd_mean, psd_std, row_psds, f_char, lambda_char, psd_char


results_no_linear_detrend = []

for i, df in enumerate(datasets):
    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]

    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Y, Z, U, V, S = df_to_grids(df)

    Uc = U[ys, xs]
    Sc = S[ys, xs]
    Zc = Z[ys, xs]

    Ezz = axial_strain(Uc, dy_mm, dz_mm)

    # Mask invalid DIC points
    Ezz = np.where(Sc == -1, np.nan, Ezz)

    # Local z coordinate starting at 0
    z = Zc[0, :] - Zc[0, 0]

    # Remove only the global mean axial strain
    # No best-fit linear trend is removed.
    Ezz_resid, mean_Ezz_removed = remove_global_mean(Ezz)

    # PSD calculated only over allowed frequencies
    freqs, psd_mean, psd_std, row_psds, f_char, lambda_char, psd_char = (
        frequency_limited_row_psd(
            Ezz_resid,
            z,
            f_max=f_max,
        )
    )

    results_no_linear_detrend.append(
        {
            "dataset": i,
            "mean_axial_strain_removed": mean_Ezz_removed,
            "lambda_min_mm": lambda_min_mm,
            "f_max_cycles_per_mm": f_max,
            "characteristic_frequency_cycles_per_mm": f_char,
            "characteristic_wavelength_mm": lambda_char,
            "psd_at_characteristic": psd_char,
            "n_rows_used": row_psds.shape[0],
        }
    )

    fig, ax = plt.subplots(dpi=120)

    for row_psd in row_psds:
        ax.plot(
            freqs,
            row_psd,
            color="0.75",
            lw=0.7,
            alpha=0.45,
        )

    ax.plot(
        freqs,
        psd_mean,
        color="k",
        lw=2.0,
        label="row-averaged PSD",
    )

    ax.axvline(f_char, color="r", ls="--", lw=1.2)
    ax.plot(f_char, psd_char, "ro", ms=5)

    ax.text(
        0.98,
        0.95,
        f"lambda = {lambda_char:.3g} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="r",
    )

    ax.set_yscale("log")
    ax.set_xlabel("z frequency [cycles/mm]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Dataset {i}: axial strain residual PSD, no linear detrend")
    ax.legend()

    fig.set_size_inches(fig.get_size_inches() * 1.25, forward=True)

    plt.show()

results_no_linear_detrend = pd.DataFrame(results_no_linear_detrend)
results_no_linear_detrend

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]

dy_mm = 0.25
dz_mm = 0.20

# Configurable wavelength band, mm
lambda_min_mm = 0.8  # shortest wavelength included
lambda_max_mm = 4.5  # longest wavelength included

# Convert wavelength limits to frequency band:
# lambda = 1/f
# lambda_min -> f_max
# lambda_max -> f_min
f_min = 1.0 / lambda_max_mm
f_max = 1.0 / lambda_min_mm


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_grids(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz)  # y displacement
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    y = np.arange(ny) * dy_mm
    z = np.arange(nz) * dz_mm
    Y, Z = np.meshgrid(y, z, indexing="ij")

    return Y, Z, U, V, S


def axial_strain(U, dy_mm, dz_mm):
    _, dU_dz = np.gradient(U, dy_mm, dz_mm, edge_order=2)
    return dU_dz


def interp_nan_row(z, row):
    valid = np.isfinite(row)

    if np.count_nonzero(valid) < 2:
        return None

    out = row.copy()
    out[~valid] = np.interp(z[~valid], z[valid], row[valid])
    return out


def remove_global_mean(Ezz):
    E = np.asarray(Ezz, dtype=float).copy()
    mean_removed = np.nanmean(E)
    E_resid = E - mean_removed
    return E_resid, mean_removed


def frequency_limited_row_psd(E, z, f_min, f_max):
    """
    Directly evaluates the Hann-windowed 1D PSD only over the requested
    frequency band corresponding to lambda_min <= lambda <= lambda_max.

    No frequencies outside [f_min, f_max] are computed.
    """
    E = np.asarray(E, dtype=float)
    z = np.asarray(z, dtype=float)

    ny, nz = E.shape
    dz = z[1] - z[0]

    df = 1.0 / (nz * dz)

    # Allowed frequencies only
    freqs = np.arange(f_min, f_max + 0.5 * df, df)

    # Remove any accidental non-positive frequencies
    freqs = freqs[freqs > 0]

    if len(freqs) == 0:
        raise ValueError(
            f"No frequencies in requested band: f_min={f_min}, f_max={f_max}"
        )

    window = np.hanning(nz)
    window_power = np.sum(window**2)

    # Direct Fourier matrix only for requested frequencies
    Fmat = np.exp(-2j * np.pi * freqs[:, None] * z[None, :])

    row_psds = []

    for j in range(ny):
        row = E[j, :]

        row_filled = interp_nan_row(z, row)

        if row_filled is None:
            continue

        # Remove row DC only, not linear trend
        row_filled = row_filled - np.mean(row_filled)

        row_win = row_filled * window

        amps = Fmat @ row_win
        psd = np.abs(amps) ** 2 / window_power

        row_psds.append(psd)

    if len(row_psds) == 0:
        raise ValueError("No valid rows for PSD.")

    row_psds = np.vstack(row_psds)

    psd_mean = np.mean(row_psds, axis=0)
    psd_std = np.std(row_psds, axis=0)

    peak_idx = np.argmax(psd_mean)

    f_char = freqs[peak_idx]
    lambda_char = 1.0 / f_char
    psd_char = psd_mean[peak_idx]

    return freqs, psd_mean, psd_std, row_psds, f_char, lambda_char, psd_char


results_psd_band = []

for i, df in enumerate(datasets):
    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]

    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Y, Z, U, V, S = df_to_grids(df)

    Uc = U[ys, xs]
    Sc = S[ys, xs]
    Zc = Z[ys, xs]

    Ezz = axial_strain(Uc, dy_mm, dz_mm)

    # Mask invalid DIC points
    Ezz = np.where(Sc == -1, np.nan, Ezz)

    # Local z coordinate starting at zero
    z = Zc[0, :] - Zc[0, 0]

    # Remove only the global mean axial strain
    Ezz_resid, mean_Ezz_removed = remove_global_mean(Ezz)

    # PSD over configurable wavelength band only
    freqs, psd_mean, psd_std, row_psds, f_char, lambda_char, psd_char = (
        frequency_limited_row_psd(
            Ezz_resid,
            z,
            f_min=f_min,
            f_max=f_max,
        )
    )

    results_psd_band.append(
        {
            "dataset": i,
            "mean_axial_strain_removed": mean_Ezz_removed,
            "lambda_min_mm": lambda_min_mm,
            "lambda_max_mm": lambda_max_mm,
            "f_min_cycles_per_mm": f_min,
            "f_max_cycles_per_mm": f_max,
            "characteristic_frequency_cycles_per_mm": f_char,
            "characteristic_wavelength_mm": lambda_char,
            "psd_at_characteristic": psd_char,
            "n_rows_used": row_psds.shape[0],
        }
    )

    fig, ax = plt.subplots(dpi=120)

    for row_psd in row_psds:
        ax.plot(
            freqs,
            row_psd,
            color="0.75",
            lw=0.7,
            alpha=0.45,
        )

    ax.plot(
        freqs,
        psd_mean,
        color="k",
        lw=2.0,
        label="row-averaged PSD",
    )

    ax.axvline(f_char, color="r", ls="--", lw=1.2)
    ax.plot(f_char, psd_char, "ro", ms=5)

    ax.text(
        0.98,
        0.95,
        f"lambda = {lambda_char:.3g} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="r",
    )

    ax.set_yscale("log")
    ax.set_xlabel("z frequency [cycles/mm]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Dataset {i}: axial strain PSD")
    ax.legend()

    fig.set_size_inches(fig.get_size_inches() * 1.25, forward=True)

    plt.show()

results_psd_band = pd.DataFrame(results_psd_band)
results_psd_band

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]

dy_mm = 0.25
dz_mm = 0.20


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_grids(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz)  # y displacement
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    y = np.arange(ny) * dy_mm
    z = np.arange(nz) * dz_mm
    Y, Z = np.meshgrid(y, z, indexing="ij")

    return Y, Z, U, V, S


def axial_strain(U, dy_mm, dz_mm):
    # epsilon_zz = du_z / dz
    _, dU_dz = np.gradient(U, dy_mm, dz_mm, edge_order=2)
    return dU_dz


results_residual_rms = []

for i, df in enumerate(datasets):
    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]

    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Y, Z, U, V, S = df_to_grids(df)

    Yc = Y[ys, xs]
    Zc = Z[ys, xs]
    Uc = U[ys, xs]
    Sc = S[ys, xs]

    # Local plot coordinates starting at zero
    Yc0 = Yc - np.nanmin(Yc)
    Zc0 = Zc - np.nanmin(Zc)
    z = Zc0[0, :]

    # Axial strain field
    Ezz = axial_strain(Uc, dy_mm, dz_mm)

    # Mask invalid DIC points
    Ezz = np.where(Sc == -1, np.nan, Ezz)

    # Macro axial strain component
    macro_Ezz = np.nanmean(Ezz)

    # Macro-removed residual strain field
    Ezz_resid = Ezz - macro_Ezz

    # Axial profile: average across width y
    Ezz_profile = np.nanmean(Ezz, axis=0)

    # Profile with macro component removed
    Ezz_profile_resid = Ezz_profile - macro_Ezz

    # Best-fit linear trend in the axial profile after macro removal
    valid_profile = np.isfinite(z) & np.isfinite(Ezz_profile_resid)

    if np.count_nonzero(valid_profile) < 2:
        raise ValueError(f"Dataset {i}: not enough valid profile points for trend fit.")

    slope, intercept = np.polyfit(
        z[valid_profile],
        Ezz_profile_resid[valid_profile],
        1,
    )

    profile_trend = slope * z + intercept

    # Detrended axial profile
    Ezz_profile_detrended = Ezz_profile_resid - profile_trend

    # Detrended residual field: remove same axial trend from every transverse row
    Ezz_resid_detrended = Ezz_resid - profile_trend[None, :]

    # RMS residual strains
    rms_resid = np.sqrt(np.nanmean(Ezz_resid**2))
    rms_resid_detrended = np.sqrt(np.nanmean(Ezz_resid_detrended**2))

    results_residual_rms.append(
        {
            "dataset": i,
            "macro_Ezz": macro_Ezz,
            "profile_linear_slope_removed": slope,
            "profile_linear_intercept_removed": intercept,
            "rms_residual_macro_removed": rms_resid,
            "rms_residual_macro_and_trend_removed": rms_resid_detrended,
        }
    )

    fig, axes = plt.subplots(1, 2, dpi=120)
    fig.set_size_inches(fig.get_size_inches() * np.array([1.8, 1.1]), forward=True)

    ax0, ax1 = axes

    # Plot residual strain field with macro component removed
    vabs = np.nanmax(np.abs(Ezz_resid))
    if not np.isfinite(vabs) or vabs == 0:
        vabs = 1.0

    norm = mpl.colors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("lightgray")

    m0 = ax0.pcolormesh(
        Yc0,
        Zc0,
        Ezz_resid,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("y [mm]")
    ax0.set_ylabel("z [mm]")
    ax0.set_title(f"RMS={rms_resid:.3e}")
    fig.colorbar(m0, ax=ax0, label=r"$\epsilon_{zz} - \langle \epsilon_{zz} \rangle$")

    # Plot axial profile before and after detrending
    ax1.plot(
        z,
        Ezz_profile_resid,
        "k-o",
        lw=1.5,
        ms=3,
        label="macro removed",
    )

    ax1.plot(
        z,
        profile_trend,
        "r--",
        lw=1.2,
        label="best-fit trend",
    )

    ax1.plot(
        z,
        Ezz_profile_detrended,
        "b-o",
        lw=1.5,
        ms=3,
        label="detrended",
    )

    ax1.axhline(0, color="0.5", lw=0.8)

    ax1.set_xlabel("z [mm]")
    ax1.set_ylabel(r"$\bar{\epsilon}_{zz}(z)$")
    ax1.set_title(f"RMS detrended={rms_resid_detrended:.3e}")
    ax1.legend(fontsize=8)

    fig.suptitle(f"Dataset {i}", y=1.02)
    fig.tight_layout()

    plt.show()

results_residual_rms = pd.DataFrame(results_residual_rms)
results_residual_rms

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]

dy_mm = 0.25
dz_mm = 0.20

# Configurable wavelength band for PSD
lambda_min_mm = 0.4
lambda_max_mm = 4.0

f_min = 1.0 / lambda_max_mm
f_max = 1.0 / lambda_min_mm

# If True, each row has its own mean removed before PSD.
# The field-level mean is always removed first.
remove_row_mean_before_psd = False


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_grids(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz)  # y displacement
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    y = np.arange(ny) * dy_mm
    z = np.arange(nz) * dz_mm
    Y, Z = np.meshgrid(y, z, indexing="ij")

    return Y, Z, U, V, S


def axial_strain(U, dy_mm, dz_mm):
    _, dU_dz = np.gradient(U, dy_mm, dz_mm, edge_order=2)
    return dU_dz


def interp_nan_row(z, row):
    valid = np.isfinite(row)

    if np.count_nonzero(valid) < 2:
        return None

    out = row.copy()
    out[~valid] = np.interp(z[~valid], z[valid], row[valid])

    return out


def frequency_limited_row_psd(E, z, f_min, f_max, remove_row_mean=False):
    """
    Directly evaluates Hann-windowed 1D PSD along z only over [f_min, f_max].
    No frequencies outside this band are computed.

    E shape: (ny, nz)
    """
    E = np.asarray(E, dtype=float)
    z = np.asarray(z, dtype=float)

    ny, nz = E.shape
    dz = z[1] - z[0]

    df = 1.0 / (nz * dz)

    freqs = np.arange(f_min, f_max + 0.5 * df, df)
    freqs = freqs[freqs > 0]

    if len(freqs) == 0:
        raise ValueError(
            f"No frequencies in requested band: f_min={f_min}, f_max={f_max}"
        )

    window = np.hanning(nz)
    window_power = np.sum(window**2)

    Fmat = np.exp(-2j * np.pi * freqs[:, None] * z[None, :])

    row_psds = []
    rows_used = []

    for j in range(ny):
        row = E[j, :]

        row_filled = interp_nan_row(z, row)

        if row_filled is None:
            continue

        if remove_row_mean:
            row_filled = row_filled - np.mean(row_filled)

        row_win = row_filled * window

        amps = Fmat @ row_win

        psd = np.abs(amps) ** 2 / window_power

        row_psds.append(psd)
        rows_used.append(j)

    if len(row_psds) == 0:
        raise ValueError("No valid rows for PSD after masking positive values.")

    row_psds = np.vstack(row_psds)
    rows_used = np.array(rows_used)

    psd_mean = np.mean(row_psds, axis=0)
    psd_std = np.std(row_psds, axis=0)

    peak_idx = np.argmax(psd_mean)

    f_char = freqs[peak_idx]
    lambda_char = 1.0 / f_char
    psd_char = psd_mean[peak_idx]

    return freqs, psd_mean, psd_std, row_psds, rows_used, f_char, lambda_char, psd_char


results_positive_ezz_psd = []

for i, df in enumerate(datasets):
    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]

    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Y, Z, U, V, S = df_to_grids(df)

    Uc = U[ys, xs]
    Sc = S[ys, xs]
    Zc = Z[ys, xs]

    # Calculate z strain
    Ezz = axial_strain(Uc, dy_mm, dz_mm)

    # Mask invalid DIC points
    Ezz = np.where(Sc == -1, np.nan, Ezz)

    # Remove mean z strain
    mean_Ezz = np.nanmean(Ezz)
    Ezz_mean_removed = Ezz - mean_Ezz

    # Only use mean-removed z strain values above zero
    Ezz_positive = np.where(Ezz_mean_removed > 0, Ezz_mean_removed, np.nan)

    # Local z coordinate
    z = Zc[0, :] - Zc[0, 0]

    freqs, psd_mean, psd_std, row_psds, rows_used, f_char, lambda_char, psd_char = (
        frequency_limited_row_psd(
            Ezz_positive,
            z,
            f_min=f_min,
            f_max=f_max,
            remove_row_mean=remove_row_mean_before_psd,
        )
    )

    results_positive_ezz_psd.append(
        {
            "dataset": i,
            "mean_Ezz_removed": mean_Ezz,
            "lambda_min_mm": lambda_min_mm,
            "lambda_max_mm": lambda_max_mm,
            "f_min_cycles_per_mm": f_min,
            "f_max_cycles_per_mm": f_max,
            "characteristic_frequency_cycles_per_mm": f_char,
            "characteristic_wavelength_mm": lambda_char,
            "psd_at_characteristic": psd_char,
            "n_rows_used": len(rows_used),
            "positive_fraction": np.count_nonzero(np.isfinite(Ezz_positive))
            / Ezz_positive.size,
        }
    )

    fig, ax = plt.subplots(dpi=120)

    for row_psd in row_psds:
        ax.plot(
            freqs,
            row_psd,
            color="0.75",
            lw=0.7,
            alpha=0.45,
        )

    ax.plot(
        freqs,
        psd_mean,
        color="k",
        lw=2.0,
        label="mean row PSD",
    )

    ax.axvline(f_char, color="r", ls="--", lw=1.2)
    ax.plot(f_char, psd_char, "ro", ms=5)

    ax.text(
        0.98,
        0.95,
        f"lambda = {lambda_char:.3g} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="r",
    )

    ax.set_yscale("log")
    ax.set_xlabel("z frequency [cycles/mm]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Dataset {i}: PSD of positive mean-removed z strain")
    ax.legend()

    fig.set_size_inches(fig.get_size_inches() * 1.25, forward=True)

    plt.show()

results_positive_ezz_psd = pd.DataFrame(results_positive_ezz_psd)
results_positive_ezz_psd

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

crops = [
    (6, 6, 9, 5),  # dataset 0
    (7, 5, 6, 6),  # dataset 1
    (9, 5, 7, 11),  # dataset 2
    (7, 3, 7, 7),  # dataset 3
]

dy_mm = 0.25
dz_mm = 0.20

# Exclude first/last this many mm in z after the existing crop
z_exclude_mm = 2.0

# PSD wavelength band
# Shortest meaningful wavelength for dz=0.2 mm is Nyquist = 2*dz = 0.4 mm.
# Use 2x Nyquist lower limit unless you want to change it.
lambda_min_mm = 4.0 * dz_mm  # 0.8 mm
lambda_max_mm = None  # None means use maximum allowed by remaining z length

remove_field_mean = True  # remove mean z strain before PSD
remove_row_mean = True  # remove each row's remaining DC before PSD


def crop_slice(lo, hi):
    return slice(lo, None if hi == 0 else -hi)


def df_to_grids(df):
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(f"Not a complete grid: len(df)={len(df_grid)}, ny*nz={ny*nz}")

    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)  # z displacement
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz)  # y displacement
    S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)

    y = np.arange(ny) * dy_mm
    z = np.arange(nz) * dz_mm

    Y, Z = np.meshgrid(y, z, indexing="ij")

    return Y, Z, U, V, S


def axial_strain(U, dy_mm, dz_mm):
    _, dU_dz = np.gradient(U, dy_mm, dz_mm, edge_order=2)
    return dU_dz


def interp_nan_row(z, row):
    valid = np.isfinite(row)

    if np.count_nonzero(valid) < 2:
        return None

    out = row.copy()
    out[~valid] = np.interp(z[~valid], z[valid], row[valid])
    return out


def row_averaged_fft_psd_z(
    E, z, lambda_min_mm=None, lambda_max_mm=None, remove_row_mean=True
):
    """
    1D Hann-windowed PSD along z for each transverse row, then averaged.

    E shape: (ny, nz)
    z shape: (nz,)
    """
    E = np.asarray(E, dtype=float)
    z = np.asarray(z, dtype=float)

    ny, nz = E.shape

    if nz < 4:
        raise ValueError(f"Need at least 4 z points after trimming; got nz={nz}")

    dz = np.mean(np.diff(z))
    z_length = (nz - 1) * dz

    freqs = np.fft.rfftfreq(nz, d=dz)

    # Convert wavelength limits to frequency limits
    if lambda_min_mm is None:
        f_max = freqs.max()
    else:
        f_max = 1.0 / lambda_min_mm

    if lambda_max_mm is None:
        # Longest nonzero FFT wavelength is approximately total sampled length
        f_min = freqs[1]
    else:
        f_min = 1.0 / lambda_max_mm

    allowed = np.isfinite(freqs) & (freqs > 0) & (freqs >= f_min) & (freqs <= f_max)

    if not np.any(allowed):
        raise ValueError(
            f"No PSD frequencies in requested band. "
            f"nz={nz}, dz={dz}, f_min={f_min}, f_max={f_max}, "
            f"freqs={freqs}"
        )

    window = np.hanning(nz)
    window_power = np.sum(window**2)

    row_psds = []
    rows_used = []

    for j in range(ny):
        row = E[j, :]

        row_filled = interp_nan_row(z, row)

        if row_filled is None:
            continue

        if remove_row_mean:
            row_filled = row_filled - np.mean(row_filled)

        row_win = row_filled * window

        F = np.fft.rfft(row_win)

        psd = np.abs(F) ** 2 / window_power

        row_psds.append(psd)
        rows_used.append(j)

    if len(row_psds) == 0:
        raise ValueError("No valid rows for PSD.")

    row_psds = np.vstack(row_psds)
    rows_used = np.array(rows_used)

    psd_mean = np.mean(row_psds, axis=0)
    psd_std = np.std(row_psds, axis=0)

    freqs_allowed = freqs[allowed]
    psd_mean_allowed = psd_mean[allowed]
    psd_std_allowed = psd_std[allowed]
    row_psds_allowed = row_psds[:, allowed]

    peak_idx = np.argmax(psd_mean_allowed)

    f_char = freqs_allowed[peak_idx]
    lambda_char = 1.0 / f_char
    psd_char = psd_mean_allowed[peak_idx]

    return {
        "freqs": freqs_allowed,
        "psd_mean": psd_mean_allowed,
        "psd_std": psd_std_allowed,
        "row_psds": row_psds_allowed,
        "rows_used": rows_used,
        "f_char": f_char,
        "lambda_char": lambda_char,
        "psd_char": psd_char,
        "z_length_mm": z_length,
        "lambda_min_mm": lambda_min_mm,
        "lambda_max_mm": 1.0 / f_min,
    }


results_ztrim_psd = []

for i, df in enumerate(datasets):
    sy_lo, sy_hi, sx_lo, sx_hi = crops[i]

    ys = crop_slice(sy_lo, sy_hi)
    xs = crop_slice(sx_lo, sx_hi)

    Y, Z, U, V, S = df_to_grids(df)

    # Apply existing crop first
    Yc = Y[ys, xs]
    Zc = Z[ys, xs]
    Uc = U[ys, xs]
    Sc = S[ys, xs]

    # Compute axial strain on the cropped displacement field
    Ezz = axial_strain(Uc, dy_mm, dz_mm)

    # Mask invalid DIC points
    Ezz = np.where(Sc == -1, np.nan, Ezz)

    # Local z coordinate starting at zero
    z_full = Zc[0, :] - Zc[0, 0]

    # Exclude first/last 2 mm in z
    z_mask = (z_full >= z_exclude_mm) & (z_full <= z_full.max() - z_exclude_mm)

    if np.count_nonzero(z_mask) < 4:
        raise ValueError(
            f"Dataset {i}: z trim leaves too few points. "
            f"z range before trim = {z_full.min()} to {z_full.max()} mm, "
            f"points left = {np.count_nonzero(z_mask)}"
        )

    Ezz_trim = Ezz[:, z_mask]
    z_trim = z_full[z_mask]

    # Re-zero trimmed z coordinate
    z_trim = z_trim - z_trim.min()

    # Remove field mean axial strain
    if remove_field_mean:
        mean_Ezz_removed = np.nanmean(Ezz_trim)
        Ezz_for_psd = Ezz_trim - mean_Ezz_removed
    else:
        mean_Ezz_removed = 0.0
        Ezz_for_psd = Ezz_trim.copy()

    psd = row_averaged_fft_psd_z(
        Ezz_for_psd,
        z_trim,
        lambda_min_mm=lambda_min_mm,
        lambda_max_mm=lambda_max_mm,
        remove_row_mean=remove_row_mean,
    )

    results_ztrim_psd.append(
        {
            "dataset": i,
            "z_exclude_each_end_mm": z_exclude_mm,
            "z_length_used_mm": psd["z_length_mm"],
            "mean_Ezz_removed": mean_Ezz_removed,
            "lambda_min_mm": psd["lambda_min_mm"],
            "lambda_max_mm": psd["lambda_max_mm"],
            "characteristic_frequency_cycles_per_mm": psd["f_char"],
            "characteristic_wavelength_mm": psd["lambda_char"],
            "psd_at_characteristic": psd["psd_char"],
            "n_rows_used": len(psd["rows_used"]),
            "n_z_points_used": Ezz_for_psd.shape[1],
        }
    )

    fig, ax = plt.subplots(dpi=120)

    for row_psd in psd["row_psds"]:
        ax.plot(
            psd["freqs"],
            row_psd,
            color="0.75",
            lw=0.7,
            alpha=0.45,
        )

    ax.plot(
        psd["freqs"],
        psd["psd_mean"],
        color="k",
        lw=2.0,
        label="mean row PSD",
    )

    ax.axvline(psd["f_char"], color="r", ls="--", lw=1.2)
    ax.plot(psd["f_char"], psd["psd_char"], "ro", ms=5)

    ax.text(
        0.98,
        0.95,
        f"lambda = {psd['lambda_char']:.3g} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="r",
    )

    ax.set_yscale("log")
    ax.set_xlabel("z frequency [cycles/mm]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Dataset {i}: z strain PSD, first/last {z_exclude_mm:g} mm removed")
    ax.legend()

    fig.set_size_inches(fig.get_size_inches() * 1.25, forward=True)

    plt.show()

results_ztrim_psd = pd.DataFrame(results_ztrim_psd)
results_ztrim_psd

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.config import VOXELSIZE

path = "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro1_production/530mpa_unint"
micro = "/Users/gtdebru/mimosa/microstructures/production/micro1_production.dat"
sim = SimResults.load(path, microstructure=micro)

dz_um = VOXELSIZE
dz_mm = dz_um / 1000.0


def simulated_z_strain_profile(sim, time_index=-1, sample_index=None):
    """
    Calculate z strain profile from simulated height maps.

    sim.height shape:
        (n_faces, n_times, nz, n_width)

    Returns one axial strain profile per face unless sample_index is specified.

    Assumption:
        height is the displacement-like field being differentiated along z.
    """

    H = np.asarray(sim.height, dtype=float)

    if H.ndim != 4:
        raise ValueError(
            f"Expected sim.height shape (n_faces, n_times, nz, n_width), got {H.shape}"
        )

    n_faces, n_times, nz, n_width = H.shape

    z_mm = np.arange(nz) * dz_mm

    if sample_index is None:
        sample_indices = range(n_faces)
    else:
        sample_indices = [sample_index]

    rows = []

    profiles = {}

    for face_idx in sample_indices:
        face_name = (
            sim.samples[face_idx] if sim.samples is not None else f"face_{face_idx}"
        )

        h = H[face_idx, time_index, :, :]  # shape: (nz, n_width)

        # Differentiate along z axis
        eps_zz_field = np.gradient(h, dz_mm, axis=0)

        # Average across transverse width
        eps_zz_profile = np.nanmean(eps_zz_field, axis=1)

        # Remove macro/mean component
        macro_eps_zz = np.nanmean(eps_zz_profile)
        eps_zz_profile_resid = eps_zz_profile - macro_eps_zz

        # Optional best-fit linear trend of the profile
        valid = np.isfinite(z_mm) & np.isfinite(eps_zz_profile_resid)

        if np.count_nonzero(valid) >= 2:
            slope, intercept = np.polyfit(
                z_mm[valid],
                eps_zz_profile_resid[valid],
                1,
            )
            trend = slope * z_mm + intercept
            eps_zz_profile_detrended = eps_zz_profile_resid - trend
        else:
            slope = np.nan
            intercept = np.nan
            trend = np.full_like(z_mm, np.nan)
            eps_zz_profile_detrended = np.full_like(z_mm, np.nan)

        rms_resid = np.sqrt(np.nanmean(eps_zz_profile_resid**2))
        rms_detrended = np.sqrt(np.nanmean(eps_zz_profile_detrended**2))

        profiles[face_name] = {
            "z_mm": z_mm,
            "eps_zz_field": eps_zz_field,
            "eps_zz_profile": eps_zz_profile,
            "macro_eps_zz": macro_eps_zz,
            "eps_zz_profile_resid": eps_zz_profile_resid,
            "trend": trend,
            "eps_zz_profile_detrended": eps_zz_profile_detrended,
            "rms_resid": rms_resid,
            "rms_detrended": rms_detrended,
            "trend_slope": slope,
            "trend_intercept": intercept,
        }

        rows.append(
            {
                "face": face_name,
                "time_index": time_index,
                "time": (
                    sim.vtk_time[time_index] if sim.vtk_time is not None else np.nan
                ),
                "macro_eps_zz": macro_eps_zz,
                "rms_residual_macro_removed": rms_resid,
                "rms_residual_macro_and_trend_removed": rms_detrended,
                "profile_linear_slope_removed": slope,
                "profile_linear_intercept_removed": intercept,
            }
        )

    summary = pd.DataFrame(rows)

    return profiles, summary


profiles, sim_z_strain_summary = simulated_z_strain_profile(
    sim,
    time_index=-1,
    sample_index=None,
)

sim_z_strain_summary

# %%
from pathlib import Path
from collections import Counter
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

# If DATA_DIR and PROFILOMETRY_SPACING_UM already exist in your notebook, this uses them.
# Otherwise, set DATA_DIR manually here.
try:
    DATA_DIR
except NameError:
    DATA_DIR = Path("/Users/gtdebru/mimosa/data")  # <-- edit if needed

try:
    PROFILOMETRY_SPACING_UM
except NameError:
    PROFILOMETRY_SPACING_UM = {
        "10x": 1.379951,
        "50x": None,
    }
DATA_DIR = Path("/Users/gtdebru/mimosa/data")
sample_type = "unint"
polish = "polished"
load = 530
mag = "10x"

exclude = []

crop = (slice(50, -50), slice(50, 750))
level = True
detrend_order = 1
skip_bad_files = True

# -1 = final time, 0 = initial time, None = plot all times
time_index = 0

print(DATA_DIR)

# "map":  Sa(z) = mean_y |h(z,y) - mean_all(h)|
# "line": Sa(z) = mean_y |h(z,y) - mean_y(h(z,y))|
center_mode = "map"

# ---------------------------------------------------------------------
# STANDALONE HEIGHT MAP LOADING
# ---------------------------------------------------------------------


@lru_cache(maxsize=256)
def _detrend_geometry(shape, spacing_um, order):
    row, column = np.indices(shape, dtype=np.float64)
    x = row.ravel() * spacing_um
    y = column.ravel() * spacing_um

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    return design, np.linalg.pinv(design)


def detrend_surface(values, spacing_um, order=1):
    design, inverse = _detrend_geometry(values.shape, float(spacing_um), order)
    coefficients = inverse @ values.ravel()
    return values - (design @ coefficients).reshape(values.shape)


def raw_height(path):
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(path, *, level=False, spacing_um=1.379951, detrend_order=1):
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um, order=detrend_order)

    return height


def _file_sort_key(path):
    try:
        return 0, float(path.stem), str(path)
    except ValueError:
        return 1, np.inf, str(path)


def load_height_maps_standalone(
    *,
    sample_type,
    polish,
    load,
    mag,
    exclude=(),
    crop=(slice(50, -50), slice(50, 750)),
    level=True,
    detrend_order=1,
    skip_bad_files=True,
):
    spacing_um = PROFILOMETRY_SPACING_UM[mag]
    path = (
        Path(DATA_DIR) / f"creep_{sample_type}_{polish}_{load}" / "profilometry" / mag
    )

    if not path.exists():
        raise FileNotFoundError(f"Profilometry path does not exist: {path}")

    # Recursively find CSVs, but we will explicitly ignore files in any "bad" folder.
    files = sorted(path.rglob("*.csv"), key=_file_sort_key)

    exclude = set(exclude)
    records = []

    for file in files:
        rel_parts = file.relative_to(path).parts

        # Expected normal structure:
        #   path / sample / time.csv
        #
        # Bad-data structure:
        #   path / sample / bad / time.csv
        #
        # Skip anything inside a folder named "bad".
        if "bad" in rel_parts:
            continue

        # Sample is the first directory under mag, not file.parent.stem.
        # Example:
        #   32a/0.csv       -> sample = 32a
        #   32a/bad/5.csv   -> skipped above
        if len(rel_parts) < 2:
            if skip_bad_files:
                warnings.warn(f"Skipping CSV not inside a sample folder: {file}")
                continue
            raise ValueError(f"CSV not inside a sample folder: {file}")

        sample = rel_parts[0]

        if sample in exclude:
            continue

        try:
            time_value = float(file.stem)
        except ValueError as exc:
            if skip_bad_files:
                warnings.warn(f"Skipping non-numeric time file {file}: {exc}")
                continue
            raise

        try:
            height = read_height(
                file,
                level=level,
                spacing_um=spacing_um,
                detrend_order=detrend_order,
            )

            height = height[crop]

            if not np.all(np.isfinite(height)):
                raise ValueError("Height map contains NaN or infinite values.")

            records.append((sample, time_value, height, file))

        except Exception as exc:
            if skip_bad_files:
                warnings.warn(f"Skipping bad profilometry file {file}: {exc}")
                continue
            raise

    if not records:
        raise ValueError("No valid height maps were loaded.")

    shape_counts = Counter(height.shape for _, _, height, _ in records)
    reference_shape, reference_count = shape_counts.most_common(1)[0]

    if len(shape_counts) > 1:
        msg = (
            f"Detected multiple cropped shapes: {dict(shape_counts)}. "
            f"Using most common shape {reference_shape}."
        )

        if skip_bad_files:
            warnings.warn(msg)
            records = [r for r in records if r[2].shape == reference_shape]
        else:
            raise ValueError(msg)

    samples = np.array(sorted({sample for sample, _, _, _ in records}))
    times = np.array(
        sorted({time_value for _, time_value, _, _ in records}), dtype=float
    )

    sample_to_index = {sample: idx for idx, sample in enumerate(samples)}
    time_to_index = {time_value: idx for idx, time_value in enumerate(times)}

    ntimes = len(times)
    nsamples = len(samples)
    nz, ny = reference_shape

    height_array = np.full((ntimes, nsamples, nz, ny), np.nan, dtype=float)
    availability_mask = np.zeros((ntimes, nsamples), dtype=bool)

    seen = set()

    for sample, time_value, height, file in records:
        ti = time_to_index[time_value]
        si = sample_to_index[sample]

        if (ti, si) in seen:
            if skip_bad_files:
                warnings.warn(
                    f"Duplicate scan for sample={sample}, time={time_value:g}; "
                    f"keeping first. File: {file}"
                )
                continue
            raise ValueError(f"Duplicate scan: sample={sample}, time={time_value:g}")

        height_array[ti, si] = height
        availability_mask[ti, si] = True
        seen.add((ti, si))

    return height_array, times, samples, availability_mask, spacing_um


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

height_array, times, samples, availability_mask, spacing_um = (
    load_height_maps_standalone(
        sample_type=sample_type,
        polish=polish,
        load=load,
        mag=mag,
        exclude=exclude,
        crop=crop,
        level=level,
        detrend_order=detrend_order,
        skip_bad_files=skip_bad_files,
    )
)

print("Loaded height array shape:", height_array.shape)
print("Samples:", samples)
print("Times:", times)
print("Spacing um:", spacing_um)

# height_array shape: (ntimes, nsamples, nz, ny)
nt, ns, nz, ny = height_array.shape
z_um = np.arange(nz) * spacing_um

# ---------------------------------------------------------------------
# CALCULATE z MEAN Sa PROFILE
# ---------------------------------------------------------------------

specimen_sa_profile = np.full((nt, ns, nz), np.nan, dtype=float)

for ti in range(nt):
    for si in range(ns):
        if not availability_mask[ti, si]:
            continue

        H = height_array[ti, si]  # shape: (nz, ny)

        if center_mode == "map":
            H0 = H - np.nanmean(H)
        elif center_mode == "line":
            H0 = H - np.nanmean(H, axis=1, keepdims=True)
        else:
            raise ValueError("center_mode must be 'map' or 'line'.")

        specimen_sa_profile[ti, si, :] = np.nanmean(np.abs(H0), axis=1)

mean_sa_profile = np.nanmean(specimen_sa_profile, axis=1)
std_sa_profile = np.nanstd(specimen_sa_profile, axis=1)

# ---------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------

if time_index is None:
    fig, ax = plt.subplots(dpi=150)

    for ti in range(nt):
        ax.plot(
            z_um,
            mean_sa_profile[ti],
            lw=2,
            label=f"t={times[ti]:g}",
        )

    ax.set_xlabel("z [µm]")
    ax.set_ylabel("mean Sa [µm]")
    ax.set_title("mean experimental Sa profile")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.set_size_inches(fig.get_size_inches() * 1.25, forward=True)
    plt.show()

else:
    ti = time_index % nt

    fig, ax = plt.subplots(dpi=150)

    for si, sample in enumerate(samples):
        if not availability_mask[ti, si]:
            continue

        ax.plot(
            z_um,
            specimen_sa_profile[ti, si],
            color="0.7",
            lw=0.8,
            alpha=0.6,
        )

    ax.plot(
        z_um,
        mean_sa_profile[ti],
        color="k",
        lw=2.5,
        label="mean",
    )

    ax.fill_between(
        z_um,
        mean_sa_profile[ti] - std_sa_profile[ti],
        mean_sa_profile[ti] + std_sa_profile[ti],
        color="k",
        alpha=0.2,
        linewidth=0,
        label="±1 std",
    )

    ax.set_xlabel("z [µm]")
    ax.set_ylabel("Sa [µm]")
    ax.set_title(f"mean experimental Sa profile, t={times[ti]:g}")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.set_size_inches(fig.get_size_inches() * 1.25, forward=True)
    plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.data_utils import MicrostructureInfo, SimResults

time_index = -1
vtk_field_name = "vm_plastic_strain"
path = "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro1_production/530mpa_unint"
micro = "/Users/gtdebru/mimosa/microstructures/production/micro1_production.dat"

sim = SimResults.load(path, microstructure=micro)


def read_vtk_field_array(vtk_path, field_name):
    vtk_path = Path(vtk_path)

    with vtk_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = iter(f)

        for line in lines:
            parts = line.split()

            if len(parts) >= 4 and parts[0] == field_name:
                ncomp = int(parts[1])
                nvals = int(parts[2])
                total = ncomp * nvals

                data = []
                while len(data) < total:
                    data.extend(float(v) for v in next(lines).split())

                arr = np.asarray(data[:total], dtype=float)

                if ncomp == 1:
                    return arr.reshape(nvals)

                return arr.reshape(nvals, ncomp)

    return None


def profile_mean_std(profiles):
    profiles = np.asarray(profiles, dtype=float)

    mean = np.nanmean(profiles, axis=0)
    std = np.nanstd(profiles, axis=0, ddof=1)

    lower = mean - std
    upper = mean + std

    return mean, std, lower, upper


# ---------------------------------------------------------------------
# Surface roughness Sa(z)
# ---------------------------------------------------------------------
heights = np.transpose(sim.height, (1, 0, 2, 3))


n_faces, n_times, nz_height, n_width = heights.shape

z_sa_um = np.arange(nz_height) * VOXELSIZE

sa_profiles = []

for face_idx in range(n_faces):
    h = heights[face_idx, time_index, :, :]

    h_mean = np.nanmean(h)
    roughness = np.abs(h - h_mean)

    sa_profiles.append(np.nanmean(roughness, axis=1))

sa_profiles = np.asarray(sa_profiles)

mean_sa, std_sa, sa_lower, sa_upper = profile_mean_std(sa_profiles)

# ---------------------------------------------------------------------
# vm_plastic_strain(z)
# ---------------------------------------------------------------------
evpfft_dir = get_evpfft_dir(sim)

vtk_files = sorted(
    evpfft_dir.glob("microstr_cell_fields_stp_*.vtk"),
    key=lambda file: int(file.stem.rsplit("_", 1)[-1]),
)

vtk_file = vtk_files[time_index]

info = MicrostructureInfo.load(sim.microstructure)
ixmin, ixmax, iymin, iymax = info.face_indices()

vm_flat = read_vtk_field_array(vtk_file, vtk_field_name)

VM = vm_flat.reshape(info.nz, info.ny, info.nx)

vm_faces = [
    VM[:, iymin : iymax + 1, ixmin],
    VM[:, iymin : iymax + 1, ixmax],
    VM[:, iymin, ixmin : ixmax + 1],
    VM[:, iymax, ixmin : ixmax + 1],
]

vm_profiles = np.asarray([np.nanmean(face, axis=1) for face in vm_faces])

z_vm_um = (np.arange(info.nz) + 0.5) * VOXELSIZE

mean_vm, std_vm, vm_lower, vm_upper = profile_mean_std(vm_profiles)


# ---------------------------------------------------------------------
# Plot Sa(z)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(dpi=150)

ax.plot(z_sa_um[1:], mean_sa[1:], color="k", lw=2.5)
ax.fill_between(
    z_sa_um[1:], sa_lower[1:], sa_upper[1:], color="k", alpha=0.20, linewidth=0
)


ax.set_xlabel("z [µm]")
ax.set_ylabel(r"$S_{a}$ [µm]")
ax.grid(True, alpha=0.25)

fig.set_size_inches(fig.get_size_inches() * 1.25, forward=True)
plt.show()

# ---------------------------------------------------------------------
# Plot vm_plastic_strain(z)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(dpi=150)

ax.plot(z_vm_um, mean_vm * 100, color="k", lw=2.5)
ax.fill_between(
    z_vm_um, vm_lower * 100, vm_upper * 100, color="k", alpha=0.20, linewidth=0
)

ax.set_xlabel("z [µm]")
ax.set_ylabel(r"$\sigma_{zz}$ [%]")
ax.grid(True, alpha=0.25)

plt.show()

# %% [markdown]
# # ----------------------------------------------------------

# %%
from pathlib import Path
from collections import Counter
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# =============================================================================
# Settings
# =============================================================================

DATA_DIR = Path("/Users/gtdebru/mimosa/data")

polish = "polished"
mag = "10x"
spacing_um = 1.379951

experiments = [
    {"load": 475, "sample_type": "int"},
    {"load": 500, "sample_type": "unint"},
    {"load": 525, "sample_type": "int"},
    {"load": 530, "sample_type": "unint"},
    {"load": 575, "sample_type": "int"},
    {"load": 588, "sample_type": "unint"},
]

crop = (slice(50, -50), slice(50, 750))
level = True
detrend_order = 1
skip_bad_files = True

exclude_dirs_named = {"bad"}

# If True, use experiment-mean strain when sample-specific strain is missing.
use_experiment_mean_strain_fallback = True

# =============================================================================
# Height-map utilities
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape, spacing_um, order):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * spacing_um
    y = column.ravel() * spacing_um

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))

    design = np.column_stack(terms)
    return design, np.linalg.pinv(design)


def detrend_surface(values, spacing_um, order=1):
    design, inverse = _detrend_geometry(values.shape, float(spacing_um), order)
    coefficients = inverse @ values.ravel()
    return values - (design @ coefficients).reshape(values.shape)


def raw_height(path):
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(path, *, level=False, spacing_um=1.379951, detrend_order=1):
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um, order=detrend_order)

    return height


def _file_sort_key(path):
    try:
        return 0, float(path.stem), str(path)
    except ValueError:
        return 1, np.inf, str(path)


def map_sa(height):
    h0 = height - np.nanmean(height)
    return np.nanmean(np.abs(h0))


def load_height_records_for_experiment(
    *,
    sample_type,
    polish,
    load,
    mag,
    crop,
    spacing_um,
    level=True,
    detrend_order=1,
    skip_bad_files=True,
):
    path = DATA_DIR / f"creep_{sample_type}_{polish}_{load}" / "profilometry" / mag

    if not path.exists():
        print(f"Missing profilometry path: {path}")
        return []

    files = sorted(path.rglob("*.csv"), key=_file_sort_key)
    records = []

    for file in files:
        rel_parts = file.relative_to(path).parts

        # Skip files in bad subdirectories
        if any(part in exclude_dirs_named for part in rel_parts):
            continue

        # Expected:
        #   sample/time.csv
        if len(rel_parts) < 2:
            continue

        sample = str(rel_parts[0]).strip()

        try:
            time_value = float(file.stem)
        except ValueError:
            continue

        try:
            height = read_height(
                file,
                level=level,
                spacing_um=spacing_um,
                detrend_order=detrend_order,
            )

            height = height[crop]

            if not np.all(np.isfinite(height)):
                continue

            records.append(
                {
                    "sample": sample,
                    "time_h": time_value,
                    "height": height,
                    "file": file,
                }
            )

        except Exception as exc:
            if skip_bad_files:
                warnings.warn(f"Skipping bad profilometry file {file}: {exc}")
                continue
            else:
                print(f"Bad profilometry file: {file}")
                print(exc)

    if records:
        shape_counts = Counter(rec["height"].shape for rec in records)
        reference_shape, _ = shape_counts.most_common(1)[0]

        if len(shape_counts) > 1:
            warnings.warn(
                f"{load} {sample_type}: multiple cropped shapes found: "
                f"{dict(shape_counts)}. Keeping {reference_shape}."
            )
            records = [rec for rec in records if rec["height"].shape == reference_shape]

    return records


# =============================================================================
# Strain utilities
# =============================================================================


def load_strain_table(sample_type, polish, load):
    strain_path = DATA_DIR / f"creep_{sample_type}_{polish}_{load}" / "strain.csv"

    df = pd.read_csv(strain_path)
    df.columns = df.columns.str.strip()

    # Strip sample column names too
    df = df.rename(columns={col: str(col).strip() for col in df.columns})

    if "time_h" not in df.columns:
        print(f"No time_h column in {strain_path}")
        return pd.DataFrame()

    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")

    for col in df.columns:
        if col != "time_h":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def interp_series_at_time(time, values, query_time):
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(time) & np.isfinite(values)

    if np.count_nonzero(valid) == 0:
        return np.nan

    t = time[valid]
    v = values[valid]

    order = np.argsort(t)
    t = t[order]
    v = v[order]

    return np.interp(query_time, t, v)


def strain_at_time_for_sample_or_mean(strain_df, sample, time_h):

    strain_time = strain_df["time_h"].to_numpy(dtype=float)

    sample_cols = [c for c in strain_df.columns if c != "time_h"]

    # Exact sample-specific match
    if sample in sample_cols:
        val = interp_series_at_time(
            strain_time,
            strain_df[sample].to_numpy(dtype=float),
            time_h,
        )
        if np.isfinite(val):
            return val, "sample"

    # Case-insensitive fallback
    lower_map = {str(c).lower(): c for c in sample_cols}
    if sample.lower() in lower_map:
        col = lower_map[sample.lower()]
        val = interp_series_at_time(
            strain_time,
            strain_df[col].to_numpy(dtype=float),
            time_h,
        )
        if np.isfinite(val):
            return val, "sample_case_insensitive"

    # Experiment-mean fallback
    if use_experiment_mean_strain_fallback and len(sample_cols) > 0:
        strain_values = strain_df[sample_cols].to_numpy(dtype=float)
        mean_strain = np.nanmean(strain_values, axis=1)

        val = interp_series_at_time(strain_time, mean_strain, time_h)

        if np.isfinite(val):
            return val, "experiment_mean"

    return np.nan, "none"


# =============================================================================
# Build all-time scatter table
# =============================================================================

rows = []
summary_rows = []

for exp in experiments:
    load = exp["load"]
    sample_type = exp["sample_type"]

    height_records = load_height_records_for_experiment(
        sample_type=sample_type,
        polish=polish,
        load=load,
        mag=mag,
        crop=crop,
        spacing_um=spacing_um,
        level=level,
        detrend_order=detrend_order,
        skip_bad_files=skip_bad_files,
    )

    strain_df = load_strain_table(sample_type, polish, load)

    if len(height_records) == 0:
        summary_rows.append(
            {
                "load_mpa": load,
                "sample_type": sample_type,
                "n_height_records": 0,
                "n_height_samples": 0,
                "n_samples_with_at_least_2_times": 0,
                "n_strain_columns": (
                    0 if strain_df.empty else len(strain_df.columns) - 1
                ),
                "n_points_plotted": 0,
                "n_sample_strain_points": 0,
                "n_experiment_mean_fallback_points": 0,
            }
        )
        continue

    sa_records = []

    for rec in height_records:
        sa_records.append(
            {
                "sample": rec["sample"],
                "time_h": rec["time_h"],
                "sa_um": map_sa(rec["height"]),
            }
        )

    sa_df = pd.DataFrame(sa_records)

    n_height_records = len(sa_df)
    n_height_samples = sa_df["sample"].nunique()
    n_points_before = len(rows)

    samples_with_2_times = []
    strain_mode_counts = Counter()

    for sample, g in sa_df.groupby("sample"):
        g = g.sort_values("time_h")

        # Require at least two profilometry times
        if g["time_h"].nunique() < 2:
            continue

        samples_with_2_times.append(sample)

        sa_initial = g["sa_um"].iloc[0]
        t_initial = g["time_h"].iloc[0]

        for _, row in g.iterrows():
            time_h = float(row["time_h"])
            sa_um = float(row["sa_um"])

            bulk_z_strain, strain_source = strain_at_time_for_sample_or_mean(
                strain_df,
                sample,
                time_h,
            )

            if not np.isfinite(bulk_z_strain):
                continue

            strain_mode_counts[strain_source] += 1

            rows.append(
                {
                    "load_mpa": load,
                    "sample_type": sample_type,
                    "polish": polish,
                    "sample": sample,
                    "time_h": time_h,
                    "initial_time_h": t_initial,
                    "sa_initial_um": sa_initial,
                    "sa_um": sa_um,
                    "delta_sa_um": sa_um - sa_initial,
                    "bulk_z_strain": bulk_z_strain,
                    "bulk_z_strain_percent": 100.0 * bulk_z_strain,
                    "strain_source": strain_source,
                }
            )

    n_points_after = len(rows)
    n_points = n_points_after - n_points_before


scatter_df = pd.DataFrame(rows)


# =============================================================================
# Scatter plot
# =============================================================================

fig, ax = plt.subplots(dpi=150)

markers = {
    "int": "o",
    "unint": "s",
}

colors = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

for (sample_type, load), group in scatter_df.groupby(["sample_type", "load_mpa"]):
    ax.scatter(
        group["bulk_z_strain_percent"],
        group["delta_sa_um"],
        s=45,
        alpha=0.8,
        marker=markers.get(sample_type, "o"),
        color=colors.get(load, None),
        label=f"{load} MPa {sample_type}",
    )

ax.axhline(0, color="0.5", lw=0.8)

ax.set_xlabel("bulk axial z strain [%]")
ax.set_ylabel(r"$\Delta S_a$ [µm]")
ax.set_title(r"$\Delta S_a$ vs bulk axial z strain")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)

fig.set_size_inches(fig.get_size_inches() * 1.35, forward=True)
plt.show()

# %%
from pathlib import Path
import numpy as np
import pandas as pd

from utils.data_utils import SimResults

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
strain_col = "epav33"

micro_runs = [
    {
        "micro_id": "micro1",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro1_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro1_production.dat"
        ),
    },
    {
        "micro_id": "micro2",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro2_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro2_production.dat"
        ),
    },
    {
        "micro_id": "micro3",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro3_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro3_production.dat"
        ),
    },
]

load_cases = [
    {"load": 475, "sample_type": "int"},
    {"load": 500, "sample_type": "unint"},
    {"load": 525, "sample_type": "int"},
    {"load": 530, "sample_type": "unint"},
    {"load": 575, "sample_type": "int"},
    {"load": 588, "sample_type": "unint"},
]


def sim_sa_from_height_map(h):
    return np.nanmean(np.abs(h - np.nanmean(h)))


sim_rows = []

for micro_run in micro_runs:
    micro_id = micro_run["micro_id"]
    sim_root = micro_run["sim_root"]
    microstructure = micro_run["microstructure"]

    for case in load_cases:
        load = case["load"]
        sample_type = case["sample_type"]

        run_dir = sim_root / f"{load}mpa_{sample_type}"

        sim_i = SimResults.load(
            run_dir,
            microstructure=microstructure,
        )

        H = np.asarray(sim_i.height, dtype=float)  # (n_faces, n_times, nz, n_width)
        n_faces, n_times, nz, n_width = H.shape

        vtk_time = np.asarray(sim_i.vtk_time, dtype=float)

        strain_time = np.asarray(sim_i.sim_time, dtype=float)
        bulk_strain_all = np.asarray(getattr(sim_i, strain_col), dtype=float)

        valid = np.isfinite(strain_time) & np.isfinite(bulk_strain_all)

        strain_time_valid = strain_time[valid]
        bulk_strain_valid = bulk_strain_all[valid]

        order = np.argsort(strain_time_valid)
        strain_time_valid = strain_time_valid[order]
        bulk_strain_valid = bulk_strain_valid[order]

        bulk_strain_vtk = np.interp(
            vtk_time,
            strain_time_valid,
            bulk_strain_valid,
        )

        n = min(n_times, len(vtk_time), len(bulk_strain_vtk))

        H = H[:, :n, :, :]
        vtk_time = vtk_time[:n]
        bulk_strain_vtk = bulk_strain_vtk[:n]

        sa_faces_time = np.full((n_faces, n), np.nan, dtype=float)

        for face_idx in range(n_faces):
            for ti in range(n):
                sa_faces_time[face_idx, ti] = sim_sa_from_height_map(H[face_idx, ti])

        sa_mean_time = np.nanmean(sa_faces_time, axis=0)
        sa_std_time = np.nanstd(sa_faces_time, axis=0, ddof=1)

        delta_sa = sa_mean_time - sa_mean_time[0]

        for ti in range(n):
            sim_rows.append(
                {
                    "micro_id": micro_id,
                    "load_mpa": load,
                    "sample_type": sample_type,
                    "time": vtk_time[ti],
                    "time_h": vtk_time[ti] / 3600
                    "bulk_z_strain": bulk_strain_vtk[ti],
                    "bulk_z_strain_percent": 100.0 * bulk_strain_vtk[ti],
                    "sa_mean_um": sa_mean_time[ti],
                    "sa_std_um": sa_std_time[ti],
                    "delta_sa_um": delta_sa[ti],
                    "run_dir": str(run_dir),
                    "microstructure": str(microstructure),
                }
            )

sim_curve_df = pd.DataFrame(sim_rows)

# %%
colors = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

linestyles = {
    "int": "-",
    "unint": "--",
}

sim_strain = []
sim_delt_sa_bystrain = []
for (sample_type, load), group in sim_curve_df.groupby(["sample_type", "load_mpa"]):
    group = group.sort_values("bulk_z_strain_percent")
    sim_strain.extend(group["bulk_z_strain_percent"])
    sim_delt_sa_bystrain.extend(group["delta_sa_um"])

# %%
fig, ax = plt.subplots(dpi=150)


for (sample_type, load), group in sim_curve_df.groupby(["sample_type", "load_mpa"]):
    group = group.sort_values("bulk_z_strain_percent")

    ax.scatter(
        group["bulk_z_strain_percent"],
        group["delta_sa_um"],
        s=45,
        alpha=0.85,
        marker=markers.get(sample_type, "o"),
        color=colors.get(load, None),
        label=f"{load} MPa {sample_type}",
    )

ax.set_xlabel("bulk axial z strain [%]")
ax.set_ylabel(r"$\Delta S_a$ [µm]")
ax.set_title(r"simulation: $\Delta S_a$ vs bulk axial z strain")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)

fig.set_size_inches(fig.get_size_inches() * 1.35, forward=True)
plt.show()

# %%
xsim = []
ysim = []

xexp = []
yexp = []

for (sample_type, load), group in sim_curve_df.groupby(["sample_type", "load_mpa"]):

    group = group.sort_values("bulk_z_strain_percent")
    xsim.extend(group["bulk_z_strain_percent"])
    ysim.extend(group["delta_sa_um"])

for (sample_type, load), group in scatter_df.groupby(["sample_type", "load_mpa"]):

    xexp.extend(group["bulk_z_strain_percent"])
    yexp.extend(group["delta_sa_um"])

xsim = np.array(xsim)
ysim = np.array(ysim)
xexp = np.array(xexp)
yexp = np.array(yexp)


msim = np.sum(xsim * ysim) / np.sum(xsim**2)
mexp = np.sum(xexp * yexp) / np.sum(xexp**2)
scale = mexp / msim

# Predictions
ysim_hat = msim * xsim
yexp_hat = mexp * xexp

# Standard R^2 using total sum of squares about mean(y)
ss_res_sim = np.sum((ysim - ysim_hat) ** 2)
ss_tot_sim = np.sum((ysim - np.mean(ysim)) ** 2)
r2_sim = 1.0 - ss_res_sim / ss_tot_sim

ss_res_exp = np.sum((yexp - yexp_hat) ** 2)
ss_tot_exp = np.sum((yexp - np.mean(yexp)) ** 2)
r2_exp = 1.0 - ss_res_exp / ss_tot_exp

# Alternative zero-intercept R^2, sometimes used for no-intercept models:
# compares residual power to sum(y^2), not variance about mean.
r2_zero_sim = 1.0 - ss_res_sim / np.sum(ysim**2)
r2_zero_exp = 1.0 - ss_res_exp / np.sum(yexp**2)

print(f"Exp. slope: {mexp:.6f}")
print(f"Sim. slope: {msim:.6f}")
print(f"Exp/Sim Scale: {mexp/msim:.6f}")

print()
print(f"R^2 sim, standard        = {r2_sim:.6f}")
print(f"R^2 exp, standard        = {r2_exp:.6f}")

print()
print(f"R^2 sim, zero-intercept  = {r2_zero_sim:.6f}")
print(f"R^2 exp, zero-intercept  = {r2_zero_exp:.6f}")

# %%


# %%
fig, ax = plt.subplots(dpi=150)
xp = np.linspace(0, 19, 100)


ax.axhline(0, color="0.5", lw=0.8)
ax.scatter(xsim, ysim * scale, marker="o", color="tab:blue", label="Sim.")
ax.scatter(xexp, yexp, marker="s", color="tab:red", label="Exp.")

ax.plot(
    xp,
    msim * xp * scale,
    color="tab:blue",
)
ax.plot(xp, mexp * xp, color="tab:red", linestyle="--")
ax.set_xlabel("Strain [%]")
ax.set_ylabel(r"$\Delta S_a$ [µm]")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)

fig.set_size_inches(fig.get_size_inches() * 1.35, forward=True)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Top row: interrupted, increasing load
# Bottom row: uninterrupted, increasing load
plot_grid = [
    [(475, "int"), (525, "int"), (575, "int")],
    [(500, "unint"), (530, "unint"), (588, "unint")],
]

colors = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}


def sim_time_to_hours(t):
    t = np.asarray(t, dtype=float)
    if np.nanmax(t) > 1.0e4:
        return t / 3600.0
    return t


fig, axes = plt.subplots(2, 3, dpi=150, sharex=False, sharey=False)

for r in range(2):
    for c in range(3):
        ax = axes[r, c]

        load, sample_type = plot_grid[r][c]
        color = colors.get(load, "k")

        exp_g = scatter_df[
            (scatter_df["load_mpa"] == load)
            & (scatter_df["sample_type"] == sample_type)
        ].copy()

        sim_g = sim_curve_df[
            (sim_curve_df["load_mpa"] == load)
            & (sim_curve_df["sample_type"] == sample_type)
        ].copy()

        # Experimental mean ± std
        if len(exp_g) > 0:
            exp_summary = (
                exp_g.groupby("time_h", as_index=False)
                .agg(
                    mean_delta_sa_um=("delta_sa_um", "mean"),
                    std_delta_sa_um=("delta_sa_um", "std"),
                    n=("delta_sa_um", "count"),
                )
                .sort_values("time_h")
            )

            exp_summary["std_delta_sa_um"] = exp_summary["std_delta_sa_um"].fillna(0.0)

            if sample_type == "unint":
                # Only first and final experimental time points with error bars
                exp_plot = exp_summary.iloc[[0, -1]].drop_duplicates("time_h")

                ax.errorbar(
                    exp_plot["time_h"],
                    exp_plot["mean_delta_sa_um"],
                    yerr=exp_plot["std_delta_sa_um"],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=4,
                    markersize=5,
                    label="exp mean ± std",
                )

            else:
                # Interrupted tests: full mean ± std curve
                x = exp_summary["time_h"].to_numpy(dtype=float)
                y = exp_summary["mean_delta_sa_um"].to_numpy(dtype=float)
                s = exp_summary["std_delta_sa_um"].to_numpy(dtype=float)

                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=2.0,
                    label="exp mean",
                )

                ax.fill_between(
                    x,
                    y - s,
                    y + s,
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    label="exp ± std",
                )

        # Simulation points
        if len(sim_g) > 0:
            sim_time_h = sim_time_to_hours(sim_g["time"].to_numpy(dtype=float))

            ax.scatter(
                sim_time_h,
                sim_g["delta_sa_um"] * scale,
                s=55,
                alpha=0.9,
                color=color,
                marker="s",
                edgecolor="k",
                linewidth=0.5,
                label="sim",
            )

        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"{load} MPa {sample_type}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

for ax in axes[1, :]:
    ax.set_xlabel("time [h]")

for ax in axes[:, 0]:
    ax.set_ylabel(r"$\Delta S_a$ [µm]")

fig.suptitle(r"$\Delta S_a$ vs time", y=1.02)
fig.set_size_inches(fig.get_size_inches() * np.array([1.6, 1.35]), forward=True)
fig.tight_layout()

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Top row: interrupted, increasing load
# Bottom row: uninterrupted, increasing load
plot_grid = [
    [(475, "int"), (525, "int"), (575, "int")],
    [(500, "unint"), (530, "unint"), (588, "unint")],
]

colors = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

fig, axes = plt.subplots(2, 3, dpi=150, sharex=False, sharey=False)

for r in range(2):
    for c in range(3):
        ax = axes[r, c]

        load, sample_type = plot_grid[r][c]
        color = colors.get(load, "k")

        exp_g = scatter_df[
            (scatter_df["load_mpa"] == load)
            & (scatter_df["sample_type"] == sample_type)
        ].copy()

        sim_g = sim_curve_df[
            (sim_curve_df["load_mpa"] == load)
            & (sim_curve_df["sample_type"] == sample_type)
        ].copy()

        # Experimental mean ± std as function of bulk z strain
        if len(exp_g) > 0:
            exp_summary = (
                exp_g.groupby("time_h", as_index=False)
                .agg(
                    mean_delta_sa_um=("delta_sa_um", "mean"),
                    std_delta_sa_um=("delta_sa_um", "std"),
                    mean_bulk_z_strain_percent=("bulk_z_strain_percent", "mean"),
                    std_bulk_z_strain_percent=("bulk_z_strain_percent", "std"),
                    n=("delta_sa_um", "count"),
                )
                .sort_values("mean_bulk_z_strain_percent")
            )

            exp_summary["std_delta_sa_um"] = exp_summary["std_delta_sa_um"].fillna(0.0)

            if sample_type == "unint":
                # Only first and final experimental points with y error bars
                exp_plot = exp_summary.iloc[[0, -1]].drop_duplicates("time_h")

                ax.errorbar(
                    exp_plot["mean_bulk_z_strain_percent"],
                    exp_plot["mean_delta_sa_um"],
                    yerr=exp_plot["std_delta_sa_um"],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=4,
                    markersize=5,
                    label="exp mean ± std",
                )

            else:
                # Interrupted tests: full mean ± std curve
                x = exp_summary["mean_bulk_z_strain_percent"].to_numpy(dtype=float)
                y = exp_summary["mean_delta_sa_um"].to_numpy(dtype=float)
                s = exp_summary["std_delta_sa_um"].to_numpy(dtype=float)

                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=2.0,
                    label="exp mean",
                    marker="o",
                    markersize=4,
                )

                ax.fill_between(
                    x,
                    y - s,
                    y + s,
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    label="exp ± std",
                )

        # Simulation points
        if len(sim_g) > 0:
            ax.scatter(
                sim_g["bulk_z_strain_percent"],
                sim_g["delta_sa_um"] * scale,
                s=55,
                alpha=0.9,
                color=color,
                marker="s",
                edgecolor="k",
                linewidth=0.5,
                label="sim",
            )

        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"{load} MPa {sample_type}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

for ax in axes[1, :]:
    ax.set_xlabel("bulk axial z strain [%]")

for ax in axes[:, 0]:
    ax.set_ylabel(r"$\Delta S_a$ [µm]")

fig.suptitle(r"$\Delta S_a$ vs bulk axial z strain", y=1.02)
fig.set_size_inches(fig.get_size_inches() * np.array([1.6, 1.35]), forward=True)
fig.tight_layout()

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

exclude_loads = {475}
yield_load = 500.0  # MPa

xsim = []
ysim = []

xexp = []
yexp = []

# ---------------------------------------------------------------------
# Simulation: final point per microstructure/load/type
# ---------------------------------------------------------------------
sim_group_cols = ["sample_type", "load_mpa"]

if "micro_id" in sim_curve_df.columns:
    sim_group_cols = ["micro_id", "sample_type", "load_mpa"]

for group_key, group in sim_curve_df.groupby(sim_group_cols):
    if "micro_id" in sim_curve_df.columns:
        micro_id, sample_type, load = group_key
    else:
        sample_type, load = group_key

    if load in exclude_loads:
        continue

    group = group.sort_values("time")
    final = group.iloc[-1]

    xsim.append(load)
    ysim.append(final["delta_sa_um"])

# ---------------------------------------------------------------------
# Experiment: final time MEAN Delta Sa per load/type
# ---------------------------------------------------------------------
for (sample_type, load), group in scatter_df.groupby(["sample_type", "load_mpa"]):
    if load in exclude_loads:
        continue

    final_time = group["time_h"].max()
    final_group = group[group["time_h"] == final_time]

    xexp.append(load)
    yexp.append(final_group["delta_sa_um"].mean())

xsim = np.asarray(xsim, dtype=float)
ysim = np.asarray(ysim, dtype=float)

xexp = np.asarray(xexp, dtype=float)
yexp = np.asarray(yexp, dtype=float)

sim_mask = np.isfinite(xsim) & np.isfinite(ysim)
exp_mask = np.isfinite(xexp) & np.isfinite(yexp)

xsim = xsim[sim_mask]
ysim = ysim[sim_mask]

xexp = xexp[exp_mask]
yexp = yexp[exp_mask]

# ---------------------------------------------------------------------
# Simulation fit: ordinary linear fit WITH free intercept
#
#   y_sim = msim * load + bsim
# ---------------------------------------------------------------------
msim, bsim = np.polyfit(xsim, ysim, 1)

ysim_hat = msim * xsim + bsim

ss_res_sim = np.sum((ysim - ysim_hat) ** 2)
ss_tot_sim = np.sum((ysim - np.mean(ysim)) ** 2)
r2_sim = 1.0 - ss_res_sim / ss_tot_sim

# ---------------------------------------------------------------------
# Experimental fit: constrained to pass through (yield_load, 0)
#
#   y_exp = mexp * (load - yield_load)
# ---------------------------------------------------------------------
xexp_shift = xexp - yield_load

mexp = np.sum(xexp_shift * yexp) / np.sum(xexp_shift**2)
bexp = -mexp * yield_load

yexp_hat = mexp * xexp_shift

ss_res_exp = np.sum((yexp - yexp_hat) ** 2)
ss_tot_exp = np.sum((yexp - np.mean(yexp)) ** 2)
r2_exp = 1.0 - ss_res_exp / ss_tot_exp

# ---------------------------------------------------------------------
# Scale simulation data/fit to experimental slope scale
# Use slope ratio only.
# ---------------------------------------------------------------------
scale = mexp / msim

# ---------------------------------------------------------------------
# Load limits with 5% spacing
# ---------------------------------------------------------------------
load_min = min(np.nanmin(xsim), np.nanmin(xexp))
load_max = max(np.nanmax(xsim), np.nanmax(xexp))

load_span = load_max - load_min
pad = 0.05 * load_span

xlim = (load_min - pad, load_max + pad)
xp = np.linspace(load_min, load_max, 200)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(dpi=150)

ax.axhline(0, color="0.5", lw=0.8)
ax.axvline(yield_load, color="0.5", lw=0.8, ls=":")

ax.scatter(
    xsim,
    ysim * scale,
    marker="o",
    color="tab:blue",
    label="Sim.",
)

ax.scatter(
    xexp,
    yexp,
    marker="s",
    color="tab:red",
    label="Exp.",
)

ax.plot(
    xp,
    (msim * xp + bsim) * scale,
    color="tab:blue",
    label=rf"Sim. fit, $R^2={r2_sim:.2f}$",
)

ax.plot(
    xp,
    mexp * (xp - yield_load),
    color="tab:red",
    linestyle="--",
    label=rf"Exp. fit, $R^2={r2_exp:.2f}$",
)

ax.set_xlim(xlim)

ax.set_xlabel("Applied stress [MPa]")
ax.set_ylabel(r"Final $\Delta S_a$ [µm]")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)

fig.set_size_inches(fig.get_size_inches() * 1.35, forward=True)
plt.show()

print(f"sim fit: y = {msim:.6e} * load + {bsim:.6e}")
print(f"exp fit: y = {mexp:.6e} * (load - {yield_load:g})")
print(f"equivalent exp: y = {mexp:.6e} * load + {bexp:.6e}")
print(f"scale mexp/msim = {scale:.6e}")
print(f"R² sim = {r2_sim:.6f}")
print(f"R² exp = {r2_exp:.6f}")

# %%
import numpy as np
import matplotlib.pyplot as plt

# Top row: interrupted, increasing load
# Bottom row: uninterrupted, increasing load
plot_grid = [
    [(475, "int"), (525, "int"), (575, "int")],
    [(500, "unint"), (530, "unint"), (588, "unint")],
]

colors = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

fig, axes = plt.subplots(2, 3, dpi=150, sharex=False, sharey=False)

for r in range(2):
    for c in range(3):
        ax = axes[r, c]

        load, sample_type = plot_grid[r][c]
        color = colors.get(load, "k")

        exp_g = scatter_df[
            (scatter_df["load_mpa"] == load)
            & (scatter_df["sample_type"] == sample_type)
        ].copy()

        sim_g = sim_curve_df[
            (sim_curve_df["load_mpa"] == load)
            & (sim_curve_df["sample_type"] == sample_type)
        ].copy()

        # Experimental mean ± std as function of bulk z strain
        if len(exp_g) > 0:
            exp_summary = (
                exp_g.groupby("time_h", as_index=False)
                .agg(
                    mean_delta_sa_um=("delta_sa_um", "mean"),
                    std_delta_sa_um=("delta_sa_um", "std"),
                    mean_bulk_z_strain_percent=("bulk_z_strain_percent", "mean"),
                    std_bulk_z_strain_percent=("bulk_z_strain_percent", "std"),
                    n=("delta_sa_um", "count"),
                )
                .sort_values("mean_bulk_z_strain_percent")
            )

            exp_summary["std_delta_sa_um"] = exp_summary["std_delta_sa_um"].fillna(0.0)

            if sample_type == "unint2":
                # Only first and final experimental points with y error bars
                exp_plot = exp_summary.iloc[[0, -1]].drop_duplicates("time_h")

                ax.errorbar(
                    exp_plot["mean_bulk_z_strain_percent"],
                    exp_plot["mean_delta_sa_um"],
                    yerr=exp_plot["std_delta_sa_um"],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=4,
                    markersize=5,
                    label="exp mean ± std",
                )

            else:
                # Interrupted tests: full mean ± std curve
                x = exp_summary["mean_bulk_z_strain_percent"].to_numpy(dtype=float)
                y = exp_summary["mean_delta_sa_um"].to_numpy(dtype=float)
                s = exp_summary["std_delta_sa_um"].to_numpy(dtype=float)

                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=2.0,
                    label="exp mean",
                    marker="o",
                    markersize=4,
                )

                ax.fill_between(
                    x,
                    y - s,
                    y + s,
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    label="exp ± std",
                )

        # Simulation mean across microstructures as curve with square markers
        if len(sim_g) > 0:
            # Average across microstructures at each simulation time/load state.
            # This assumes corresponding microstructures have matching time/strain points.
            sim_summary = (
                sim_g.groupby("time", as_index=False)
                .agg(
                    mean_bulk_z_strain_percent=("bulk_z_strain_percent", "mean"),
                    mean_delta_sa_um=("delta_sa_um", "mean"),
                )
                .sort_values("mean_bulk_z_strain_percent")
            )

            # ax.plot(
            #     sim_summary["mean_bulk_z_strain_percent"],
            #     sim_summary["mean_delta_sa_um"] * scale,
            #     color="black",
            #     lw=2.0,
            #     marker="s",
            #     markersize=5,
            #     markeredgecolor="k",
            #     markeredgewidth=0.5,
            #     label="sim mean",
            #     linestyle="--",
            #     zorder=0,
            #     alpha=0.5,
            # )

            # ax.scatter(
            #     sim_summary["mean_bulk_z_strain_percent"],
            #     sim_summary["mean_delta_sa_um"] * scale,
            #     color=color,
            #     lw=2.0,
            #     marker="s",
            #     markersize=5,
            #     markeredgecolor="k",
            #     markeredgewidth=0.5,
            #     label="sim mean",
            #     linestyle="--",
            #     zorder=0,
            #     alpha=0.5,
            # )

            ax.scatter(
                sim_summary["mean_bulk_z_strain_percent"],
                sim_summary["mean_delta_sa_um"] * scale,
                s=55,
                alpha=0.5,
                color=color,
                marker="s",
                edgecolor="k",
                linewidth=0.5,
                label="sim",
            )

        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"{load} MPa {sample_type}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

for ax in axes[1, :]:
    ax.set_xlabel("bulk axial z strain [%]")

for ax in axes[:, 0]:
    ax.set_ylabel(r"$\Delta S_a$ [µm]")

fig.suptitle(r"$\Delta S_a$ vs bulk axial z strain", y=1.02)
fig.set_size_inches(fig.get_size_inches() * np.array([1.6, 1.35]), forward=True)
fig.tight_layout()

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Top row: interrupted, increasing load
# Bottom row: uninterrupted, increasing load
plot_grid = [
    [(475, "int"), (525, "int"), (575, "int")],
    [(500, "unint"), (530, "unint"), (588, "unint")],
]

colors = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

panel_labels = [["A", "B", "C"], ["D", "E", "F"]]

fig, axes = plt.subplots(2, 3, dpi=150, sharex=False, sharey=False)

for r in range(2):
    for c in range(3):
        ax = axes[r, c]

        load, sample_type = plot_grid[r][c]
        color = colors.get(load, "k")

        exp_g = scatter_df[
            (scatter_df["load_mpa"] == load)
            & (scatter_df["sample_type"] == sample_type)
        ].copy()

        sim_g = sim_curve_df[
            (sim_curve_df["load_mpa"] == load)
            & (sim_curve_df["sample_type"] == sample_type)
        ].copy()

        # Experimental mean ± std as function of bulk z strain
        if len(exp_g) > 0:
            exp_summary = (
                exp_g.groupby("time_h", as_index=False)
                .agg(
                    mean_delta_sa_um=("delta_sa_um", "mean"),
                    std_delta_sa_um=("delta_sa_um", "std"),
                    mean_bulk_z_strain_percent=("bulk_z_strain_percent", "mean"),
                    std_bulk_z_strain_percent=("bulk_z_strain_percent", "std"),
                    n=("delta_sa_um", "count"),
                )
                .sort_values("mean_bulk_z_strain_percent")
            )

            exp_summary["std_delta_sa_um"] = exp_summary["std_delta_sa_um"].fillna(0.0)

            if sample_type == "unint2":
                # Only first and final experimental points with y error bars
                exp_plot = exp_summary.iloc[[0, -1]].drop_duplicates("time_h")

                ax.errorbar(
                    exp_plot["mean_bulk_z_strain_percent"],
                    exp_plot["mean_delta_sa_um"],
                    yerr=exp_plot["std_delta_sa_um"],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=4,
                    markersize=5,
                )

            else:
                # Interrupted tests: full mean ± std curve
                x = exp_summary["mean_bulk_z_strain_percent"].to_numpy(dtype=float)
                y = exp_summary["mean_delta_sa_um"].to_numpy(dtype=float)
                s = exp_summary["std_delta_sa_um"].to_numpy(dtype=float)

                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=2.0,
                    marker="o",
                    markersize=4,
                )

                ax.fill_between(
                    x,
                    y - s,
                    y + s,
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                )

        # Simulation mean across microstructures as curve with square markers
        if len(sim_g) > 0:
            sim_summary = (
                sim_g.groupby("time", as_index=False)
                .agg(
                    mean_bulk_z_strain_percent=("bulk_z_strain_percent", "mean"),
                    mean_delta_sa_um=("delta_sa_um", "mean"),
                )
                .sort_values("mean_bulk_z_strain_percent")
            )

            ax.scatter(
                sim_summary["mean_bulk_z_strain_percent"],
                sim_summary["mean_delta_sa_um"] * scale,
                s=55,
                alpha=0.5,
                color=color,
                marker="s",
                edgecolor="k",
                linewidth=0.5,
            )

        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"{load} MPa {sample_type}")
        ax.grid(True, alpha=0.25)

        # Boldface panel label in the top-left of each axis
        ax.text(
            0.03,
            0.95,
            panel_labels[r][c],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

for ax in axes[1, :]:
    ax.set_xlabel(r"$\epsilon_{zz}$ [%]")

for ax in axes[:, 0]:
    ax.set_ylabel(r"$\Delta S_a$ [µm]")

# Single external legend, black symbols
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

# fig.suptitle(r"$\Delta S_a$ vs $\epsilon_{zz}$", y=1.02)
fig.set_size_inches(fig.get_size_inches() * np.array([1.6, 1.35]), forward=True)

# Leave room on the right for the external legend
fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])

plt.show()

# %%
# =============================================================================
# Roughness--plastic-strain analysis and publication figures
# =============================================================================
#
# What this script does:
#   4.  Calculates finite-difference roughening rates, Delta Sa / Delta eps_p.
#   5.  Fits post-yield linear, threshold-linear, and threshold-power-law models.
#   6.  Handles the fact that you have below-yield / yield / above-yield load
#       groups rather than dense transition data.
#   7.  Generates representative surface height maps at selected plastic strains.
#   8.  Generates height/profile plots parallel and transverse to loading.
#   9.  Calculates directionality/anisotropy metrics.
#   10. Calculates Sa, Sq, Sz, Ssk, Sku vs plastic strain.
#   11. Generates height-distribution/histogram figures.
#   12. Calculates and plots autocorrelation/correlation lengths.
#   13. Calculates PSD power restricted to wavelength bands.
#   14. Calculates optional representative grain-size normalization from EBSD
#       feature statistics.
#   15. Provides optional local Sa(z) vs local DIC strain correlation tools.
#
# Notes:
#   - This assumes the axial/loading direction is the first image axis of the
#     cropped height map, here labeled z. The transverse in-plane direction is
#     the second image axis, here labeled y.
#   - Your strain.csv values are treated as plastic axial strain. If they are
#     engineering plastic strain, labels below are appropriate. If true plastic
#     strain, change STRAIN_LABEL.
#   - Constant-load creep tests do not give a conventional stress-strain curve
#     per specimen, but stress/load is still useful as a grouping/threshold
#     variable. The primary x-axis here is plastic strain.
#
# Literature sources for power-law / grain-size roughening models:
#   - Osakada, K. and Oyane, M., "On the Roughening Phenomenon of Free Surface
#     in Deformation Process," Bulletin of JSME, 14, 1971.
#   - Stoudt, M. R. and Ricker, R. E., "The relationship between grain size and
#     the surface roughening behavior of Al-Mg alloys," Metallurgical and
#     Materials Transactions A, 33A, 2002.
#   - Wouters, O., Vellinga, W. P., Van Tijum, R., and De Hosson, J. T. M.,
#     "On the evolution of surface roughness during deformation of polycrystalline
#     aluminum alloys," Acta Materialia, 53, 2005.
#
# Common forms used in that literature are effectively:
#     Delta R ~ d * epsilon_p
# or more generally:
#     Delta S_a = A * epsilon_p^n
# and, when including an onset threshold:
#     Delta S_a = A * max(epsilon_p - epsilon_c, 0)^n.
#
# =============================================================================

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from collections import Counter
import inspect
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from scipy.stats import t, bootstrap
from scipy.interpolate import interp1d

# =============================================================================
# User settings
# =============================================================================

DATA_DIR = Path("/Users/gtdebru/mimosa/data")
MICRO_STATS_CSV = Path("merged_feature_stats_all.csv")

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

polish = "polished"
mag = "10x"
spacing_um = 1.379951

experiments = [
    {"load": 475, "sample_type": "int"},
    {"load": 500, "sample_type": "unint"},
    {"load": 525, "sample_type": "int"},
    {"load": 530, "sample_type": "unint"},
    {"load": 575, "sample_type": "int"},
    {"load": 588, "sample_type": "unint"},
]

crop = (slice(50, -50), slice(50, 750))
level = True
detrend_order = 1
skip_bad_files = True
exclude_dirs_named = {"bad"}

# Treat load < this as below-yield. Adjust if needed.
YIELD_LOAD_MPA = 500.0

# If True, samples without a sample-specific strain column use experiment mean strain.
use_experiment_mean_strain_fallback = True

# Require each sample to have an initial near-zero profilometry measurement.
REQUIRE_INITIAL_ZERO = True
INITIAL_TIME_TOL_H = 1.0e-8

# Use strain as fraction internally; plot as percent.
STRAIN_LABEL = r"bulk axial plastic strain, $\epsilon_p$ (%)"

# PSD settings.
RUN_FULL_PSD_BAND_ANALYSIS = True
RUN_FULL_AUTOCORRELATION_ANALYSIS = True

# If None, representative grain size is estimated from merged_feature_stats_all.csv.
# If you prefer to manually set it, e.g. GRAIN_SIZE_UM_OVERRIDE = 35.0.
GRAIN_SIZE_UM_OVERRIDE = None

# PSD band definition mode:
#   "grain_scaled": bands use representative grain size d.
#   "fixed": use FIXED_WAVELENGTH_BANDS_UM.
PSD_BAND_MODE = "grain_scaled"

FIXED_WAVELENGTH_BANDS_UM = {
    "subgrain_3_10um": (3.0, 10.0),
    "intragranular_10_50um": (10.0, 50.0),
    "mesoscale_50_200um": (50.0, 200.0),
    "macroscale_200um_plus": (200.0, np.inf),
}

# If your long-wavelength PSD is contaminated by global deformation, you can cap
# all bands at this wavelength. None means no cap except finite map size.
PSD_MAX_WAVELENGTH_CAP_UM = None

# Representative map/profile/histogram figures.
N_REPRESENTATIVE_STRAINS = 5

# Optional DIC local-strain analysis. Leave as None unless you want to run it.
DIC_CSV_PATH = None
DIC_START_ROW = 1086
DIC_N_DATASETS = 4
HEIGHT_FOR_DIC_PATH = None


# =============================================================================
# Plot style
# =============================================================================

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

TYPE_MARKERS = {
    "int": "o",
    "unint": "s",
}


# =============================================================================
# Height-map utilities
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * spacing_um
    y = column.ravel() * spacing_um

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    design, inverse = _detrend_geometry(values.shape, float(spacing_um), int(order))
    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: Path,
    *,
    level: bool = False,
    spacing_um: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um, order=detrend_order)

    return height


def _file_sort_key(path: Path):
    try:
        return 0, float(path.stem), str(path)
    except ValueError:
        return 1, np.inf, str(path)


def load_height_records_for_experiment(
    *,
    sample_type: str,
    polish: str,
    load: int | float,
    mag: str,
    crop: tuple[slice, slice],
    spacing_um: float,
    level: bool = True,
    detrend_order: int = 1,
    skip_bad_files: bool = True,
) -> list[dict]:
    path = DATA_DIR / f"creep_{sample_type}_{polish}_{load}" / "profilometry" / mag

    if not path.exists():
        print(f"Missing profilometry path: {path}")
        return []

    files = sorted(path.rglob("*.csv"), key=_file_sort_key)
    records = []

    for file in files:
        rel_parts = file.relative_to(path).parts

        if any(part in exclude_dirs_named for part in rel_parts):
            continue

        if len(rel_parts) < 2:
            continue

        sample = str(rel_parts[0]).strip()

        try:
            time_value = float(file.stem)
        except ValueError:
            continue

        try:
            height = read_height(
                file,
                level=level,
                spacing_um=spacing_um,
                detrend_order=detrend_order,
            )
            height = height[crop]

            if not np.all(np.isfinite(height)):
                continue

            records.append(
                {
                    "sample": sample,
                    "time_h": time_value,
                    "height_path": file,
                    "height_shape": height.shape,
                }
            )

        except Exception as exc:
            if skip_bad_files:
                warnings.warn(f"Skipping bad profilometry file {file}: {exc}")
                continue
            raise

    if records:
        shape_counts = Counter(rec["height_shape"] for rec in records)
        reference_shape, _ = shape_counts.most_common(1)[0]

        if len(shape_counts) > 1:
            warnings.warn(
                f"{load} {sample_type}: multiple cropped shapes found: "
                f"{dict(shape_counts)}. Keeping {reference_shape}."
            )

        records = [rec for rec in records if rec["height_shape"] == reference_shape]

    return records


def load_cropped_height_from_record(record: dict) -> np.ndarray:
    height = read_height(
        Path(record["height_path"]),
        level=level,
        spacing_um=spacing_um,
        detrend_order=detrend_order,
    )
    return height[crop]


# =============================================================================
# Strain utilities
# =============================================================================


def load_strain_table(sample_type: str, polish: str, load: int | float) -> pd.DataFrame:
    strain_path = DATA_DIR / f"creep_{sample_type}_{polish}_{load}" / "strain.csv"

    if not strain_path.exists():
        warnings.warn(f"Missing strain file: {strain_path}")
        return pd.DataFrame()

    df = pd.read_csv(strain_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={col: str(col).strip() for col in df.columns})

    if "time_h" not in df.columns:
        warnings.warn(f"No time_h column in {strain_path}")
        return pd.DataFrame()

    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")

    for col in df.columns:
        if col != "time_h":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def interp_series_at_time(
    time: np.ndarray, values: np.ndarray, query_time: float
) -> float:
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(time) & np.isfinite(values)

    if np.count_nonzero(valid) == 0:
        return np.nan

    t_valid = time[valid]
    v_valid = values[valid]

    order = np.argsort(t_valid)
    t_valid = t_valid[order]
    v_valid = v_valid[order]

    return float(np.interp(query_time, t_valid, v_valid))


def strain_at_time_for_sample_or_mean(
    strain_df: pd.DataFrame,
    sample: str,
    time_h: float,
) -> tuple[float, str]:
    if strain_df.empty:
        return np.nan, "none"

    strain_time = strain_df["time_h"].to_numpy(dtype=float)
    sample_cols = [c for c in strain_df.columns if c != "time_h"]

    if sample in sample_cols:
        val = interp_series_at_time(
            strain_time, strain_df[sample].to_numpy(dtype=float), time_h
        )
        if np.isfinite(val):
            return val, "sample"

    lower_map = {str(c).lower(): c for c in sample_cols}

    if sample.lower() in lower_map:
        col = lower_map[sample.lower()]
        val = interp_series_at_time(
            strain_time, strain_df[col].to_numpy(dtype=float), time_h
        )
        if np.isfinite(val):
            return val, "sample_case_insensitive"

    if use_experiment_mean_strain_fallback and len(sample_cols) > 0:
        strain_values = strain_df[sample_cols].to_numpy(dtype=float)
        mean_strain = np.nanmean(strain_values, axis=1)
        val = interp_series_at_time(strain_time, mean_strain, time_h)

        if np.isfinite(val):
            return val, "experiment_mean"

    return np.nan, "none"


# =============================================================================
# Surface roughness metrics
# =============================================================================


def surface_metrics(height_um: np.ndarray) -> dict:
    z = np.asarray(height_um, dtype=float)
    z = z - np.nanmean(z)

    abs_z = np.abs(z)

    sa = np.nanmean(abs_z)
    sq = np.sqrt(np.nanmean(z**2))
    sz = np.nanpercentile(z, 99.5) - np.nanpercentile(z, 0.5)

    if sq > 0:
        ssk = np.nanmean(z**3) / sq**3
        sku = np.nanmean(z**4) / sq**4
    else:
        ssk = np.nan
        sku = np.nan

    return {
        "sa_um": sa,
        "sq_um": sq,
        "sz_robust_99p5_0p5_um": sz,
        "ssk": ssk,
        "sku": sku,
    }


def directional_line_roughness(height_um: np.ndarray) -> dict:
    """
    Returns mean 1D profile roughness for lines parallel and transverse to loading.

    Assumption:
      height shape = (nz, ny)
      axis 0 = z/loading direction
      axis 1 = y/transverse direction
    """
    H = np.asarray(height_um, dtype=float)

    # Parallel to loading: fixed y-column, profile along z.
    col_centered = H - np.nanmean(H, axis=0, keepdims=True)
    ra_parallel_z = np.nanmean(np.nanmean(np.abs(col_centered), axis=0))
    rq_parallel_z = np.nanmean(np.sqrt(np.nanmean(col_centered**2, axis=0)))

    # Transverse to loading: fixed z-row, profile along y.
    row_centered = H - np.nanmean(H, axis=1, keepdims=True)
    ra_transverse_y = np.nanmean(np.nanmean(np.abs(row_centered), axis=1))
    rq_transverse_y = np.nanmean(np.sqrt(np.nanmean(row_centered**2, axis=1)))

    return {
        "ra_parallel_z_um": ra_parallel_z,
        "rq_parallel_z_um": rq_parallel_z,
        "ra_transverse_y_um": ra_transverse_y,
        "rq_transverse_y_um": rq_transverse_y,
        "profile_ra_anisotropy_parallel_over_transverse": (
            ra_parallel_z / ra_transverse_y if ra_transverse_y > 0 else np.nan
        ),
        "profile_rq_anisotropy_parallel_over_transverse": (
            rq_parallel_z / rq_transverse_y if rq_transverse_y > 0 else np.nan
        ),
    }


def calculate_surface_metrics_for_record(record: dict) -> dict:
    H = load_cropped_height_from_record(record)
    metrics = surface_metrics(H)
    metrics.update(directional_line_roughness(H))
    return metrics


# =============================================================================
# Build all-time data table
# =============================================================================

rows = []
summary_rows = []

for exp in experiments:
    load = exp["load"]
    sample_type = exp["sample_type"]

    height_records = load_height_records_for_experiment(
        sample_type=sample_type,
        polish=polish,
        load=load,
        mag=mag,
        crop=crop,
        spacing_um=spacing_um,
        level=level,
        detrend_order=detrend_order,
        skip_bad_files=skip_bad_files,
    )

    strain_df = load_strain_table(sample_type, polish, load)

    if len(height_records) == 0:
        summary_rows.append(
            {
                "load_mpa": load,
                "sample_type": sample_type,
                "n_height_records": 0,
                "n_height_samples": 0,
                "n_samples_with_valid_initial": 0,
                "n_points": 0,
            }
        )
        continue

    metric_records = []

    for rec in height_records:
        try:
            metrics = calculate_surface_metrics_for_record(rec)
        except Exception as exc:
            warnings.warn(f"Failed metric calculation for {rec['height_path']}: {exc}")
            continue

        metric_records.append({**rec, **metrics})

    metric_df = pd.DataFrame(metric_records)

    if metric_df.empty:
        continue

    n_before = len(rows)
    n_valid_initial_samples = 0

    for sample, g in metric_df.groupby("sample"):
        g = g.sort_values("time_h")

        if (
            REQUIRE_INITIAL_ZERO
            and np.nanmin(g["time_h"].to_numpy(dtype=float)) > INITIAL_TIME_TOL_H
        ):
            continue

        if g["time_h"].nunique() < 2:
            continue

        n_valid_initial_samples += 1

        initial_row = g.iloc[0]
        t_initial = float(initial_row["time_h"])
        sa_initial = float(initial_row["sa_um"])
        sq_initial = float(initial_row["sq_um"])
        sz_initial = float(initial_row["sz_robust_99p5_0p5_um"])

        for _, row in g.iterrows():
            time_h = float(row["time_h"])

            bulk_z_strain, strain_source = strain_at_time_for_sample_or_mean(
                strain_df,
                sample,
                time_h,
            )

            if not np.isfinite(bulk_z_strain):
                continue

            rows.append(
                {
                    "load_mpa": load,
                    "sample_type": sample_type,
                    "polish": polish,
                    "sample": sample,
                    "time_h": time_h,
                    "initial_time_h": t_initial,
                    "height_path": row["height_path"],
                    "bulk_z_strain": bulk_z_strain,
                    "bulk_z_strain_percent": 100.0 * bulk_z_strain,
                    "strain_source": strain_source,
                    "sa_initial_um": sa_initial,
                    "sq_initial_um": sq_initial,
                    "sz_initial_um": sz_initial,
                    "sa_um": float(row["sa_um"]),
                    "sq_um": float(row["sq_um"]),
                    "sz_robust_99p5_0p5_um": float(row["sz_robust_99p5_0p5_um"]),
                    "ssk": float(row["ssk"]),
                    "sku": float(row["sku"]),
                    "delta_sa_um": float(row["sa_um"] - sa_initial),
                    "delta_sq_um": float(row["sq_um"] - sq_initial),
                    "delta_sz_robust_um": float(
                        row["sz_robust_99p5_0p5_um"] - sz_initial
                    ),
                    "ra_parallel_z_um": float(row["ra_parallel_z_um"]),
                    "rq_parallel_z_um": float(row["rq_parallel_z_um"]),
                    "ra_transverse_y_um": float(row["ra_transverse_y_um"]),
                    "rq_transverse_y_um": float(row["rq_transverse_y_um"]),
                    "profile_ra_anisotropy_parallel_over_transverse": float(
                        row["profile_ra_anisotropy_parallel_over_transverse"]
                    ),
                    "profile_rq_anisotropy_parallel_over_transverse": float(
                        row["profile_rq_anisotropy_parallel_over_transverse"]
                    ),
                }
            )

    n_after = len(rows)

    summary_rows.append(
        {
            "load_mpa": load,
            "sample_type": sample_type,
            "n_height_records": len(metric_df),
            "n_height_samples": metric_df["sample"].nunique(),
            "n_samples_with_valid_initial": n_valid_initial_samples,
            "n_points": n_after - n_before,
        }
    )

point_df = pd.DataFrame(rows)
summary_df = pd.DataFrame(summary_rows)

point_df.to_csv(OUTPUT_DIR / "roughness_plastic_strain_point_table.csv", index=False)
summary_df.to_csv(
    OUTPUT_DIR / "roughness_plastic_strain_loading_summary.csv", index=False
)

print("Summary:")
print(summary_df)
print()
print("point_df shape:", point_df.shape)
print(point_df.head())


# =============================================================================
# Representative grain-size estimate from EBSD feature stats
# =============================================================================


def representative_grain_size_from_feature_stats(csv_path: Path) -> dict:
    if not csv_path.exists():
        warnings.warn(f"Microstructure stats CSV not found: {csv_path}")
        return {
            "d_ref_um": np.nan,
            "d_ref_source": "not_available",
            "grain_stats_df": pd.DataFrame(),
        }

    g = pd.read_csv(csv_path)

    if "complete_non_surface" in g.columns:
        mask = g["complete_non_surface"].astype(str).str.lower().eq("true")
        g_use = g[mask].copy()
    else:
        g_use = g.copy()

    if "EquivalentDiameters" not in g_use.columns:
        warnings.warn("EquivalentDiameters column not found in microstructure stats.")
        return {
            "d_ref_um": np.nan,
            "d_ref_source": "not_available",
            "grain_stats_df": g,
        }

    # Feature stats appear to be in micrometers in your file.
    d = pd.to_numeric(g_use["EquivalentDiameters"], errors="coerce").to_numpy(
        dtype=float
    )
    d = d[np.isfinite(d) & (d > 0)]

    if d.size == 0:
        return {
            "d_ref_um": np.nan,
            "d_ref_source": "not_available",
            "grain_stats_df": g,
        }

    d_ref = float(np.median(d))

    return {
        "d_ref_um": d_ref,
        "d_ref_mean_um": float(np.mean(d)),
        "d_ref_std_um": float(np.std(d, ddof=1)),
        "d_ref_q25_um": float(np.percentile(d, 25)),
        "d_ref_q75_um": float(np.percentile(d, 75)),
        "d_ref_source": "median_equivalent_diameter_complete_non_surface",
        "grain_stats_df": g,
    }


grain_info = representative_grain_size_from_feature_stats(MICRO_STATS_CSV)

if GRAIN_SIZE_UM_OVERRIDE is not None:
    d_ref_um = float(GRAIN_SIZE_UM_OVERRIDE)
    d_ref_source = "manual_override"
else:
    d_ref_um = grain_info["d_ref_um"]
    d_ref_source = grain_info["d_ref_source"]

print()
print(f"Representative grain size d_ref = {d_ref_um:.4g} um ({d_ref_source})")

if np.isfinite(d_ref_um) and d_ref_um > 0:
    point_df["sa_over_d_ref"] = point_df["sa_um"] / d_ref_um
    point_df["delta_sa_over_d_ref"] = point_df["delta_sa_um"] / d_ref_um
    point_df["sq_over_d_ref"] = point_df["sq_um"] / d_ref_um
    point_df["delta_sq_over_d_ref"] = point_df["delta_sq_um"] / d_ref_um
else:
    point_df["sa_over_d_ref"] = np.nan
    point_df["delta_sa_over_d_ref"] = np.nan
    point_df["sq_over_d_ref"] = np.nan
    point_df["delta_sq_over_d_ref"] = np.nan


# =============================================================================
# Generic plotting/stat helpers
# =============================================================================


def savefig(fig: plt.Figure, filename: str):
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")


def mean_ci_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    rows = []

    for key, group in df.groupby(group_cols):
        y = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]

        if y.size == 0:
            continue

        mean = float(np.mean(y))

        if y.size > 1:
            std = float(np.std(y, ddof=1))
            sem = std / np.sqrt(y.size)
            tcrit = t.ppf(1.0 - (1.0 - confidence) / 2.0, df=y.size - 1)
            low = mean - tcrit * sem
            high = mean + tcrit * sem
        else:
            std = np.nan
            low = np.nan
            high = np.nan

        if not isinstance(key, tuple):
            key = (key,)

        row = {col: val for col, val in zip(group_cols, key)}
        row.update(
            {
                f"{value_col}_mean": mean,
                f"{value_col}_std": std,
                f"{value_col}_ci_low": low,
                f"{value_col}_ci_high": high,
                "n": int(y.size),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def plot_scatter_by_load_type(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    alpha: float = 0.75,
    s: float = 35,
    label_prefix: str = "",
):
    for (sample_type, load), group in df.groupby(["sample_type", "load_mpa"]):
        ax.scatter(
            group[x_col],
            group[y_col],
            s=s,
            alpha=alpha,
            marker=TYPE_MARKERS.get(sample_type, "o"),
            color=LOAD_COLORS.get(load, None),
            label=f"{label_prefix}{load} MPa {sample_type}",
        )


def plot_mean_by_time_or_strain(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    *,
    x_col: str = "bulk_z_strain_percent",
    plot_individual_points: bool = True,
):
    if plot_individual_points:
        plot_scatter_by_load_type(ax, df, x_col, y_col, alpha=0.35, s=18)

    summary = (
        df.groupby(["load_mpa", "sample_type", "time_h"], as_index=False)
        .agg(
            x_mean=(x_col, "mean"),
            x_std=(x_col, "std"),
            y_mean=(y_col, "mean"),
            y_std=(y_col, "std"),
            n=(y_col, "count"),
        )
        .sort_values(["load_mpa", "sample_type", "x_mean"])
    )

    for (sample_type, load), group in summary.groupby(["sample_type", "load_mpa"]):
        group = group.sort_values("x_mean")
        color = LOAD_COLORS.get(load, None)
        marker = TYPE_MARKERS.get(sample_type, "o")

        ax.errorbar(
            group["x_mean"],
            group["y_mean"],
            yerr=group["y_std"].fillna(0.0),
            fmt=marker + "-",
            color=color,
            ecolor=color,
            lw=1.6,
            ms=4,
            capsize=3,
            label=f"mean {load} MPa {sample_type}",
        )


# =============================================================================
# 4, 5, 6: Roughening rate and model fits
# =============================================================================


def add_incremental_rates(df: pd.DataFrame) -> pd.DataFrame:
    rate_rows = []

    for key, g in df.groupby(["load_mpa", "sample_type", "sample"]):
        g = g.sort_values("bulk_z_strain")

        if len(g) < 2:
            continue

        load, sample_type, sample = key

        eps = g["bulk_z_strain"].to_numpy(dtype=float)
        eps_pct = g["bulk_z_strain_percent"].to_numpy(dtype=float)
        sa = g["sa_um"].to_numpy(dtype=float)
        dsa = g["delta_sa_um"].to_numpy(dtype=float)
        time_h = g["time_h"].to_numpy(dtype=float)

        for i in range(1, len(g)):
            de = eps[i] - eps[i - 1]
            de_pct = eps_pct[i] - eps_pct[i - 1]

            if not np.isfinite(de) or de <= 0:
                continue

            rate_rows.append(
                {
                    "load_mpa": load,
                    "sample_type": sample_type,
                    "sample": sample,
                    "time_h_mid": 0.5 * (time_h[i] + time_h[i - 1]),
                    "strain_mid": 0.5 * (eps[i] + eps[i - 1]),
                    "strain_mid_percent": 0.5 * (eps_pct[i] + eps_pct[i - 1]),
                    "delta_strain": de,
                    "delta_strain_percent": de_pct,
                    "delta_sa_step_um": sa[i] - sa[i - 1],
                    "delta_delta_sa_step_um": dsa[i] - dsa[i - 1],
                    "rate_um_per_strain": (sa[i] - sa[i - 1]) / de,
                    "rate_um_per_percent_strain": (sa[i] - sa[i - 1]) / de_pct,
                    "rate_delta_um_per_percent_strain": (dsa[i] - dsa[i - 1]) / de_pct,
                }
            )

    return pd.DataFrame(rate_rows)


rate_df = add_incremental_rates(point_df)
rate_df.to_csv(OUTPUT_DIR / "roughening_incremental_rates.csv", index=False)

fig, ax = plt.subplots(figsize=(6.6, 4.2))
plot_scatter_by_load_type(
    ax,
    rate_df,
    "strain_mid_percent",
    "rate_um_per_percent_strain",
    alpha=0.75,
    s=42,
)
ax.axhline(0.0, color="0.4", lw=0.8)
ax.set_xlabel(STRAIN_LABEL)
ax.set_ylabel(r"finite roughening rate, $\Delta S_a / \Delta \epsilon_p$ [$\mu$m / %]")
ax.set_title(r"Finite-difference roughening rate")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
savefig(fig, "04_finite_difference_roughening_rate.png")
plt.show()


def model_linear_free(x, m, b):
    return m * x + b


def model_threshold_linear(x, A, eps_c):
    return A * np.maximum(x - eps_c, 0.0)


def model_threshold_power(x, A, eps_c, n):
    return A * np.maximum(x - eps_c, 0.0) ** n


def fit_model(
    name: str,
    func,
    x: np.ndarray,
    y: np.ndarray,
    p0: tuple,
    bounds=(-np.inf, np.inf),
) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]

    if len(x_fit) < len(p0) + 1:
        return {
            "name": name,
            "success": False,
            "message": "not enough data",
        }

    try:
        popt, pcov = curve_fit(
            func,
            x_fit,
            y_fit,
            p0=p0,
            bounds=bounds,
            maxfev=200000,
        )

        y_hat = func(x_fit, *popt)
        resid = y_fit - y_hat

        rss = float(np.sum(resid**2))
        tss = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else np.nan

        n_obs = len(y_fit)
        k = len(popt)

        if rss <= 0:
            aic = -np.inf
            bic = -np.inf
        else:
            aic = n_obs * np.log(rss / n_obs) + 2 * k
            bic = n_obs * np.log(rss / n_obs) + k * np.log(n_obs)

        stderr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(k, np.nan)

        return {
            "name": name,
            "success": True,
            "func": func,
            "popt": popt,
            "stderr": stderr,
            "pcov": pcov,
            "rss": rss,
            "r2": r2,
            "aic": aic,
            "bic": bic,
            "n_obs": n_obs,
            "k": k,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "y_hat": y_hat,
        }

    except Exception as exc:
        return {
            "name": name,
            "success": False,
            "message": str(exc),
        }


fit_df = point_df.copy()
fit_df = fit_df[
    np.isfinite(fit_df["bulk_z_strain"]) & np.isfinite(fit_df["delta_sa_um"])
].copy()

x_all = fit_df["bulk_z_strain"].to_numpy(dtype=float)
y_all = fit_df["delta_sa_um"].to_numpy(dtype=float)

post_yield_mask = fit_df["load_mpa"].to_numpy(dtype=float) >= YIELD_LOAD_MPA
x_post = x_all[post_yield_mask]
y_post = y_all[post_yield_mask]

# Post-yield linear fit excluding below-yield observations.
linear_post_fit = fit_model(
    "post_yield_linear_free_intercept",
    model_linear_free,
    x_post,
    y_post,
    p0=(np.nanmax(y_post) / max(np.nanmax(x_post), 1e-12), 0.0),
)

# Threshold-linear model uses all data and learns epsilon_c.
threshold_linear_fit = fit_model(
    "all_data_threshold_linear",
    model_threshold_linear,
    x_all,
    y_all,
    p0=(np.nanmax(y_all) / max(np.nanmax(x_all), 1e-12), np.nanpercentile(x_all, 10)),
    bounds=([0.0, 0.0], [np.inf, np.nanmax(x_all)]),
)

# Threshold-power model uses all data and learns epsilon_c and n.
threshold_power_fit = fit_model(
    "all_data_threshold_power",
    model_threshold_power,
    x_all,
    y_all,
    p0=(
        np.nanmax(y_all) / max(np.nanmax(x_all), 1e-12),
        np.nanpercentile(x_all, 10),
        1.0,
    ),
    bounds=([0.0, 0.0, 0.25], [np.inf, np.nanmax(x_all), 3.0]),
)

fits = [linear_post_fit, threshold_linear_fit, threshold_power_fit]

fit_summary_rows = []

for fit in fits:
    if not fit["success"]:
        fit_summary_rows.append(
            {
                "model": fit["name"],
                "success": False,
                "message": fit.get("message", ""),
            }
        )
        continue

    row = {
        "model": fit["name"],
        "success": True,
        "rss": fit["rss"],
        "r2": fit["r2"],
        "aic": fit["aic"],
        "bic": fit["bic"],
        "n_obs": fit["n_obs"],
    }

    for i, value in enumerate(fit["popt"]):
        row[f"p{i}"] = value
        row[f"p{i}_stderr"] = fit["stderr"][i]

    fit_summary_rows.append(row)

fit_summary = pd.DataFrame(fit_summary_rows)
fit_summary.to_csv(OUTPUT_DIR / "roughening_model_fit_summary.csv", index=False)

print()
print("Model fit summary:")
print(fit_summary)

fig, ax = plt.subplots(figsize=(6.8, 4.6))
plot_scatter_by_load_type(
    ax, fit_df, "bulk_z_strain_percent", "delta_sa_um", alpha=0.55, s=30
)

x_plot = np.linspace(0.0, np.nanmax(x_all) * 1.03, 400)

for fit, color, linestyle in zip(
    fits, ["black", "tab:blue", "tab:red"], ["-", "--", "-."]
):
    if not fit["success"]:
        continue

    y_plot = fit["func"](x_plot, *fit["popt"])
    label = f"{fit['name']}, $R^2$={fit['r2']:.2f}, AIC={fit['aic']:.1f}"

    if fit["name"] == "all_data_threshold_power":
        A, eps_c, n_exp = fit["popt"]
        label += rf", $n={n_exp:.2f}$, $\epsilon_c={100*eps_c:.2f}\%$"
    elif fit["name"] == "all_data_threshold_linear":
        A, eps_c = fit["popt"]
        label += rf", $\epsilon_c={100*eps_c:.2f}\%$"

    ax.plot(
        100.0 * x_plot, y_plot, color=color, linestyle=linestyle, lw=2.0, label=label
    )

ax.axhline(0.0, color="0.4", lw=0.8)
ax.axvline(0.0, color="0.4", lw=0.8)
ax.set_xlabel(STRAIN_LABEL)
ax.set_ylabel(r"$\Delta S_a$ [$\mu$m]")
ax.set_title(r"Linear vs threshold roughening models")
ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "05_06_linear_threshold_power_model_comparison.png")
plt.show()


# =============================================================================
# Main Delta Sa vs strain figure
# =============================================================================

fig, ax = plt.subplots(figsize=(6.8, 4.4))
plot_mean_by_time_or_strain(
    ax,
    point_df,
    y_col="delta_sa_um",
    x_col="bulk_z_strain_percent",
    plot_individual_points=True,
)
ax.axhline(0.0, color="0.4", lw=0.8)
ax.set_xlabel(STRAIN_LABEL)
ax.set_ylabel(r"$\Delta S_a$ [$\mu$m]")
ax.set_title(r"Roughness increment vs plastic strain")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
savefig(fig, "main_delta_sa_vs_plastic_strain.png")
plt.show()


# =============================================================================
# 7: Representative surface height maps
# =============================================================================


def pick_representative_records_by_strain(
    df: pd.DataFrame,
    n: int = 5,
    *,
    exclude_initial: bool = True,
) -> pd.DataFrame:
    d = df.copy()

    if exclude_initial:
        d = d[d["bulk_z_strain"] > 0].copy()

    d = d[np.isfinite(d["bulk_z_strain"])].copy()

    if d.empty:
        return d

    targets = np.linspace(d["bulk_z_strain"].min(), d["bulk_z_strain"].max(), n)

    chosen_indices = []

    for target in targets:
        idx = (d["bulk_z_strain"] - target).abs().idxmin()
        chosen_indices.append(idx)

    chosen = (
        d.loc[chosen_indices]
        .drop_duplicates("height_path")
        .sort_values("bulk_z_strain")
    )

    return chosen


representative_df = pick_representative_records_by_strain(
    point_df,
    n=N_REPRESENTATIVE_STRAINS,
    exclude_initial=False,
)
representative_df.to_csv(
    OUTPUT_DIR / "representative_records_for_maps.csv", index=False
)

heights_for_maps = []
for _, rec in representative_df.iterrows():
    H = load_cropped_height_from_record(rec)
    H0 = H - np.nanmean(H)
    heights_for_maps.append(H0)

if len(heights_for_maps) > 0:
    all_values = np.concatenate([H.ravel() for H in heights_for_maps])
    lim = float(np.nanpercentile(np.abs(all_values), 99.0))
    lim = max(lim, np.finfo(float).eps)

    n_maps = len(heights_for_maps)
    ncols = min(3, n_maps)
    nrows = int(np.ceil(n_maps / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.5 * nrows), squeeze=False
    )
    extent = None

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (_, rec), H0 in zip(
        axes.ravel(), representative_df.iterrows(), heights_for_maps
    ):
        nz, ny = H0.shape
        extent = [0, ny * spacing_um, 0, nz * spacing_um]
        image = ax.imshow(
            H0,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            interpolation="nearest",
            rasterized=True,
            aspect="equal",
        )
        ax.axis("on")
        ax.set_xlabel(r"$y$ [$\mu$m]")
        ax.set_ylabel(r"$z$ loading [$\mu$m]")
        ax.set_title(
            f"{rec['load_mpa']} MPa {rec['sample_type']} {rec['sample']}\n"
            rf"$\epsilon_p$={rec['bulk_z_strain_percent']:.2f}%, "
            rf"$\Delta S_a$={rec['delta_sa_um']:.2f} $\mu$m"
        )
        ax.annotate(
            "loading",
            xy=(0.92, 0.15),
            xytext=(0.92, 0.35),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="k", lw=1.2),
            ha="center",
            va="center",
            fontsize=8,
        )

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label(r"leveled height, $h-\bar{h}$ [$\mu$m]")
    fig.suptitle("Representative surface height maps", y=0.995)
    savefig(fig, "07_representative_surface_height_maps.png")
    plt.show()


# =============================================================================
# 8: Line/profile plots parallel and transverse to loading
# =============================================================================


def representative_line_profiles(height_um: np.ndarray) -> dict:
    H = np.asarray(height_um, dtype=float)
    H0 = H - np.nanmean(H)

    nz, ny = H0.shape
    z_um = np.arange(nz) * spacing_um
    y_um = np.arange(ny) * spacing_um

    z_profile_mean_over_y = np.nanmean(H0, axis=1)
    y_profile_mean_over_z = np.nanmean(H0, axis=0)

    center_y_index = ny // 2
    center_z_index = nz // 2

    z_centerline = H0[:, center_y_index]
    y_centerline = H0[center_z_index, :]

    return {
        "z_um": z_um,
        "y_um": y_um,
        "z_profile_mean_over_y": z_profile_mean_over_y,
        "y_profile_mean_over_z": y_profile_mean_over_z,
        "z_centerline": z_centerline,
        "y_centerline": y_centerline,
    }


if len(representative_df) > 0:
    n_profiles = len(representative_df)
    fig, axes = plt.subplots(
        n_profiles, 2, figsize=(8.0, 2.1 * n_profiles), squeeze=False
    )

    for row_idx, ((_, rec), ax_row) in enumerate(
        zip(representative_df.iterrows(), axes)
    ):
        H = load_cropped_height_from_record(rec)
        profiles = representative_line_profiles(H)

        ax = ax_row[0]
        ax.plot(
            profiles["z_um"],
            profiles["z_centerline"],
            color="0.65",
            lw=0.8,
            label="centerline",
        )
        ax.plot(
            profiles["z_um"],
            profiles["z_profile_mean_over_y"],
            color="k",
            lw=1.8,
            label="mean over transverse",
        )
        ax.set_xlabel(r"$z$ loading [$\mu$m]")
        ax.set_ylabel(r"$h-\bar{h}$ [$\mu$m]")
        ax.set_title(
            f"{rec['load_mpa']} MPa, "
            rf"$\epsilon_p$={rec['bulk_z_strain_percent']:.2f}%: parallel"
        )
        ax.legend(fontsize=7)

        ax = ax_row[1]
        ax.plot(
            profiles["y_um"],
            profiles["y_centerline"],
            color="0.65",
            lw=0.8,
            label="centerline",
        )
        ax.plot(
            profiles["y_um"],
            profiles["y_profile_mean_over_z"],
            color="k",
            lw=1.8,
            label="mean over loading",
        )
        ax.set_xlabel(r"$y$ transverse [$\mu$m]")
        ax.set_ylabel(r"$h-\bar{h}$ [$\mu$m]")
        ax.set_title(
            f"{rec['load_mpa']} MPa, "
            rf"$\epsilon_p$={rec['bulk_z_strain_percent']:.2f}%: transverse"
        )
        ax.legend(fontsize=7)

    fig.tight_layout()
    savefig(fig, "08_parallel_transverse_height_profiles.png")
    plt.show()


fig, ax = plt.subplots(figsize=(6.8, 4.4))
plot_mean_by_time_or_strain(
    ax,
    point_df,
    y_col="profile_ra_anisotropy_parallel_over_transverse",
    x_col="bulk_z_strain_percent",
    plot_individual_points=True,
)
ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
ax.set_xlabel(STRAIN_LABEL)
ax.set_ylabel(r"profile roughness anisotropy, $R_a^\parallel/R_a^\perp$")
ax.set_title("Directional profile roughness anisotropy")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
savefig(fig, "08_profile_roughness_anisotropy_vs_strain.png")
plt.show()


# =============================================================================
# 9, 12: Autocorrelation and correlation-length anisotropy
# =============================================================================


def normalized_autocorrelation_2d(
    height_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = np.asarray(height_um, dtype=float)
    H0 = H - np.nanmean(H)

    F = np.fft.fft2(H0)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)

    cz, cy = np.array(ac.shape) // 2

    if ac[cz, cy] == 0:
        ac[:] = np.nan
    else:
        ac = ac / ac[cz, cy]

    nz, ny = H.shape
    z_lags_um = (np.arange(nz) - cz) * spacing_um
    y_lags_um = (np.arange(ny) - cy) * spacing_um

    return z_lags_um, y_lags_um, ac


def first_one_over_e_length(lags: np.ndarray, values: np.ndarray) -> tuple[float, bool]:
    lags = np.asarray(lags, dtype=float)
    values = np.asarray(values, dtype=float)

    center = np.argmin(np.abs(lags))
    x = lags[center:]
    y = values[center:]

    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, False

    target = np.exp(-1)

    for k in range(1, len(x)):
        if y[k] <= target:
            x0, x1 = x[k - 1], x[k]
            y0, y1 = y[k - 1], y[k]

            if y1 == y0:
                return float(x1), True

            x_cross = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            return float(x_cross), True

    return float(x[-1]), False


def autocorrelation_metrics(height_um: np.ndarray) -> dict:
    z_lags, y_lags, ac = normalized_autocorrelation_2d(height_um)

    cz = np.argmin(np.abs(z_lags))
    cy = np.argmin(np.abs(y_lags))

    ac_z = ac[:, cy]
    ac_y = ac[cz, :]

    corr_z, crossed_z = first_one_over_e_length(z_lags, ac_z)
    corr_y, crossed_y = first_one_over_e_length(y_lags, ac_y)

    return {
        "corr_length_parallel_z_um": corr_z,
        "corr_length_transverse_y_um": corr_y,
        "corr_crossed_parallel_z": crossed_z,
        "corr_crossed_transverse_y": crossed_y,
        "corr_length_anisotropy_parallel_over_transverse": (
            corr_z / corr_y
            if np.isfinite(corr_z) and np.isfinite(corr_y) and corr_y > 0
            else np.nan
        ),
    }


if RUN_FULL_AUTOCORRELATION_ANALYSIS:
    ac_rows = []

    for idx, rec in point_df.iterrows():
        try:
            H = load_cropped_height_from_record(rec)
            acm = autocorrelation_metrics(H)
            ac_rows.append(
                {
                    "row_index": idx,
                    "load_mpa": rec["load_mpa"],
                    "sample_type": rec["sample_type"],
                    "sample": rec["sample"],
                    "time_h": rec["time_h"],
                    "bulk_z_strain": rec["bulk_z_strain"],
                    "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                    "delta_sa_um": rec["delta_sa_um"],
                    **acm,
                }
            )
        except Exception as exc:
            warnings.warn(f"Autocorrelation failed for {rec['height_path']}: {exc}")

    ac_df = pd.DataFrame(ac_rows)
    ac_df.to_csv(OUTPUT_DIR / "autocorrelation_metrics.csv", index=False)

    for col, ylabel, fname in [
        (
            "corr_length_parallel_z_um",
            r"parallel correlation length, $\xi_\parallel$ [$\mu$m]",
            "12_corr_length_parallel_vs_strain.png",
        ),
        (
            "corr_length_transverse_y_um",
            r"transverse correlation length, $\xi_\perp$ [$\mu$m]",
            "12_corr_length_transverse_vs_strain.png",
        ),
        (
            "corr_length_anisotropy_parallel_over_transverse",
            r"$\xi_\parallel/\xi_\perp$",
            "09_12_corr_length_anisotropy_vs_strain.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        plot_mean_by_time_or_strain(
            ax,
            ac_df,
            y_col=col,
            x_col="bulk_z_strain_percent",
            plot_individual_points=True,
        )
        if "anisotropy" in col:
            ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
        ax.set_xlabel(STRAIN_LABEL)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        savefig(fig, fname)
        plt.show()

    if len(representative_df) > 0:
        ac_maps = []
        ac_meta = []

        for _, rec in representative_df.iterrows():
            H = load_cropped_height_from_record(rec)
            z_lags, y_lags, ac = normalized_autocorrelation_2d(H)
            ac_maps.append((z_lags, y_lags, ac))
            ac_meta.append(rec)

        n_maps = len(ac_maps)
        ncols = min(3, n_maps)
        nrows = int(np.ceil(n_maps / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.3 * ncols, 3.2 * nrows), squeeze=False
        )

        for ax in axes.ravel():
            ax.axis("off")

        for ax, rec, (z_lags, y_lags, ac) in zip(axes.ravel(), ac_meta, ac_maps):
            extent = [y_lags.min(), y_lags.max(), z_lags.min(), z_lags.max()]
            im = ax.imshow(
                ac,
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=-0.2,
                vmax=1.0,
                interpolation="nearest",
                rasterized=True,
                aspect="equal",
            )
            ax.axis("on")
            ax.set_xlim(-250, 250)
            ax.set_ylim(-250, 250)
            ax.set_xlabel(r"$\Delta y$ [$\mu$m]")
            ax.set_ylabel(r"$\Delta z$ [$\mu$m]")
            ax.set_title(
                f"{rec['load_mpa']} MPa\n"
                rf"$\epsilon_p$={rec['bulk_z_strain_percent']:.2f}%"
            )

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label("normalized autocorrelation")
        fig.suptitle("2D autocorrelation maps", y=0.995)
        savefig(fig, "12_representative_autocorrelation_maps.png")
        plt.show()


# =============================================================================
# 10: Additional areal roughness parameters vs plastic strain
# =============================================================================

metric_plot_specs = [
    ("sa_um", r"$S_a$ [$\mu$m]"),
    ("delta_sa_um", r"$\Delta S_a$ [$\mu$m]"),
    ("sq_um", r"$S_q$ [$\mu$m]"),
    ("delta_sq_um", r"$\Delta S_q$ [$\mu$m]"),
    ("sz_robust_99p5_0p5_um", r"$S_z^{0.5-99.5}$ [$\mu$m]"),
    ("ssk", r"$S_{sk}$"),
    ("sku", r"$S_{ku}$"),
]

n_metrics = len(metric_plot_specs)
ncols = 2
nrows = int(np.ceil(n_metrics / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 3.0 * nrows), squeeze=False)

for ax in axes.ravel():
    ax.axis("off")

for ax, (col, ylabel) in zip(axes.ravel(), metric_plot_specs):
    ax.axis("on")
    plot_mean_by_time_or_strain(
        ax,
        point_df,
        y_col=col,
        x_col="bulk_z_strain_percent",
        plot_individual_points=False,
    )
    ax.set_xlabel(STRAIN_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=6, ncol=2)

fig.tight_layout()
savefig(fig, "10_surface_parameter_suite_vs_plastic_strain.png")
plt.show()


# =============================================================================
# 11: Height distributions
# =============================================================================

if len(representative_df) > 0:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for _, rec in representative_df.iterrows():
        H = load_cropped_height_from_record(rec)
        H0 = H - np.nanmean(H)
        values = H0.ravel()
        values = values[np.isfinite(values)]

        lo, hi = np.percentile(values, [0.5, 99.5])
        values_clip = values[(values >= lo) & (values <= hi)]

        ax.hist(
            values_clip,
            bins=90,
            density=True,
            histtype="step",
            lw=1.6,
            label=(
                f"{rec['load_mpa']} MPa, "
                rf"$\epsilon_p$={rec['bulk_z_strain_percent']:.2f}%"
            ),
        )

    ax.set_xlabel(r"leveled height, $h-\bar{h}$ [$\mu$m]")
    ax.set_ylabel("probability density")
    ax.set_title("Height distributions at representative strain states")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "11_height_distributions_representative_strains.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for _, rec in representative_df.iterrows():
        H = load_cropped_height_from_record(rec)
        H0 = H - np.nanmean(H)
        sq = np.sqrt(np.nanmean(H0**2))

        if sq <= 0:
            continue

        values = (H0 / sq).ravel()
        values = values[np.isfinite(values)]
        values = values[
            (values >= np.percentile(values, 0.5))
            & (values <= np.percentile(values, 99.5))
        ]

        ax.hist(
            values,
            bins=90,
            density=True,
            histtype="step",
            lw=1.6,
            label=(
                f"{rec['load_mpa']} MPa, "
                rf"$\epsilon_p$={rec['bulk_z_strain_percent']:.2f}%"
            ),
        )

    ax.set_xlabel(r"standardized height, $(h-\bar{h})/S_q$")
    ax.set_ylabel("probability density")
    ax.set_title("Standardized height distributions")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "11_standardized_height_distributions_representative_strains.png")
    plt.show()


# =============================================================================
# 13: PSD band powers
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    nz, ny = shape
    wz = np.hanning(nz)
    wy = np.hanning(ny)
    W = wz[:, None] * wy[None, :]
    rms = np.sqrt(np.mean(W**2))

    if rms > 0:
        W = W / rms

    return W


def psd2d(
    height_um: np.ndarray, spacing_um: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Returns:
      fz, fy, PSD2D, dfz, dfy

    Normalization is chosen so that:
      sum(PSD2D) * dfz * dfy ~= mean((windowed height)^2)
    """
    H = np.asarray(height_um, dtype=float)
    H0 = H - np.nanmean(H)

    nz, ny = H0.shape
    W = hann2d(H0.shape)
    Hw = H0 * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um * spacing_um
    n_pixels = nz * ny

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    fz = np.fft.fftfreq(nz, d=spacing_um)
    fy = np.fft.fftfreq(ny, d=spacing_um)

    dfz = 1.0 / (nz * spacing_um)
    dfy = 1.0 / (ny * spacing_um)

    return fz, fy, PSD, dfz, dfy


def make_psd_bands_um(
    d_ref_um: float, spacing_um: float, map_shape: tuple[int, int]
) -> dict[str, tuple[float, float]]:
    nz, ny = map_shape
    min_map_length_um = min(nz, ny) * spacing_um
    shortest_resolved_um = 2.0 * spacing_um

    if PSD_BAND_MODE == "fixed" or not np.isfinite(d_ref_um) or d_ref_um <= 0:
        bands = FIXED_WAVELENGTH_BANDS_UM.copy()
    else:
        d = d_ref_um
        bands = {
            "subgrain": (shortest_resolved_um, 0.5 * d),
            "grain_scale": (0.5 * d, 2.0 * d),
            "mesoscale_2d_10d": (2.0 * d, 10.0 * d),
            "long_wavelength_over_10d": (10.0 * d, np.inf),
        }

    cleaned = {}

    for name, (lam_min, lam_max) in bands.items():
        lam_min_eff = max(float(lam_min), shortest_resolved_um)

        if np.isinf(lam_max):
            lam_max_eff = min_map_length_um
        else:
            lam_max_eff = min(float(lam_max), min_map_length_um)

        if PSD_MAX_WAVELENGTH_CAP_UM is not None:
            lam_max_eff = min(lam_max_eff, float(PSD_MAX_WAVELENGTH_CAP_UM))

        if lam_max_eff > lam_min_eff:
            cleaned[name] = (lam_min_eff, lam_max_eff)

    return cleaned


def psd_band_powers(
    height_um: np.ndarray, spacing_um: float, bands_um: dict[str, tuple[float, float]]
) -> dict:
    fz, fy, PSD, dfz, dfy = psd2d(height_um, spacing_um)

    FY, FZ = np.meshgrid(fy, fz)
    FR = np.sqrt(FZ**2 + FY**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / FR

    out = {}

    for band_name, (lam_min, lam_max) in bands_um.items():
        mask = (
            np.isfinite(wavelength)
            & (FR > 0)
            & (wavelength >= lam_min)
            & (wavelength < lam_max)
        )

        power = float(np.nansum(PSD[mask]) * dfz * dfy)
        out[f"psd_power_{band_name}_um2"] = power
        out[f"psd_rms_{band_name}_um"] = np.sqrt(power) if power >= 0 else np.nan
        out[f"psd_band_lambda_min_{band_name}_um"] = lam_min
        out[f"psd_band_lambda_max_{band_name}_um"] = lam_max
        out[f"psd_modes_{band_name}"] = int(np.count_nonzero(mask))

    return out


if RUN_FULL_PSD_BAND_ANALYSIS:
    # Use first available height shape to define bands.
    first_H = load_cropped_height_from_record(point_df.iloc[0])
    psd_bands_um = make_psd_bands_um(d_ref_um, spacing_um, first_H.shape)

    print()
    print("PSD wavelength bands [um]:")
    for name, band in psd_bands_um.items():
        print(f"  {name}: {band[0]:.3g} to {band[1]:.3g} um")

    psd_rows = []

    for idx, rec in point_df.iterrows():
        try:
            H = load_cropped_height_from_record(rec)
            powers = psd_band_powers(H, spacing_um, psd_bands_um)
            psd_rows.append(
                {
                    "row_index": idx,
                    "load_mpa": rec["load_mpa"],
                    "sample_type": rec["sample_type"],
                    "sample": rec["sample"],
                    "time_h": rec["time_h"],
                    "bulk_z_strain": rec["bulk_z_strain"],
                    "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                    "delta_sa_um": rec["delta_sa_um"],
                    **powers,
                }
            )
        except Exception as exc:
            warnings.warn(f"PSD band power failed for {rec['height_path']}: {exc}")

    psd_band_df = pd.DataFrame(psd_rows)
    psd_band_df.to_csv(OUTPUT_DIR / "13_psd_band_powers.csv", index=False)

    # Add per-sample PSD-power increments relative to initial.
    for band_name in psd_bands_um:
        power_col = f"psd_power_{band_name}_um2"
        rms_col = f"psd_rms_{band_name}_um"

        psd_band_df[f"delta_{power_col}"] = np.nan
        psd_band_df[f"gain_{power_col}"] = np.nan
        psd_band_df[f"delta_{rms_col}"] = np.nan

        for key, g in psd_band_df.groupby(["load_mpa", "sample_type", "sample"]):
            g = g.sort_values("time_h")
            idxs = g.index.to_numpy()

            p0 = float(g[power_col].iloc[0])
            r0 = float(g[rms_col].iloc[0])

            psd_band_df.loc[idxs, f"delta_{power_col}"] = (
                g[power_col].to_numpy(dtype=float) - p0
            )
            psd_band_df.loc[idxs, f"gain_{power_col}"] = (
                g[power_col].to_numpy(dtype=float) / p0 if p0 > 0 else np.nan
            )
            psd_band_df.loc[idxs, f"delta_{rms_col}"] = (
                g[rms_col].to_numpy(dtype=float) - r0
            )

    psd_band_df.to_csv(
        OUTPUT_DIR / "13_psd_band_powers_with_increments.csv", index=False
    )

    band_names = list(psd_bands_um.keys())
    n_bands = len(band_names)

    fig, axes = plt.subplots(n_bands, 1, figsize=(6.8, 3.2 * n_bands), squeeze=False)

    for ax, band_name in zip(axes.ravel(), band_names):
        col = f"delta_psd_rms_{band_name}_um"
        plot_mean_by_time_or_strain(
            ax,
            psd_band_df,
            y_col=col,
            x_col="bulk_z_strain_percent",
            plot_individual_points=True,
        )
        lam_min, lam_max = psd_bands_um[band_name]
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xlabel(STRAIN_LABEL)
        ax.set_ylabel(r"$\Delta \sqrt{P_{\mathrm{band}}}$ [$\mu$m]")
        ax.set_title(
            f"PSD band roughness amplitude: {band_name}, "
            rf"$\lambda \in [{lam_min:.2g}, {lam_max:.2g}]$ $\mu$m"
        )
        ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    savefig(fig, "13_psd_band_rms_increment_vs_strain.png")
    plt.show()

    fig, axes = plt.subplots(n_bands, 1, figsize=(6.8, 3.2 * n_bands), squeeze=False)

    for ax, band_name in zip(axes.ravel(), band_names):
        col = f"gain_psd_power_{band_name}_um2"
        plot_mean_by_time_or_strain(
            ax,
            psd_band_df,
            y_col=col,
            x_col="bulk_z_strain_percent",
            plot_individual_points=True,
        )
        ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
        ax.set_xlabel(STRAIN_LABEL)
        ax.set_ylabel(r"PSD power gain, $P/P_0$")
        ax.set_yscale("log")
        lam_min, lam_max = psd_bands_um[band_name]
        ax.set_title(
            f"PSD band gain: {band_name}, "
            rf"$\lambda \in [{lam_min:.2g}, {lam_max:.2g}]$ $\mu$m"
        )
        ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    savefig(fig, "13_psd_band_power_gain_vs_strain.png")
    plt.show()


# =============================================================================
# 14: Grain-size-normalized roughening
# =============================================================================

if np.isfinite(d_ref_um) and d_ref_um > 0:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    plot_mean_by_time_or_strain(
        ax,
        point_df,
        y_col="delta_sa_over_d_ref",
        x_col="bulk_z_strain_percent",
        plot_individual_points=True,
    )
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel(STRAIN_LABEL)
    ax.set_ylabel(r"$\Delta S_a / d_{\mathrm{ref}}$")
    ax.set_title(
        rf"Grain-size-normalized roughening, "
        rf"$d_{{\mathrm{{ref}}}}={d_ref_um:.2f}\,\mu$m"
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    savefig(fig, "14_delta_sa_normalized_by_representative_grain_size.png")
    plt.show()

    # Also save a short text note clarifying interpretation.
    with open(OUTPUT_DIR / "14_grain_size_normalization_note.txt", "w") as f:
        f.write(
            "Grain-size normalization uses one representative EBSD-derived grain size, "
            f"d_ref = {d_ref_um:.6g} um, source = {d_ref_source}.\n"
            "This is acceptable as a nondimensionalization if all tested specimens "
            "are from the same material condition and are expected to share the same "
            "grain-size distribution. It should not be described as specimen-resolved "
            "grain-size normalization unless each tested specimen has its own EBSD map.\n"
        )


# =============================================================================
# 15: Optional local Sa vs local DIC strain analysis
# =============================================================================


def is_blank_line(line: str) -> bool:
    if not line.strip():
        return True

    cells = line.rstrip(" \n\r").split(",")
    return all(cell.strip().strip('"') == "" for cell in cells)


def read_dic_blocks(
    filename: str | Path,
    *,
    start_row: int,
    n_datasets: int,
) -> list[pd.DataFrame]:
    filename = Path(filename)
    skip_before = start_row - 1

    with filename.open("r", newline="") as f:
        lines = f.readlines()

    lines = lines[skip_before:]

    blocks = []
    current_block = []

    for line in lines:
        if is_blank_line(line):
            if current_block:
                blocks.append(current_block)
                current_block = []

            if len(blocks) == n_datasets:
                break
        else:
            current_block.append(line)

    if current_block and len(blocks) < n_datasets:
        blocks.append(current_block)

    from io import StringIO

    datasets = []

    for block in blocks:
        df = pd.read_csv(StringIO("".join(block)), skipinitialspace=True)
        df.columns = df.columns.str.strip().str.strip('"').str.strip()
        datasets.append(df)

    return datasets


def dic_strain_from_df(df: pd.DataFrame) -> dict:
    """
    Assumes:
      x_c -> z position
      y_c -> y position
      u_c -> z displacement
      v_c -> y displacement

    Returns small-strain components on a regular grid.
    """
    df_grid = df.sort_values(["y_c", "x_c"]).copy()

    z_vals = np.sort(df_grid["x_c"].unique())
    y_vals = np.sort(df_grid["y_c"].unique())

    nz = len(z_vals)
    ny = len(y_vals)

    if len(df_grid) != ny * nz:
        raise ValueError(
            f"DIC data are not a complete grid: len={len(df_grid)}, ny*nz={ny*nz}"
        )

    Z = df_grid["x_c"].to_numpy(dtype=float).reshape(ny, nz)
    Y = df_grid["y_c"].to_numpy(dtype=float).reshape(ny, nz)
    U = df_grid["u_c"].to_numpy(dtype=float).reshape(ny, nz)
    V = df_grid["v_c"].to_numpy(dtype=float).reshape(ny, nz)

    if "sigma" in df_grid.columns:
        S = df_grid["sigma"].to_numpy(dtype=float).reshape(ny, nz)
    else:
        S = np.zeros_like(U)

    z = Z[0, :]
    y = Y[:, 0]

    dU_dy, dU_dz = np.gradient(U, y, z, edge_order=2)
    dV_dy, dV_dz = np.gradient(V, y, z, edge_order=2)

    eps_zz = dU_dz
    eps_yy = dV_dy
    gamma_yz = dU_dy + dV_dz

    invalid = S == -1

    eps_zz = np.where(invalid, np.nan, eps_zz)
    eps_yy = np.where(invalid, np.nan, eps_yy)
    gamma_yz = np.where(invalid, np.nan, gamma_yz)

    return {
        "Y": Y,
        "Z": Z,
        "U": U,
        "V": V,
        "sigma": S,
        "eps_zz": eps_zz,
        "eps_yy": eps_yy,
        "gamma_yz": gamma_yz,
    }


def local_sa_profile_z(height_um: np.ndarray, n_bins: int = 40) -> pd.DataFrame:
    """
    Calculates local Sa in axial bins:
      Sa(z_bin) = mean_{pixels in bin} |h - mean(h_bin)|.
    """
    H = np.asarray(height_um, dtype=float)
    nz, ny = H.shape

    z_um = np.arange(nz) * spacing_um
    edges = np.linspace(z_um.min(), z_um.max(), n_bins + 1)

    rows = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask_z = (z_um >= lo) & (z_um < hi if i < n_bins - 1 else z_um <= hi)

        if not np.any(mask_z):
            continue

        block = H[mask_z, :]
        block_centered = block - np.nanmean(block)

        rows.append(
            {
                "z_center_um": 0.5 * (lo + hi),
                "z_norm": (0.5 * (lo + hi) - z_um.min()) / (z_um.max() - z_um.min()),
                "local_sa_um": np.nanmean(np.abs(block_centered)),
                "local_sq_um": np.sqrt(np.nanmean(block_centered**2)),
                "n_pixels": int(np.count_nonzero(np.isfinite(block))),
            }
        )

    return pd.DataFrame(rows)


def dic_axial_strain_profile_z(dic_result: dict) -> pd.DataFrame:
    Z = dic_result["Z"]
    E = dic_result["eps_zz"]

    z = Z[0, :]
    e_profile = np.nanmean(E, axis=0)

    return pd.DataFrame(
        {
            "z_dic": z,
            "z_norm": (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z)),
            "eps_zz_mean": e_profile,
        }
    )


def plot_local_sa_vs_dic_strain(
    height_path: str | Path,
    dic_df: pd.DataFrame,
    *,
    n_sa_bins: int = 40,
    title: str = "local Sa vs local DIC strain",
):
    H = read_height(
        Path(height_path),
        level=level,
        spacing_um=spacing_um,
        detrend_order=detrend_order,
    )
    H = H[crop]

    sa_prof = local_sa_profile_z(H, n_bins=n_sa_bins)
    dic_res = dic_strain_from_df(dic_df)
    eps_prof = dic_axial_strain_profile_z(dic_res)

    interp_eps = interp1d(
        eps_prof["z_norm"],
        eps_prof["eps_zz_mean"],
        bounds_error=False,
        fill_value=np.nan,
    )

    sa_prof["eps_zz_interp"] = interp_eps(sa_prof["z_norm"])

    valid = np.isfinite(sa_prof["local_sa_um"]) & np.isfinite(sa_prof["eps_zz_interp"])

    corr = (
        np.corrcoef(
            sa_prof.loc[valid, "local_sa_um"],
            sa_prof.loc[valid, "eps_zz_interp"],
        )[0, 1]
        if np.count_nonzero(valid) >= 3
        else np.nan
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    ax = axes[0]
    ax.plot(sa_prof["z_norm"], sa_prof["local_sa_um"], "k-o", ms=3)
    ax.set_xlabel("normalized axial coordinate")
    ax.set_ylabel(r"local $S_a$ [$\mu$m]")
    ax.set_title("local roughness profile")

    ax = axes[1]
    ax.plot(eps_prof["z_norm"], eps_prof["eps_zz_mean"], "r-o", ms=3)
    ax.set_xlabel("normalized axial coordinate")
    ax.set_ylabel(r"local $\epsilon_{zz}$")
    ax.set_title("DIC axial strain profile")

    ax = axes[2]
    ax.scatter(
        sa_prof.loc[valid, "eps_zz_interp"],
        sa_prof.loc[valid, "local_sa_um"],
        c=sa_prof.loc[valid, "z_norm"],
        cmap="viridis",
    )
    ax.set_xlabel(r"local $\epsilon_{zz}$")
    ax.set_ylabel(r"local $S_a$ [$\mu$m]")
    ax.set_title(rf"local correlation, $\rho={corr:.2f}$")
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("normalized z")

    fig.suptitle(title, y=1.03)
    fig.tight_layout()

    return fig, axes, sa_prof, eps_prof, corr


if DIC_CSV_PATH is not None and HEIGHT_FOR_DIC_PATH is not None:
    dic_datasets = read_dic_blocks(
        DIC_CSV_PATH,
        start_row=DIC_START_ROW,
        n_datasets=DIC_N_DATASETS,
    )

    for i, dic_df in enumerate(dic_datasets):
        fig, axes, sa_prof, eps_prof, corr = plot_local_sa_vs_dic_strain(
            HEIGHT_FOR_DIC_PATH,
            dic_df,
            n_sa_bins=40,
            title=f"local Sa vs local DIC strain, DIC dataset {i}",
        )
        savefig(fig, f"15_local_sa_vs_dic_strain_dataset_{i}.png")
        plt.show()

        sa_prof.to_csv(OUTPUT_DIR / f"15_local_sa_profile_dataset_{i}.csv", index=False)
        eps_prof.to_csv(
            OUTPUT_DIR / f"15_dic_axial_strain_profile_dataset_{i}.csv", index=False
        )


# =============================================================================
# Final compact summary table for manuscript
# =============================================================================

manuscript_summary = (
    point_df.groupby(["load_mpa", "sample_type", "time_h"], as_index=False)
    .agg(
        n=("delta_sa_um", "count"),
        mean_plastic_strain_percent=("bulk_z_strain_percent", "mean"),
        std_plastic_strain_percent=("bulk_z_strain_percent", "std"),
        mean_sa_um=("sa_um", "mean"),
        std_sa_um=("sa_um", "std"),
        mean_delta_sa_um=("delta_sa_um", "mean"),
        std_delta_sa_um=("delta_sa_um", "std"),
        mean_sq_um=("sq_um", "mean"),
        std_sq_um=("sq_um", "std"),
        mean_szk_um=("sz_robust_99p5_0p5_um", "mean"),
        std_szk_um=("sz_robust_99p5_0p5_um", "std"),
        mean_ssk=("ssk", "mean"),
        std_ssk=("ssk", "std"),
        mean_sku=("sku", "mean"),
        std_sku=("sku", "std"),
        mean_profile_anisotropy=(
            "profile_ra_anisotropy_parallel_over_transverse",
            "mean",
        ),
        std_profile_anisotropy=(
            "profile_ra_anisotropy_parallel_over_transverse",
            "std",
        ),
    )
    .sort_values(["sample_type", "load_mpa", "time_h"])
)

manuscript_summary.to_csv(
    OUTPUT_DIR / "manuscript_summary_by_load_time.csv", index=False
)

print()
print("Saved manuscript summary:")
print(OUTPUT_DIR / "manuscript_summary_by_load_time.csv")

print()
print("Analysis complete. Output directory:")
print(OUTPUT_DIR.resolve())

# %%
# =============================================================================
# Four-panel PSD figure:
#
#   A. Radial PSD at selected plastic-strain states
#   B. PSD gain relative to initial condition
#   C. Band-limited RMS roughness amplitude increments vs plastic strain
#   D. Fraction of total PSD power in each wavelength band vs plastic strain
#
# Excludes 475 MPa sub-yield load.
#
# Requirements:
#   - This script can use point_df if it already exists in memory.
#   - Otherwise it tries to load:
#       roughness_strain_publication_figures/roughness_plastic_strain_point_table.csv
#   - point_df must contain:
#       load_mpa, sample_type, sample, time_h, bulk_z_strain_percent,
#       delta_sa_um, height_path
#
# Assumptions:
#   - Height maps are in CSV format with 19 header rows, as in your notebook.
#   - Cropped height map axes are:
#       axis 0 = z/loading direction
#       axis 1 = y/transverse direction
#   - Plastic strain is already in point_df["bulk_z_strain_percent"].
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import distance_transform_edt

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"
MICRO_STATS_CSV = Path(
    "/Users/gtdebru/mimosa/data/EBSD-AM316L/merged_stats/merged_feature_stats_all.csv"
)

EXCLUDED_LOADS = {475}

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))
level = True
detrend_order = 1

# Maximum number of maps used per strain group for panels A/B.
# Increase or set to None if you want every map included.
MAX_MAPS_PER_STRAIN_GROUP = 12

# Maximum number of maps used per load/time state for panels C/D.
# This keeps the PSD-band calculation manageable while preserving load/time trends.
# Set to None to use all maps.
MAX_MAPS_PER_LOAD_TIME = None

# Long-wavelength cap for PSD bands.
# Wavelengths approaching the full field of view are often dominated by specimen-scale
# shape or deformation rather than microstructural roughness.
MAX_WAVELENGTH_FRACTION_OF_SHORT_FOV = 1.0 / 3.0

# Number of radial PSD bins for panels A/B.
N_RADIAL_BINS = 70

# Number of strain bins for panels C/D.
N_STRAIN_BINS_FOR_BAND_SUMMARY = 8

# If you want to override the representative grain size from EBSD, set this to a number.
# Otherwise, the script estimates d_ref from merged_feature_stats_all.csv.
GRAIN_SIZE_UM_OVERRIDE = None

# Plot settings
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

BAND_COLORS = {
    "subgrain": "tab:blue",
    "grain_scale": "tab:orange",
    "mesoscale": "tab:green",
    "long_wavelength": "tab:red",
}

STRAIN_LABEL = r"bulk axial plastic strain, $\epsilon_p$ (%)"


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist. "
            "Run the roughness table construction first, or update POINT_TABLE_PATH."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_cols = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "delta_sa_um",
    "height_path",
}

missing = required_cols - set(point_df.columns)
if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()
df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"], errors="coerce"
)
df["delta_sa_um"] = pd.to_numeric(df["delta_sa_um"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["bulk_z_strain_percent"])
    & np.isfinite(df["time_h"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()

if df.empty:
    raise ValueError("No data remain after excluding the specified loads.")

df = df.sort_values(["load_mpa", "sample_type", "sample", "time_h"]).reset_index(
    drop=True
)

print("Using records:")
print(df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height-map loading
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape, spacing_um_value, order):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(values, spacing_um_value, order=1):
    values = np.asarray(values, dtype=np.float64)
    design, inverse = _detrend_geometry(
        values.shape, float(spacing_um_value), int(order)
    )
    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path):
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(path, *, level=True, spacing_um_value=spacing_um, detrend_order=1):
    path = Path(path)
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_cropped_height_from_path(path):
    H = read_height(
        Path(path),
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )
    H = H[crop]
    H = H - np.nanmean(H)
    return H


# =============================================================================
# Representative grain size and PSD bands
# =============================================================================


def representative_grain_size_from_feature_stats(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        warnings.warn(f"Could not find {csv_path}; using fixed wavelength bands.")
        return np.nan, "not_available"

    g = pd.read_csv(csv_path)

    if "EquivalentDiameters" not in g.columns:
        warnings.warn(
            "EquivalentDiameters column not found; using fixed wavelength bands."
        )
        return np.nan, "not_available"

    if "complete_non_surface" in g.columns:
        mask = g["complete_non_surface"].astype(str).str.lower().eq("true")
        g = g[mask].copy()

    d = pd.to_numeric(g["EquivalentDiameters"], errors="coerce").to_numpy(dtype=float)
    d = d[np.isfinite(d) & (d > 0)]

    if d.size == 0:
        return np.nan, "not_available"

    return float(np.median(d)), "median EquivalentDiameters, complete_non_surface"


if GRAIN_SIZE_UM_OVERRIDE is not None:
    d_ref_um = float(GRAIN_SIZE_UM_OVERRIDE)
    d_ref_source = "manual override"
else:
    d_ref_um, d_ref_source = representative_grain_size_from_feature_stats(
        MICRO_STATS_CSV
    )

first_H = load_cropped_height_from_path(df.iloc[0]["height_path"])
nz, ny = first_H.shape
Lz_um = nz * spacing_um
Ly_um = ny * spacing_um
short_fov_um = min(Lz_um, Ly_um)
long_wavelength_cap_um = MAX_WAVELENGTH_FRACTION_OF_SHORT_FOV * short_fov_um

nyquist_shortest_wavelength_um = 2.0 * spacing_um

if np.isfinite(d_ref_um) and d_ref_um > 0:
    raw_bands = {
        "subgrain": (nyquist_shortest_wavelength_um, 0.5 * d_ref_um),
        "grain_scale": (0.5 * d_ref_um, 2.0 * d_ref_um),
        "mesoscale": (2.0 * d_ref_um, 10.0 * d_ref_um),
        "long_wavelength": (10.0 * d_ref_um, long_wavelength_cap_um),
    }
else:
    raw_bands = {
        "subgrain": (nyquist_shortest_wavelength_um, 10.0),
        "grain_scale": (10.0, 50.0),
        "mesoscale": (50.0, 200.0),
        "long_wavelength": (200.0, long_wavelength_cap_um),
    }

bands_um = {}

for name, (lam_min, lam_max) in raw_bands.items():
    lam_min = max(float(lam_min), nyquist_shortest_wavelength_um)
    lam_max = min(float(lam_max), long_wavelength_cap_um)

    if lam_max > lam_min:
        bands_um[name] = (lam_min, lam_max)

print()
print(f"Representative grain size d_ref = {d_ref_um:.4g} um ({d_ref_source})")
print(f"Map size: Lz = {Lz_um:.1f} um, Ly = {Ly_um:.1f} um")
print(f"Long-wavelength cap = {long_wavelength_cap_um:.1f} um")
print("PSD bands:")
for name, (lo, hi) in bands_um.items():
    print(f"  {name:16s}: {lo:8.3g} to {hi:8.3g} um")


# =============================================================================
# PSD functions
# =============================================================================


def hann2d(shape):
    nz, ny = shape
    wz = np.hanning(nz)
    wy = np.hanning(ny)
    W = wz[:, None] * wy[None, :]

    rms = np.sqrt(np.mean(W**2))
    if rms > 0:
        W = W / rms

    return W


def psd2d(height_um, spacing_um_value):
    """
    Returns fz, fy, PSD, dfz, dfy.

    PSD normalization is chosen such that:
        sum(PSD) * dfz * dfy ~= mean((windowed height)^2)
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    nz, ny = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = nz * ny

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    fz = np.fft.fftfreq(nz, d=spacing_um_value)
    fy = np.fft.fftfreq(ny, d=spacing_um_value)

    dfz = 1.0 / (nz * spacing_um_value)
    dfy = 1.0 / (ny * spacing_um_value)

    return fz, fy, PSD, dfz, dfy


def radial_bin_edges_for_shape(shape, spacing_um_value, n_bins=70):
    nz, ny = shape

    fz = np.fft.fftfreq(nz, d=spacing_um_value)
    fy = np.fft.fftfreq(ny, d=spacing_um_value)

    FY, FZ = np.meshgrid(fy, fz)
    FR = np.sqrt(FZ**2 + FY**2)

    positive = FR[(FR > 0) & np.isfinite(FR)]

    f_min = positive.min()
    f_max = min(0.5 / spacing_um_value, positive.max())

    edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)

    return edges


def radial_psd(height_um, spacing_um_value, bin_edges):
    fz, fy, PSD, dfz, dfy = psd2d(height_um, spacing_um_value)

    FY, FZ = np.meshgrid(fy, fz)
    FR = np.sqrt(FZ**2 + FY**2)

    f_flat = FR.ravel()
    p_flat = PSD.ravel()

    nbins = len(bin_edges) - 1
    bin_index = np.searchsorted(bin_edges, f_flat, side="right") - 1

    radial_frequency = np.full(nbins, np.nan)
    radial_psd_value = np.full(nbins, np.nan)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        mask = (
            (bin_index == i) & np.isfinite(f_flat) & np.isfinite(p_flat) & (f_flat > 0)
        )

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] > 0:
            radial_frequency[i] = np.mean(f_flat[mask])
            radial_psd_value[i] = np.mean(p_flat[mask])

    wavelength_um = 1.0 / radial_frequency

    return radial_frequency, wavelength_um, radial_psd_value, modes


def psd_band_powers(height_um, spacing_um_value, bands):
    fz, fy, PSD, dfz, dfy = psd2d(height_um, spacing_um_value)

    FY, FZ = np.meshgrid(fy, fz)
    FR = np.sqrt(FZ**2 + FY**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength_um = 1.0 / FR

    out = {}

    for band_name, (lam_min, lam_max) in bands.items():
        mask = (
            np.isfinite(wavelength_um)
            & (FR > 0)
            & (wavelength_um >= lam_min)
            & (wavelength_um < lam_max)
        )

        power = float(np.nansum(PSD[mask]) * dfz * dfy)
        rms = np.sqrt(power) if power >= 0 else np.nan

        out[f"power_{band_name}_um2"] = power
        out[f"rms_{band_name}_um"] = rms
        out[f"modes_{band_name}"] = int(np.count_nonzero(mask))

    return out


# =============================================================================
# Select strain groups for panels A/B
# =============================================================================


def make_strain_groups_for_psd(input_df, n_positive_groups=3):
    """
    Initial group plus quantile-based positive-strain groups.
    """
    d = input_df.copy()
    d = d[np.isfinite(d["bulk_z_strain_percent"])].copy()

    initial = d[np.abs(d["bulk_z_strain_percent"]) <= 0.05].copy()
    positive = d[d["bulk_z_strain_percent"] > 0.05].copy()

    groups = []

    if not initial.empty:
        groups.append(
            {
                "name": "initial",
                "label": r"initial, $\epsilon_p \approx 0$",
                "df": initial,
                "color": "black",
            }
        )

    if positive.empty:
        return groups

    q_edges = np.unique(
        np.nanquantile(
            positive["bulk_z_strain_percent"].to_numpy(dtype=float),
            np.linspace(0.0, 1.0, n_positive_groups + 1),
        )
    )

    colors = ["tab:blue", "tab:orange", "tab:red"]

    for i in range(len(q_edges) - 1):
        lo = q_edges[i]
        hi = q_edges[i + 1]

        if i == len(q_edges) - 2:
            mask = (positive["bulk_z_strain_percent"] >= lo) & (
                positive["bulk_z_strain_percent"] <= hi
            )
        else:
            mask = (positive["bulk_z_strain_percent"] >= lo) & (
                positive["bulk_z_strain_percent"] < hi
            )

        g = positive[mask].copy()

        if g.empty:
            continue

        groups.append(
            {
                "name": f"group_{i+1}",
                "label": rf"$\epsilon_p$ = {lo:.2g}--{hi:.2g}%",
                "df": g,
                "color": colors[min(i, len(colors) - 1)],
            }
        )

    return groups


strain_groups = make_strain_groups_for_psd(df, n_positive_groups=3)

print()
print("Strain groups for radial PSD:")
for g in strain_groups:
    print(f"  {g['label']}: n = {len(g['df'])}")


def sample_group_records(group_df, max_records=None, seed=123):
    if max_records is None or len(group_df) <= max_records:
        return group_df.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(group_df.index.to_numpy(), size=max_records, replace=False)
    return group_df.loc[idx].copy()


# =============================================================================
# Compute group radial PSDs for panels A/B
# =============================================================================

bin_edges = radial_bin_edges_for_shape(first_H.shape, spacing_um, n_bins=N_RADIAL_BINS)

group_psd_results = []
baseline_psd_mean = None
baseline_wavelength = None

for group_idx, group in enumerate(strain_groups):
    gdf = sample_group_records(
        group["df"],
        max_records=MAX_MAPS_PER_STRAIN_GROUP,
        seed=123 + group_idx,
    )

    curves = []
    wavelength_ref = None

    for _, rec in gdf.iterrows():
        try:
            H = load_cropped_height_from_path(rec["height_path"])
            freq, wavelength, psd_curve, modes = radial_psd(H, spacing_um, bin_edges)

            valid = (
                np.isfinite(wavelength)
                & np.isfinite(psd_curve)
                & (wavelength > 0)
                & (psd_curve > 0)
                & (modes >= 8)
            )

            if wavelength_ref is None:
                wavelength_ref = wavelength

            curve = np.full_like(psd_curve, np.nan, dtype=float)
            curve[valid] = psd_curve[valid]
            curves.append(curve)

        except Exception as exc:
            warnings.warn(f"Radial PSD failed for {rec['height_path']}: {exc}")

    if len(curves) == 0:
        continue

    curves = np.vstack(curves)
    mean_curve = np.nanmean(curves, axis=0)
    std_curve = np.nanstd(curves, axis=0, ddof=1)

    group_psd_results.append(
        {
            "name": group["name"],
            "label": group["label"],
            "color": group["color"],
            "n": curves.shape[0],
            "wavelength_um": wavelength_ref,
            "mean_psd_um4": mean_curve,
            "std_psd_um4": std_curve,
            "curves": curves,
        }
    )

    if group["name"] == "initial":
        baseline_psd_mean = mean_curve
        baseline_wavelength = wavelength_ref

if baseline_psd_mean is None:
    raise ValueError("No initial PSD group was found. Cannot compute PSD gain.")


# =============================================================================
# Compute PSD band powers for panels C/D
# =============================================================================

band_rows = []

group_cols = ["load_mpa", "sample_type", "time_h"]

for group_key, g in df.groupby(group_cols):
    g_use = sample_group_records(
        g,
        max_records=MAX_MAPS_PER_LOAD_TIME,
        seed=int(abs(hash(str(group_key))) % (2**32)),
    )

    for _, rec in g_use.iterrows():
        try:
            H = load_cropped_height_from_path(rec["height_path"])
            powers = psd_band_powers(H, spacing_um, bands_um)

            band_rows.append(
                {
                    "load_mpa": rec["load_mpa"],
                    "sample_type": rec["sample_type"],
                    "sample": rec["sample"],
                    "time_h": rec["time_h"],
                    "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                    "delta_sa_um": rec["delta_sa_um"],
                    "height_path": rec["height_path"],
                    **powers,
                }
            )

        except Exception as exc:
            warnings.warn(
                f"PSD band calculation failed for {rec['height_path']}: {exc}"
            )

band_df = pd.DataFrame(band_rows)

if band_df.empty:
    raise ValueError("PSD band calculation produced no rows.")

# Add per-sample initial-referenced increments.
for band_name in bands_um:
    power_col = f"power_{band_name}_um2"
    rms_col = f"rms_{band_name}_um"

    band_df[f"delta_{power_col}"] = np.nan
    band_df[f"gain_{power_col}"] = np.nan
    band_df[f"delta_{rms_col}"] = np.nan

    for key, g in band_df.groupby(["load_mpa", "sample_type", "sample"]):
        g = g.sort_values("time_h")
        idx = g.index.to_numpy()

        p0 = float(g[power_col].iloc[0])
        r0 = float(g[rms_col].iloc[0])

        band_df.loc[idx, f"delta_{power_col}"] = g[power_col].to_numpy(dtype=float) - p0
        band_df.loc[idx, f"gain_{power_col}"] = (
            g[power_col].to_numpy(dtype=float) / p0 if p0 > 0 else np.nan
        )
        band_df.loc[idx, f"delta_{rms_col}"] = g[rms_col].to_numpy(dtype=float) - r0

power_cols = [f"power_{band_name}_um2" for band_name in bands_um]

band_df["total_band_power_um2"] = band_df[power_cols].sum(axis=1)

for band_name in bands_um:
    band_df[f"fraction_{band_name}"] = (
        band_df[f"power_{band_name}_um2"] / band_df["total_band_power_um2"]
    )

band_df.to_csv(
    OUTPUT_DIR / "panel_abcd_psd_band_table_excluding_475mpa.csv", index=False
)


def binned_band_summary(input_df, value_col, n_bins=8):
    d = input_df.copy()
    d = d[np.isfinite(d["bulk_z_strain_percent"]) & np.isfinite(d[value_col])].copy()

    if d.empty:
        return pd.DataFrame()

    x = d["bulk_z_strain_percent"].to_numpy(dtype=float)

    # Include zero as its own bin if present.
    zero_mask = np.abs(x) <= 0.05
    positive = d[~zero_mask].copy()

    rows = []

    if np.any(zero_mask):
        gz = d[zero_mask]
        y = gz[value_col].to_numpy(dtype=float)
        rows.append(
            {
                "x_mean": float(np.nanmean(gz["bulk_z_strain_percent"])),
                "x_min": float(np.nanmin(gz["bulk_z_strain_percent"])),
                "x_max": float(np.nanmax(gz["bulk_z_strain_percent"])),
                "y_mean": float(np.nanmean(y)),
                "y_std": (
                    float(np.nanstd(y, ddof=1))
                    if np.count_nonzero(np.isfinite(y)) > 1
                    else 0.0
                ),
                "n": int(np.count_nonzero(np.isfinite(y))),
            }
        )

    if not positive.empty:
        q = np.unique(
            np.nanquantile(
                positive["bulk_z_strain_percent"].to_numpy(dtype=float),
                np.linspace(0.0, 1.0, n_bins + 1),
            )
        )

        for i in range(len(q) - 1):
            lo = q[i]
            hi = q[i + 1]

            if i == len(q) - 2:
                mask = (positive["bulk_z_strain_percent"] >= lo) & (
                    positive["bulk_z_strain_percent"] <= hi
                )
            else:
                mask = (positive["bulk_z_strain_percent"] >= lo) & (
                    positive["bulk_z_strain_percent"] < hi
                )

            g = positive[mask]

            if g.empty:
                continue

            y = g[value_col].to_numpy(dtype=float)
            y = y[np.isfinite(y)]

            if y.size == 0:
                continue

            rows.append(
                {
                    "x_mean": float(np.nanmean(g["bulk_z_strain_percent"])),
                    "x_min": float(np.nanmin(g["bulk_z_strain_percent"])),
                    "x_max": float(np.nanmax(g["bulk_z_strain_percent"])),
                    "y_mean": float(np.nanmean(y)),
                    "y_std": float(np.nanstd(y, ddof=1)) if y.size > 1 else 0.0,
                    "n": int(y.size),
                }
            )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values("x_mean")

    return out


# =============================================================================
# Four-panel figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
axA, axB, axC, axD = axes.ravel()

# -----------------------------------------------------------------------------
# Panel A: radial PSD at selected strain states
# -----------------------------------------------------------------------------

for res in group_psd_results:
    wavelength = res["wavelength_um"]
    mean_psd = res["mean_psd_um4"]
    std_psd = res["std_psd_um4"]

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(mean_psd)
        & (wavelength > 0)
        & (mean_psd > 0)
    )

    order = np.argsort(wavelength[valid])
    lam = wavelength[valid][order]
    mean = mean_psd[valid][order]

    axA.loglog(
        lam,
        mean,
        color=res["color"],
        lw=2.0,
        label=f"{res['label']} (n={res['n']})",
    )

axA.axvline(
    d_ref_um, color="0.35", ls=":", lw=1.0, label=rf"$d_{{ref}}$={d_ref_um:.1f} $\mu$m"
)
axA.axvline(
    long_wavelength_cap_um, color="0.55", ls="--", lw=1.0, label=r"$L_{\min}/3$"
)
axA.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
axA.set_ylabel(r"radial PSD, $C_{\mathrm{iso}}$ [$\mu$m$^4$]")
axA.set_title("A. Radial PSD at selected strain states")
axA.legend(fontsize=6)

# -----------------------------------------------------------------------------
# Panel B: PSD gain relative to initial
# -----------------------------------------------------------------------------

for res in group_psd_results:
    if res["name"] == "initial":
        continue

    wavelength = res["wavelength_um"]
    mean_psd = res["mean_psd_um4"]

    gain = mean_psd / baseline_psd_mean

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(gain)
        & (wavelength > 0)
        & (gain > 0)
        & np.isfinite(baseline_psd_mean)
        & (baseline_psd_mean > 0)
    )

    order = np.argsort(wavelength[valid])
    lam = wavelength[valid][order]
    gain_plot = gain[valid][order]

    axB.loglog(
        lam,
        gain_plot,
        color=res["color"],
        lw=2.0,
        label=res["label"],
    )

axB.axhline(1.0, color="0.4", ls="--", lw=1.0)
axB.axvline(d_ref_um, color="0.35", ls=":", lw=1.0)
axB.axvline(long_wavelength_cap_um, color="0.55", ls="--", lw=1.0)
axB.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
axB.set_ylabel(r"PSD gain, $C(\epsilon_p)/C_0$")
axB.set_title("B. PSD gain relative to initial")
axB.legend(fontsize=6)

# -----------------------------------------------------------------------------
# Panel C: band-limited RMS amplitude increment vs plastic strain
# -----------------------------------------------------------------------------

for band_name, (lam_min, lam_max) in bands_um.items():
    col = f"delta_rms_{band_name}_um"
    summary = binned_band_summary(
        band_df,
        col,
        n_bins=N_STRAIN_BINS_FOR_BAND_SUMMARY,
    )

    if summary.empty:
        continue

    color = BAND_COLORS.get(band_name, None)

    axC.errorbar(
        summary["x_mean"],
        summary["y_mean"],
        yerr=summary["y_std"],
        fmt="o-",
        lw=1.8,
        ms=4,
        capsize=3,
        color=color,
        label=(f"{band_name}: " rf"{lam_min:.1f}$<\lambda<${lam_max:.1f} $\mu$m"),
    )

axC.axhline(0.0, color="0.4", lw=0.8)
axC.set_xlabel(STRAIN_LABEL)
axC.set_ylabel(r"band RMS increment, $\Delta A_{\mathrm{band}}$ [$\mu$m]")
axC.set_title("C. Band-limited roughness amplitude")
axC.legend(fontsize=6)

# -----------------------------------------------------------------------------
# Panel D: fraction of total band power vs plastic strain
# -----------------------------------------------------------------------------

for band_name, (lam_min, lam_max) in bands_um.items():
    col = f"fraction_{band_name}"
    summary = binned_band_summary(
        band_df,
        col,
        n_bins=N_STRAIN_BINS_FOR_BAND_SUMMARY,
    )

    if summary.empty:
        continue

    color = BAND_COLORS.get(band_name, None)

    axD.errorbar(
        summary["x_mean"],
        summary["y_mean"],
        yerr=summary["y_std"],
        fmt="o-",
        lw=1.8,
        ms=4,
        capsize=3,
        color=color,
        label=band_name,
    )

axD.set_xlabel(STRAIN_LABEL)
axD.set_ylabel(r"fraction of analyzed PSD power, $P_{\mathrm{band}}/\sum P_i$")
axD.set_ylim(0.0, 1.05)
axD.set_title("D. Redistribution of roughness power by scale")
axD.legend(fontsize=6)

# -----------------------------------------------------------------------------
# Shared formatting and save
# -----------------------------------------------------------------------------

for ax in axes.ravel():
    ax.grid(True, which="both", alpha=0.25)

fig.suptitle(
    "Scale-resolved surface roughening with plastic strain\n"
    "475 MPa sub-yield condition excluded",
    y=1.02,
)

fig.tight_layout()

outpath = OUTPUT_DIR / "Figure_PSD_A_D_excluding_475MPa.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Raw Delta h PSD analysis with NO registration correction
# =============================================================================
#
# This version assumes repeated profilometry maps are close enough spatially that
# direct subtraction is acceptable as a sensitivity analysis:
#
#     Delta h(x,y; eps_p) = h_i(x,y) - h_0(x,y)
#
# No translation correction.
# No rotation correction.
#
# The script:
#   1. Loads point_df or roughness_plastic_strain_point_table.csv.
#   2. Excludes 475 MPa.
#   3. For each sample, uses the earliest scan as h_0.
#   4. Computes raw signed Delta h for later scans.
#   5. Computes radial PSDs of Delta h.
#   6. Computes band-limited PSD powers/RMS amplitudes.
#   7. Produces one 4-panel figure:
#        A. Representative raw Delta h maps
#        B. Radial PSD of raw Delta h grouped by plastic strain
#        C. Band-limited RMS amplitude of raw Delta h vs plastic strain
#        D. Fraction of raw Delta h PSD power in each wavelength band
#
# Important:
#   - This is intentionally raw Delta h. If residual offsets/rotation are present,
#     short-wavelength PSD content may include misregistration artifacts.
#   - PSD is computed from signed Delta h, not |Delta h|.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from collections import Counter
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.ndimage import distance_transform_edt
from scipy.signal.windows import hann

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"
MICRO_STATS_CSV = Path(
    "/Users/gtdebru/mimosa/data/EBSD-AM316L/merged_stats/merged_feature_stats_all.csv"
)

EXCLUDED_LOADS = {475}

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))
level = True
detrend_order = 1

# Raw Delta h options.
# If maps are not perfectly overlapping, you may crop a small margin.
# Set to 0 for truly raw same-array subtraction.
RAW_DELTA_CROP_MARGIN_PIXELS = 0

# Plane-level Delta h after subtraction?
# This removes residual tilt between scans. If you want to preserve absolute
# long-wavelength shape changes, set this False.
LEVEL_DELTA_H = True
DELTA_H_DETREND_ORDER = 1

# PSD settings.
N_RADIAL_BINS = 80
MIN_MODES_PER_RADIAL_BIN = 8
N_POSITIVE_STRAIN_GROUPS = 4
MAX_CURVES_PER_GROUP = 25

# Grain-size / wavelength-band settings.
# If None, the script estimates d_ref from MICRO_STATS_CSV.
GRAIN_SIZE_UM_OVERRIDE = None

# d_ref method:
#   "area_weighted_mean" is recommended for PSD banding.
#   "mean" and "median" are also available.
D_REF_METHOD = "area_weighted_mean"

# Band multipliers relative to d_ref.
# Recommended given your median ESD ~4 um, mean ESD ~9.6 um:
# use a weighted/mean d_ref and allow grain band to extend to 3*d_ref.
BAND_DEFINITIONS_AS_MULTIPLES_OF_D = {
    "subgrain": (0.0, 0.5),
    "grain-scale": (0.5, 3.0),
    "mesoscale": (3.0, 10.0),
    "long-wavelength": (10.0, np.inf),
}

# Avoid interpreting wavelengths approaching the full field of view.
MAX_WAVELENGTH_FRACTION_OF_SHORT_FOV = 1.0 / 3.0

# Representative Delta h maps for panel A.
N_REPRESENTATIVE_MAPS = 4

# Plot style.
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

BAND_COLORS = {
    "subgrain": "tab:blue",
    "grain-scale": "tab:orange",
    "mesoscale": "tab:green",
    "long-wavelength": "tab:red",
}

STRAIN_LABEL = r"bulk axial plastic strain, $\epsilon_p$ (%)"


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist. "
            "Run the roughness table construction first or update POINT_TABLE_PATH."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing_columns = required_columns - set(point_df.columns)

if missing_columns:
    raise ValueError(f"point_df is missing required columns: {sorted(missing_columns)}")

df = point_df.copy()

df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"], errors="coerce"
)

if "delta_sa_um" in df.columns:
    df["delta_sa_um"] = pd.to_numeric(df["delta_sa_um"], errors="coerce")

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()
df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No data remain after filtering.")

print("Records after filtering:")
print(df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height-map reading and leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    design, inverse = _detrend_geometry(
        values.shape, float(spacing_um_value), int(order)
    )
    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_cropped_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )
    H = H[crop]
    H = H - np.nanmean(H)

    return H


def crop_margin(A: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return A

    if A.shape[0] <= 2 * margin or A.shape[1] <= 2 * margin:
        raise ValueError(f"Margin {margin} too large for shape {A.shape}")

    return A[margin:-margin, margin:-margin]


def compute_raw_delta_h(
    initial_path: str | Path, current_path: str | Path
) -> np.ndarray:
    H0 = load_cropped_height(initial_path)
    H1 = load_cropped_height(current_path)

    if H0.shape != H1.shape:
        raise ValueError(f"Shape mismatch: initial {H0.shape}, current {H1.shape}")

    H0 = crop_margin(H0, RAW_DELTA_CROP_MARGIN_PIXELS)
    H1 = crop_margin(H1, RAW_DELTA_CROP_MARGIN_PIXELS)

    delta_h = H1 - H0

    if LEVEL_DELTA_H:
        delta_h = detrend_surface(delta_h, spacing_um, order=DELTA_H_DETREND_ORDER)

    delta_h = delta_h - np.nanmean(delta_h)

    return delta_h


# =============================================================================
# Grain-size reference and bands
# =============================================================================


def representative_grain_size_um(
    csv_path: str | Path,
    method: str = "area_weighted_mean",
):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        warnings.warn(f"{csv_path} not found. Falling back to fixed d_ref = NaN.")
        return np.nan, "not_available"

    g = pd.read_csv(csv_path)

    if "EquivalentDiameters" not in g.columns:
        warnings.warn("EquivalentDiameters not found. Falling back to d_ref = NaN.")
        return np.nan, "not_available"

    if "complete_non_surface" in g.columns:
        mask = g["complete_non_surface"].astype(str).str.lower().eq("true")
        g = g[mask].copy()

    d = pd.to_numeric(g["EquivalentDiameters"], errors="coerce").to_numpy(dtype=float)

    if "SizeVolumes" in g.columns:
        weights = pd.to_numeric(g["SizeVolumes"], errors="coerce").to_numpy(dtype=float)
    elif "ShapeVolumes" in g.columns:
        weights = pd.to_numeric(g["ShapeVolumes"], errors="coerce").to_numpy(
            dtype=float
        )
    else:
        weights = np.ones_like(d)

    valid = np.isfinite(d) & (d > 0) & np.isfinite(weights) & (weights > 0)
    d = d[valid]
    weights = weights[valid]

    if d.size == 0:
        return np.nan, "not_available"

    method = method.lower()

    if method == "median":
        return float(np.median(d)), "number median ESD"
    elif method == "mean":
        return float(np.mean(d)), "number mean ESD"
    elif method == "area_weighted_mean":
        return (
            float(np.sum(weights * d) / np.sum(weights)),
            "SizeVolumes/ShapeVolumes-weighted mean ESD",
        )
    else:
        raise ValueError("method must be 'median', 'mean', or 'area_weighted_mean'")


def make_wavelength_bands(shape: tuple[int, int]):
    n0, n1 = shape
    short_fov_um = min(n0, n1) * spacing_um
    long_cap_um = MAX_WAVELENGTH_FRACTION_OF_SHORT_FOV * short_fov_um
    nyquist_um = 2.0 * spacing_um

    if GRAIN_SIZE_UM_OVERRIDE is not None:
        d_ref = float(GRAIN_SIZE_UM_OVERRIDE)
        d_ref_source = "manual override"
    else:
        d_ref, d_ref_source = representative_grain_size_um(
            MICRO_STATS_CSV,
            method=D_REF_METHOD,
        )

    if not np.isfinite(d_ref) or d_ref <= 0:
        warnings.warn("No valid d_ref. Using fixed fallback bands.")
        raw_bands = {
            "subgrain": (nyquist_um, 10.0),
            "grain-scale": (10.0, 50.0),
            "mesoscale": (50.0, 200.0),
            "long-wavelength": (200.0, long_cap_um),
        }
    else:
        raw_bands = {}

        for name, (m0, m1) in BAND_DEFINITIONS_AS_MULTIPLES_OF_D.items():
            lo = nyquist_um if m0 <= 0 else m0 * d_ref
            hi = long_cap_um if np.isinf(m1) else m1 * d_ref
            raw_bands[name] = (lo, hi)

    bands = {}

    for name, (lo, hi) in raw_bands.items():
        lo = max(float(lo), nyquist_um)
        hi = min(float(hi), long_cap_um)

        if hi > lo:
            bands[name] = (lo, hi)

    return bands, d_ref, d_ref_source, long_cap_um, nyquist_um


# =============================================================================
# PSD utilities
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = hann(n0, sym=False)
    w1 = hann(n1, sym=False)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))

    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um: np.ndarray, spacing_um_value: float):
    """
    Returns f0, f1, PSD2D, df0, df1.

    PSD units are um^4 when height and spacing are in um.
    Normalization approximately satisfies:
        sum(PSD2D) * df0 * df1 = mean((windowed height)^2)
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    df0 = 1.0 / (n0 * spacing_um_value)
    df1 = 1.0 / (n1 * spacing_um_value)

    return f0, f1, PSD, df0, df1


def radial_bin_edges(shape: tuple[int, int], spacing_um_value: float, n_bins: int):
    n0, n1 = shape

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    positive = FR[(FR > 0) & np.isfinite(FR)]

    f_min = positive.min()
    f_max = min(0.5 / spacing_um_value, positive.max())

    return np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)


def radial_psd_from_height(
    height_um: np.ndarray,
    spacing_um_value: float,
    edges: np.ndarray,
    min_modes: int = MIN_MODES_PER_RADIAL_BIN,
):
    f0, f1, PSD, df0, df1 = psd2d_height(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    f_flat = FR.ravel()
    p_flat = PSD.ravel()

    nbins = len(edges) - 1
    bin_index = np.searchsorted(edges, f_flat, side="right") - 1

    freq = np.full(nbins, np.nan)
    psd_radial = np.full(nbins, np.nan)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        mask = (
            (bin_index == i) & np.isfinite(f_flat) & np.isfinite(p_flat) & (f_flat > 0)
        )

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            freq[i] = np.mean(f_flat[mask])
            psd_radial[i] = np.mean(p_flat[mask])

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / freq

    return freq, wavelength, psd_radial, modes


def band_powers_from_delta_h(delta_h: np.ndarray, bands: dict):
    f0, f1, PSD, df0, df1 = psd2d_height(delta_h, spacing_um)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / FR

    out = {}

    for name, (lo, hi) in bands.items():
        mask = (
            np.isfinite(wavelength) & (FR > 0) & (wavelength >= lo) & (wavelength < hi)
        )

        power = float(np.nansum(PSD[mask]) * df0 * df1)
        rms = np.sqrt(power) if power >= 0 else np.nan

        out[f"power_{name}_um2"] = power
        out[f"rms_{name}_um"] = rms
        out[f"modes_{name}"] = int(np.count_nonzero(mask))

    total_power = sum(out[f"power_{name}_um2"] for name in bands)

    for name in bands:
        out[f"fraction_{name}"] = (
            out[f"power_{name}_um2"] / total_power if total_power > 0 else np.nan
        )

    out["total_band_power_um2"] = total_power
    out["total_band_rms_um"] = np.sqrt(total_power) if total_power >= 0 else np.nan

    return out


# =============================================================================
# Compute raw Delta h maps, PSD curves, and band powers
# =============================================================================

delta_rows = []
radial_curve_records = []
delta_map_records = []

first_delta_h = None

for (sample_type, load, sample), g in df.groupby(["sample_type", "load_mpa", "sample"]):
    g = g.sort_values("time_h").copy()

    if len(g) < 2:
        continue

    initial = g.iloc[0]
    initial_time = float(initial["time_h"])
    initial_path = initial["height_path"]

    # Use only samples whose earliest scan is approximately t = 0.
    # Remove this condition if you want earliest available scan as reference.
    if initial_time > 1.0e-8:
        continue

    for _, current in g.iloc[1:].iterrows():
        current_path = current["height_path"]

        try:
            delta_h = compute_raw_delta_h(initial_path, current_path)

            if first_delta_h is None:
                first_delta_h = delta_h.copy()
                psd_edges = radial_bin_edges(
                    first_delta_h.shape, spacing_um, N_RADIAL_BINS
                )
                bands_um, d_ref_um, d_ref_source, long_cap_um, nyquist_um = (
                    make_wavelength_bands(first_delta_h.shape)
                )

                print()
                print(f"d_ref = {d_ref_um:.4g} um ({d_ref_source})")
                print(f"Nyquist shortest wavelength = {nyquist_um:.3g} um")
                print(f"Long-wavelength cap = {long_cap_um:.3g} um")
                print("Bands:")
                for band_name, (lo, hi) in bands_um.items():
                    print(f"  {band_name:16s}: {lo:8.3g} to {hi:8.3g} um")

            freq, wavelength, psd_radial, modes = radial_psd_from_height(
                delta_h,
                spacing_um,
                psd_edges,
                min_modes=MIN_MODES_PER_RADIAL_BIN,
            )

            powers = band_powers_from_delta_h(delta_h, bands_um)

            delta_sa_from_delta_h = float(
                np.nanmean(np.abs(delta_h - np.nanmean(delta_h)))
            )
            delta_sq_from_delta_h = float(
                np.sqrt(np.nanmean((delta_h - np.nanmean(delta_h)) ** 2))
            )

            row = {
                "load_mpa": load,
                "sample_type": sample_type,
                "sample": sample,
                "initial_time_h": initial_time,
                "time_h": float(current["time_h"]),
                "bulk_z_strain_percent": float(current["bulk_z_strain_percent"]),
                "height_path_initial": initial_path,
                "height_path_current": current_path,
                "raw_delta_h_sa_um": delta_sa_from_delta_h,
                "raw_delta_h_sq_um": delta_sq_from_delta_h,
                "raw_delta_h_shape_axis0": int(delta_h.shape[0]),
                "raw_delta_h_shape_axis1": int(delta_h.shape[1]),
                **powers,
            }

            if "delta_sa_um" in current.index:
                row["delta_sa_um"] = float(current["delta_sa_um"])

            delta_rows.append(row)

            radial_curve_records.append(
                {
                    "load_mpa": load,
                    "sample_type": sample_type,
                    "sample": sample,
                    "time_h": float(current["time_h"]),
                    "bulk_z_strain_percent": float(current["bulk_z_strain_percent"]),
                    "frequency_um_inv": freq,
                    "wavelength_um": wavelength,
                    "psd_um4": psd_radial,
                    "modes": modes,
                }
            )

            delta_map_records.append(
                {
                    "load_mpa": load,
                    "sample_type": sample_type,
                    "sample": sample,
                    "time_h": float(current["time_h"]),
                    "bulk_z_strain_percent": float(current["bulk_z_strain_percent"]),
                    "delta_h": delta_h,
                }
            )

        except Exception as exc:
            warnings.warn(
                f"Raw Delta h PSD failed for sample={sample}, load={load}, "
                f"time={current['time_h']}: {exc}"
            )

if first_delta_h is None or len(delta_rows) == 0:
    raise RuntimeError("No raw Delta h records were successfully computed.")

delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(OUTPUT_DIR / "raw_delta_h_psd_metrics_no_registration.csv", index=False)

print()
print("Raw Delta h table:")
print(delta_df.head())
print(f"Saved: {OUTPUT_DIR / 'raw_delta_h_psd_metrics_no_registration.csv'}")


# =============================================================================
# Long-form radial PSD table
# =============================================================================

curve_rows = []

for rec in radial_curve_records:
    wavelength = rec["wavelength_um"]
    frequency = rec["frequency_um_inv"]
    psd = rec["psd_um4"]
    modes = rec["modes"]

    for i in range(len(wavelength)):
        curve_rows.append(
            {
                "load_mpa": rec["load_mpa"],
                "sample_type": rec["sample_type"],
                "sample": rec["sample"],
                "time_h": rec["time_h"],
                "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                "frequency_um_inv": frequency[i],
                "wavelength_um": wavelength[i],
                "psd_raw_delta_h_um4": psd[i],
                "modes": modes[i],
            }
        )

radial_df = pd.DataFrame(curve_rows)
radial_df.to_csv(
    OUTPUT_DIR / "raw_delta_h_radial_psd_no_registration_long.csv", index=False
)

print(f"Saved: {OUTPUT_DIR / 'raw_delta_h_radial_psd_no_registration_long.csv'}")


# =============================================================================
# Helpers for plotting
# =============================================================================


def make_strain_groups_for_curves(input_df: pd.DataFrame, n_groups: int):
    d = input_df[np.isfinite(input_df["bulk_z_strain_percent"])].copy()

    q = np.unique(
        np.nanquantile(
            d["bulk_z_strain_percent"].to_numpy(dtype=float),
            np.linspace(0.0, 1.0, n_groups + 1),
        )
    )

    groups = []
    colors = mpl.colormaps["viridis"](np.linspace(0.12, 0.9, max(len(q) - 1, 1)))

    for i in range(len(q) - 1):
        lo = q[i]
        hi = q[i + 1]

        if i == len(q) - 2:
            mask = (d["bulk_z_strain_percent"] >= lo) & (
                d["bulk_z_strain_percent"] <= hi
            )
        else:
            mask = (d["bulk_z_strain_percent"] >= lo) & (
                d["bulk_z_strain_percent"] < hi
            )

        g = d[mask].copy()

        if g.empty:
            continue

        groups.append(
            {
                "lo": lo,
                "hi": hi,
                "label": rf"{lo:.2g}--{hi:.2g}% $\epsilon_p$",
                "df": g,
                "color": colors[i],
            }
        )

    return groups


def find_radial_curve(row):
    for rec in radial_curve_records:
        if (
            rec["load_mpa"] == row["load_mpa"]
            and rec["sample_type"] == row["sample_type"]
            and rec["sample"] == row["sample"]
            and np.isclose(rec["time_h"], row["time_h"])
        ):
            return rec
    return None


def sample_rows(input_df: pd.DataFrame, max_rows: int | None, seed: int):
    if max_rows is None or len(input_df) <= max_rows:
        return input_df.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(input_df.index.to_numpy(), size=max_rows, replace=False)

    return input_df.loc[idx].copy()


def summarize_by_strain_bins(input_df: pd.DataFrame, value_col: str, n_bins: int = 8):
    d = input_df[
        np.isfinite(input_df["bulk_z_strain_percent"])
        & np.isfinite(input_df[value_col])
    ].copy()

    if d.empty:
        return pd.DataFrame()

    q = np.unique(
        np.nanquantile(
            d["bulk_z_strain_percent"].to_numpy(dtype=float),
            np.linspace(0.0, 1.0, n_bins + 1),
        )
    )

    rows = []

    for i in range(len(q) - 1):
        lo = q[i]
        hi = q[i + 1]

        if i == len(q) - 2:
            mask = (d["bulk_z_strain_percent"] >= lo) & (
                d["bulk_z_strain_percent"] <= hi
            )
        else:
            mask = (d["bulk_z_strain_percent"] >= lo) & (
                d["bulk_z_strain_percent"] < hi
            )

        g = d[mask].copy()

        if g.empty:
            continue

        y = g[value_col].to_numpy(dtype=float)
        y = y[np.isfinite(y)]

        rows.append(
            {
                "x_mean": float(np.nanmean(g["bulk_z_strain_percent"])),
                "x_min": float(np.nanmin(g["bulk_z_strain_percent"])),
                "x_max": float(np.nanmax(g["bulk_z_strain_percent"])),
                "y_mean": float(np.nanmean(y)),
                "y_std": float(np.nanstd(y, ddof=1)) if y.size > 1 else 0.0,
                "n": int(y.size),
            }
        )

    return pd.DataFrame(rows).sort_values("x_mean")


def pick_representative_delta_maps(delta_map_records, n_maps: int):
    if len(delta_map_records) <= n_maps:
        return delta_map_records

    strains = np.array(
        [r["bulk_z_strain_percent"] for r in delta_map_records], dtype=float
    )
    targets = np.linspace(np.nanmin(strains), np.nanmax(strains), n_maps)

    chosen = []
    used = set()

    for target in targets:
        order = np.argsort(np.abs(strains - target))

        for idx in order:
            key = (
                delta_map_records[idx]["load_mpa"],
                delta_map_records[idx]["sample_type"],
                delta_map_records[idx]["sample"],
                delta_map_records[idx]["time_h"],
            )

            if key not in used:
                chosen.append(delta_map_records[idx])
                used.add(key)
                break

    return chosen


# =============================================================================
# Four-panel figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))
axA, axB, axC, axD = axes.ravel()

# -----------------------------------------------------------------------------
# Panel A: Representative raw Delta h maps
# -----------------------------------------------------------------------------

rep_maps = pick_representative_delta_maps(delta_map_records, N_REPRESENTATIVE_MAPS)

if len(rep_maps) > 0:
    values = np.concatenate([r["delta_h"].ravel() for r in rep_maps])
    lim = float(np.nanpercentile(np.abs(values), 99.0))
    lim = max(lim, np.finfo(float).eps)

    # Draw mini image grid inside panel A.
    axA.axis("off")

    n = len(rep_maps)
    inset_width = 0.46
    inset_height = 0.38

    positions = [
        (0.02, 0.56),
        (0.52, 0.56),
        (0.02, 0.08),
        (0.52, 0.08),
    ]

    image_for_colorbar = None

    for i, rec in enumerate(rep_maps[:4]):
        x0, y0 = positions[i]
        inset = axA.inset_axes([x0, y0, inset_width, inset_height])

        H = rec["delta_h"]
        n0, n1 = H.shape
        extent = [0.0, n1 * spacing_um, 0.0, n0 * spacing_um]

        image_for_colorbar = inset.imshow(
            H,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            interpolation="nearest",
            rasterized=True,
            aspect="equal",
        )

        inset.set_title(
            f"{rec['load_mpa']} MPa, {rec['sample']}\n"
            rf"$\epsilon_p$={rec['bulk_z_strain_percent']:.2f}%",
            fontsize=7,
        )
        inset.set_xticks([])
        inset.set_yticks([])

    cax = axA.inset_axes([0.08, -0.03, 0.84, 0.035])
    cbar = fig.colorbar(image_for_colorbar, cax=cax, orientation="horizontal")
    cbar.set_label(r"raw $\Delta h$ [$\mu$m]", fontsize=8)

    axA.set_title("A. Representative raw height-change maps", loc="left")

# -----------------------------------------------------------------------------
# Panel B: Radial PSDs of raw Delta h grouped by plastic strain
# -----------------------------------------------------------------------------

strain_groups = make_strain_groups_for_curves(delta_df, N_POSITIVE_STRAIN_GROUPS)

for group_idx, group in enumerate(strain_groups):
    g = sample_rows(group["df"], MAX_CURVES_PER_GROUP, seed=2000 + group_idx)

    curves = []
    wavelength_ref = None

    for _, row in g.iterrows():
        curve = find_radial_curve(row)

        if curve is None:
            continue

        wavelength = curve["wavelength_um"]
        psd = curve["psd_um4"]

        valid = (
            np.isfinite(wavelength) & np.isfinite(psd) & (wavelength > 0) & (psd > 0)
        )

        if np.count_nonzero(valid) == 0:
            continue

        order = np.argsort(wavelength[valid])

        axB.loglog(
            wavelength[valid][order],
            psd[valid][order],
            color=group["color"],
            alpha=0.14,
            lw=0.6,
        )

        curve_full = np.full_like(psd, np.nan, dtype=float)
        curve_full[valid] = psd[valid]
        curves.append(curve_full)

        if wavelength_ref is None:
            wavelength_ref = wavelength

    if len(curves) == 0:
        continue

    curves = np.vstack(curves)
    mean_curve = np.nanmean(curves, axis=0)

    valid = (
        np.isfinite(wavelength_ref)
        & np.isfinite(mean_curve)
        & (wavelength_ref > 0)
        & (mean_curve > 0)
    )

    order = np.argsort(wavelength_ref[valid])

    axB.loglog(
        wavelength_ref[valid][order],
        mean_curve[valid][order],
        color=group["color"],
        lw=2.2,
        label=f"{group['label']} mean, n={len(curves)}",
    )

for band_name, (lo, hi) in bands_um.items():
    axB.axvspan(
        lo,
        hi,
        color=BAND_COLORS.get(band_name, "0.8"),
        alpha=0.055,
        lw=0,
    )

if np.isfinite(d_ref_um) and d_ref_um > 0:
    axB.axvline(
        d_ref_um,
        color="0.25",
        ls=":",
        lw=1.0,
        label=rf"$d_{{ref}}={d_ref_um:.1f}\,\mu$m",
    )

axB.axvline(
    long_cap_um,
    color="0.45",
    ls="--",
    lw=1.0,
    label=r"long-$\lambda$ cap",
)

axB.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
axB.set_ylabel(r"PSD of raw $\Delta h$, $C_{\Delta h}$ [$\mu$m$^4$]")
axB.set_title("B. Radial PSD of raw height-change fields", loc="left")
axB.legend(fontsize=6)

# -----------------------------------------------------------------------------
# Panel C: Band-limited RMS amplitude of raw Delta h vs strain
# -----------------------------------------------------------------------------

for band_name, (lo, hi) in bands_um.items():
    col = f"rms_{band_name}_um"

    if col not in delta_df.columns:
        continue

    summary = summarize_by_strain_bins(delta_df, col, n_bins=8)

    if summary.empty:
        continue

    color = BAND_COLORS.get(band_name, None)

    axC.scatter(
        delta_df["bulk_z_strain_percent"],
        delta_df[col],
        s=18,
        color=color,
        alpha=0.22,
    )

    axC.errorbar(
        summary["x_mean"],
        summary["y_mean"],
        yerr=summary["y_std"],
        fmt="o-",
        color=color,
        lw=1.8,
        ms=4,
        capsize=3,
        label=rf"{band_name}: {lo:.1f}$<\lambda<${hi:.1f} $\mu$m",
    )

axC.set_xlabel(STRAIN_LABEL)
axC.set_ylabel(r"band RMS of raw $\Delta h$, $A_{\Delta h,\mathrm{band}}$ [$\mu$m]")
axC.set_title("C. Scale-resolved amplitude of raw $\\Delta h$", loc="left")
axC.legend(fontsize=6)

# -----------------------------------------------------------------------------
# Panel D: Fraction of raw Delta h PSD power in each band
# -----------------------------------------------------------------------------

for band_name, (lo, hi) in bands_um.items():
    col = f"fraction_{band_name}"

    if col not in delta_df.columns:
        continue

    summary = summarize_by_strain_bins(delta_df, col, n_bins=8)

    if summary.empty:
        continue

    color = BAND_COLORS.get(band_name, None)

    axD.errorbar(
        summary["x_mean"],
        summary["y_mean"],
        yerr=summary["y_std"],
        fmt="o-",
        color=color,
        lw=1.8,
        ms=4,
        capsize=3,
        label=band_name,
    )

axD.set_xlabel(STRAIN_LABEL)
axD.set_ylabel(r"fraction of raw $\Delta h$ PSD power, $P_i/\sum P_i$")
axD.set_ylim(0.0, 1.05)
axD.set_title("D. Redistribution of raw $\\Delta h$ power by scale", loc="left")
axD.legend(fontsize=6)

# -----------------------------------------------------------------------------
# Final formatting
# -----------------------------------------------------------------------------

for ax in [axB, axC, axD]:
    ax.grid(True, which="both", alpha=0.25)

fig.suptitle(
    "Raw height-change PSD analysis without registration correction\n"
    "475 MPa sub-yield condition excluded",
    y=1.02,
)

fig.tight_layout()

outpath = OUTPUT_DIR / "raw_delta_h_psd_four_panel_no_registration.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()


# =============================================================================
# Extra quick check: raw Delta h scalar amplitude vs conventional Delta Sa
# =============================================================================

if "delta_sa_um" in delta_df.columns:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))

    ax.scatter(
        delta_df["delta_sa_um"],
        delta_df["raw_delta_h_sa_um"],
        c=delta_df["bulk_z_strain_percent"],
        cmap="viridis",
        s=35,
        alpha=0.8,
    )

    finite = np.isfinite(delta_df["delta_sa_um"]) & np.isfinite(
        delta_df["raw_delta_h_sa_um"]
    )

    if np.count_nonzero(finite) >= 2:
        x = delta_df.loc[finite, "delta_sa_um"].to_numpy(dtype=float)
        y = delta_df.loc[finite, "raw_delta_h_sa_um"].to_numpy(dtype=float)

        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))

        ax.plot([lo, hi], [lo, hi], color="0.4", lw=1.0, ls="--")

    ax.set_xlabel(r"conventional $\Delta S_a = S_a(\epsilon)-S_a(0)$ [$\mu$m]")
    ax.set_ylabel(r"$S_a$ of raw $\Delta h$ [$\mu$m]")
    ax.set_title(r"Comparison of scalar roughness increments")
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label(STRAIN_LABEL)

    fig.tight_layout()
    outpath = OUTPUT_DIR / "raw_delta_h_sa_vs_delta_sa.png"
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath.resolve()}")
    plt.show()


print()
print("Complete.")
print(f"Output directory: {OUTPUT_DIR.resolve()}")

# %%
# =============================================================================
# Two-panel height-field PSD figure
#
# Left:
#   Radial PSD of plane-leveled, mean-removed height maps vs wavelength,
#   grouped by plastic strain level, including initial height PSD.
#
# Right:
#   PSD gain vs wavelength for each strain-level mean PSD curve,
#   relative to the initial mean PSD.
#
# Notes:
#   - PSD is computed from signed height h, not |h|.
#   - Height maps are plane leveled, cropped, and mean removed before PSD.
#   - Individual traces are plotted faintly; group means are plotted bold.
#   - No vertical min/max wavelength guide lines are plotted.
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

# Keep this if you want to exclude the sub-yield condition.
# Set EXCLUDED_LOADS = set() to include all loads.
EXCLUDED_LOADS = {475}

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

# Plane leveling settings.
level = True
detrend_order = 1

# Strain grouping.
INITIAL_STRAIN_TOL_PERCENT = 0.05
N_POSITIVE_STRAIN_GROUPS = 4
MAX_TRACES_PER_GROUP = None  # set to e.g. 25 to randomly thin traces

# PSD settings.
N_RADIAL_BINS = 85
MIN_MODES_PER_BIN = 8

# Plot settings.
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)
if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()
df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()
df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No data remain after filtering.")

print("Records used:")
print(df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height loading and plane leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    design, inverse = _detrend_geometry(
        values.shape, float(spacing_um_value), int(order)
    )
    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD functions
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))
    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um: np.ndarray, spacing_um_value: float):
    """
    Returns f0, f1, PSD2D, df0, df1.

    PSD units are um^4 if height and spacing are in um.
    Normalization approximately satisfies:
        sum(PSD2D) * df0 * df1 = mean((windowed height)^2)
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    df0 = 1.0 / (n0 * spacing_um_value)
    df1 = 1.0 / (n1 * spacing_um_value)

    return f0, f1, PSD, df0, df1


def radial_bin_edges(shape: tuple[int, int], spacing_um_value: float, n_bins: int):
    n0, n1 = shape

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    positive = FR[(FR > 0) & np.isfinite(FR)]

    f_min = positive.min()
    f_max = min(0.5 / spacing_um_value, positive.max())

    return np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)


def radial_psd_from_height(
    height_um: np.ndarray,
    spacing_um_value: float,
    edges: np.ndarray,
    min_modes: int = MIN_MODES_PER_BIN,
):
    f0, f1, PSD, df0, df1 = psd2d_height(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    f_flat = FR.ravel()
    p_flat = PSD.ravel()

    nbins = len(edges) - 1
    bin_index = np.searchsorted(edges, f_flat, side="right") - 1

    freq = np.full(nbins, np.nan)
    psd_radial = np.full(nbins, np.nan)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        mask = (
            (bin_index == i) & np.isfinite(f_flat) & np.isfinite(p_flat) & (f_flat > 0)
        )

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            freq[i] = np.mean(f_flat[mask])
            psd_radial[i] = np.mean(p_flat[mask])

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / freq

    return freq, wavelength, psd_radial, modes


# =============================================================================
# Strain grouping
# =============================================================================


def make_strain_groups(input_df: pd.DataFrame):
    d = input_df.copy()

    initial = d[np.abs(d["bulk_z_strain_percent"]) <= INITIAL_STRAIN_TOL_PERCENT].copy()
    positive = d[d["bulk_z_strain_percent"] > INITIAL_STRAIN_TOL_PERCENT].copy()

    groups = []

    if not initial.empty:
        groups.append(
            {
                "name": "initial",
                "label": r"initial, $\epsilon_p \approx 0$",
                "df": initial,
                "color": "black",
            }
        )

    if not positive.empty:
        q = np.unique(
            np.nanquantile(
                positive["bulk_z_strain_percent"].to_numpy(dtype=float),
                np.linspace(0.0, 1.0, N_POSITIVE_STRAIN_GROUPS + 1),
            )
        )

        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.18, 0.9, max(len(q) - 1, 1)))

        for i in range(len(q) - 1):
            lo = q[i]
            hi = q[i + 1]

            if i == len(q) - 2:
                mask = (positive["bulk_z_strain_percent"] >= lo) & (
                    positive["bulk_z_strain_percent"] <= hi
                )
            else:
                mask = (positive["bulk_z_strain_percent"] >= lo) & (
                    positive["bulk_z_strain_percent"] < hi
                )

            g = positive[mask].copy()

            if g.empty:
                continue

            groups.append(
                {
                    "name": f"group_{i+1}",
                    "label": rf"$\epsilon_p$ = {lo:.2g}--{hi:.2g}%",
                    "df": g,
                    "color": colors[i],
                }
            )

    return groups


def sample_group_rows(group_df: pd.DataFrame, max_rows: int | None, seed: int):
    if max_rows is None or len(group_df) <= max_rows:
        return group_df.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(group_df.index.to_numpy(), size=max_rows, replace=False)

    return group_df.loc[idx].copy()


groups = make_strain_groups(df)

print()
print("PSD strain groups:")
for group in groups:
    print(f"  {group['label']}: n = {len(group['df'])}")

if not any(group["name"] == "initial" for group in groups):
    raise ValueError(
        "No initial group found. Increase INITIAL_STRAIN_TOL_PERCENT or check your point table."
    )


# =============================================================================
# Compute radial PSDs
# =============================================================================

first_height = load_plane_leveled_mean_removed_height(df.iloc[0]["height_path"])
edges = radial_bin_edges(first_height.shape, spacing_um, N_RADIAL_BINS)

group_results = []

for group_index, group in enumerate(groups):
    group_df = sample_group_rows(
        group["df"],
        MAX_TRACES_PER_GROUP,
        seed=1000 + group_index,
    )

    curves = []
    meta_rows = []
    wavelength_ref = None
    freq_ref = None

    for _, rec in group_df.iterrows():
        try:
            H = load_plane_leveled_mean_removed_height(rec["height_path"])
            freq, wavelength, psd_curve, modes = radial_psd_from_height(
                H,
                spacing_um,
                edges,
                min_modes=MIN_MODES_PER_BIN,
            )

            if wavelength_ref is None:
                wavelength_ref = wavelength
                freq_ref = freq

            valid = (
                np.isfinite(wavelength)
                & np.isfinite(psd_curve)
                & (wavelength > 0)
                & (psd_curve > 0)
            )

            curve = np.full_like(psd_curve, np.nan, dtype=float)
            curve[valid] = psd_curve[valid]

            curves.append(curve)

            meta_rows.append(
                {
                    "load_mpa": rec["load_mpa"],
                    "sample_type": rec["sample_type"],
                    "sample": rec["sample"],
                    "time_h": rec["time_h"],
                    "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                    "height_path": rec["height_path"],
                }
            )

        except Exception as exc:
            warnings.warn(f"PSD failed for {rec['height_path']}: {exc}")

    if len(curves) == 0:
        continue

    curves = np.vstack(curves)

    mean_curve = np.nanmean(curves, axis=0)
    std_curve = (
        np.nanstd(curves, axis=0, ddof=1)
        if curves.shape[0] > 1
        else np.zeros_like(mean_curve)
    )

    group_results.append(
        {
            "name": group["name"],
            "label": group["label"],
            "color": group["color"],
            "n": curves.shape[0],
            "frequency_um_inv": freq_ref,
            "wavelength_um": wavelength_ref,
            "curves": curves,
            "mean_psd_um4": mean_curve,
            "std_psd_um4": std_curve,
            "meta": pd.DataFrame(meta_rows),
        }
    )

# Baseline initial mean PSD.
initial_result = next(result for result in group_results if result["name"] == "initial")
baseline_psd = initial_result["mean_psd_um4"]
baseline_wavelength = initial_result["wavelength_um"]

# Save group mean curves.
summary_rows = []
trace_rows = []

for result in group_results:
    wavelength = result["wavelength_um"]
    frequency = result["frequency_um_inv"]
    mean_psd = result["mean_psd_um4"]
    std_psd = result["std_psd_um4"]

    gain_mean = mean_psd / baseline_psd

    for i in range(len(wavelength)):
        summary_rows.append(
            {
                "group": result["name"],
                "label": result["label"],
                "n": result["n"],
                "frequency_um_inv": frequency[i],
                "wavelength_um": wavelength[i],
                "mean_psd_um4": mean_psd[i],
                "std_psd_um4": std_psd[i],
                "gain_vs_initial": gain_mean[i],
            }
        )

    for curve_index, curve in enumerate(result["curves"]):
        gain_curve = curve / baseline_psd

        meta = result["meta"].iloc[curve_index]

        for i in range(len(wavelength)):
            trace_rows.append(
                {
                    "group": result["name"],
                    "label": result["label"],
                    "curve_index": curve_index,
                    "load_mpa": meta["load_mpa"],
                    "sample_type": meta["sample_type"],
                    "sample": meta["sample"],
                    "time_h": meta["time_h"],
                    "bulk_z_strain_percent": meta["bulk_z_strain_percent"],
                    "frequency_um_inv": frequency[i],
                    "wavelength_um": wavelength[i],
                    "psd_um4": curve[i],
                    "gain_vs_initial": gain_curve[i],
                }
            )

psd_summary_df = pd.DataFrame(summary_rows)
psd_trace_df = pd.DataFrame(trace_rows)

psd_summary_df.to_csv(
    OUTPUT_DIR / "height_radial_psd_group_means_by_strain.csv", index=False
)
psd_trace_df.to_csv(
    OUTPUT_DIR / "height_radial_psd_individual_traces_by_strain.csv", index=False
)

print()
print(f"Saved: {OUTPUT_DIR / 'height_radial_psd_group_means_by_strain.csv'}")
print(f"Saved: {OUTPUT_DIR / 'height_radial_psd_individual_traces_by_strain.csv'}")


# =============================================================================
# Two-panel figure
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
ax_psd, ax_gain = axes

# -----------------------------------------------------------------------------
# Left panel: radial PSD traces and means
# -----------------------------------------------------------------------------

for result in group_results:
    wavelength = result["wavelength_um"]
    curves = result["curves"]
    mean_curve = result["mean_psd_um4"]
    color = result["color"]

    valid_base = np.isfinite(wavelength) & (wavelength > 0)
    order_base = np.argsort(wavelength[valid_base])

    wavelength_plot_base = wavelength[valid_base][order_base]

    for curve in curves:
        valid = valid_base & np.isfinite(curve) & (curve > 0)

        if np.count_nonzero(valid) == 0:
            continue

        order = np.argsort(wavelength[valid])

        # ax_psd.loglog(
        #     wavelength[valid][order],
        #     curve[valid][order],
        #     color=color,
        #     alpha=0.14 if result["name"] != "initial" else 0.10,
        #     lw=0.7,
        # )

    valid_mean = valid_base & np.isfinite(mean_curve) & (mean_curve > 0)

    if np.count_nonzero(valid_mean) > 0:
        order = np.argsort(wavelength[valid_mean])

        ax_psd.loglog(
            wavelength[valid_mean][order],
            mean_curve[valid_mean][order],
            color=color,
            lw=2.3,
            label=f"{result['label']} mean, n={result['n']}",
        )

ax_psd.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
ax_psd.set_ylabel(r"radial PSD of height, $C_h$ [$\mu$m$^4$]")
# ax_psd.set_title("A. Height-field radial PSD")
ax_psd.grid(True, which="both", alpha=0.25)
ax_psd.legend(fontsize=6)

# -----------------------------------------------------------------------------
# Right panel: gain traces and mean gain vs initial mean PSD
# -----------------------------------------------------------------------------

for result in group_results:
    if result["name"] == "initial":
        continue
    wavelength = result["wavelength_um"]
    curves = result["curves"]
    mean_curve = result["mean_psd_um4"]
    color = result["color"]

    valid_base = (
        np.isfinite(wavelength)
        & (wavelength > 0)
        & np.isfinite(baseline_psd)
        & (baseline_psd > 0)
    )

    for curve in curves:
        gain_curve = curve / baseline_psd

        valid = valid_base & np.isfinite(gain_curve) & (gain_curve > 0)

        if np.count_nonzero(valid) == 0:
            continue

        order = np.argsort(wavelength[valid])

        # ax_gain.loglog(
        #     wavelength[valid][order],
        #     gain_curve[valid][order],
        #     color=color,
        #     alpha=0.12 if result["name"] != "initial" else 0.08,
        #     lw=0.7,
        # )

    mean_gain = mean_curve / baseline_psd

    valid_mean = valid_base & np.isfinite(mean_gain) & (mean_gain > 0)

    if np.count_nonzero(valid_mean) > 0:
        order = np.argsort(wavelength[valid_mean])

        ax_gain.loglog(
            wavelength[valid_mean][order],
            mean_gain[valid_mean][order],
            color=color,
            lw=2.3,
            label=f"{result['label']} mean gain",
        )

ax_gain.axhline(1.0, color="0.45", lw=0.9, ls="--")
ax_gain.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
ax_gain.set_ylabel(r"PSD gain, $C_h(\epsilon_p)/C_h(0)$")
# ax_gain.set_title("B. PSD gain relative to initial mean")
ax_gain.grid(True, which="both", alpha=0.25)
ax_gain.legend(fontsize=6)

# fig.suptitle(
#     "Plane-leveled, mean-removed height PSD grouped by plastic strain",
#     y=1.02,
# )
fig.tight_layout()

outpath = OUTPUT_DIR / "two_panel_height_psd_and_gain_by_strain.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Recompute two-panel height PSD and gain out to 400 um
#
# This version explicitly constructs radial bins in wavelength space from
# LAMBDA_MIN_PLOT_UM to LAMBDA_MAX_PLOT_UM, instead of relying on the previous
# group_results. This is necessary because changing only the x-axis limit does
# not create PSD bins at longer wavelengths.
#
# Important:
#   - Long-wavelength radial PSD bins have few Fourier modes, so they are noisy.
#   - MIN_MODES_PER_BIN_LONG = 1 is used to retain the longest wavelengths.
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.ndimage import distance_transform_edt

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

EXCLUDED_LOADS = {475}

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

level = True
detrend_order = 1

INITIAL_STRAIN_TOL_PERCENT = 0.05
N_POSITIVE_STRAIN_GROUPS = 4
MAX_TRACES_PER_GROUP = None

LAMBDA_MIN_PLOT_UM = 3.0
LAMBDA_MAX_PLOT_UM = 400.0

# Number of wavelength bins over the requested plotting range.
N_WAVELENGTH_BINS = 90

# Use low threshold to keep long-wavelength bins.
# Long wavelengths naturally have few Fourier modes.
MIN_MODES_PER_BIN_LONG = 1

x_ticks = np.array([3, 5, 10, 20, 50, 100, 200, 250, 300, 400], dtype=float)
x_ticks = x_ticks[(x_ticks >= LAMBDA_MIN_PLOT_UM) & (x_ticks <= LAMBDA_MAX_PLOT_UM)]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)
if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()

df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()
df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No data remain after filtering.")

print("Records used:")
print(df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height loading and plane leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(values, spacing_um_value, order=1):
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path):
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path,
    *,
    level=True,
    spacing_um_value=spacing_um,
    detrend_order=1,
):
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path):
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD functions
# =============================================================================


def hann2d(shape):
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))

    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um, spacing_um_value):
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    df0 = 1.0 / (n0 * spacing_um_value)
    df1 = 1.0 / (n1 * spacing_um_value)

    return f0, f1, PSD, df0, df1


def wavelength_bin_edges(lambda_min_um, lambda_max_um, n_bins):
    """
    Log-spaced wavelength bin edges.
    Converted to frequency bins internally.
    """
    return np.logspace(
        np.log10(lambda_min_um),
        np.log10(lambda_max_um),
        n_bins + 1,
    )


def radial_psd_from_height_wavelength_binned(
    height_um,
    spacing_um_value,
    lambda_edges_um,
    min_modes=1,
):
    """
    Radial PSD using explicit wavelength bins.

    A bin is:
        lambda_edges[i] <= lambda < lambda_edges[i+1]

    Since lambda = 1/f, this corresponds to:
        1/lambda_hi < f <= 1/lambda_lo
    """
    f0, f1, PSD, df0, df1 = psd2d_height(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / FR

    nbins = len(lambda_edges_um) - 1

    lambda_center = np.sqrt(lambda_edges_um[:-1] * lambda_edges_um[1:])
    psd_radial = np.full(nbins, np.nan)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        lo = lambda_edges_um[i]
        hi = lambda_edges_um[i + 1]

        mask = (
            np.isfinite(wavelength)
            & np.isfinite(PSD)
            & (FR > 0)
            & (wavelength >= lo)
            & (wavelength < hi)
        )

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            psd_radial[i] = np.nanmean(PSD[mask])

    frequency_center = 1.0 / lambda_center

    return frequency_center, lambda_center, psd_radial, modes


# =============================================================================
# Strain grouping
# =============================================================================


def make_strain_groups(input_df):
    d = input_df.copy()

    initial = d[np.abs(d["bulk_z_strain_percent"]) <= INITIAL_STRAIN_TOL_PERCENT].copy()
    positive = d[d["bulk_z_strain_percent"] > INITIAL_STRAIN_TOL_PERCENT].copy()

    groups = []

    if not initial.empty:
        groups.append(
            {
                "name": "initial",
                "label": r"initial, $\epsilon_p \approx 0$",
                "df": initial,
                "color": "black",
            }
        )

    if not positive.empty:
        q = np.unique(
            np.nanquantile(
                positive["bulk_z_strain_percent"].to_numpy(dtype=float),
                np.linspace(0.0, 1.0, N_POSITIVE_STRAIN_GROUPS + 1),
            )
        )

        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.18, 0.9, max(len(q) - 1, 1)))

        for i in range(len(q) - 1):
            lo = q[i]
            hi = q[i + 1]

            if i == len(q) - 2:
                mask = (positive["bulk_z_strain_percent"] >= lo) & (
                    positive["bulk_z_strain_percent"] <= hi
                )
            else:
                mask = (positive["bulk_z_strain_percent"] >= lo) & (
                    positive["bulk_z_strain_percent"] < hi
                )

            g = positive[mask].copy()

            if g.empty:
                continue

            groups.append(
                {
                    "name": f"group_{i+1}",
                    "label": rf"$\epsilon_p$ = {lo:.2g}--{hi:.2g}%",
                    "df": g,
                    "color": colors[i],
                }
            )

    return groups


def sample_group_rows(group_df, max_rows, seed):
    if max_rows is None or len(group_df) <= max_rows:
        return group_df.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(group_df.index.to_numpy(), size=max_rows, replace=False)

    return group_df.loc[idx].copy()


groups = make_strain_groups(df)

print()
print("PSD strain groups:")
for group in groups:
    print(f"  {group['label']}: n = {len(group['df'])}")

if not any(group["name"] == "initial" for group in groups):
    raise ValueError(
        "No initial group found. Increase INITIAL_STRAIN_TOL_PERCENT or check your point table."
    )


# =============================================================================
# Compute radial PSDs with explicit wavelength range
# =============================================================================

lambda_edges = wavelength_bin_edges(
    LAMBDA_MIN_PLOT_UM,
    LAMBDA_MAX_PLOT_UM,
    N_WAVELENGTH_BINS,
)

group_results = []

for group_index, group in enumerate(groups):
    group_df = sample_group_rows(
        group["df"],
        MAX_TRACES_PER_GROUP,
        seed=1000 + group_index,
    )

    curves = []
    meta_rows = []
    wavelength_ref = None
    freq_ref = None

    for _, rec in group_df.iterrows():
        try:
            H = load_plane_leveled_mean_removed_height(rec["height_path"])

            freq, wavelength, psd_curve, modes = (
                radial_psd_from_height_wavelength_binned(
                    H,
                    spacing_um,
                    lambda_edges,
                    min_modes=MIN_MODES_PER_BIN_LONG,
                )
            )

            if wavelength_ref is None:
                wavelength_ref = wavelength
                freq_ref = freq

            valid = (
                np.isfinite(wavelength)
                & np.isfinite(psd_curve)
                & (wavelength > 0)
                & (wavelength <= LAMBDA_MAX_PLOT_UM)
                & (psd_curve > 0)
            )

            curve = np.full_like(psd_curve, np.nan, dtype=float)
            curve[valid] = psd_curve[valid]

            curves.append(curve)

            meta_rows.append(
                {
                    "load_mpa": rec["load_mpa"],
                    "sample_type": rec["sample_type"],
                    "sample": rec["sample"],
                    "time_h": rec["time_h"],
                    "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                    "height_path": rec["height_path"],
                }
            )

        except Exception as exc:
            warnings.warn(f"PSD failed for {rec['height_path']}: {exc}")

    if len(curves) == 0:
        continue

    curves = np.vstack(curves)

    mean_curve = np.nanmean(curves, axis=0)
    std_curve = (
        np.nanstd(curves, axis=0, ddof=1)
        if curves.shape[0] > 1
        else np.zeros_like(mean_curve)
    )

    group_results.append(
        {
            "name": group["name"],
            "label": group["label"],
            "color": group["color"],
            "n": curves.shape[0],
            "frequency_um_inv": freq_ref,
            "wavelength_um": wavelength_ref,
            "curves": curves,
            "mean_psd_um4": mean_curve,
            "std_psd_um4": std_curve,
            "meta": pd.DataFrame(meta_rows),
        }
    )

initial_result = next(result for result in group_results if result["name"] == "initial")
baseline_psd = initial_result["mean_psd_um4"]


# =============================================================================
# Save curves
# =============================================================================

summary_rows = []
trace_rows = []

for result in group_results:
    wavelength = result["wavelength_um"]
    frequency = result["frequency_um_inv"]
    mean_psd = result["mean_psd_um4"]
    std_psd = result["std_psd_um4"]

    gain_mean = mean_psd / baseline_psd

    for i in range(len(wavelength)):
        summary_rows.append(
            {
                "group": result["name"],
                "label": result["label"],
                "n": result["n"],
                "frequency_um_inv": frequency[i],
                "wavelength_um": wavelength[i],
                "mean_psd_um4": mean_psd[i],
                "std_psd_um4": std_psd[i],
                "gain_vs_initial": gain_mean[i],
            }
        )

    for curve_index, curve in enumerate(result["curves"]):
        gain_curve = curve / baseline_psd
        meta = result["meta"].iloc[curve_index]

        for i in range(len(wavelength)):
            trace_rows.append(
                {
                    "group": result["name"],
                    "label": result["label"],
                    "curve_index": curve_index,
                    "load_mpa": meta["load_mpa"],
                    "sample_type": meta["sample_type"],
                    "sample": meta["sample"],
                    "time_h": meta["time_h"],
                    "bulk_z_strain_percent": meta["bulk_z_strain_percent"],
                    "frequency_um_inv": frequency[i],
                    "wavelength_um": wavelength[i],
                    "psd_um4": curve[i],
                    "gain_vs_initial": gain_curve[i],
                }
            )

psd_summary_df = pd.DataFrame(summary_rows)
psd_trace_df = pd.DataFrame(trace_rows)

psd_summary_df.to_csv(
    OUTPUT_DIR / "height_radial_psd_group_means_by_strain_to_400um.csv",
    index=False,
)
psd_trace_df.to_csv(
    OUTPUT_DIR / "height_radial_psd_individual_traces_by_strain_to_400um.csv",
    index=False,
)


# =============================================================================
# Plot two-panel figure
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
ax_psd, ax_gain = axes

# Left panel
for result in group_results:
    wavelength = result["wavelength_um"]
    curves = result["curves"]
    mean_curve = result["mean_psd_um4"]
    color = result["color"]

    valid_base = (
        np.isfinite(wavelength)
        & (wavelength >= LAMBDA_MIN_PLOT_UM)
        & (wavelength <= LAMBDA_MAX_PLOT_UM)
    )

    for curve in curves:
        valid = valid_base & np.isfinite(curve) & (curve > 0)

        if np.count_nonzero(valid) == 0:
            continue

        order = np.argsort(wavelength[valid])

        ax_psd.loglog(
            wavelength[valid][order],
            curve[valid][order],
            color=color,
            alpha=0.14 if result["name"] != "initial" else 0.10,
            lw=0.7,
        )

    valid_mean = valid_base & np.isfinite(mean_curve) & (mean_curve > 0)

    if np.count_nonzero(valid_mean) > 0:
        order = np.argsort(wavelength[valid_mean])

        ax_psd.loglog(
            wavelength[valid_mean][order],
            mean_curve[valid_mean][order],
            color=color,
            lw=2.3,
            label=f"{result['label']} mean, n={result['n']}",
        )

ax_psd.set_xlim(LAMBDA_MIN_PLOT_UM, LAMBDA_MAX_PLOT_UM)
ax_psd.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
ax_psd.set_ylabel(r"radial PSD of height, $C_h$ [$\mu$m$^4$]")
ax_psd.grid(True, which="both", alpha=0.25)
ax_psd.legend(fontsize=6)

ax_psd.xaxis.set_major_locator(FixedLocator(x_ticks))
ax_psd.xaxis.set_major_formatter(FixedFormatter([f"{int(t)}" for t in x_ticks]))
ax_psd.xaxis.set_minor_formatter(NullFormatter())

# Right panel
for result in group_results:
    if result["name"] == "initial":
        continue

    wavelength = result["wavelength_um"]
    curves = result["curves"]
    mean_curve = result["mean_psd_um4"]
    color = result["color"]

    valid_base = (
        np.isfinite(wavelength)
        & (wavelength >= LAMBDA_MIN_PLOT_UM)
        & (wavelength <= LAMBDA_MAX_PLOT_UM)
        & np.isfinite(baseline_psd)
        & (baseline_psd > 0)
    )

    for curve in curves:
        gain_curve = curve / baseline_psd

        valid = valid_base & np.isfinite(gain_curve) & (gain_curve > 0)

        if np.count_nonzero(valid) == 0:
            continue

        order = np.argsort(wavelength[valid])

        ax_gain.loglog(
            wavelength[valid][order],
            gain_curve[valid][order],
            color=color,
            alpha=0.12,
            lw=0.7,
        )

    mean_gain = mean_curve / baseline_psd

    valid_mean = valid_base & np.isfinite(mean_gain) & (mean_gain > 0)

    if np.count_nonzero(valid_mean) > 0:
        order = np.argsort(wavelength[valid_mean])

        ax_gain.loglog(
            wavelength[valid_mean][order],
            mean_gain[valid_mean][order],
            color=color,
            lw=2.3,
            label=f"{result['label']} mean gain",
        )

ax_gain.axhline(1.0, color="0.45", lw=0.9, ls="--")
ax_gain.set_xlim(LAMBDA_MIN_PLOT_UM, LAMBDA_MAX_PLOT_UM)
ax_gain.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
ax_gain.set_ylabel(r"PSD gain, $C_h(\epsilon_p)/C_h(0)$")
ax_gain.grid(True, which="both", alpha=0.25)
ax_gain.legend(fontsize=6)

ax_gain.xaxis.set_major_locator(FixedLocator(x_ticks))
ax_gain.xaxis.set_major_formatter(FixedFormatter([f"{int(t)}" for t in x_ticks]))
ax_gain.xaxis.set_minor_formatter(NullFormatter())

fig.tight_layout()

outpath = OUTPUT_DIR / "two_panel_height_psd_and_gain_by_strain_to_400um.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Two-panel PSD figure for raw Delta h fields
#
# Left:
#   Radial PSD of raw Delta h fields,
#       Delta h(x,y; eps_p) = h(x,y; eps_p) - h(x,y; 0),
#   grouped by plastic strain level.
#
# Right:
#   PSD gain of each Delta h strain-group mean relative to the lowest nonzero
#   strain Delta h group.
#
# Important:
#   - Initial Delta h is identically zero, so there is no initial Delta h PSD
#     to plot on a log scale and no valid initial Delta h gain denominator.
#   - PSD is computed from signed Delta h, not |Delta h|.
#   - This version does raw subtraction with no translation/rotation correction.
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

# Keep this if you want to exclude the sub-yield condition.
EXCLUDED_LOADS = {475}

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

# Individual height-map plane leveling.
level = True
detrend_order = 1

# Delta h settings.
# If you want literal raw array subtraction, keep margin = 0.
RAW_DELTA_CROP_MARGIN_PIXELS = 0

# Plane-level Delta h after subtraction.
# Set False if you want to retain residual long-wavelength tilt/shape mismatch.
LEVEL_DELTA_H = True
DELTA_H_DETREND_ORDER = 1

# Strain grouping.
N_POSITIVE_STRAIN_GROUPS = 4
MAX_TRACES_PER_GROUP = None  # set to e.g. 25 to thin traces

# PSD settings.
N_RADIAL_BINS = 85
MIN_MODES_PER_BIN = 8

# Plot settings.
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)

if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()
df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()
df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No data remain after filtering.")

print("Records used:")
print(df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height loading and leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


def crop_margin(A: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return A

    if A.shape[0] <= 2 * margin or A.shape[1] <= 2 * margin:
        raise ValueError(f"Margin {margin} too large for shape {A.shape}")

    return A[margin:-margin, margin:-margin]


def compute_raw_delta_h(
    initial_path: str | Path, current_path: str | Path
) -> np.ndarray:
    H0 = load_plane_leveled_mean_removed_height(initial_path)
    H1 = load_plane_leveled_mean_removed_height(current_path)

    if H0.shape != H1.shape:
        raise ValueError(f"Shape mismatch: initial {H0.shape}, current {H1.shape}")

    H0 = crop_margin(H0, RAW_DELTA_CROP_MARGIN_PIXELS)
    H1 = crop_margin(H1, RAW_DELTA_CROP_MARGIN_PIXELS)

    dH = H1 - H0

    if LEVEL_DELTA_H:
        dH = detrend_surface(dH, spacing_um, order=DELTA_H_DETREND_ORDER)

    dH = dH - np.nanmean(dH)

    return dH


# =============================================================================
# PSD functions
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))

    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um: np.ndarray, spacing_um_value: float):
    """
    Returns f0, f1, PSD2D, df0, df1.

    PSD units are um^4 if height and spacing are in um.
    Normalization approximately satisfies:
        sum(PSD2D) * df0 * df1 = mean((windowed height)^2)
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    df0 = 1.0 / (n0 * spacing_um_value)
    df1 = 1.0 / (n1 * spacing_um_value)

    return f0, f1, PSD, df0, df1


def radial_bin_edges(shape: tuple[int, int], spacing_um_value: float, n_bins: int):
    n0, n1 = shape

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    positive = FR[(FR > 0) & np.isfinite(FR)]

    f_min = positive.min()
    f_max = min(0.5 / spacing_um_value, positive.max())

    return np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)


def radial_psd_from_height(
    height_um: np.ndarray,
    spacing_um_value: float,
    edges: np.ndarray,
    min_modes: int = MIN_MODES_PER_BIN,
):
    f0, f1, PSD, df0, df1 = psd2d_height(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    f_flat = FR.ravel()
    p_flat = PSD.ravel()

    nbins = len(edges) - 1
    bin_index = np.searchsorted(edges, f_flat, side="right") - 1

    freq = np.full(nbins, np.nan)
    psd_radial = np.full(nbins, np.nan)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        mask = (
            (bin_index == i) & np.isfinite(f_flat) & np.isfinite(p_flat) & (f_flat > 0)
        )

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            freq[i] = np.mean(f_flat[mask])
            psd_radial[i] = np.mean(p_flat[mask])

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / freq

    return freq, wavelength, psd_radial, modes


# =============================================================================
# Compute raw Delta h radial PSDs
# =============================================================================

delta_rows = []
delta_curve_records = []
first_delta_h = None
edges = None

for (sample_type, load, sample), g in df.groupby(["sample_type", "load_mpa", "sample"]):
    g = g.sort_values("time_h").copy()

    if len(g) < 2:
        continue

    initial = g.iloc[0]
    initial_time = float(initial["time_h"])
    initial_path = initial["height_path"]

    # Use only samples with an actual initial scan.
    if initial_time > 1.0e-8:
        continue

    for _, current in g.iloc[1:].iterrows():
        try:
            dH = compute_raw_delta_h(initial_path, current["height_path"])

            if first_delta_h is None:
                first_delta_h = dH.copy()
                edges = radial_bin_edges(first_delta_h.shape, spacing_um, N_RADIAL_BINS)

            freq, wavelength, psd_curve, modes = radial_psd_from_height(
                dH,
                spacing_um,
                edges,
                min_modes=MIN_MODES_PER_BIN,
            )

            row = {
                "load_mpa": load,
                "sample_type": sample_type,
                "sample": sample,
                "initial_time_h": initial_time,
                "time_h": float(current["time_h"]),
                "bulk_z_strain_percent": float(current["bulk_z_strain_percent"]),
                "height_path_initial": initial_path,
                "height_path_current": current["height_path"],
                "raw_delta_h_sa_um": float(np.nanmean(np.abs(dH - np.nanmean(dH)))),
                "raw_delta_h_sq_um": float(
                    np.sqrt(np.nanmean((dH - np.nanmean(dH)) ** 2))
                ),
            }

            if "delta_sa_um" in current.index:
                row["delta_sa_um"] = float(current["delta_sa_um"])

            delta_rows.append(row)

            delta_curve_records.append(
                {
                    **row,
                    "frequency_um_inv": freq,
                    "wavelength_um": wavelength,
                    "psd_um4": psd_curve,
                    "modes": modes,
                }
            )

        except Exception as exc:
            warnings.warn(
                f"Raw Delta h PSD failed for sample={sample}, load={load}, "
                f"time={current['time_h']}: {exc}"
            )

if first_delta_h is None or len(delta_curve_records) == 0:
    raise RuntimeError("No Delta h PSD records were successfully computed.")

delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(OUTPUT_DIR / "raw_delta_h_psd_point_table.csv", index=False)

print()
print("Delta h PSD records:")
print(delta_df.groupby(["sample_type", "load_mpa"]).size())
print(f"Saved: {OUTPUT_DIR / 'raw_delta_h_psd_point_table.csv'}")


# =============================================================================
# Strain grouping for Delta h curves
# =============================================================================


def make_delta_h_strain_groups(input_df: pd.DataFrame):
    d = input_df.copy()
    d = d[np.isfinite(d["bulk_z_strain_percent"])].copy()

    if d.empty:
        return []

    q = np.unique(
        np.nanquantile(
            d["bulk_z_strain_percent"].to_numpy(dtype=float),
            np.linspace(0.0, 1.0, N_POSITIVE_STRAIN_GROUPS + 1),
        )
    )

    groups = []
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.18, 0.9, max(len(q) - 1, 1)))

    for i in range(len(q) - 1):
        lo = q[i]
        hi = q[i + 1]

        if i == len(q) - 2:
            mask = (d["bulk_z_strain_percent"] >= lo) & (
                d["bulk_z_strain_percent"] <= hi
            )
        else:
            mask = (d["bulk_z_strain_percent"] >= lo) & (
                d["bulk_z_strain_percent"] < hi
            )

        g = d[mask].copy()

        if g.empty:
            continue

        groups.append(
            {
                "name": f"group_{i+1}",
                "label": rf"$\epsilon_p$ = {lo:.2g}--{hi:.2g}%",
                "df": g,
                "color": colors[i],
            }
        )

    return groups


def sample_group_rows(group_df: pd.DataFrame, max_rows: int | None, seed: int):
    if max_rows is None or len(group_df) <= max_rows:
        return group_df.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(group_df.index.to_numpy(), size=max_rows, replace=False)

    return group_df.loc[idx].copy()


def find_curve_for_row(row: pd.Series):
    for rec in delta_curve_records:
        if (
            rec["load_mpa"] == row["load_mpa"]
            and rec["sample_type"] == row["sample_type"]
            and rec["sample"] == row["sample"]
            and np.isclose(rec["time_h"], row["time_h"])
        ):
            return rec

    return None


groups = make_delta_h_strain_groups(delta_df)

print()
print("Delta h PSD strain groups:")
for group in groups:
    print(f"  {group['label']}: n = {len(group['df'])}")

if len(groups) < 1:
    raise ValueError("No nonzero Delta h strain groups were found.")


# =============================================================================
# Build group results
# =============================================================================

group_results = []

for group_index, group in enumerate(groups):
    group_df = sample_group_rows(
        group["df"],
        MAX_TRACES_PER_GROUP,
        seed=3000 + group_index,
    )

    curves = []
    meta_rows = []
    wavelength_ref = None
    freq_ref = None

    for _, row in group_df.iterrows():
        rec = find_curve_for_row(row)

        if rec is None:
            continue

        wavelength = rec["wavelength_um"]
        freq = rec["frequency_um_inv"]
        psd_curve = rec["psd_um4"]

        valid = (
            np.isfinite(wavelength)
            & np.isfinite(psd_curve)
            & (wavelength > 0)
            & (psd_curve > 0)
        )

        curve = np.full_like(psd_curve, np.nan, dtype=float)
        curve[valid] = psd_curve[valid]

        curves.append(curve)

        if wavelength_ref is None:
            wavelength_ref = wavelength
            freq_ref = freq

        meta_rows.append(
            {
                "load_mpa": row["load_mpa"],
                "sample_type": row["sample_type"],
                "sample": row["sample"],
                "time_h": row["time_h"],
                "bulk_z_strain_percent": row["bulk_z_strain_percent"],
            }
        )

    if len(curves) == 0:
        continue

    curves = np.vstack(curves)

    group_results.append(
        {
            "name": group["name"],
            "label": group["label"],
            "color": group["color"],
            "n": curves.shape[0],
            "frequency_um_inv": freq_ref,
            "wavelength_um": wavelength_ref,
            "curves": curves,
            "mean_psd_um4": np.nanmean(curves, axis=0),
            "std_psd_um4": (
                np.nanstd(curves, axis=0, ddof=1)
                if curves.shape[0] > 1
                else np.zeros(curves.shape[1])
            ),
            "meta": pd.DataFrame(meta_rows),
        }
    )

if len(group_results) < 1:
    raise RuntimeError("No group PSD results were generated.")

# Reference for gain is the lowest nonzero strain group.
reference_result = group_results[0]
reference_psd = reference_result["mean_psd_um4"]

print()
print(f"Gain reference group: {reference_result['label']}")


# =============================================================================
# Save group means and traces
# =============================================================================

summary_rows = []
trace_rows = []

for result in group_results:
    wavelength = result["wavelength_um"]
    frequency = result["frequency_um_inv"]
    mean_psd = result["mean_psd_um4"]
    std_psd = result["std_psd_um4"]
    mean_gain = mean_psd / reference_psd

    for i in range(len(wavelength)):
        summary_rows.append(
            {
                "group": result["name"],
                "label": result["label"],
                "n": result["n"],
                "gain_reference_label": reference_result["label"],
                "frequency_um_inv": frequency[i],
                "wavelength_um": wavelength[i],
                "mean_psd_raw_delta_h_um4": mean_psd[i],
                "std_psd_raw_delta_h_um4": std_psd[i],
                "gain_vs_lowest_delta_h_group": mean_gain[i],
            }
        )

    for curve_index, curve in enumerate(result["curves"]):
        gain_curve = curve / reference_psd
        meta = result["meta"].iloc[curve_index]

        for i in range(len(wavelength)):
            trace_rows.append(
                {
                    "group": result["name"],
                    "label": result["label"],
                    "curve_index": curve_index,
                    "gain_reference_label": reference_result["label"],
                    "load_mpa": meta["load_mpa"],
                    "sample_type": meta["sample_type"],
                    "sample": meta["sample"],
                    "time_h": meta["time_h"],
                    "bulk_z_strain_percent": meta["bulk_z_strain_percent"],
                    "frequency_um_inv": frequency[i],
                    "wavelength_um": wavelength[i],
                    "psd_raw_delta_h_um4": curve[i],
                    "gain_vs_lowest_delta_h_group": gain_curve[i],
                }
            )

delta_psd_summary_df = pd.DataFrame(summary_rows)
delta_psd_trace_df = pd.DataFrame(trace_rows)

delta_psd_summary_df.to_csv(
    OUTPUT_DIR / "raw_delta_h_radial_psd_group_means_by_strain.csv",
    index=False,
)
delta_psd_trace_df.to_csv(
    OUTPUT_DIR / "raw_delta_h_radial_psd_individual_traces_by_strain.csv",
    index=False,
)

print(f"Saved: {OUTPUT_DIR / 'raw_delta_h_radial_psd_group_means_by_strain.csv'}")
print(f"Saved: {OUTPUT_DIR / 'raw_delta_h_radial_psd_individual_traces_by_strain.csv'}")


# =============================================================================
# Two-panel figure
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
ax_psd, ax_gain = axes

# -----------------------------------------------------------------------------
# Left panel: Delta h radial PSD traces and means
# -----------------------------------------------------------------------------

for result in group_results:
    wavelength = result["wavelength_um"]
    curves = result["curves"]
    mean_curve = result["mean_psd_um4"]
    color = result["color"]

    for curve in curves:
        valid = (
            np.isfinite(wavelength)
            & np.isfinite(curve)
            & (wavelength > 0)
            & (curve > 0)
        )

        if np.count_nonzero(valid) == 0:
            continue

        order = np.argsort(wavelength[valid])

        ax_psd.loglog(
            wavelength[valid][order],
            curve[valid][order],
            color=color,
            alpha=0.14,
            lw=0.7,
        )

    valid_mean = (
        np.isfinite(wavelength)
        & np.isfinite(mean_curve)
        & (wavelength > 0)
        & (mean_curve > 0)
    )

    if np.count_nonzero(valid_mean) > 0:
        order = np.argsort(wavelength[valid_mean])

        ax_psd.loglog(
            wavelength[valid_mean][order],
            mean_curve[valid_mean][order],
            color=color,
            lw=2.3,
            label=f"{result['label']} mean, n={result['n']}",
        )

ax_psd.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
ax_psd.set_ylabel(r"radial PSD of $\Delta h$, $C_{\Delta h}$ [$\mu$m$^4$]")
# ax_psd.set_title(r"A. Raw $\Delta h$ radial PSD")
ax_psd.grid(True, which="both", alpha=0.25)
ax_psd.legend(fontsize=6)

# -----------------------------------------------------------------------------
# Right panel: Delta h PSD gain
# Exclude the reference group from the gain plot because its gain is 1 by definition.
# -----------------------------------------------------------------------------

for result in group_results:
    if result["name"] == reference_result["name"]:
        continue

    wavelength = result["wavelength_um"]
    curves = result["curves"]
    mean_curve = result["mean_psd_um4"]
    color = result["color"]

    valid_base = (
        np.isfinite(wavelength)
        & (wavelength > 0)
        & np.isfinite(reference_psd)
        & (reference_psd > 0)
    )

    for curve in curves:
        gain_curve = curve / reference_psd

        valid = valid_base & np.isfinite(gain_curve) & (gain_curve > 0)

        if np.count_nonzero(valid) == 0:
            continue

        order = np.argsort(wavelength[valid])

        ax_gain.loglog(
            wavelength[valid][order],
            gain_curve[valid][order],
            color=color,
            alpha=0.12,
            lw=0.7,
        )

    mean_gain = mean_curve / reference_psd

    valid_mean = valid_base & np.isfinite(mean_gain) & (mean_gain > 0)

    if np.count_nonzero(valid_mean) > 0:
        order = np.argsort(wavelength[valid_mean])

        ax_gain.loglog(
            wavelength[valid_mean][order],
            mean_gain[valid_mean][order],
            color=color,
            lw=2.3,
            label=f"{result['label']} mean gain",
        )

ax_gain.axhline(1.0, color="0.45", lw=0.9, ls="--")
ax_gain.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
ax_gain.set_ylabel(
    rf"PSD gain, $C_{{\Delta h}}(\epsilon_p)/C_{{\Delta h}}$"
    rf"({reference_result['label']})"
)
# ax_gain.set_title(r"B. Raw $\Delta h$ PSD gain")
ax_gain.grid(True, which="both", alpha=0.25)
ax_gain.legend(fontsize=6)

# fig.suptitle(
#     rf"Raw $\Delta h$ PSD grouped by plastic strain; gain reference: {reference_result['label']}",
#     y=1.02,
# )
fig.tight_layout()

outpath = OUTPUT_DIR / "two_panel_raw_delta_h_psd_and_gain_by_strain.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# PSD band-gain plot using EBSD-based wavelength brackets
#
# Uses height-field PSD, not Delta h.
#
# PSD gain:
#   G_band(eps_p) = P_band(eps_p) / P_band(0)
#
# where P_band is the integrated PSD power of the signed, plane-leveled,
# mean-removed height field over the specified wavelength bracket.
#
# EBSD-based wavelength brackets:
#   sub-grain:       2*pixel_size <= lambda < 0.5*d_ref
#   grain-scale:     0.5*d_ref    <= lambda < 3*d_ref
#   mesoscale:       3*d_ref      <= lambda < 10*d_ref
#   macroscale:      10*d_ref     <= lambda < L_min/3
#
# For d_ref = 9.6 um:
#   sub-grain:       ~2.76--4.8 um
#   grain-scale:     4.8--28.8 um
#   mesoscale:       28.8--96 um
#   macroscale:      96--L_min/3 um
# =============================================================================

from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

EXCLUDED_LOADS = {475}

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

level = True
detrend_order = 1

INITIAL_TIME_TOL_H = 1.0e-8

# Use your EBSD representative total average ESD.
D_REF_UM = 9.6

# Long-wavelength cap to avoid field-of-view-scale artifacts.
MAX_WAVELENGTH_FRACTION_OF_SHORT_FOV = 1.0 / 3.0

# Number of strain groups shown as different bar colors.
N_STRAIN_GROUPS = 4

# Plot options.
USE_LOG_Y = True
SHOW_INDIVIDUAL_POINTS = True
ERROR_BAR = "std"  # "std", "sem", or "ci95"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

BAND_COLORS = {
    "sub-grain": "tab:blue",
    "grain-scale": "tab:orange",
    "mesoscale": "tab:green",
    "macroscale": "tab:red",
}

STRAIN_LABEL = r"bulk axial plastic strain, $\epsilon_p$ (%)"

# -----------------------------------------------------------------------------
# Load point table
# -----------------------------------------------------------------------------

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)

if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()
df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()
df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No data remain after filtering.")

print("Records used:")
print(df.groupby(["sample_type", "load_mpa"]).size())

# -----------------------------------------------------------------------------
# Height loading and plane leveling
# -----------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


# -----------------------------------------------------------------------------
# EBSD-based wavelength bands
# -----------------------------------------------------------------------------

first_height = load_plane_leveled_mean_removed_height(df.iloc[0]["height_path"])
n0, n1 = first_height.shape

L0_um = n0 * spacing_um
L1_um = n1 * spacing_um
L_min_um = min(L0_um, L1_um)

nyquist_shortest_wavelength_um = 2.0 * spacing_um
long_wavelength_cap_um = MAX_WAVELENGTH_FRACTION_OF_SHORT_FOV * L_min_um

bands_um = {
    "sub-grain": (
        nyquist_shortest_wavelength_um,
        0.5 * D_REF_UM,
    ),
    "grain-scale": (
        0.5 * D_REF_UM,
        3.0 * D_REF_UM,
    ),
    "mesoscale": (
        3.0 * D_REF_UM,
        10.0 * D_REF_UM,
    ),
    "macroscale": (
        10.0 * D_REF_UM,
        long_wavelength_cap_um,
    ),
}

# Clean bands so none are below Nyquist or above the FOV cap.
clean_bands_um = {}

for name, (lam_min, lam_max) in bands_um.items():
    lam_min = max(float(lam_min), nyquist_shortest_wavelength_um)
    lam_max = min(float(lam_max), long_wavelength_cap_um)

    if lam_max > lam_min:
        clean_bands_um[name] = (lam_min, lam_max)

bands_um = clean_bands_um
band_names = list(bands_um.keys())

if len(band_names) == 0:
    raise ValueError("No valid wavelength bands after applying Nyquist/FOV limits.")

print()
print(f"d_ref = {D_REF_UM:.3f} um")
print(f"Pixel spacing = {spacing_um:.4f} um")
print(f"Nyquist shortest wavelength = {nyquist_shortest_wavelength_um:.3f} um")
print(f"Cropped FOV = {L0_um:.1f} x {L1_um:.1f} um")
print(f"Long-wavelength cap L_min/3 = {long_wavelength_cap_um:.1f} um")
print("EBSD-based wavelength bands:")
for band_name, (lo, hi) in bands_um.items():
    print(f"  {band_name:14s}: {lo:8.3f} to {hi:8.3f} um")

# -----------------------------------------------------------------------------
# PSD band power functions
# -----------------------------------------------------------------------------


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))

    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um: np.ndarray, spacing_um_value: float):
    """
    Returns f0, f1, PSD2D, df0, df1.

    PSD units are um^4 if height and spacing are in um.
    Normalization approximately satisfies:
        sum(PSD2D) * df0 * df1 = mean((windowed height)^2)
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    df0 = 1.0 / (n0 * spacing_um_value)
    df1 = 1.0 / (n1 * spacing_um_value)

    return f0, f1, PSD, df0, df1


def band_powers_from_height(
    height_um: np.ndarray, bands: dict[str, tuple[float, float]]
):
    f0, f1, PSD, df0, df1 = psd2d_height(height_um, spacing_um)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength_um = 1.0 / FR

    out = {}

    for name, (lo, hi) in bands.items():
        mask = (
            np.isfinite(wavelength_um)
            & (FR > 0)
            & (wavelength_um >= lo)
            & (wavelength_um < hi)
        )

        power = float(np.nansum(PSD[mask]) * df0 * df1)

        out[f"power_{name}_um2"] = power
        out[f"rms_{name}_um"] = np.sqrt(power) if power >= 0 else np.nan
        out[f"modes_{name}"] = int(np.count_nonzero(mask))

    return out


# -----------------------------------------------------------------------------
# Compute per-sample band gain relative to initial scan
# -----------------------------------------------------------------------------

rows = []

for (sample_type, load, sample), g in df.groupby(["sample_type", "load_mpa", "sample"]):
    g = g.sort_values("time_h").copy()

    if len(g) < 2:
        continue

    initial = g.iloc[0]

    if float(initial["time_h"]) > INITIAL_TIME_TOL_H:
        continue

    H0 = load_plane_leveled_mean_removed_height(initial["height_path"])
    p0 = band_powers_from_height(H0, bands_um)

    for _, rec in g.iterrows():
        H = load_plane_leveled_mean_removed_height(rec["height_path"])
        p = band_powers_from_height(H, bands_um)

        row = {
            "load_mpa": load,
            "sample_type": sample_type,
            "sample": sample,
            "time_h": float(rec["time_h"]),
            "bulk_z_strain_percent": float(rec["bulk_z_strain_percent"]),
            "height_path": rec["height_path"],
        }

        if "delta_sa_um" in rec.index:
            row["delta_sa_um"] = rec["delta_sa_um"]

        for band_name in band_names:
            current_power = p[f"power_{band_name}_um2"]
            initial_power = p0[f"power_{band_name}_um2"]

            row[f"power_{band_name}_um2"] = current_power
            row[f"initial_power_{band_name}_um2"] = initial_power
            row[f"gain_{band_name}"] = (
                current_power / initial_power if initial_power > 0 else np.nan
            )
            row[f"rms_gain_{band_name}"] = (
                np.sqrt(current_power / initial_power)
                if initial_power > 0 and current_power >= 0
                else np.nan
            )

        rows.append(row)

gain_df = pd.DataFrame(rows)

if gain_df.empty:
    raise RuntimeError("No PSD gain rows were computed.")

gain_df.to_csv(OUTPUT_DIR / "psd_band_gain_ebsd_brackets.csv", index=False)
print()
print(f"Saved: {OUTPUT_DIR / 'psd_band_gain_ebsd_brackets.csv'}")

# -----------------------------------------------------------------------------
# Assign non-initial data to strain groups
# -----------------------------------------------------------------------------

plot_df = gain_df[gain_df["bulk_z_strain_percent"] > 0.05].copy()

if plot_df.empty:
    raise ValueError("No non-initial records available for plotting.")

q = np.unique(
    np.nanquantile(
        plot_df["bulk_z_strain_percent"].to_numpy(dtype=float),
        np.linspace(0.0, 1.0, N_STRAIN_GROUPS + 1),
    )
)

plot_df["strain_group"] = None
strain_group_labels = []

for i in range(len(q) - 1):
    lo = q[i]
    hi = q[i + 1]

    if i == len(q) - 2:
        mask = (plot_df["bulk_z_strain_percent"] >= lo) & (
            plot_df["bulk_z_strain_percent"] <= hi
        )
    else:
        mask = (plot_df["bulk_z_strain_percent"] >= lo) & (
            plot_df["bulk_z_strain_percent"] < hi
        )

    label = rf"{lo:.2g}--{hi:.2g}%"
    plot_df.loc[mask, "strain_group"] = label
    strain_group_labels.append(label)

plot_df = plot_df[plot_df["strain_group"].notna()].copy()

# -----------------------------------------------------------------------------
# Long-form table and summary
# -----------------------------------------------------------------------------

long_rows = []

for _, rec in plot_df.iterrows():
    for band_name in band_names:
        lo, hi = bands_um[band_name]

        long_rows.append(
            {
                "load_mpa": rec["load_mpa"],
                "sample_type": rec["sample_type"],
                "sample": rec["sample"],
                "time_h": rec["time_h"],
                "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                "strain_group": rec["strain_group"],
                "band": band_name,
                "lambda_min_um": lo,
                "lambda_max_um": hi,
                "gain": rec[f"gain_{band_name}"],
                "rms_gain": rec[f"rms_gain_{band_name}"],
            }
        )

gain_long = pd.DataFrame(long_rows)
gain_long.to_csv(OUTPUT_DIR / "psd_band_gain_ebsd_brackets_long.csv", index=False)

summary_rows = []

for (strain_group, band), g in gain_long.groupby(["strain_group", "band"]):
    y = pd.to_numeric(g["gain"], errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        continue

    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    sem = std / np.sqrt(y.size) if y.size > 1 else 0.0

    if ERROR_BAR == "std":
        err = std
    elif ERROR_BAR == "sem":
        err = sem
    elif ERROR_BAR == "ci95":
        err = 1.96 * sem
    else:
        raise ValueError("ERROR_BAR must be 'std', 'sem', or 'ci95'.")

    summary_rows.append(
        {
            "strain_group": strain_group,
            "band": band,
            "gain_mean": mean,
            "gain_err": err,
            "gain_std": std,
            "gain_sem": sem,
            "n": int(y.size),
        }
    )

summary = pd.DataFrame(summary_rows)

summary["strain_group"] = pd.Categorical(
    summary["strain_group"],
    categories=strain_group_labels,
    ordered=True,
)

summary["band"] = pd.Categorical(
    summary["band"],
    categories=band_names,
    ordered=True,
)

summary = summary.sort_values(["band", "strain_group"]).reset_index(drop=True)
summary.to_csv(OUTPUT_DIR / "psd_band_gain_ebsd_brackets_summary.csv", index=False)

print(f"Saved: {OUTPUT_DIR / 'psd_band_gain_ebsd_brackets_long.csv'}")
print(f"Saved: {OUTPUT_DIR / 'psd_band_gain_ebsd_brackets_summary.csv'}")

# -----------------------------------------------------------------------------
# Plot: grouped bar chart
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.6, 4.9))

x = np.arange(len(band_names), dtype=float)
n_groups = len(strain_group_labels)
bar_width = 0.82 / max(n_groups, 1)

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.18, 0.9, n_groups))

rng = np.random.default_rng(12345)

for j, strain_group in enumerate(strain_group_labels):
    group_summary = summary[summary["strain_group"] == strain_group].copy()

    offsets = x - 0.41 + bar_width / 2.0 + j * bar_width

    y_means = []
    y_errs = []

    for band_name in band_names:
        row = group_summary[group_summary["band"] == band_name]

        if row.empty:
            y_means.append(np.nan)
            y_errs.append(np.nan)
        else:
            y_means.append(float(row["gain_mean"].iloc[0]))
            y_errs.append(float(row["gain_err"].iloc[0]))

    y_means = np.asarray(y_means, dtype=float)
    y_errs = np.asarray(y_errs, dtype=float)

    ax.bar(
        offsets,
        y_means,
        width=bar_width * 0.92,
        color=colors[j],
        alpha=0.82,
        edgecolor="black",
        linewidth=0.5,
        label=rf"$\epsilon_p$ = {strain_group}",
        zorder=2,
    )

    ax.errorbar(
        offsets,
        y_means,
        yerr=y_errs,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2.5,
        zorder=3,
    )

    if SHOW_INDIVIDUAL_POINTS:
        for i, band_name in enumerate(band_names):
            vals = gain_long[
                (gain_long["strain_group"] == strain_group)
                & (gain_long["band"] == band_name)
            ]["gain"].to_numpy(dtype=float)

            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                continue

            jitter = rng.uniform(
                low=-0.35 * bar_width,
                high=0.35 * bar_width,
                size=vals.size,
            )

            ax.scatter(
                np.full(vals.size, offsets[i]) + jitter,
                vals,
                s=10,
                color="black",
                alpha=0.25,
                linewidths=0,
                zorder=4,
            )

ax.axhline(1.0, color="0.45", lw=1.0, ls="--", zorder=1)

band_tick_labels = []

for band_name in band_names:
    lo, hi = bands_um[band_name]
    band_tick_labels.append(f"{band_name}\n" rf"{lo:.1f}--{hi:.1f} $\mu$m")

ax.set_xticks(x)
ax.set_xticklabels(band_tick_labels)

if USE_LOG_Y:
    ax.set_yscale("log")

ax.set_xlabel("EBSD-based wavelength bracket")
ax.set_ylabel(r"PSD band-power gain, $P_{\mathrm{band}}/P_{\mathrm{band},0}$")
ax.set_title(
    "PSD gain by EBSD-based wavelength bracket\n"
    rf"$d_{{ref}}={D_REF_UM:.1f}\,\mu$m; 475 MPa excluded"
)

ax.grid(True, axis="y", which="both", alpha=0.25)
ax.legend(title="strain group", fontsize=7, title_fontsize=8)

fig.tight_layout()

outpath = OUTPUT_DIR / "psd_gain_by_ebsd_wavelength_bracket_bar_chart.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# -----------------------------------------------------------------------------
# Optional companion: connected-line version
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.8, 4.7))

for j, strain_group in enumerate(strain_group_labels):
    group_summary = summary[summary["strain_group"] == strain_group].copy()

    y_means = []
    y_errs = []

    for band_name in band_names:
        row = group_summary[group_summary["band"] == band_name]

        if row.empty:
            y_means.append(np.nan)
            y_errs.append(np.nan)
        else:
            y_means.append(float(row["gain_mean"].iloc[0]))
            y_errs.append(float(row["gain_err"].iloc[0]))

    y_means = np.asarray(y_means, dtype=float)
    y_errs = np.asarray(y_errs, dtype=float)

    ax.errorbar(
        x,
        y_means,
        yerr=y_errs,
        fmt="o-",
        color=colors[j],
        lw=1.8,
        ms=5,
        capsize=3,
        label=rf"$\epsilon_p$ = {strain_group}",
    )

ax.axhline(1.0, color="0.45", lw=1.0, ls="--")
ax.set_xticks(x)
ax.set_xticklabels(band_tick_labels)

if USE_LOG_Y:
    ax.set_yscale("log")

ax.set_xlabel("EBSD-based wavelength bracket")
ax.set_ylabel(r"PSD band-power gain, $P_{\mathrm{band}}/P_{\mathrm{band},0}$")
ax.set_title("PSD gain by EBSD-based wavelength bracket")
ax.grid(True, axis="y", which="both", alpha=0.25)
ax.legend(fontsize=7)

fig.tight_layout()

outpath = OUTPUT_DIR / "psd_gain_by_ebsd_wavelength_bracket_line_plot.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Height-distribution plot over the same plastic-strain brackets
#
# Produces height probability-density distributions for plane-leveled,
# mean-removed height fields:
#
#     h'(x,y) = h(x,y) - mean(h)
#
# grouped by plastic-strain brackets.
#
# Includes an initial bracket plus N positive-strain brackets.
# Excludes 475 MPa by default.
#
# Individual map distributions are shown as faint traces.
# Group-mean distributions are shown as bold curves.
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

EXCLUDED_LOADS = {475}

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

level = True
detrend_order = 1

# Same style of strain brackets as the PSD plots:
# initial group plus quantile-based positive-strain groups.
INITIAL_STRAIN_TOL_PERCENT = 0.05
N_POSITIVE_STRAIN_GROUPS = 4

# Histogram settings.
N_HEIGHT_BINS = 140

# To avoid one huge concatenated array, use random pixel subsampling only to
# estimate common histogram limits. Histograms themselves use all pixels.
N_PIXELS_PER_MAP_FOR_GLOBAL_LIMITS = 15000
GLOBAL_HEIGHT_PERCENTILE_LIMITS = (0.25, 99.75)

# Optional: thin number of individual traces per group.
MAX_INDIVIDUAL_TRACES_PER_GROUP = 35

# Also make standardized height plot?
MAKE_STANDARDIZED_HEIGHT_PLOT = True

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)

if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()
df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()
df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No data remain after filtering.")

print("Records used:")
print(df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height loading and plane leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


# =============================================================================
# Make strain brackets
# =============================================================================


def make_strain_groups(input_df: pd.DataFrame):
    d = input_df.copy()

    initial = d[np.abs(d["bulk_z_strain_percent"]) <= INITIAL_STRAIN_TOL_PERCENT].copy()
    positive = d[d["bulk_z_strain_percent"] > INITIAL_STRAIN_TOL_PERCENT].copy()

    groups = []

    if not initial.empty:
        groups.append(
            {
                "name": "initial",
                "label": r"initial, $\epsilon_p \approx 0$",
                "df": initial,
                "color": "black",
            }
        )

    if not positive.empty:
        q = np.unique(
            np.nanquantile(
                positive["bulk_z_strain_percent"].to_numpy(dtype=float),
                np.linspace(0.0, 1.0, N_POSITIVE_STRAIN_GROUPS + 1),
            )
        )

        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.18, 0.9, max(len(q) - 1, 1)))

        for i in range(len(q) - 1):
            lo = q[i]
            hi = q[i + 1]

            if i == len(q) - 2:
                mask = (positive["bulk_z_strain_percent"] >= lo) & (
                    positive["bulk_z_strain_percent"] <= hi
                )
            else:
                mask = (positive["bulk_z_strain_percent"] >= lo) & (
                    positive["bulk_z_strain_percent"] < hi
                )

            g = positive[mask].copy()

            if g.empty:
                continue

            groups.append(
                {
                    "name": f"group_{i+1}",
                    "label": rf"$\epsilon_p$ = {lo:.2g}--{hi:.2g}%",
                    "df": g,
                    "color": colors[i],
                }
            )

    return groups


groups = make_strain_groups(df)

print()
print("Height-distribution strain groups:")
for group in groups:
    print(f"  {group['label']}: n = {len(group['df'])}")

if len(groups) == 0:
    raise ValueError("No strain groups were generated.")


# =============================================================================
# Estimate global height limits for common bins
# =============================================================================

rng = np.random.default_rng(12345)
sampled_values = []

for group in groups:
    for _, rec in group["df"].iterrows():
        try:
            H = load_plane_leveled_mean_removed_height(rec["height_path"])
            values = H.ravel()
            values = values[np.isfinite(values)]

            if values.size == 0:
                continue

            if values.size > N_PIXELS_PER_MAP_FOR_GLOBAL_LIMITS:
                idx = rng.choice(
                    values.size,
                    size=N_PIXELS_PER_MAP_FOR_GLOBAL_LIMITS,
                    replace=False,
                )
                values = values[idx]

            sampled_values.append(values)

        except Exception as exc:
            warnings.warn(f"Failed loading {rec['height_path']} for limits: {exc}")

if len(sampled_values) == 0:
    raise RuntimeError("Could not load any height values.")

sampled_values = np.concatenate(sampled_values)

h_lo, h_hi = np.nanpercentile(sampled_values, GLOBAL_HEIGHT_PERCENTILE_LIMITS)
h_abs = max(abs(h_lo), abs(h_hi))
h_lo, h_hi = -h_abs, h_abs

height_bin_edges = np.linspace(h_lo, h_hi, N_HEIGHT_BINS + 1)
height_bin_centers = 0.5 * (height_bin_edges[:-1] + height_bin_edges[1:])

# Standardized height bins.
standardized_bin_edges = np.linspace(-5.0, 5.0, N_HEIGHT_BINS + 1)
standardized_bin_centers = 0.5 * (
    standardized_bin_edges[:-1] + standardized_bin_edges[1:]
)


# =============================================================================
# Compute per-map histograms and group summaries
# =============================================================================


def histogram_density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.full(len(edges) - 1, np.nan)

    hist, _ = np.histogram(values, bins=edges, density=True)

    return hist.astype(float)


height_hist_rows = []
standardized_hist_rows = []

for group_index, group in enumerate(groups):
    g = group["df"].copy()

    # Optional thinning for individual traces and computation.
    # Set MAX_INDIVIDUAL_TRACES_PER_GROUP = None to use all traces.
    if (
        MAX_INDIVIDUAL_TRACES_PER_GROUP is not None
        and len(g) > MAX_INDIVIDUAL_TRACES_PER_GROUP
    ):
        idx = rng.choice(
            g.index.to_numpy(),
            size=MAX_INDIVIDUAL_TRACES_PER_GROUP,
            replace=False,
        )
        g = g.loc[idx].copy()

    for _, rec in g.iterrows():
        try:
            H = load_plane_leveled_mean_removed_height(rec["height_path"])
            values = H.ravel()
            values = values[np.isfinite(values)]

            hist = histogram_density(values, height_bin_edges)

            for x, y in zip(height_bin_centers, hist):
                height_hist_rows.append(
                    {
                        "group_name": group["name"],
                        "group_label": group["label"],
                        "group_color": group["color"],
                        "load_mpa": rec["load_mpa"],
                        "sample_type": rec["sample_type"],
                        "sample": rec["sample"],
                        "time_h": rec["time_h"],
                        "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                        "height_um": x,
                        "density": y,
                    }
                )

            if MAKE_STANDARDIZED_HEIGHT_PLOT:
                sq = np.sqrt(np.nanmean(values**2))

                if np.isfinite(sq) and sq > 0:
                    zstd = values / sq
                    hist_std = histogram_density(zstd, standardized_bin_edges)

                    for x, y in zip(standardized_bin_centers, hist_std):
                        standardized_hist_rows.append(
                            {
                                "group_name": group["name"],
                                "group_label": group["label"],
                                "group_color": group["color"],
                                "load_mpa": rec["load_mpa"],
                                "sample_type": rec["sample_type"],
                                "sample": rec["sample"],
                                "time_h": rec["time_h"],
                                "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                                "standardized_height": x,
                                "density": y,
                            }
                        )

        except Exception as exc:
            warnings.warn(f"Failed histogram for {rec['height_path']}: {exc}")

height_hist_df = pd.DataFrame(height_hist_rows)
height_hist_df.to_csv(
    OUTPUT_DIR / "height_distribution_histograms_by_strain_group.csv", index=False
)

if MAKE_STANDARDIZED_HEIGHT_PLOT:
    standardized_hist_df = pd.DataFrame(standardized_hist_rows)
    standardized_hist_df.to_csv(
        OUTPUT_DIR / "standardized_height_distribution_histograms_by_strain_group.csv",
        index=False,
    )

print()
print(f"Saved: {OUTPUT_DIR / 'height_distribution_histograms_by_strain_group.csv'}")
if MAKE_STANDARDIZED_HEIGHT_PLOT:
    print(
        f"Saved: {OUTPUT_DIR / 'standardized_height_distribution_histograms_by_strain_group.csv'}"
    )


# =============================================================================
# Plot height distributions
# =============================================================================

fig, ax = plt.subplots(figsize=(7.2, 4.8))

for group in groups:
    group_hist = height_hist_df[height_hist_df["group_name"] == group["name"]].copy()

    if group_hist.empty:
        continue

    # Plot individual traces.
    for key, trace in group_hist.groupby(
        ["load_mpa", "sample_type", "sample", "time_h"]
    ):
        trace = trace.sort_values("height_um")

        ax.plot(
            trace["height_um"],
            trace["density"],
            color=group["color"],
            alpha=0.12 if group["name"] != "initial" else 0.10,
            lw=0.7,
        )

    # Mean and std across maps at each bin.
    summary = (
        group_hist.groupby("height_um", as_index=False)
        .agg(
            density_mean=("density", "mean"),
            density_std=("density", "std"),
            n=("density", "count"),
        )
        .sort_values("height_um")
    )

    summary["density_std"] = summary["density_std"].fillna(0.0)

    ax.plot(
        summary["height_um"],
        summary["density_mean"],
        color=group["color"],
        lw=2.3,
        label=f"{group['label']} mean, n={int(summary['n'].max())}",
    )

    ax.fill_between(
        summary["height_um"],
        np.maximum(summary["density_mean"] - summary["density_std"], 0.0),
        summary["density_mean"] + summary["density_std"],
        color=group["color"],
        alpha=0.10,
        linewidth=0,
    )

ax.set_xlabel(r"plane-leveled height, $h-\bar{h}$ [$\mu$m]")
ax.set_ylabel("probability density")
ax.set_title("Height distributions by plastic-strain bracket")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=7)

fig.tight_layout()

outpath = OUTPUT_DIR / "height_distribution_by_strain_bracket.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()


# =============================================================================
# Optional standardized height-distribution plot
# =============================================================================

if MAKE_STANDARDIZED_HEIGHT_PLOT:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for group in groups:
        group_hist = standardized_hist_df[
            standardized_hist_df["group_name"] == group["name"]
        ].copy()

        if group_hist.empty:
            continue

        # Plot individual traces.
        for key, trace in group_hist.groupby(
            ["load_mpa", "sample_type", "sample", "time_h"]
        ):
            trace = trace.sort_values("standardized_height")

            ax.plot(
                trace["standardized_height"],
                trace["density"],
                color=group["color"],
                alpha=0.12 if group["name"] != "initial" else 0.10,
                lw=0.7,
            )

        summary = (
            group_hist.groupby("standardized_height", as_index=False)
            .agg(
                density_mean=("density", "mean"),
                density_std=("density", "std"),
                n=("density", "count"),
            )
            .sort_values("standardized_height")
        )

        summary["density_std"] = summary["density_std"].fillna(0.0)

        ax.plot(
            summary["standardized_height"],
            summary["density_mean"],
            color=group["color"],
            lw=2.3,
            label=f"{group['label']} mean, n={int(summary['n'].max())}",
        )

        ax.fill_between(
            summary["standardized_height"],
            np.maximum(summary["density_mean"] - summary["density_std"], 0.0),
            summary["density_mean"] + summary["density_std"],
            color=group["color"],
            alpha=0.10,
            linewidth=0,
        )

    ax.set_xlabel(r"standardized height, $(h-\bar{h})/S_q$")
    ax.set_ylabel("probability density")
    ax.set_title("Standardized height distributions by plastic-strain bracket")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    fig.tight_layout()

    outpath = OUTPUT_DIR / "standardized_height_distribution_by_strain_bracket.png"
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath.resolve()}")

    plt.show()

# %%
# =============================================================================
# Six-panel figure: 2D normalized autocorrelation maps of raw Delta h
#
# Top row:    interrupted tests:   475, 525, 575 MPa
# Bottom row: uninterrupted tests: 500, 530, 588 MPa
#
# For each experiment, one representative sample is chosen automatically:
#   - sample must have an initial scan at t ~ 0 and at least one later scan
#   - final scan is used
#   - representative sample is the sample whose final plastic strain is closest
#     to the experiment median final plastic strain
#
# Delta h is computed as:
#   Delta h = h_final - h_initial
#
# No registration correction is applied.
#
# The plotted autocorrelation is:
#   C(Delta y, Delta z) =
#       <Delta h(y,z) Delta h(y+Delta y,z+Delta z)> / <Delta h^2>
#
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

# Individual height-map leveling.
level = True
detrend_order = 1

# Raw Delta h options.
RAW_DELTA_CROP_MARGIN_PIXELS = 0

# Plane-level Delta h after subtraction.
LEVEL_DELTA_H = True
DELTA_H_DETREND_ORDER = 1

# Initial-time tolerance.
INITIAL_TIME_TOL_H = 1.0e-8

# Autocorrelation plot lag limit.
# Use None for full lag range. A finite value makes the maps easier to interpret.
LAG_LIMIT_UM = 300.0

# Color scale for normalized autocorrelation.
AC_VMIN = -0.25
AC_VMAX = 1.0

# Experiment layout.
plot_grid = [
    [(475, "int"), (525, "int"), (575, "int")],
    [(500, "unint"), (530, "unint"), (588, "unint")],
]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)

if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()
df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

if "delta_sa_um" in df.columns:
    df["delta_sa_um"] = pd.to_numeric(df["delta_sa_um"], errors="coerce")

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

print("Records available:")
print(df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height-map reading and leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


def crop_margin(A: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return A

    if A.shape[0] <= 2 * margin or A.shape[1] <= 2 * margin:
        raise ValueError(f"Margin {margin} too large for shape {A.shape}.")

    return A[margin:-margin, margin:-margin]


def compute_raw_delta_h(initial_path: str | Path, final_path: str | Path) -> np.ndarray:
    H0 = load_plane_leveled_mean_removed_height(initial_path)
    Hf = load_plane_leveled_mean_removed_height(final_path)

    if H0.shape != Hf.shape:
        raise ValueError(f"Shape mismatch: initial {H0.shape}, final {Hf.shape}")

    H0 = crop_margin(H0, RAW_DELTA_CROP_MARGIN_PIXELS)
    Hf = crop_margin(Hf, RAW_DELTA_CROP_MARGIN_PIXELS)

    dH = Hf - H0

    if LEVEL_DELTA_H:
        dH = detrend_surface(dH, spacing_um, order=DELTA_H_DETREND_ORDER)

    dH = dH - np.nanmean(dH)

    return dH


# =============================================================================
# Representative sample selection
# =============================================================================


def select_representative_sample_for_experiment(
    table: pd.DataFrame,
    load_mpa: int | float,
    sample_type: str,
) -> dict:
    exp_df = table[
        (table["load_mpa"] == load_mpa)
        & (table["sample_type"].astype(str) == sample_type)
    ].copy()

    if exp_df.empty:
        raise ValueError(f"No records found for {load_mpa} MPa {sample_type}")

    candidates = []

    for sample, g in exp_df.groupby("sample"):
        g = g.sort_values("time_h").copy()

        if len(g) < 2:
            continue

        initial = g.iloc[0]
        final = g.iloc[-1]

        if float(initial["time_h"]) > INITIAL_TIME_TOL_H:
            continue

        if not np.isfinite(final["bulk_z_strain_percent"]):
            continue

        candidates.append(
            {
                "load_mpa": load_mpa,
                "sample_type": sample_type,
                "sample": sample,
                "initial_time_h": float(initial["time_h"]),
                "final_time_h": float(final["time_h"]),
                "initial_height_path": initial["height_path"],
                "final_height_path": final["height_path"],
                "final_bulk_z_strain_percent": float(final["bulk_z_strain_percent"]),
                "final_delta_sa_um": (
                    float(final["delta_sa_um"])
                    if "delta_sa_um" in final.index
                    and np.isfinite(final["delta_sa_um"])
                    else np.nan
                ),
            }
        )

    if len(candidates) == 0:
        raise ValueError(
            f"No candidate samples with initial and final scans for {load_mpa} MPa {sample_type}"
        )

    candidate_df = pd.DataFrame(candidates)
    median_strain = candidate_df["final_bulk_z_strain_percent"].median()

    candidate_df["distance_to_median_strain"] = (
        candidate_df["final_bulk_z_strain_percent"] - median_strain
    ).abs()

    selected = (
        candidate_df.sort_values(["distance_to_median_strain", "sample"])
        .iloc[0]
        .to_dict()
    )

    return selected


selected_records = []

for row in plot_grid:
    for load, sample_type in row:
        try:
            selected_records.append(
                select_representative_sample_for_experiment(df, load, sample_type)
            )
        except Exception as exc:
            warnings.warn(str(exc))

selected_df = pd.DataFrame(selected_records)
selected_df.to_csv(
    OUTPUT_DIR / "selected_representative_samples_for_delta_h_autocorrelation.csv",
    index=False,
)

print()
print("Selected representative samples:")
print(selected_df)
print(
    f"Saved: {OUTPUT_DIR / 'selected_representative_samples_for_delta_h_autocorrelation.csv'}"
)


# =============================================================================
# Autocorrelation calculation
# =============================================================================


def normalized_autocorrelation_2d(A: np.ndarray):
    """
    Computes normalized 2D autocorrelation using FFT.

    A shape:
        axis 0 = z/loading direction
        axis 1 = y/transverse direction

    Returns:
        z_lags_um, y_lags_um, ac
    """
    A = np.asarray(A, dtype=float)
    A = A - np.nanmean(A)

    # Fill nonfinite values with zero after mean removal.
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    F = np.fft.fft2(A)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)

    center0, center1 = np.array(ac.shape) // 2
    center_value = ac[center0, center1]

    if not np.isfinite(center_value) or center_value == 0:
        ac[:] = np.nan
    else:
        ac = ac / center_value

    n0, n1 = A.shape

    z_lags_um = (np.arange(n0) - center0) * spacing_um
    y_lags_um = (np.arange(n1) - center1) * spacing_um

    return z_lags_um, y_lags_um, ac


def first_one_over_e_length(lags: np.ndarray, values: np.ndarray):
    lags = np.asarray(lags, dtype=float)
    values = np.asarray(values, dtype=float)

    center = np.argmin(np.abs(lags))
    x = lags[center:]
    y = values[center:]

    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, False

    target = np.exp(-1.0)

    for k in range(1, len(x)):
        if y[k] <= target:
            x0, x1 = x[k - 1], x[k]
            y0, y1 = y[k - 1], y[k]

            if y1 == y0:
                return float(x1), True

            x_cross = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            return float(x_cross), True

    return float(x[-1]), False


def autocorrelation_summary_metrics(z_lags_um, y_lags_um, ac):
    center0 = np.argmin(np.abs(z_lags_um))
    center1 = np.argmin(np.abs(y_lags_um))

    ac_z = ac[:, center1]
    ac_y = ac[center0, :]

    xi_z, crossed_z = first_one_over_e_length(z_lags_um, ac_z)
    xi_y, crossed_y = first_one_over_e_length(y_lags_um, ac_y)

    return {
        "corr_length_parallel_z_um": xi_z,
        "corr_length_transverse_y_um": xi_y,
        "corr_length_parallel_z_crossed_1e": crossed_z,
        "corr_length_transverse_y_crossed_1e": crossed_y,
        "corr_length_anisotropy_z_over_y": (
            xi_z / xi_y
            if np.isfinite(xi_z) and np.isfinite(xi_y) and xi_y > 0
            else np.nan
        ),
    }


# =============================================================================
# Compute Delta h and autocorrelation maps for selected samples
# =============================================================================

ac_records = []
ac_maps = {}

for rec in selected_records:
    try:
        dH = compute_raw_delta_h(
            rec["initial_height_path"],
            rec["final_height_path"],
        )

        z_lags_um, y_lags_um, ac = normalized_autocorrelation_2d(dH)
        metrics = autocorrelation_summary_metrics(z_lags_um, y_lags_um, ac)

        key = (rec["load_mpa"], rec["sample_type"])
        ac_maps[key] = {
            "delta_h": dH,
            "z_lags_um": z_lags_um,
            "y_lags_um": y_lags_um,
            "ac": ac,
            "metadata": rec,
            "metrics": metrics,
        }

        ac_records.append(
            {
                **rec,
                "raw_delta_h_sa_um": float(np.nanmean(np.abs(dH - np.nanmean(dH)))),
                "raw_delta_h_sq_um": float(
                    np.sqrt(np.nanmean((dH - np.nanmean(dH)) ** 2))
                ),
                **metrics,
            }
        )

    except Exception as exc:
        warnings.warn(
            f"Failed autocorrelation for {rec['load_mpa']} MPa {rec['sample_type']} "
            f"sample {rec['sample']}: {exc}"
        )

ac_summary_df = pd.DataFrame(ac_records)
ac_summary_df.to_csv(
    OUTPUT_DIR / "delta_h_autocorrelation_representative_summary.csv", index=False
)

print()
print("Autocorrelation summary:")
print(ac_summary_df)
print(f"Saved: {OUTPUT_DIR / 'delta_h_autocorrelation_representative_summary.csv'}")


# =============================================================================
# Six-panel plot
# =============================================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(10.8, 6.8),
    constrained_layout=True,
)

image_for_colorbar = None

for r in range(2):
    for c in range(3):
        ax = axes[r, c]
        load, sample_type = plot_grid[r][c]
        key = (load, sample_type)

        if key not in ac_maps:
            ax.axis("off")
            ax.set_title(f"{load} MPa {sample_type}\nnot available")
            continue

        item = ac_maps[key]
        z_lags_um = item["z_lags_um"]
        y_lags_um = item["y_lags_um"]
        ac = item["ac"]
        rec = item["metadata"]
        metrics = item["metrics"]

        extent = [
            y_lags_um.min(),
            y_lags_um.max(),
            z_lags_um.min(),
            z_lags_um.max(),
        ]

        image_for_colorbar = ax.imshow(
            ac,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=AC_VMIN,
            vmax=AC_VMAX,
            interpolation="nearest",
            rasterized=True,
            aspect="equal",
        )

        if LAG_LIMIT_UM is not None:
            ax.set_xlim(-LAG_LIMIT_UM, LAG_LIMIT_UM)
            ax.set_ylim(-LAG_LIMIT_UM, LAG_LIMIT_UM)

        ax.axhline(0.0, color="white", lw=0.6, alpha=0.55)
        ax.axvline(0.0, color="white", lw=0.6, alpha=0.55)

        # Mark 1/e directional correlation lengths if available.
        xi_z = metrics["corr_length_parallel_z_um"]
        xi_y = metrics["corr_length_transverse_y_um"]

        if np.isfinite(xi_z):
            ax.axhline(xi_z, color="red", lw=0.9, ls="--", alpha=0.85)
            ax.axhline(-xi_z, color="red", lw=0.9, ls="--", alpha=0.85)

        if np.isfinite(xi_y):
            ax.axvline(xi_y, color="red", lw=0.9, ls="--", alpha=0.85)
            ax.axvline(-xi_y, color="red", lw=0.9, ls="--", alpha=0.85)

        ax.set_title(
            f"{load} MPa {sample_type}, sample {rec['sample']}\n"
            rf"$\epsilon_p$={rec['final_bulk_z_strain_percent']:.2f}%, "
            rf"$\xi_z/\xi_y$={metrics['corr_length_anisotropy_z_over_y']:.2f}"
        )

        ax.set_xlabel(r"transverse lag, $\Delta y$ [$\mu$m]")
        ax.set_ylabel(r"loading-direction lag, $\Delta z$ [$\mu$m]")

if image_for_colorbar is not None:
    cbar = fig.colorbar(
        image_for_colorbar,
        ax=axes.ravel().tolist(),
        shrink=0.88,
        pad=0.02,
    )
    cbar.set_label(r"normalized autocorrelation of $\Delta h$")

# fig.suptitle(
#     r"2D normalized autocorrelation maps of raw $\Delta h$"
#     "\nTop: interrupted; bottom: uninterrupted",
#     y=1.04,
# )

outpath = OUTPUT_DIR / "six_panel_delta_h_2d_autocorrelation_maps.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
from pathlib import Path

# =============================================================================
# Thresholded linear fit:
#   Delta Sa = K * max(eps_zz - eps_c, 0)
#
# Uses strain in percent units.
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Choose data source
# -------------------------------------------------------------------------
# Prefer scatter_df if it exists; otherwise use point_df.
if "scatter_df" in globals():
    fit_df = scatter_df.copy()
elif "point_df" in globals():
    fit_df = point_df.copy()
else:
    raise NameError("Neither scatter_df nor point_df exists in memory.")

required_cols = ["bulk_z_strain_percent", "delta_sa_um", "load_mpa"]
missing = [c for c in required_cols if c not in fit_df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

fit_df["bulk_z_strain_percent"] = pd.to_numeric(
    fit_df["bulk_z_strain_percent"], errors="coerce"
)
fit_df["delta_sa_um"] = pd.to_numeric(fit_df["delta_sa_um"], errors="coerce")
fit_df["load_mpa"] = pd.to_numeric(fit_df["load_mpa"], errors="coerce")

fit_df = fit_df[
    np.isfinite(fit_df["bulk_z_strain_percent"])
    & np.isfinite(fit_df["delta_sa_um"])
    & np.isfinite(fit_df["load_mpa"])
].copy()

# Optional: remove exactly initial points if you do not want repeated zeros
# dominating the threshold fit.
# Usually I keep them because they anchor Delta Sa = 0 at eps = 0.
KEEP_INITIAL_POINTS = True
if not KEEP_INITIAL_POINTS:
    fit_df = fit_df[fit_df["bulk_z_strain_percent"] > 0.05].copy()

# Optional: include/exclude 475 MPa.
# For estimating threshold, I recommend including 475 MPa because it provides
# the "no roughening below threshold" information.
INCLUDE_475_FOR_THRESHOLD = True
if not INCLUDE_475_FOR_THRESHOLD:
    fit_df = fit_df[fit_df["load_mpa"] != 475].copy()

x = fit_df["bulk_z_strain_percent"].to_numpy(dtype=float)
y = fit_df["delta_sa_um"].to_numpy(dtype=float)

# Prevent negative roughness increments from overly controlling the onset fit.
# If you want to preserve negative values from measurement scatter, set this False.
CLIP_NEGATIVE_DELTA_SA_FOR_FIT = False
if CLIP_NEGATIVE_DELTA_SA_FOR_FIT:
    y_fitdata = np.maximum(y, 0.0)
else:
    y_fitdata = y.copy()


# -------------------------------------------------------------------------
# Model definitions
# -------------------------------------------------------------------------
def threshold_linear(eps_percent, K, eps_c_percent):
    return K * np.maximum(eps_percent - eps_c_percent, 0.0)


def threshold_power(eps_percent, A, eps_c_percent, n):
    return A * np.maximum(eps_percent - eps_c_percent, 0.0) ** n


def post_threshold_linear_with_intercept(eps_percent, K, eps_c_percent, b):
    return b + K * np.maximum(eps_percent - eps_c_percent, 0.0)


# -------------------------------------------------------------------------
# Fit thresholded linear model
# -------------------------------------------------------------------------
x_max = np.nanmax(x)
y_max = np.nanmax(y_fitdata)

# Initial guesses.
K0 = y_max / max(x_max, 1e-12)
eps_c0 = np.nanpercentile(x, 10)

bounds_linear = ([0.0, 0.0], [np.inf, x_max])

popt, pcov = curve_fit(
    threshold_linear,
    x,
    y_fitdata,
    p0=(K0, eps_c0),
    bounds=bounds_linear,
    maxfev=200000,
)

K_hat, eps_c_hat = popt
K_se, eps_c_se = np.sqrt(np.diag(pcov))

y_hat = threshold_linear(x, K_hat, eps_c_hat)
resid = y_fitdata - y_hat

rss = np.sum(resid**2)
tss = np.sum((y_fitdata - np.mean(y_fitdata)) ** 2)
r2 = 1.0 - rss / tss if tss > 0 else np.nan

n_obs = len(x)
k_params = 2
aic = n_obs * np.log(rss / n_obs) + 2 * k_params if rss > 0 else -np.inf
bic = n_obs * np.log(rss / n_obs) + k_params * np.log(n_obs) if rss > 0 else -np.inf

print("Thresholded linear fit:")
print(f"  Delta Sa = K * max(eps_zz - eps_c, 0)")
print(f"  K       = {K_hat:.6g} um / % strain")
print(f"  eps_c   = {eps_c_hat:.6g} % strain")
print(f"  K SE    = {K_se:.6g}")
print(f"  eps_c SE= {eps_c_se:.6g} % strain")
print(f"  R^2     = {r2:.6f}")
print(f"  AIC     = {aic:.3f}")
print(f"  BIC     = {bic:.3f}")

# -------------------------------------------------------------------------
# Bootstrap confidence interval for eps_c
# -------------------------------------------------------------------------
N_BOOT = 5000
rng = np.random.default_rng(12345)

boot_params = []

for _ in range(N_BOOT):
    idx = rng.integers(0, len(x), size=len(x))
    xb = x[idx]
    yb = y_fitdata[idx]

    try:
        popt_b, _ = curve_fit(
            threshold_linear,
            xb,
            yb,
            p0=(K_hat, eps_c_hat),
            bounds=bounds_linear,
            maxfev=50000,
        )
        boot_params.append(popt_b)
    except Exception:
        continue

boot_params = np.asarray(boot_params, dtype=float)

if boot_params.shape[0] > 20:
    K_ci = np.percentile(boot_params[:, 0], [2.5, 50.0, 97.5])
    eps_c_ci = np.percentile(boot_params[:, 1], [2.5, 50.0, 97.5])

    print()
    print(f"Bootstrap successful fits: {boot_params.shape[0]} / {N_BOOT}")
    print(f"  K 95% CI       = [{K_ci[0]:.6g}, {K_ci[2]:.6g}] um / % strain")
    print(f"  eps_c 95% CI   = [{eps_c_ci[0]:.6g}, {eps_c_ci[2]:.6g}] % strain")
    print(f"  eps_c median   = {eps_c_ci[1]:.6g} % strain")
else:
    K_ci = np.array([np.nan, np.nan, np.nan])
    eps_c_ci = np.array([np.nan, np.nan, np.nan])
    print("Bootstrap failed or produced too few successful fits.")

# -------------------------------------------------------------------------
# Recommended minimum strain for correlation-length plot
# -------------------------------------------------------------------------
# Option 1: use fitted onset.
min_strain_fit_percent = eps_c_hat

# Option 2: conservative use upper 95% CI of onset.
if np.isfinite(eps_c_ci[2]):
    min_strain_conservative_percent = eps_c_ci[2]
else:
    min_strain_conservative_percent = eps_c_hat

print()
print("Recommended cutoffs for correlation-length plot:")
print(f"  fitted threshold cutoff:       eps_zz >= {min_strain_fit_percent:.4g} %")
print(
    f"  conservative threshold cutoff: eps_zz >= {min_strain_conservative_percent:.4g} %"
)

# -------------------------------------------------------------------------
# Compare with post-yield linear-only fit, excluding below-yield load
# -------------------------------------------------------------------------
post_df = fit_df[fit_df["load_mpa"] >= 500].copy()
xp = post_df["bulk_z_strain_percent"].to_numpy(dtype=float)
yp = post_df["delta_sa_um"].to_numpy(dtype=float)

if CLIP_NEGATIVE_DELTA_SA_FOR_FIT:
    yp = np.maximum(yp, 0.0)


def linear_free(eps_percent, m, b):
    return m * eps_percent + b


popt_post, pcov_post = curve_fit(
    linear_free,
    xp,
    yp,
    p0=(K_hat, 0.0),
    maxfev=200000,
)

m_post, b_post = popt_post
yp_hat = linear_free(xp, m_post, b_post)
rss_post = np.sum((yp - yp_hat) ** 2)
tss_post = np.sum((yp - np.mean(yp)) ** 2)
r2_post = 1.0 - rss_post / tss_post if tss_post > 0 else np.nan

print()
print("Post-yield free linear fit, load >= 500 MPa:")
print(f"  Delta Sa = m * eps_zz + b")
print(f"  m   = {m_post:.6g} um / % strain")
print(f"  b   = {b_post:.6g} um")
print(f"  R^2 = {r2_post:.6f}")

# -------------------------------------------------------------------------
# Save fit summary
# -------------------------------------------------------------------------
fit_summary = pd.DataFrame(
    [
        {
            "model": "threshold_linear",
            "K_um_per_percent": K_hat,
            "K_se": K_se,
            "eps_c_percent": eps_c_hat,
            "eps_c_se_percent": eps_c_se,
            "eps_c_ci_low_percent": eps_c_ci[0],
            "eps_c_ci_median_percent": eps_c_ci[1],
            "eps_c_ci_high_percent": eps_c_ci[2],
            "r2": r2,
            "aic": aic,
            "bic": bic,
            "n": n_obs,
            "min_strain_fit_percent": min_strain_fit_percent,
            "min_strain_conservative_percent": min_strain_conservative_percent,
        },
        {
            "model": "post_yield_linear_free_intercept",
            "K_um_per_percent": m_post,
            "K_se": np.sqrt(np.diag(pcov_post))[0],
            "eps_c_percent": np.nan,
            "eps_c_se_percent": np.nan,
            "eps_c_ci_low_percent": np.nan,
            "eps_c_ci_median_percent": np.nan,
            "eps_c_ci_high_percent": np.nan,
            "r2": r2_post,
            "aic": np.nan,
            "bic": np.nan,
            "n": len(xp),
            "min_strain_fit_percent": np.nan,
            "min_strain_conservative_percent": np.nan,
        },
    ]
)

fit_summary.to_csv(
    OUTPUT_DIR / "threshold_linear_delta_sa_fit_summary.csv", index=False
)
print()
print(f"Saved: {OUTPUT_DIR / 'threshold_linear_delta_sa_fit_summary.csv'}")

# -------------------------------------------------------------------------
# Plot fit
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.5), dpi=150)

# Color points by load.
load_colors = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

for load, g in fit_df.groupby("load_mpa"):
    ax.scatter(
        g["bulk_z_strain_percent"],
        g["delta_sa_um"],
        s=34,
        alpha=0.75,
        color=load_colors.get(load, "0.4"),
        label=f"{int(load)} MPa",
    )

x_plot = np.linspace(0.0, np.nanmax(x) * 1.05, 400)
y_plot = threshold_linear(x_plot, K_hat, eps_c_hat)

ax.plot(
    x_plot,
    y_plot,
    color="black",
    lw=2.2,
    label=rf"threshold-linear fit, $\epsilon_c={eps_c_hat:.2f}\%$",
)

# Bootstrap CI band if available.
if boot_params.shape[0] > 20:
    y_boot = np.array(
        [threshold_linear(x_plot, K_b, eps_c_b) for K_b, eps_c_b in boot_params]
    )

    y_low, y_high = np.percentile(y_boot, [2.5, 97.5], axis=0)

    ax.fill_between(
        x_plot,
        y_low,
        y_high,
        color="black",
        alpha=0.15,
        linewidth=0,
        label="95% bootstrap band",
    )

ax.axvline(
    eps_c_hat,
    color="black",
    lw=1.2,
    ls="--",
)

if np.isfinite(eps_c_ci[2]):
    ax.axvline(
        eps_c_ci[2],
        color="tab:red",
        lw=1.2,
        ls=":",
        label=rf"conservative cutoff, $\epsilon_c^{{97.5}}={eps_c_ci[2]:.2f}\%$",
    )

ax.axhline(0.0, color="0.5", lw=0.8)

ax.set_xlabel(r"$\epsilon_{zz}$ [%]")
ax.set_ylabel(r"$\Delta S_a$ [$\mu$m]")
ax.set_title(r"Thresholded linear roughening fit")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=7, ncol=2)

fig.tight_layout()

outpath = OUTPUT_DIR / "threshold_linear_delta_sa_vs_strain_fit.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# -------------------------------------------------------------------------
# Example: filter a correlation-length dataframe using the cutoff
# -------------------------------------------------------------------------
# If corr_df exists, this creates two filtered versions:
#   corr_df_threshold_fit
#   corr_df_threshold_conservative
# -------------------------------------------------------------------------
if "corr_df" in globals():
    corr_df_threshold_fit = corr_df[
        corr_df["bulk_z_strain_percent"] >= min_strain_fit_percent
    ].copy()

    corr_df_threshold_conservative = corr_df[
        corr_df["bulk_z_strain_percent"] >= min_strain_conservative_percent
    ].copy()

    print()
    print("Correlation dataframe filtering:")
    print(f"  original rows:              {len(corr_df)}")
    print(f"  fit-threshold rows:         {len(corr_df_threshold_fit)}")
    print(f"  conservative-threshold rows:{len(corr_df_threshold_conservative)}")

# %%
# =============================================================================
# Mean log10 PSD gain vs wavelength bracket
#
# - No threshold fit
# - No heatmap
# - Excludes 475 MPa
# - Uses d_ref from actual EBSD microstructural statistics
# - Uses 20 um wavelength brackets from 2x Nyquist to 30*d_ref
# - Plots mean log10 PSD gain for four strain groups
#
# PSD gain:
#   G_band = P_band(epsilon_zz) / P_band(0)
#
# Plotted:
#   mean[log10(G_band)]
# =============================================================================

from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

MICRO_STATS_CSV = Path(
    "/Users/gtdebru/mimosa/data/EBSD-AM316L/merged_stats/merged_feature_stats_all.csv"
)

EXCLUDED_LOADS_FOR_PSD = {475}

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))
level = True
detrend_order = 1

INITIAL_TIME_TOL_H = 1.0e-8
INITIAL_STRAIN_TOL_PERCENT = 0.05

# Use the arithmetic mean EquivalentDiameters over all features to reproduce ~9.65 um.
PLANE_SELECTION = "all"
USE_COMPLETE_NON_SURFACE_ONLY = False
D_REF_METHOD = "number_mean"

BRACKET_WIDTH_UM = 20.0
END_MULTIPLE_OF_D = 30.0

N_STRAIN_GROUPS = 4

SHOW_INDIVIDUAL_POINTS = False
ERROR_BAR = "sem"  # "std", "sem", or "ci95"
USE_SYMMETRIC_Y_LIMITS = False

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)
if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df_all = point_df.copy()

df_all["load_mpa"] = pd.to_numeric(df_all["load_mpa"], errors="coerce")
df_all["time_h"] = pd.to_numeric(df_all["time_h"], errors="coerce")
df_all["bulk_z_strain_percent"] = pd.to_numeric(
    df_all["bulk_z_strain_percent"],
    errors="coerce",
)

df_all = df_all[
    np.isfinite(df_all["load_mpa"])
    & np.isfinite(df_all["time_h"])
    & np.isfinite(df_all["bulk_z_strain_percent"])
].copy()

df_all = df_all.sort_values(
    ["sample_type", "load_mpa", "sample", "time_h"]
).reset_index(drop=True)

df_psd = df_all[~df_all["load_mpa"].isin(EXCLUDED_LOADS_FOR_PSD)].copy()

if df_psd.empty:
    raise ValueError("No data remain after PSD load filtering.")

print("PSD records used:")
print(df_psd.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height loading and plane leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


# =============================================================================
# d_ref from EBSD microstructural statistics
# =============================================================================


def load_microstructural_esd_statistics(
    csv_path: str | Path,
    *,
    plane_selection="all",
    complete_non_surface_only: bool = False,
) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find microstructural statistics file: {csv_path}"
        )

    stats = pd.read_csv(csv_path)

    required = {"plane", "EquivalentDiameters"}
    missing = required - set(stats.columns)

    if missing:
        raise ValueError(
            f"Microstructural statistics file is missing columns: {missing}"
        )

    stats = stats.copy()

    stats["EquivalentDiameters"] = pd.to_numeric(
        stats["EquivalentDiameters"],
        errors="coerce",
    )

    stats = stats[np.isfinite(stats["EquivalentDiameters"])].copy()
    stats = stats[stats["EquivalentDiameters"] > 0].copy()

    if plane_selection != "all":
        if isinstance(plane_selection, str):
            plane_selection = [plane_selection]
        stats = stats[stats["plane"].isin(plane_selection)].copy()

    if complete_non_surface_only:
        if "complete_non_surface" not in stats.columns:
            raise ValueError(
                "complete_non_surface_only=True, but file does not contain "
                "'complete_non_surface'."
            )

        mask = stats["complete_non_surface"].astype(str).str.lower().eq("true")
        stats = stats[mask].copy()

    if stats.empty:
        raise ValueError("No valid microstructural features remain after filtering.")

    return stats


micro_stats = load_microstructural_esd_statistics(
    MICRO_STATS_CSV,
    plane_selection=PLANE_SELECTION,
    complete_non_surface_only=USE_COMPLETE_NON_SURFACE_ONLY,
)

d = micro_stats["EquivalentDiameters"].to_numpy(dtype=float)

if D_REF_METHOD == "number_mean":
    D_REF_UM = float(np.mean(d))
elif D_REF_METHOD == "number_median":
    D_REF_UM = float(np.median(d))
else:
    raise ValueError(
        "This cleaned version only supports 'number_mean' or 'number_median'."
    )

d_ref_summary = pd.DataFrame(
    [
        {
            "n_features": len(d),
            "number_mean_um": float(np.mean(d)),
            "number_median_um": float(np.median(d)),
            "number_std_um": float(np.std(d, ddof=1)),
            "selected_d_ref_um": D_REF_UM,
            "selected_method": D_REF_METHOD,
            "plane_selection": str(PLANE_SELECTION),
            "complete_non_surface_only": USE_COMPLETE_NON_SURFACE_ONLY,
        }
    ]
)

d_ref_summary.to_csv(
    OUTPUT_DIR / "d_ref_from_microstructural_statistics_for_20um_brackets.csv",
    index=False,
)

print()
print("Representative grain-size calculation:")
print(f"  file: {MICRO_STATS_CSV}")
print(f"  method: {D_REF_METHOD}")
print(f"  complete_non_surface only: {USE_COMPLETE_NON_SURFACE_ONLY}")
print(f"  n features used: {len(d)}")
print(f"  number mean ESD:   {np.mean(d):.6g} um")
print(f"  number median ESD: {np.median(d):.6g} um")
print(f"  selected d_ref:    {D_REF_UM:.6g} um")


# =============================================================================
# Wavelength brackets: 20 um brackets from 2x Nyquist to 30*d_ref
# =============================================================================

first_height = load_plane_leveled_mean_removed_height(df_psd.iloc[0]["height_path"])
n0, n1 = first_height.shape

L0_um = n0 * spacing_um
L1_um = n1 * spacing_um
L_min_um = min(L0_um, L1_um)

nyquist_shortest_wavelength_um = 2.0 * spacing_um
twice_nyquist_shortest_wavelength_um = 2.0 * nyquist_shortest_wavelength_um

lambda_start_um = twice_nyquist_shortest_wavelength_um
lambda_end_um = END_MULTIPLE_OF_D * D_REF_UM

if lambda_end_um <= lambda_start_um:
    raise ValueError(
        f"Invalid wavelength range: start={lambda_start_um}, end={lambda_end_um}"
    )

edges = [lambda_start_um]

while edges[-1] < lambda_end_um:
    edges.append(min(edges[-1] + BRACKET_WIDTH_UM, lambda_end_um))

edges = np.asarray(edges, dtype=float)

bands_um = {}

for i in range(len(edges) - 1):
    name = f"{edges[i]:.1f}-{edges[i + 1]:.1f} µm"
    bands_um[name] = (edges[i], edges[i + 1])

band_names = list(bands_um.keys())

print()
print("20 um wavelength brackets:")
print(f"  pixel spacing = {spacing_um:.6g} um")
print(f"  Nyquist shortest wavelength = {nyquist_shortest_wavelength_um:.6g} um")
print(
    f"  twice Nyquist shortest wavelength = {twice_nyquist_shortest_wavelength_um:.6g} um"
)
print(f"  d_ref = {D_REF_UM:.6g} um")
print(f"  upper limit 30*d_ref = {lambda_end_um:.6g} um")
print(f"  FOV = {L0_um:.1f} x {L1_um:.1f} um")
print(f"  number of brackets = {len(band_names)}")


# =============================================================================
# PSD band power functions
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))

    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um: np.ndarray, spacing_um_value: float):
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    df0 = 1.0 / (n0 * spacing_um_value)
    df1 = 1.0 / (n1 * spacing_um_value)

    return f0, f1, PSD, df0, df1


def band_powers_from_height(
    height_um: np.ndarray, bands: dict[str, tuple[float, float]]
):
    f0, f1, PSD, df0, df1 = psd2d_height(height_um, spacing_um)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength_um = 1.0 / FR

    out = {}

    for name, (lo, hi) in bands.items():
        mask = (
            np.isfinite(wavelength_um)
            & (FR > 0)
            & (wavelength_um >= lo)
            & (wavelength_um < hi)
        )

        power = float(np.nansum(PSD[mask]) * df0 * df1)

        out[f"power_{name}_um2"] = power
        out[f"rms_{name}_um"] = np.sqrt(power) if power >= 0 else np.nan
        out[f"modes_{name}"] = int(np.count_nonzero(mask))

    return out


# =============================================================================
# Compute per-sample log10 PSD gain relative to initial scan
# =============================================================================

rows = []

for (sample_type, load, sample), g in df_psd.groupby(
    ["sample_type", "load_mpa", "sample"]
):
    g = g.sort_values("time_h").copy()

    if len(g) < 2:
        continue

    initial = g.iloc[0]

    if float(initial["time_h"]) > INITIAL_TIME_TOL_H:
        continue

    H0 = load_plane_leveled_mean_removed_height(initial["height_path"])
    p0 = band_powers_from_height(H0, bands_um)

    for _, rec in g.iterrows():
        H = load_plane_leveled_mean_removed_height(rec["height_path"])
        p = band_powers_from_height(H, bands_um)

        row = {
            "load_mpa": load,
            "sample_type": sample_type,
            "sample": sample,
            "time_h": float(rec["time_h"]),
            "bulk_z_strain_percent": float(rec["bulk_z_strain_percent"]),
            "height_path": rec["height_path"],
        }

        for band_name in band_names:
            current_power = p[f"power_{band_name}_um2"]
            initial_power = p0[f"power_{band_name}_um2"]

            gain = current_power / initial_power if initial_power > 0 else np.nan
            log10_gain = np.log10(gain) if np.isfinite(gain) and gain > 0 else np.nan

            row[f"power_{band_name}_um2"] = current_power
            row[f"initial_power_{band_name}_um2"] = initial_power
            row[f"gain_{band_name}"] = gain
            row[f"log10_gain_{band_name}"] = log10_gain

        rows.append(row)

gain_df = pd.DataFrame(rows)

if gain_df.empty:
    raise RuntimeError("No PSD gain rows were computed.")

gain_df.to_csv(OUTPUT_DIR / "psd_log10_gain_20um_brackets_all_rows.csv", index=False)

print()
print(f"Saved: {OUTPUT_DIR / 'psd_log10_gain_20um_brackets_all_rows.csv'}")


# =============================================================================
# Assign four non-initial strain groups
# =============================================================================

plot_df = gain_df[gain_df["bulk_z_strain_percent"] > INITIAL_STRAIN_TOL_PERCENT].copy()

if plot_df.empty:
    raise ValueError("No non-initial rows remain for plotting.")

q = np.unique(
    np.nanquantile(
        plot_df["bulk_z_strain_percent"].to_numpy(dtype=float),
        np.linspace(0.0, 1.0, N_STRAIN_GROUPS + 1),
    )
)

plot_df["strain_group"] = None
strain_group_labels = []

for i in range(len(q) - 1):
    lo = q[i]
    hi = q[i + 1]

    if i == len(q) - 2:
        mask = (plot_df["bulk_z_strain_percent"] >= lo) & (
            plot_df["bulk_z_strain_percent"] <= hi
        )
    else:
        mask = (plot_df["bulk_z_strain_percent"] >= lo) & (
            plot_df["bulk_z_strain_percent"] < hi
        )

    label = rf"{lo:.2g}--{hi:.2g}%"
    plot_df.loc[mask, "strain_group"] = label
    strain_group_labels.append(label)

plot_df = plot_df[plot_df["strain_group"].notna()].copy()


# =============================================================================
# Long-form table and mean summary
# =============================================================================

long_rows = []

for _, rec in plot_df.iterrows():
    for band_name in band_names:
        lo, hi = bands_um[band_name]
        val = rec[f"log10_gain_{band_name}"]

        long_rows.append(
            {
                "load_mpa": rec["load_mpa"],
                "sample_type": rec["sample_type"],
                "sample": rec["sample"],
                "time_h": rec["time_h"],
                "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                "strain_group": rec["strain_group"],
                "band": band_name,
                "lambda_min_um": lo,
                "lambda_max_um": hi,
                "lambda_center_um": 0.5 * (lo + hi),
                "log10_gain": val,
                "gain": 10.0 ** val if np.isfinite(val) else np.nan,
            }
        )

gain_long = pd.DataFrame(long_rows)
gain_long = gain_long[np.isfinite(gain_long["log10_gain"])].copy()

gain_long.to_csv(OUTPUT_DIR / "psd_log10_gain_20um_brackets_long.csv", index=False)

summary_rows = []

for (strain_group, band), g in gain_long.groupby(["strain_group", "band"]):
    y = g["log10_gain"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        continue

    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    sem = std / np.sqrt(y.size) if y.size > 1 else 0.0

    if ERROR_BAR == "std":
        err = std
    elif ERROR_BAR == "sem":
        err = sem
    elif ERROR_BAR == "ci95":
        err = 1.96 * sem
    else:
        raise ValueError("ERROR_BAR must be 'std', 'sem', or 'ci95'.")

    first = g.iloc[0]

    summary_rows.append(
        {
            "strain_group": strain_group,
            "band": band,
            "lambda_min_um": first["lambda_min_um"],
            "lambda_max_um": first["lambda_max_um"],
            "lambda_center_um": first["lambda_center_um"],
            "log10_gain_mean": mean,
            "log10_gain_err": err,
            "log10_gain_std": std,
            "log10_gain_sem": sem,
            "n": int(y.size),
        }
    )

summary = pd.DataFrame(summary_rows)

summary["strain_group"] = pd.Categorical(
    summary["strain_group"],
    categories=strain_group_labels,
    ordered=True,
)

summary["band"] = pd.Categorical(
    summary["band"],
    categories=band_names,
    ordered=True,
)

summary = summary.sort_values(["strain_group", "lambda_center_um"]).reset_index(
    drop=True
)

summary.to_csv(OUTPUT_DIR / "psd_log10_gain_20um_brackets_summary.csv", index=False)

print(f"Saved: {OUTPUT_DIR / 'psd_log10_gain_20um_brackets_long.csv'}")
print(f"Saved: {OUTPUT_DIR / 'psd_log10_gain_20um_brackets_summary.csv'}")


# =============================================================================
# Plot: mean log10 PSD gain vs wavelength bracket for each strain group
# =============================================================================

fig, ax = plt.subplots(figsize=(8.6, 4.8))

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.18, 0.9, len(strain_group_labels)))

rng = np.random.default_rng(12345)

for j, strain_group in enumerate(strain_group_labels):
    s = summary[summary["strain_group"] == strain_group].copy()
    s = s.sort_values("lambda_center_um")

    ax.errorbar(
        s["lambda_center_um"],
        s["log10_gain_mean"],
        yerr=s["log10_gain_err"],
        fmt="o-",
        lw=1.8,
        ms=4.5,
        capsize=3,
        color=colors[j],
        label=rf"$\epsilon_{{zz}}$ = {strain_group}",
    )

    if SHOW_INDIVIDUAL_POINTS:
        g = gain_long[gain_long["strain_group"] == strain_group].copy()

        for _, row in s.iterrows():
            vals = g[g["band"] == row["band"]]["log10_gain"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                continue

            jitter = rng.normal(
                loc=0.0,
                scale=0.7,
                size=vals.size,
            )

            ax.scatter(
                np.full(vals.size, row["lambda_center_um"]) + jitter,
                vals,
                s=8,
                color=colors[j],
                alpha=0.18,
                linewidths=0,
            )

ax.axhline(0.0, color="0.45", lw=1.0, ls="--")

ax.set_xlabel(r"wavelength bracket center, $\lambda$ [$\mu$m]")
ax.set_ylabel(r"mean $\log_{10}$ PSD gain")
ax.set_title(
    r"Mean $\log_{10}$ PSD gain in 20 $\mu$m wavelength brackets"
    "\n"
    rf"brackets from $4\Delta x$ to $30d_{{ref}}$; "
    rf"$d_{{ref}}={D_REF_UM:.2f}\,\mu$m"
)

ax.grid(True, alpha=0.25)
ax.legend(fontsize=7)

if USE_SYMMETRIC_Y_LIMITS:
    all_y = gain_long["log10_gain"].to_numpy(dtype=float)
    all_y = all_y[np.isfinite(all_y)]

    if all_y.size > 0:
        qlo, qhi = np.percentile(all_y, [2.0, 98.0])
        ymax = max(abs(qlo), abs(qhi), 0.25)
        ax.set_ylim(-1.1 * ymax, 1.1 * ymax)

fig.tight_layout()

outpath = OUTPUT_DIR / "mean_log10_psd_gain_20um_brackets_by_strain.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Strain sensitivity of PSD gain vs wavelength
# =============================================================================
#
# Goal:
#   Use individual sample/time observations, not strain-group means.
#   For each radial PSD wavelength, fit:
#
#       log10[ C_h(lambda, epsilon) / C_h(lambda, 0) ]
#           = beta(lambda) * epsilon_zz
#
#   where epsilon_zz is in percent strain.
#
#   beta(lambda) is the strain sensitivity of PSD gain at each wavelength:
#
#       units: decades per percent strain
#
#   Interpretation:
#       beta = 0.05 means PSD gain increases by 10^0.05 per 1% strain.
#
# Outputs:
#   1. psd_gain_strain_sensitivity_vs_wavelength.csv
#   2. psd_gain_strain_sensitivity_vs_wavelength.png
#   3. psd_gain_sensitivity_fit_data_long.csv
#
# Notes:
#   - PSD is computed from signed, plane-leveled, mean-removed height fields.
#   - PSD gain is computed specimen-by-specimen relative to that same specimen's
#     initial scan.
#   - The fit is performed separately at each wavelength.
#   - Bootstrap confidence intervals resample specimens, not wavelength bins.
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.ndimage import distance_transform_edt

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

# Exclude sub-yield load if desired.
EXCLUDED_LOADS = {475}

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

level = True
detrend_order = 1

INITIAL_TIME_TOL_H = 1.0e-8

# Wavelength range for radial PSD sensitivity.
LAMBDA_MIN_UM = 3.0
LAMBDA_MAX_UM = 400.0
N_WAVELENGTH_BINS = 90

# Long-wavelength bins have few Fourier modes. Use 1 if you want to retain
# wavelengths approaching hundreds of microns.
MIN_MODES_PER_BIN = 1

# Fit options.
# If False, fit is constrained through origin:
#   log10(gain) = beta * strain
#
# If True:
#   log10(gain) = alpha + beta * strain
#
# For PSD gain relative to the initial state, FIT_INTERCEPT=False is usually
# more physically consistent because log10(gain)=0 at strain=0.
FIT_INTERCEPT = False

# Exclude exact initial points from the regression?
# For FIT_INTERCEPT=False, initial points do not affect the slope because x=0.
EXCLUDE_INITIAL_FROM_FIT = False
INITIAL_STRAIN_TOL_PERCENT = 0.05

# Cluster bootstrap by specimen.
DO_CLUSTER_BOOTSTRAP = True
N_BOOT = 2000
BOOT_SEED = 12345
CONFIDENCE_LEVEL = 0.95

# Plot options.
SHOW_ZERO_LINE = True
SHOW_GAIN_FACTOR_AXIS = True

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)
if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()

df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()

df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No data remain after filtering.")

df["specimen_id"] = (
    df["sample_type"].astype(str)
    + "_"
    + df["load_mpa"].astype(int).astype(str)
    + "MPa_"
    + df["sample"].astype(str)
)

print("Records used:")
print(df.groupby(["sample_type", "load_mpa"]).size())
print(f"Number of specimens: {df['specimen_id'].nunique()}")


# =============================================================================
# Height loading and plane leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD functions with explicit wavelength bins
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))
    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um: np.ndarray, spacing_um_value: float):
    """
    Returns f0, f1, PSD2D, df0, df1.

    PSD units are um^4 if height and spacing are in um.
    Normalization approximately satisfies:
        sum(PSD2D) * df0 * df1 = mean((windowed height)^2)
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    df0 = 1.0 / (n0 * spacing_um_value)
    df1 = 1.0 / (n1 * spacing_um_value)

    return f0, f1, PSD, df0, df1


def wavelength_bin_edges(lambda_min_um: float, lambda_max_um: float, n_bins: int):
    return np.logspace(
        np.log10(lambda_min_um),
        np.log10(lambda_max_um),
        n_bins + 1,
    )


def radial_psd_from_height_wavelength_binned(
    height_um: np.ndarray,
    spacing_um_value: float,
    lambda_edges_um: np.ndarray,
    min_modes: int = 1,
):
    """
    Radial PSD using explicit wavelength bins.

    Wavelength bin:
        lambda_edges[i] <= lambda < lambda_edges[i+1]
    """
    f0, f1, PSD, df0, df1 = psd2d_height(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / FR

    nbins = len(lambda_edges_um) - 1
    lambda_center = np.sqrt(lambda_edges_um[:-1] * lambda_edges_um[1:])
    frequency_center = 1.0 / lambda_center

    psd_radial = np.full(nbins, np.nan)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        lo = lambda_edges_um[i]
        hi = lambda_edges_um[i + 1]

        mask = (
            np.isfinite(wavelength)
            & np.isfinite(PSD)
            & (FR > 0)
            & (wavelength >= lo)
            & (wavelength < hi)
        )

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            psd_radial[i] = np.nanmean(PSD[mask])

    return frequency_center, lambda_center, psd_radial, modes


lambda_edges = wavelength_bin_edges(
    LAMBDA_MIN_UM,
    LAMBDA_MAX_UM,
    N_WAVELENGTH_BINS,
)


# =============================================================================
# Compute specimen-normalized PSD gains
# =============================================================================

gain_rows = []

for specimen_id, g in df.groupby("specimen_id"):
    g = g.sort_values("time_h").copy()

    if len(g) < 2:
        continue

    initial = g.iloc[0]

    if float(initial["time_h"]) > INITIAL_TIME_TOL_H:
        continue

    try:
        H0 = load_plane_leveled_mean_removed_height(initial["height_path"])
        f0, wavelength0, psd0, modes0 = radial_psd_from_height_wavelength_binned(
            H0,
            spacing_um,
            lambda_edges,
            min_modes=MIN_MODES_PER_BIN,
        )
    except Exception as exc:
        warnings.warn(f"Initial PSD failed for specimen {specimen_id}: {exc}")
        continue

    for _, rec in g.iterrows():
        try:
            H = load_plane_leveled_mean_removed_height(rec["height_path"])
            f, wavelength, psd, modes = radial_psd_from_height_wavelength_binned(
                H,
                spacing_um,
                lambda_edges,
                min_modes=MIN_MODES_PER_BIN,
            )

            gain = psd / psd0

            for i in range(len(wavelength)):
                if not (
                    np.isfinite(wavelength[i])
                    and np.isfinite(gain[i])
                    and gain[i] > 0
                    and np.isfinite(psd0[i])
                    and psd0[i] > 0
                ):
                    continue

                gain_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "load_mpa": rec["load_mpa"],
                        "sample_type": rec["sample_type"],
                        "sample": rec["sample"],
                        "time_h": rec["time_h"],
                        "bulk_z_strain_percent": rec["bulk_z_strain_percent"],
                        "frequency_um_inv": f[i],
                        "wavelength_um": wavelength[i],
                        "wavelength_index": i,
                        "psd_um4": psd[i],
                        "initial_psd_um4": psd0[i],
                        "gain": gain[i],
                        "log10_gain": np.log10(gain[i]),
                        "modes": modes[i],
                    }
                )

        except Exception as exc:
            warnings.warn(
                f"PSD gain failed for specimen {specimen_id}, time={rec['time_h']}: {exc}"
            )

gain_df = pd.DataFrame(gain_rows)

if gain_df.empty:
    raise RuntimeError("No PSD gain rows were computed.")

if EXCLUDE_INITIAL_FROM_FIT:
    gain_df = gain_df[
        gain_df["bulk_z_strain_percent"] > INITIAL_STRAIN_TOL_PERCENT
    ].copy()

gain_df.to_csv(
    OUTPUT_DIR / "psd_gain_sensitivity_fit_data_long.csv",
    index=False,
)

print()
print(f"Saved: {OUTPUT_DIR / 'psd_gain_sensitivity_fit_data_long.csv'}")
print(f"Gain rows: {len(gain_df)}")
print(f"Specimens in gain table: {gain_df['specimen_id'].nunique()}")


# =============================================================================
# Fit log10 gain vs continuous strain at each wavelength
# =============================================================================


def fit_slope_at_wavelength(d: pd.DataFrame, fit_intercept: bool):
    x = d["bulk_z_strain_percent"].to_numpy(dtype=float)
    y = d["log10_gain"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return {
            "success": False,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "n": len(x),
        }

    if fit_intercept:
        coeff = np.polyfit(x, y, deg=1)
        slope = float(coeff[0])
        intercept = float(coeff[1])
        yhat = slope * x + intercept
    else:
        denom = np.sum(x**2)
        if denom <= 0:
            return {
                "success": False,
                "slope": np.nan,
                "intercept": 0.0,
                "r2": np.nan,
                "n": len(x),
            }

        slope = float(np.sum(x * y) / denom)
        intercept = 0.0
        yhat = slope * x

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "success": True,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n": len(x),
    }


def cluster_bootstrap_slope(
    d: pd.DataFrame,
    fit_intercept: bool,
    n_boot: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    specimen_ids = np.array(sorted(d["specimen_id"].unique()))

    if specimen_ids.size < 3:
        return np.array([])

    boot_slopes = []

    grouped = {sid: d[d["specimen_id"] == sid] for sid in specimen_ids}

    for _ in range(n_boot):
        sampled_ids = rng.choice(specimen_ids, size=specimen_ids.size, replace=True)
        boot_df = pd.concat([grouped[sid] for sid in sampled_ids], ignore_index=True)

        fit = fit_slope_at_wavelength(boot_df, fit_intercept)

        if fit["success"] and np.isfinite(fit["slope"]):
            boot_slopes.append(fit["slope"])

    return np.asarray(boot_slopes, dtype=float)


sensitivity_rows = []

alpha = 1.0 - CONFIDENCE_LEVEL
ci_percentiles = [100.0 * alpha / 2.0, 50.0, 100.0 * (1.0 - alpha / 2.0)]

for wavelength_index, dlam in gain_df.groupby("wavelength_index"):
    dlam = dlam.copy()

    fit = fit_slope_at_wavelength(dlam, FIT_INTERCEPT)

    wavelength_um = float(np.nanmean(dlam["wavelength_um"]))
    frequency_um_inv = float(np.nanmean(dlam["frequency_um_inv"]))
    n_obs = int(len(dlam))
    n_specimens = int(dlam["specimen_id"].nunique())

    slope_ci_low = np.nan
    slope_ci_median = np.nan
    slope_ci_high = np.nan

    if fit["success"] and DO_CLUSTER_BOOTSTRAP:
        boot_slopes = cluster_bootstrap_slope(
            dlam,
            FIT_INTERCEPT,
            N_BOOT,
            seed=BOOT_SEED + int(wavelength_index),
        )

        if boot_slopes.size > 20:
            slope_ci_low, slope_ci_median, slope_ci_high = np.percentile(
                boot_slopes,
                ci_percentiles,
            )

    sensitivity_rows.append(
        {
            "wavelength_index": int(wavelength_index),
            "wavelength_um": wavelength_um,
            "frequency_um_inv": frequency_um_inv,
            "slope_log10_gain_per_percent_strain": fit["slope"],
            "intercept_log10_gain": fit["intercept"],
            "r2": fit["r2"],
            "n_observations": n_obs,
            "n_specimens": n_specimens,
            "slope_ci_low": slope_ci_low,
            "slope_ci_median": slope_ci_median,
            "slope_ci_high": slope_ci_high,
            "fit_intercept": FIT_INTERCEPT,
        }
    )

sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_df = sensitivity_df.sort_values("wavelength_um").reset_index(drop=True)

sensitivity_df["gain_factor_per_percent_strain"] = (
    10.0 ** sensitivity_df["slope_log10_gain_per_percent_strain"]
)
sensitivity_df["gain_factor_ci_low"] = 10.0 ** sensitivity_df["slope_ci_low"]
sensitivity_df["gain_factor_ci_high"] = 10.0 ** sensitivity_df["slope_ci_high"]

sensitivity_df.to_csv(
    OUTPUT_DIR / "psd_gain_strain_sensitivity_vs_wavelength.csv",
    index=False,
)

print(f"Saved: {OUTPUT_DIR / 'psd_gain_strain_sensitivity_vs_wavelength.csv'}")


# =============================================================================
# Plot strain sensitivity vs wavelength
# =============================================================================

plot_df = sensitivity_df[
    np.isfinite(sensitivity_df["wavelength_um"])
    & np.isfinite(sensitivity_df["slope_log10_gain_per_percent_strain"])
].copy()

fig, ax = plt.subplots(figsize=(7.2, 4.8))

x = plot_df["wavelength_um"].to_numpy(dtype=float)
y = plot_df["slope_log10_gain_per_percent_strain"].to_numpy(dtype=float)

if DO_CLUSTER_BOOTSTRAP and np.any(np.isfinite(plot_df["slope_ci_low"])):
    y_low = plot_df["slope_ci_low"].to_numpy(dtype=float)
    y_high = plot_df["slope_ci_high"].to_numpy(dtype=float)

    valid_ci = np.isfinite(x) & np.isfinite(y_low) & np.isfinite(y_high)

    ax.fill_between(
        x[valid_ci],
        y_low[valid_ci],
        y_high[valid_ci],
        color="tab:blue",
        alpha=0.20,
        linewidth=0,
        label=f"{int(CONFIDENCE_LEVEL * 100)}% specimen-bootstrap CI",
    )

ax.plot(
    x,
    y,
    color="tab:blue",
    lw=2.0,
    label="fit slope",
)

ax.scatter(
    x,
    y,
    color="tab:blue",
    s=18,
    alpha=0.8,
)

if SHOW_ZERO_LINE:
    ax.axhline(0.0, color="0.45", lw=1.0, ls="--")

ax.set_xscale("log")
ax.set_xlim(LAMBDA_MIN_UM, LAMBDA_MAX_UM)

x_ticks = np.array([3, 5, 10, 20, 50, 100, 200, 300, 400], dtype=float)
x_ticks = x_ticks[(x_ticks >= LAMBDA_MIN_UM) & (x_ticks <= LAMBDA_MAX_UM)]

ax.xaxis.set_major_locator(FixedLocator(x_ticks))
ax.xaxis.set_major_formatter(FixedFormatter([f"{int(t)}" for t in x_ticks]))
ax.xaxis.set_minor_formatter(NullFormatter())

ax.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
ax.set_ylabel(
    r"strain sensitivity, " r"$d\log_{10}(C_h/C_{h,0})/d\epsilon_{zz}$ [decades / %]"
)

fit_type = "free-intercept" if FIT_INTERCEPT else "zero-intercept"
ax.set_title(f"PSD gain strain sensitivity vs wavelength ({fit_type} fit)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=7)

fig.tight_layout()

outpath = OUTPUT_DIR / "psd_gain_strain_sensitivity_vs_wavelength.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()


# =============================================================================
# Optional secondary axis: gain factor per 1% strain
# =============================================================================

if SHOW_GAIN_FACTOR_AXIS:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    y_factor = plot_df["gain_factor_per_percent_strain"].to_numpy(dtype=float)

    if DO_CLUSTER_BOOTSTRAP and np.any(np.isfinite(plot_df["gain_factor_ci_low"])):
        y_low = plot_df["gain_factor_ci_low"].to_numpy(dtype=float)
        y_high = plot_df["gain_factor_ci_high"].to_numpy(dtype=float)

        valid_ci = np.isfinite(x) & np.isfinite(y_low) & np.isfinite(y_high)

        ax.fill_between(
            x[valid_ci],
            y_low[valid_ci],
            y_high[valid_ci],
            color="tab:purple",
            alpha=0.20,
            linewidth=0,
            label=f"{int(CONFIDENCE_LEVEL * 100)}% specimen-bootstrap CI",
        )

    ax.plot(
        x,
        y_factor,
        color="tab:purple",
        lw=2.0,
        label="gain factor per 1% strain",
    )

    ax.scatter(
        x,
        y_factor,
        color="tab:purple",
        s=18,
        alpha=0.8,
    )

    ax.axhline(1.0, color="0.45", lw=1.0, ls="--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(LAMBDA_MIN_UM, LAMBDA_MAX_UM)

    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{int(t)}" for t in x_ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
    ax.set_ylabel(r"PSD gain factor per 1% strain, $10^\beta$")
    ax.set_title("PSD gain factor per unit strain vs wavelength")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)

    fig.tight_layout()

    outpath = OUTPUT_DIR / "psd_gain_factor_per_percent_strain_vs_wavelength.png"
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath.resolve()}")

    plt.show()

# %%
# =============================================================================
# Per-sample RADIAL autocorrelation plots with radial correlation length marked
# =============================================================================
#
# Saves one figure per sample under:
#
#   /Users/gtdebru/mimosa/data/{load}mpa_{sample_type}/10x/{sample_id}/
#
# Example:
#   /Users/gtdebru/mimosa/data/475mpa_int/10x/sample_01/
#   /Users/gtdebru/mimosa/data/500mpa_unint/10x/sample_03/
#
# Figure contents:
#   A. 2D normalized autocorrelation map
#   B. Radially averaged autocorrelation C(r), with radial xi marked
#
# Radial correlation length definition:
#   xi_r is the first positive radial lag where C(r) <= 1/e.
#
# If C(r) does not cross 1/e within the valid range, xi_r is reported as NaN
# and the figure is labeled "no 1/e crossing".
#
# FIELD_MODE options:
#   "height_final"  -> radial autocorrelation of final height map for each sample
#   "height_all"    -> radial autocorrelation of every height map for each sample/time
#   "delta_h_final" -> radial autocorrelation of raw Delta h = h_final - h_initial
#   "delta_h_all"   -> radial autocorrelation of raw Delta h = h_time - h_initial
#
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# =============================================================================
# User settings
# =============================================================================

DATA_ROOT = Path("/Users/gtdebru/mimosa/data")
OUTPUT_ROOT = DATA_ROOT

POINT_TABLE_PATH = (
    Path("roughness_strain_publication_figures")
    / "roughness_plastic_strain_point_table.csv"
)

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

level = True
detrend_order = 1

# Choose field.
# FIELD_MODE = "height_final"
# FIELD_MODE = "height_all"
# FIELD_MODE = "delta_h_final"
FIELD_MODE = "delta_h_all"

EXCLUDED_LOADS = set()
# EXCLUDED_LOADS = {475}

INITIAL_TIME_TOL_H = 1.0e-8

# Delta h settings.
LEVEL_DELTA_H = True
DELTA_H_DETREND_ORDER = 1
DELTA_H_CROP_MARGIN_PIXELS = 0

# Autocorrelation settings.
CORR_TARGET = np.exp(-1.0)
LAG_LIMIT_UM = 400.0
MAX_LAG_FOR_XI_UM = 400.0
RADIAL_BIN_WIDTH_UM = 2.0

AC_VMIN = -0.25
AC_VMAX = 1.0

MIN_VALID_PIXELS = 100

SAVE_DPI = 300
SHOW_FIGURES = False

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": SAVE_DPI,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)

if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()

df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

if "delta_sa_um" in df.columns:
    df["delta_sa_um"] = pd.to_numeric(df["delta_sa_um"], errors="coerce")

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()

df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No valid rows remain after filtering.")

print("Records available:")
print(df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height loading and leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray,
    spacing_um_value: float,
    order: int = 1,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


def crop_margin(A: np.ndarray, margin_pixels: int) -> np.ndarray:
    if margin_pixels <= 0:
        return A

    if A.shape[0] <= 2 * margin_pixels or A.shape[1] <= 2 * margin_pixels:
        raise ValueError(f"Margin {margin_pixels} too large for shape {A.shape}.")

    return A[margin_pixels:-margin_pixels, margin_pixels:-margin_pixels]


def compute_raw_delta_h(
    initial_path: str | Path, current_path: str | Path
) -> np.ndarray:
    H0 = load_plane_leveled_mean_removed_height(initial_path)
    H1 = load_plane_leveled_mean_removed_height(current_path)

    if H0.shape != H1.shape:
        raise ValueError(f"Shape mismatch: initial {H0.shape}, current {H1.shape}")

    H0 = crop_margin(H0, DELTA_H_CROP_MARGIN_PIXELS)
    H1 = crop_margin(H1, DELTA_H_CROP_MARGIN_PIXELS)

    dH = H1 - H0

    if LEVEL_DELTA_H:
        dH = detrend_surface(dH, spacing_um, order=DELTA_H_DETREND_ORDER)

    dH = dH - np.nanmean(dH)

    return dH


# =============================================================================
# Autocorrelation
# =============================================================================


def normalized_autocorrelation_2d(A: np.ndarray):
    """
    Returns:
        z_lags_um, y_lags_um, ac

    Convention:
        axis 0 = z/loading direction
        axis 1 = y/transverse direction
    """
    A = np.asarray(A, dtype=float)
    A = A - np.nanmean(A)

    if np.count_nonzero(np.isfinite(A)) < MIN_VALID_PIXELS:
        raise ValueError("Too few finite pixels for autocorrelation.")

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    F = np.fft.fft2(A)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)

    center0, center1 = np.array(ac.shape) // 2
    center_value = ac[center0, center1]

    if not np.isfinite(center_value) or center_value == 0:
        ac[:] = np.nan
    else:
        ac = ac / center_value

    n0, n1 = A.shape

    z_lags_um = (np.arange(n0) - center0) * spacing_um
    y_lags_um = (np.arange(n1) - center1) * spacing_um

    return z_lags_um, y_lags_um, ac


def radial_average_autocorrelation(
    z_lags_um: np.ndarray,
    y_lags_um: np.ndarray,
    ac: np.ndarray,
    *,
    bin_width_um: float = RADIAL_BIN_WIDTH_UM,
    max_lag_um: float | None = MAX_LAG_FOR_XI_UM,
):
    """
    Radially average the 2D autocorrelation map.

    C(r) = average of C(Delta z, Delta y) over annuli of radius r.

    Returns:
        r_centers_um, c_radial, counts
    """
    Y, Z = np.meshgrid(y_lags_um, z_lags_um)
    R = np.sqrt(Z**2 + Y**2)

    valid = np.isfinite(R) & np.isfinite(ac)

    if max_lag_um is None:
        r_max = np.nanmax(R[valid])
    else:
        r_max = float(max_lag_um)

    edges = np.arange(0.0, r_max + bin_width_um, bin_width_um)

    if len(edges) < 2:
        raise ValueError("Not enough radial bins.")

    r_centers = 0.5 * (edges[:-1] + edges[1:])
    c_radial = np.full_like(r_centers, np.nan, dtype=float)
    counts = np.zeros_like(r_centers, dtype=int)

    for i in range(len(r_centers)):
        mask = valid & (R >= edges[i]) & (R < edges[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            c_radial[i] = float(np.nanmean(ac[mask]))

    return r_centers, c_radial, counts


def first_radial_threshold_crossing(
    r_um: np.ndarray,
    c_radial: np.ndarray,
    *,
    target: float = CORR_TARGET,
):
    """
    Returns:
        xi_r_um, crossed

    If no crossing occurs, returns np.nan, False.
    """
    r = np.asarray(r_um, dtype=float)
    c = np.asarray(c_radial, dtype=float)

    valid = np.isfinite(r) & np.isfinite(c) & (r >= 0)
    r = r[valid]
    c = c[valid]

    if len(r) < 2:
        return np.nan, False

    # Start after the first bin, which contains r~0 and should be near 1.
    for k in range(1, len(r)):
        if c[k] <= target:
            r0, r1 = r[k - 1], r[k]
            c0, c1 = c[k - 1], c[k]

            if c1 == c0:
                return float(r1), True

            xi = r0 + (target - c0) * (r1 - r0) / (c1 - c0)
            return float(xi), True

    return np.nan, False


def radial_autocorrelation_metrics(A: np.ndarray) -> dict:
    z_lags_um, y_lags_um, ac = normalized_autocorrelation_2d(A)

    r_um, c_radial, counts = radial_average_autocorrelation(
        z_lags_um,
        y_lags_um,
        ac,
        bin_width_um=RADIAL_BIN_WIDTH_UM,
        max_lag_um=MAX_LAG_FOR_XI_UM,
    )

    xi_r_um, crossed = first_radial_threshold_crossing(
        r_um,
        c_radial,
        target=CORR_TARGET,
    )

    return {
        "z_lags_um": z_lags_um,
        "y_lags_um": y_lags_um,
        "ac": ac,
        "r_um": r_um,
        "c_radial": c_radial,
        "radial_counts": counts,
        "xi_radial_um": xi_r_um,
        "crossed_1e_radial": crossed,
    }


# =============================================================================
# Output path utilities
# =============================================================================


def experiment_name(load_mpa, sample_type: str) -> str:
    return f"{int(load_mpa)}mpa_{sample_type}"


def sample_output_dir(load_mpa, sample_type: str, sample_id: str) -> Path:
    return OUTPUT_ROOT / experiment_name(load_mpa, sample_type) / "10x" / str(sample_id)


def safe_time_label(time_h: float) -> str:
    return f"{time_h:.6g}h".replace(".", "p").replace("-", "m")


# =============================================================================
# Select records to plot
# =============================================================================

plot_records = []

for (sample_type, load, sample), g in df.groupby(["sample_type", "load_mpa", "sample"]):
    g = g.sort_values("time_h").copy()

    if g.empty:
        continue

    if FIELD_MODE in {"height_final", "delta_h_final"}:
        recs = [g.iloc[-1]]
    elif FIELD_MODE in {"height_all", "delta_h_all"}:
        recs = [row for _, row in g.iterrows()]
    else:
        raise ValueError(
            "FIELD_MODE must be one of: "
            "'height_final', 'height_all', 'delta_h_final', 'delta_h_all'."
        )

    if FIELD_MODE.startswith("delta_h"):
        if len(g) < 2:
            continue

        initial = g.iloc[0]

        if float(initial["time_h"]) > INITIAL_TIME_TOL_H:
            warnings.warn(
                f"Skipping Delta h for {load} {sample_type} sample {sample}: "
                f"earliest time is {initial['time_h']}, not initial."
            )
            continue

        initial_path = initial["height_path"]

        for rec in recs:
            if float(rec["time_h"]) <= INITIAL_TIME_TOL_H:
                continue

            plot_records.append(
                {
                    "load_mpa": load,
                    "sample_type": sample_type,
                    "sample": sample,
                    "time_h": float(rec["time_h"]),
                    "bulk_z_strain_percent": float(rec["bulk_z_strain_percent"]),
                    "height_path": rec["height_path"],
                    "initial_height_path": initial_path,
                    "field_mode": FIELD_MODE,
                }
            )

    else:
        for rec in recs:
            plot_records.append(
                {
                    "load_mpa": load,
                    "sample_type": sample_type,
                    "sample": sample,
                    "time_h": float(rec["time_h"]),
                    "bulk_z_strain_percent": float(rec["bulk_z_strain_percent"]),
                    "height_path": rec["height_path"],
                    "initial_height_path": None,
                    "field_mode": FIELD_MODE,
                }
            )

print()
print(f"Number of radial autocorrelation figures to generate: {len(plot_records)}")


# =============================================================================
# Plotting function
# =============================================================================


def plot_radial_autocorrelation_for_record(record: dict):
    load = record["load_mpa"]
    sample_type = record["sample_type"]
    sample = record["sample"]
    time_h = record["time_h"]
    strain_percent = record["bulk_z_strain_percent"]

    if record["field_mode"].startswith("delta_h"):
        A = compute_raw_delta_h(
            record["initial_height_path"],
            record["height_path"],
        )
        field_label = r"$\Delta h$"
        filename_prefix = "radial_autocorrelation_delta_h"
    else:
        A = load_plane_leveled_mean_removed_height(record["height_path"])
        field_label = r"$h-\bar{h}$"
        filename_prefix = "radial_autocorrelation_height"

    metrics = radial_autocorrelation_metrics(A)

    z_lags_um = metrics["z_lags_um"]
    y_lags_um = metrics["y_lags_um"]
    ac = metrics["ac"]
    r_um = metrics["r_um"]
    c_radial = metrics["c_radial"]
    counts = metrics["radial_counts"]
    xi_r = metrics["xi_radial_um"]
    crossed = metrics["crossed_1e_radial"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.8),
        gridspec_kw={"width_ratios": [1.05, 1.25]},
    )

    ax_map, ax_rad = axes

    extent = [
        y_lags_um.min(),
        y_lags_um.max(),
        z_lags_um.min(),
        z_lags_um.max(),
    ]

    im = ax_map.imshow(
        ac,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=AC_VMIN,
        vmax=AC_VMAX,
        interpolation="nearest",
        rasterized=True,
        aspect="equal",
    )

    if LAG_LIMIT_UM is not None:
        ax_map.set_xlim(-LAG_LIMIT_UM, LAG_LIMIT_UM)
        ax_map.set_ylim(-LAG_LIMIT_UM, LAG_LIMIT_UM)

    ax_map.axhline(0.0, color="white", lw=0.6, alpha=0.65)
    ax_map.axvline(0.0, color="white", lw=0.6, alpha=0.65)

    # Mark radial xi on the map as a circle if it exists.
    if crossed and np.isfinite(xi_r):
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        ax_map.plot(
            xi_r * np.cos(theta),
            xi_r * np.sin(theta),
            color="red",
            lw=1.2,
            ls="--",
        )

    ax_map.set_xlabel(r"transverse lag, $\Delta y$ [$\mu$m]")
    ax_map.set_ylabel(r"loading lag, $\Delta z$ [$\mu$m]")
    ax_map.set_title("2D normalized autocorrelation")

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("normalized autocorrelation")

    # Radial autocorrelation
    valid = np.isfinite(r_um) & np.isfinite(c_radial)

    ax_rad.plot(
        r_um[valid],
        c_radial[valid],
        color="black",
        lw=1.8,
        label=r"radial $C(r)$",
    )

    ax_rad.axhline(
        CORR_TARGET,
        color="0.45",
        lw=1.0,
        ls="--",
        label=r"$1/e$",
    )

    if crossed and np.isfinite(xi_r):
        ax_rad.axvline(
            xi_r,
            color="red",
            lw=1.3,
            ls="--",
            label=rf"$\xi_r={xi_r:.1f}\,\mu$m",
        )

        ax_rad.annotate(
            rf"$\xi_r={xi_r:.1f}\,\mu$m",
            xy=(xi_r, CORR_TARGET),
            xytext=(0.53, 0.72),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.9),
            color="red",
            fontsize=9,
        )
    else:
        ax_rad.text(
            0.05,
            0.12,
            "no 1/e crossing",
            transform=ax_rad.transAxes,
            color="red",
            fontsize=9,
        )

    if MAX_LAG_FOR_XI_UM is not None:
        ax_rad.set_xlim(0.0, MAX_LAG_FOR_XI_UM)
    elif LAG_LIMIT_UM is not None:
        ax_rad.set_xlim(0.0, LAG_LIMIT_UM)

    ax_rad.set_ylim(-0.3, 1.05)
    ax_rad.set_xlabel(r"radial lag, $r=\sqrt{\Delta z^2+\Delta y^2}$ [$\mu$m]")
    ax_rad.set_ylabel(r"radially averaged autocorrelation, $C(r)$")
    ax_rad.set_title("Radial autocorrelation")
    ax_rad.legend(fontsize=7)

    fig.suptitle(
        f"{int(load)} MPa {sample_type}, sample {sample}, "
        f"time={time_h:.4g} h, "
        rf"$\epsilon_{{zz}}={strain_percent:.3g}\%$, "
        f"{field_label}",
        y=1.04,
    )

    fig.tight_layout()

    outdir = sample_output_dir(load, sample_type, sample)
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"{filename_prefix}_{safe_time_label(time_h)}.png"
    fig.savefig(outpath, bbox_inches="tight")

    summary = {
        "load_mpa": load,
        "sample_type": sample_type,
        "sample": sample,
        "time_h": time_h,
        "bulk_z_strain_percent": strain_percent,
        "field_mode": record["field_mode"],
        "height_path": record["height_path"],
        "initial_height_path": record["initial_height_path"],
        "output_path": str(outpath),
        "xi_radial_um": xi_r,
        "crossed_1e_radial": crossed,
        "target_correlation": CORR_TARGET,
        "radial_bin_width_um": RADIAL_BIN_WIDTH_UM,
        "max_lag_for_xi_um": MAX_LAG_FOR_XI_UM,
    }

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return summary


# =============================================================================
# Generate and save all figures
# =============================================================================

summary_rows = []

for i, record in enumerate(plot_records, start=1):
    try:
        summary = plot_radial_autocorrelation_for_record(record)
        summary_rows.append(summary)

        print(
            f"[{i}/{len(plot_records)}] Saved {summary['output_path']} "
            f"xi_r={summary['xi_radial_um']}"
        )

    except Exception as exc:
        warnings.warn(
            f"Failed radial autocorrelation plot for "
            f"{record['load_mpa']} MPa {record['sample_type']} sample {record['sample']} "
            f"time={record['time_h']}: {exc}"
        )

summary_df = pd.DataFrame(summary_rows)

summary_csv = OUTPUT_ROOT / "radial_autocorrelation_correlation_length_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print()
print("Complete. Saved summary CSV:")
print(summary_csv)

# %%
# =============================================================================
# Experiment-aggregated radial autocorrelation curves
#
# For each experiment, this computes per-sample:
#
#   C_initial(r)  = radial ACF of initial height map
#   C_final(r)    = radial ACF of final height map
#   C_delta_h(r)  = radial ACF of Delta h = h_final - h_initial
#
# and derived curves:
#
#   Delta C(r) = C_final(r) - C_initial(r)
#   Gain C(r)  = C_final(r) / C_initial(r)
#
# Then it aggregates over all samples in each experiment and plots mean and
# median curves.
#
# Output directory per experiment:
#
#   /Users/gtdebru/mimosa/data/{load}mpa_{sample_type}/10x/
#
# Example:
#   /Users/gtdebru/mimosa/data/475mpa_int/10x/
#   /Users/gtdebru/mimosa/data/500mpa_unint/10x/
#
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# =============================================================================
# User settings
# =============================================================================

DATA_ROOT = Path("/Users/gtdebru/mimosa/data")
OUTPUT_ROOT = DATA_ROOT

POINT_TABLE_PATH = (
    Path("roughness_strain_publication_figures")
    / "roughness_plastic_strain_point_table.csv"
)

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

level = True
detrend_order = 1

INITIAL_TIME_TOL_H = 1.0e-8

# Optional load exclusion.
EXCLUDED_LOADS = set()
# EXCLUDED_LOADS = {475}

# Delta h settings.
LEVEL_DELTA_H = True
DELTA_H_DETREND_ORDER = 1
DELTA_H_CROP_MARGIN_PIXELS = 0

# Radial ACF settings.
RADIAL_BIN_WIDTH_UM = 2.0
MAX_LAG_UM = 400.0

# ACF gain denominator mask:
# Gain C_final / C_initial is unstable where C_initial ~ 0.
MIN_ABS_INITIAL_ACF_FOR_GAIN = 0.05

# Plot settings.
SAVE_DPI = 300
SHOW_FIGURES = False

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": SAVE_DPI,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)

if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()

df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

if "delta_sa_um" in df.columns:
    df["delta_sa_um"] = pd.to_numeric(df["delta_sa_um"], errors="coerce")

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()

df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No valid rows remain after filtering.")

print("Records available:")
print(df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height loading and leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray,
    spacing_um_value: float,
    order: int = 1,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


def crop_margin(A: np.ndarray, margin_pixels: int) -> np.ndarray:
    if margin_pixels <= 0:
        return A

    if A.shape[0] <= 2 * margin_pixels or A.shape[1] <= 2 * margin_pixels:
        raise ValueError(f"Margin {margin_pixels} too large for shape {A.shape}.")

    return A[margin_pixels:-margin_pixels, margin_pixels:-margin_pixels]


def compute_raw_delta_h(initial_path: str | Path, final_path: str | Path) -> np.ndarray:
    H0 = load_plane_leveled_mean_removed_height(initial_path)
    Hf = load_plane_leveled_mean_removed_height(final_path)

    if H0.shape != Hf.shape:
        raise ValueError(f"Shape mismatch: initial {H0.shape}, final {Hf.shape}")

    H0 = crop_margin(H0, DELTA_H_CROP_MARGIN_PIXELS)
    Hf = crop_margin(Hf, DELTA_H_CROP_MARGIN_PIXELS)

    dH = Hf - H0

    if LEVEL_DELTA_H:
        dH = detrend_surface(dH, spacing_um, order=DELTA_H_DETREND_ORDER)

    dH = dH - np.nanmean(dH)

    return dH


# =============================================================================
# Autocorrelation functions
# =============================================================================


def normalized_autocorrelation_2d(A: np.ndarray):
    """
    Returns:
        z_lags_um, y_lags_um, ac

    Convention:
        axis 0 = z/loading direction
        axis 1 = y/transverse direction
    """
    A = np.asarray(A, dtype=float)
    A = A - np.nanmean(A)

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    F = np.fft.fft2(A)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)

    center0, center1 = np.array(ac.shape) // 2
    center_value = ac[center0, center1]

    if not np.isfinite(center_value) or center_value == 0:
        ac[:] = np.nan
    else:
        ac = ac / center_value

    n0, n1 = A.shape

    z_lags_um = (np.arange(n0) - center0) * spacing_um
    y_lags_um = (np.arange(n1) - center1) * spacing_um

    return z_lags_um, y_lags_um, ac


def radial_average_autocorrelation(
    z_lags_um: np.ndarray,
    y_lags_um: np.ndarray,
    ac: np.ndarray,
    *,
    bin_width_um: float = RADIAL_BIN_WIDTH_UM,
    max_lag_um: float = MAX_LAG_UM,
):
    Y, Z = np.meshgrid(y_lags_um, z_lags_um)
    R = np.sqrt(Z**2 + Y**2)

    valid = np.isfinite(R) & np.isfinite(ac)

    edges = np.arange(0.0, max_lag_um + bin_width_um, bin_width_um)

    r_centers = 0.5 * (edges[:-1] + edges[1:])
    c_radial = np.full_like(r_centers, np.nan, dtype=float)
    counts = np.zeros_like(r_centers, dtype=int)

    for i in range(len(r_centers)):
        mask = valid & (R >= edges[i]) & (R < edges[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            c_radial[i] = float(np.nanmean(ac[mask]))

    return r_centers, c_radial, counts


def radial_acf(A: np.ndarray):
    z_lags_um, y_lags_um, ac = normalized_autocorrelation_2d(A)

    r_um, c_radial, counts = radial_average_autocorrelation(
        z_lags_um,
        y_lags_um,
        ac,
        bin_width_um=RADIAL_BIN_WIDTH_UM,
        max_lag_um=MAX_LAG_UM,
    )

    return r_um, c_radial, counts


# =============================================================================
# Output helpers
# =============================================================================


def experiment_name(load_mpa, sample_type: str) -> str:
    return f"{int(load_mpa)}mpa_{sample_type}"


def experiment_output_dir(load_mpa, sample_type: str) -> Path:
    return OUTPUT_ROOT / experiment_name(load_mpa, sample_type) / "10x"


# =============================================================================
# Compute per-sample curves
# =============================================================================

sample_rows = []
curve_rows = []

for (sample_type, load, sample), g in df.groupby(["sample_type", "load_mpa", "sample"]):
    g = g.sort_values("time_h").copy()

    if len(g) < 2:
        continue

    initial = g.iloc[0]
    final = g.iloc[-1]

    if float(initial["time_h"]) > INITIAL_TIME_TOL_H:
        warnings.warn(
            f"Skipping {load} MPa {sample_type} sample {sample}: "
            f"earliest time is {initial['time_h']}, not initial."
        )
        continue

    try:
        H0 = load_plane_leveled_mean_removed_height(initial["height_path"])
        Hf = load_plane_leveled_mean_removed_height(final["height_path"])
        dH = compute_raw_delta_h(initial["height_path"], final["height_path"])

        r_initial, c_initial, counts_initial = radial_acf(H0)
        r_final, c_final, counts_final = radial_acf(Hf)
        r_delta_h, c_delta_h, counts_delta_h = radial_acf(dH)

        if not (
            np.allclose(r_initial, r_final, equal_nan=True)
            and np.allclose(r_initial, r_delta_h, equal_nan=True)
        ):
            raise ValueError("Radial lag grids do not match.")

        r_um = r_initial

        delta_acf = c_final - c_initial

        gain_acf = np.full_like(c_final, np.nan, dtype=float)
        denom_ok = np.isfinite(c_initial) & (
            np.abs(c_initial) >= MIN_ABS_INITIAL_ACF_FOR_GAIN
        )
        numer_ok = np.isfinite(c_final)
        gain_acf[denom_ok & numer_ok] = (
            c_final[denom_ok & numer_ok] / c_initial[denom_ok & numer_ok]
        )

        final_strain = float(final["bulk_z_strain_percent"])
        final_time = float(final["time_h"])
        final_delta_sa = (
            float(final["delta_sa_um"])
            if "delta_sa_um" in final.index and np.isfinite(final["delta_sa_um"])
            else np.nan
        )

        sample_rows.append(
            {
                "load_mpa": load,
                "sample_type": sample_type,
                "sample": sample,
                "initial_time_h": float(initial["time_h"]),
                "final_time_h": final_time,
                "final_bulk_z_strain_percent": final_strain,
                "final_delta_sa_um": final_delta_sa,
                "initial_height_path": initial["height_path"],
                "final_height_path": final["height_path"],
            }
        )

        for i, r in enumerate(r_um):
            curve_rows.extend(
                [
                    {
                        "load_mpa": load,
                        "sample_type": sample_type,
                        "sample": sample,
                        "r_um": r,
                        "curve_type": "initial_height_acf",
                        "acf": c_initial[i],
                        "final_bulk_z_strain_percent": final_strain,
                        "final_time_h": final_time,
                        "final_delta_sa_um": final_delta_sa,
                    },
                    {
                        "load_mpa": load,
                        "sample_type": sample_type,
                        "sample": sample,
                        "r_um": r,
                        "curve_type": "final_height_acf",
                        "acf": c_final[i],
                        "final_bulk_z_strain_percent": final_strain,
                        "final_time_h": final_time,
                        "final_delta_sa_um": final_delta_sa,
                    },
                    {
                        "load_mpa": load,
                        "sample_type": sample_type,
                        "sample": sample,
                        "r_um": r,
                        "curve_type": "delta_h_acf",
                        "acf": c_delta_h[i],
                        "final_bulk_z_strain_percent": final_strain,
                        "final_time_h": final_time,
                        "final_delta_sa_um": final_delta_sa,
                    },
                    {
                        "load_mpa": load,
                        "sample_type": sample_type,
                        "sample": sample,
                        "r_um": r,
                        "curve_type": "delta_acf_final_minus_initial",
                        "acf": delta_acf[i],
                        "final_bulk_z_strain_percent": final_strain,
                        "final_time_h": final_time,
                        "final_delta_sa_um": final_delta_sa,
                    },
                    {
                        "load_mpa": load,
                        "sample_type": sample_type,
                        "sample": sample,
                        "r_um": r,
                        "curve_type": "gain_acf_final_over_initial",
                        "acf": gain_acf[i],
                        "final_bulk_z_strain_percent": final_strain,
                        "final_time_h": final_time,
                        "final_delta_sa_um": final_delta_sa,
                    },
                ]
            )

    except Exception as exc:
        warnings.warn(
            f"Failed ACF calculation for {load} MPa {sample_type} sample {sample}: {exc}"
        )

sample_df = pd.DataFrame(sample_rows)
curves_df = pd.DataFrame(curve_rows)

if curves_df.empty:
    raise RuntimeError("No ACF curves were computed.")

summary_csv = OUTPUT_ROOT / "experiment_radial_acf_sample_curve_long.csv"
curves_df.to_csv(summary_csv, index=False)

sample_csv = OUTPUT_ROOT / "experiment_radial_acf_sample_summary.csv"
sample_df.to_csv(sample_csv, index=False)

print()
print(f"Saved long curve table: {summary_csv}")
print(f"Saved sample summary table: {sample_csv}")


# =============================================================================
# Aggregate and plot by experiment
# =============================================================================


def aggregate_curves_for_experiment(exp_df: pd.DataFrame):
    rows = []

    for (curve_type, r_um), g in exp_df.groupby(["curve_type", "r_um"]):
        y = pd.to_numeric(g["acf"], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]

        if y.size == 0:
            continue

        rows.append(
            {
                "curve_type": curve_type,
                "r_um": r_um,
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
                "n": int(y.size),
            }
        )

    return pd.DataFrame(rows).sort_values(["curve_type", "r_um"]).reset_index(drop=True)


curve_colors = {
    "initial_height_acf": "black",
    "final_height_acf": "tab:blue",
    "delta_h_acf": "tab:orange",
    "delta_acf_final_minus_initial": "tab:green",
    "gain_acf_final_over_initial": "tab:red",
}

curve_labels = {
    "initial_height_acf": r"$C_{h_0}(r)$",
    "final_height_acf": r"$C_{h_f}(r)$",
    "delta_h_acf": r"$C_{\Delta h}(r)$",
    "delta_acf_final_minus_initial": r"$C_{h_f}(r)-C_{h_0}(r)$",
    "gain_acf_final_over_initial": r"$C_{h_f}(r)/C_{h_0}(r)$",
}

aggregate_rows = []

for (sample_type, load), exp_df in curves_df.groupby(["sample_type", "load_mpa"]):
    exp_df = exp_df.copy()
    agg = aggregate_curves_for_experiment(exp_df)

    if agg.empty:
        continue

    agg["load_mpa"] = load
    agg["sample_type"] = sample_type
    aggregate_rows.append(agg)

    outdir = experiment_output_dir(load, sample_type)
    outdir.mkdir(parents=True, exist_ok=True)

    agg_csv = outdir / "experiment_radial_acf_aggregate_mean_median.csv"
    agg.to_csv(agg_csv, index=False)

    n_samples = exp_df["sample"].nunique()
    mean_final_strain = (
        exp_df[["sample", "final_bulk_z_strain_percent"]]
        .drop_duplicates()["final_bulk_z_strain_percent"]
        .mean()
    )

    # -------------------------------------------------------------------------
    # Figure: three panels
    #   A. Initial/final/delta_h ACFs
    #   B. Delta ACF = final - initial
    #   C. Gain ACF = final / initial
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    ax0, ax1, ax2 = axes

    # Panel A: three ACF curves
    for curve_type in ["initial_height_acf", "final_height_acf", "delta_h_acf"]:
        s = agg[agg["curve_type"] == curve_type].sort_values("r_um")

        if s.empty:
            continue

        color = curve_colors[curve_type]
        label = curve_labels[curve_type]

        ax0.plot(
            s["r_um"],
            s["mean"],
            color=color,
            lw=2.0,
            label=f"{label} mean",
        )

        ax0.plot(
            s["r_um"],
            s["median"],
            color=color,
            lw=1.5,
            ls="--",
            label=f"{label} median",
        )

    ax0.axhline(0.0, color="0.55", lw=0.8)
    ax0.axhline(np.exp(-1.0), color="0.55", lw=0.8, ls=":")
    ax0.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax0.set_ylabel(r"radial ACF, $C(r)$")
    ax0.set_title("A. Initial, final, and $\\Delta h$ ACFs")
    ax0.set_xlim(0.0, MAX_LAG_UM)
    ax0.set_ylim(-0.35, 1.05)
    ax0.grid(True, alpha=0.25)

    # Panel B: delta ACF
    curve_type = "delta_acf_final_minus_initial"
    s = agg[agg["curve_type"] == curve_type].sort_values("r_um")

    if not s.empty:
        color = curve_colors[curve_type]
        label = curve_labels[curve_type]

        ax1.plot(
            s["r_um"],
            s["mean"],
            color=color,
            lw=2.0,
            label=f"{label} mean",
        )

        ax1.plot(
            s["r_um"],
            s["median"],
            color=color,
            lw=1.5,
            ls="--",
            label=f"{label} median",
        )

    ax1.axhline(0.0, color="0.55", lw=0.8)
    ax1.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax1.set_ylabel(r"$\Delta C(r)$")
    ax1.set_title(r"B. ACF difference, $C_{h_f}-C_{h_0}$")
    ax1.set_xlim(0.0, MAX_LAG_UM)
    ax1.grid(True, alpha=0.25)

    # Panel C: gain ACF
    curve_type = "gain_acf_final_over_initial"
    s = agg[agg["curve_type"] == curve_type].sort_values("r_um")

    if not s.empty:
        color = curve_colors[curve_type]
        label = curve_labels[curve_type]

        ax2.plot(
            s["r_um"],
            s["mean"],
            color=color,
            lw=2.0,
            label=f"{label} mean",
        )

        ax2.plot(
            s["r_um"],
            s["median"],
            color=color,
            lw=1.5,
            ls="--",
            label=f"{label} median",
        )

        yvals = np.concatenate(
            [
                s["mean"].to_numpy(dtype=float),
                s["median"].to_numpy(dtype=float),
            ]
        )
        yvals = yvals[np.isfinite(yvals)]

        if yvals.size > 0:
            lo, hi = np.nanpercentile(yvals, [2, 98])
            span = max(abs(lo - 1.0), abs(hi - 1.0), 0.5)
            ax2.set_ylim(1.0 - 1.15 * span, 1.0 + 1.15 * span)

    ax2.axhline(1.0, color="0.55", lw=0.8, ls="--")
    ax2.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax2.set_ylabel(r"ACF gain, $C_{h_f}(r)/C_{h_0}(r)$")
    ax2.set_title(r"C. ACF gain")
    ax2.set_xlim(0.0, MAX_LAG_UM)
    ax2.grid(True, alpha=0.25)

    for ax in axes:
        ax.legend(fontsize=6)

    fig.suptitle(
        f"{int(load)} MPa {sample_type}: experiment-aggregated radial ACFs\n"
        f"n={n_samples} samples, mean final "
        rf"$\epsilon_{{zz}}={mean_final_strain:.3g}\%$",
        y=1.04,
    )

    fig.tight_layout()

    outpath = outdir / "experiment_radial_acf_initial_final_delta_gain_mean_median.png"
    fig.savefig(outpath, bbox_inches="tight")

    print(f"Saved: {outpath}")
    print(f"Saved: {agg_csv}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

if len(aggregate_rows) > 0:
    all_aggregate_df = pd.concat(aggregate_rows, ignore_index=True)
    all_aggregate_csv = (
        OUTPUT_ROOT / "all_experiments_radial_acf_aggregate_mean_median.csv"
    )
    all_aggregate_df.to_csv(all_aggregate_csv, index=False)
    print()
    print(f"Saved all-experiment aggregate table: {all_aggregate_csv}")

print()
print("Complete.")

# %%
# =============================================================================
# Two-panel figure:
#   Left:  bracketed PSD log10 gain curves by strain level
#   Right: radial ACF change curves by strain level
#
# Separate figure:
#   Scatter plot of all radial ACF correlation lengths vs strain
#
# Definitions:
#   PSD gain:
#       G_band(eps) = P_band(eps) / P_band(0)
#       plotted as mean log10(G_band) by strain group
#
#   ACF change:
#       Delta C(r; eps) = C_h(r; eps) - C_h(r; 0)
#       plotted as mean Delta C(r) by strain group
#
#   Radial ACF length:
#       xi_r = first radial lag where C_h(r) <= 1/e
#
# Notes:
#   - Uses signed, plane-leveled, mean-removed height maps.
#   - Uses each specimen's own initial scan as reference.
#   - Excludes 475 MPa by default.
#   - Uses four non-initial strain groups.
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

EXCLUDED_LOADS = {475}

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

level = True
detrend_order = 1

INITIAL_TIME_TOL_H = 1.0e-8
INITIAL_STRAIN_TOL_PERCENT = 0.05

N_STRAIN_GROUPS = 4

# PSD wavelength brackets.
# This follows the previous bracketed PSD-gain figure:
# 20 um brackets from 2 * Nyquist shortest wavelength to 300 um.
PSD_BRACKET_WIDTH_UM = 20.0
PSD_MAX_WAVELENGTH_UM = 300.0

# ACF settings.
ACF_RADIAL_BIN_WIDTH_UM = 2.0
ACF_MAX_LAG_UM = 100.0
ACF_TARGET = np.exp(-1.0)

# For ACF gain/difference computations, use all lags up to this value.
# The ACF length is set to NaN if no 1/e crossing occurs.
MAX_LAG_FOR_XI_UM = 400.0

ERROR_BAR = "sem"  # "std", "sem", or "ci95"
SHOW_INDIVIDUAL_POINTS_ON_SCATTER = True
COLOR_SCATTER_BY_LOAD = True

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)
if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()

df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

if "delta_sa_um" in df.columns:
    df["delta_sa_um"] = pd.to_numeric(df["delta_sa_um"], errors="coerce")

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()
df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No valid rows remain after filtering.")

df["specimen_id"] = (
    df["sample_type"].astype(str)
    + "_"
    + df["load_mpa"].astype(int).astype(str)
    + "MPa_"
    + df["sample"].astype(str)
)

print("Records used:")
print(df.groupby(["sample_type", "load_mpa"]).size())
print(f"Number of specimens: {df['specimen_id'].nunique()}")


# =============================================================================
# Height loading and leveling
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, column = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = column.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend((x * x, x * y, y * y))
    elif order != 1:
        raise ValueError("detrend_order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coefficients = inverse @ values.ravel()
    trend = (design @ coefficients).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def read_height(
    path: str | Path,
    *,
    level: bool = True,
    spacing_um_value: float = spacing_um,
    detrend_order: int = 1,
) -> np.ndarray:
    height = raw_height(path)

    missing = ~np.isfinite(height)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        height = height[tuple(nearest)]

    if level:
        height = detrend_surface(height, spacing_um_value, order=detrend_order)

    return height


def load_plane_leveled_mean_removed_height(path: str | Path) -> np.ndarray:
    H = read_height(
        path,
        level=level,
        spacing_um_value=spacing_um,
        detrend_order=detrend_order,
    )

    H = H[crop]
    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD functions
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))
    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um: np.ndarray, spacing_um_value: float):
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    df0 = 1.0 / (n0 * spacing_um_value)
    df1 = 1.0 / (n1 * spacing_um_value)

    return f0, f1, PSD, df0, df1


def psd_band_powers(height_um: np.ndarray, bands: dict[str, tuple[float, float]]):
    f0, f1, PSD, df0, df1 = psd2d_height(height_um, spacing_um)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength_um = 1.0 / FR

    out = {}

    for name, (lo, hi) in bands.items():
        mask = (
            np.isfinite(wavelength_um)
            & (FR > 0)
            & (wavelength_um >= lo)
            & (wavelength_um < hi)
        )

        power = float(np.nansum(PSD[mask]) * df0 * df1)

        out[name] = power

    return out


def make_psd_wavelength_bands():
    nyquist_shortest_wavelength_um = 2.0 * spacing_um
    twice_nyquist_um = 2.0 * nyquist_shortest_wavelength_um

    lambda_start_um = twice_nyquist_um
    lambda_end_um = PSD_MAX_WAVELENGTH_UM

    if lambda_end_um <= lambda_start_um:
        raise ValueError(
            f"Invalid PSD wavelength range: start={lambda_start_um}, end={lambda_end_um}"
        )

    edges = [lambda_start_um]

    while edges[-1] < lambda_end_um:
        edges.append(min(edges[-1] + PSD_BRACKET_WIDTH_UM, lambda_end_um))

    edges = np.asarray(edges, dtype=float)

    bands = {}
    for i in range(len(edges) - 1):
        name = f"{edges[i]:.1f}-{edges[i+1]:.1f} µm"
        bands[name] = (edges[i], edges[i + 1])

    return bands


psd_bands_um = make_psd_wavelength_bands()
psd_band_names = list(psd_bands_um.keys())

print()
print("PSD wavelength brackets:")
for name, (lo, hi) in psd_bands_um.items():
    print(f"  {name}: {lo:.3f} to {hi:.3f} um")


# =============================================================================
# Radial ACF functions
# =============================================================================


def normalized_autocorrelation_2d(A: np.ndarray):
    A = np.asarray(A, dtype=float)
    A = A - np.nanmean(A)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    F = np.fft.fft2(A)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)

    center0, center1 = np.array(ac.shape) // 2
    center_value = ac[center0, center1]

    if not np.isfinite(center_value) or center_value == 0:
        ac[:] = np.nan
    else:
        ac = ac / center_value

    n0, n1 = A.shape
    z_lags_um = (np.arange(n0) - center0) * spacing_um
    y_lags_um = (np.arange(n1) - center1) * spacing_um

    return z_lags_um, y_lags_um, ac


def radial_average_autocorrelation(
    z_lags_um: np.ndarray,
    y_lags_um: np.ndarray,
    ac: np.ndarray,
    *,
    bin_width_um: float = ACF_RADIAL_BIN_WIDTH_UM,
    max_lag_um: float = ACF_MAX_LAG_UM,
):
    Y, Z = np.meshgrid(y_lags_um, z_lags_um)
    R = np.sqrt(Z**2 + Y**2)

    valid = np.isfinite(R) & np.isfinite(ac)

    edges = np.arange(0.0, max_lag_um + bin_width_um, bin_width_um)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    c_radial = np.full_like(r_centers, np.nan, dtype=float)
    counts = np.zeros_like(r_centers, dtype=int)

    for i in range(len(r_centers)):
        mask = valid & (R >= edges[i]) & (R < edges[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            c_radial[i] = float(np.nanmean(ac[mask]))

    return r_centers, c_radial, counts


def radial_acf(height_um: np.ndarray):
    z_lags_um, y_lags_um, ac = normalized_autocorrelation_2d(height_um)

    r_um, c_radial, counts = radial_average_autocorrelation(
        z_lags_um,
        y_lags_um,
        ac,
        bin_width_um=ACF_RADIAL_BIN_WIDTH_UM,
        max_lag_um=ACF_MAX_LAG_UM,
    )

    return r_um, c_radial, counts


def first_radial_1e_crossing(r_um: np.ndarray, c_radial: np.ndarray):
    r = np.asarray(r_um, dtype=float)
    c = np.asarray(c_radial, dtype=float)

    valid = np.isfinite(r) & np.isfinite(c) & (r >= 0) & (r <= MAX_LAG_FOR_XI_UM)

    r = r[valid]
    c = c[valid]

    if len(r) < 2:
        return np.nan, False

    for k in range(1, len(r)):
        if c[k] <= ACF_TARGET:
            r0, r1 = r[k - 1], r[k]
            c0, c1 = c[k - 1], c[k]

            if c1 == c0:
                return float(r1), True

            xi = r0 + (ACF_TARGET - c0) * (r1 - r0) / (c1 - c0)
            return float(xi), True

    return np.nan, False


# =============================================================================
# Helper functions
# =============================================================================


def error_from_values(y: np.ndarray, mode: str):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if y.size <= 1:
        return 0.0

    std = float(np.std(y, ddof=1))
    sem = std / np.sqrt(y.size)

    if mode == "std":
        return std
    if mode == "sem":
        return sem
    if mode == "ci95":
        return 1.96 * sem

    raise ValueError("ERROR_BAR must be 'std', 'sem', or 'ci95'.")


def assign_strain_groups(input_df: pd.DataFrame, strain_col: str):
    d = input_df[input_df[strain_col] > INITIAL_STRAIN_TOL_PERCENT].copy()

    if d.empty:
        raise ValueError("No non-initial rows available for strain grouping.")

    q = np.unique(
        np.nanquantile(
            d[strain_col].to_numpy(dtype=float),
            np.linspace(0.0, 1.0, N_STRAIN_GROUPS + 1),
        )
    )

    d["strain_group"] = None
    labels = []

    for i in range(len(q) - 1):
        lo = q[i]
        hi = q[i + 1]

        if i == len(q) - 2:
            mask = (d[strain_col] >= lo) & (d[strain_col] <= hi)
        else:
            mask = (d[strain_col] >= lo) & (d[strain_col] < hi)

        label = rf"{lo:.2g}--{hi:.2g}%"
        d.loc[mask, "strain_group"] = label
        labels.append(label)

    d = d[d["strain_group"].notna()].copy()

    d["strain_group"] = pd.Categorical(
        d["strain_group"],
        categories=labels,
        ordered=True,
    )

    return d, labels


# =============================================================================
# Compute per-sample/time PSD gain, ACF change, and radial ACF length
# =============================================================================

psd_rows = []
acf_change_rows = []
acf_length_rows = []

for specimen_id, g in df.groupby("specimen_id"):
    g = g.sort_values("time_h").copy()

    if len(g) < 2:
        continue

    initial = g.iloc[0]

    if float(initial["time_h"]) > INITIAL_TIME_TOL_H:
        warnings.warn(
            f"Skipping specimen {specimen_id}: earliest time is "
            f"{initial['time_h']}, not initial."
        )
        continue

    try:
        H0 = load_plane_leveled_mean_removed_height(initial["height_path"])
        p0 = psd_band_powers(H0, psd_bands_um)
        r0, c0, _ = radial_acf(H0)
    except Exception as exc:
        warnings.warn(f"Initial reference failed for specimen {specimen_id}: {exc}")
        continue

    for _, rec in g.iterrows():
        try:
            H = load_plane_leveled_mean_removed_height(rec["height_path"])

            strain = float(rec["bulk_z_strain_percent"])
            time_h = float(rec["time_h"])

            # PSD gain.
            p = psd_band_powers(H, psd_bands_um)

            for band_name in psd_band_names:
                initial_power = p0[band_name]
                current_power = p[band_name]

                gain = (
                    current_power / initial_power
                    if np.isfinite(initial_power) and initial_power > 0
                    else np.nan
                )

                log10_gain = (
                    np.log10(gain) if np.isfinite(gain) and gain > 0 else np.nan
                )

                lo, hi = psd_bands_um[band_name]

                psd_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "load_mpa": rec["load_mpa"],
                        "sample_type": rec["sample_type"],
                        "sample": rec["sample"],
                        "time_h": time_h,
                        "bulk_z_strain_percent": strain,
                        "band": band_name,
                        "lambda_min_um": lo,
                        "lambda_max_um": hi,
                        "lambda_center_um": 0.5 * (lo + hi),
                        "log10_gain": log10_gain,
                        "gain": gain,
                    }
                )

            # ACF change and ACF length.
            r, c, _ = radial_acf(H)

            if not np.allclose(r, r0, equal_nan=True):
                raise ValueError("ACF radial grids do not match.")

            delta_c = c - c0
            xi_r, crossed = first_radial_1e_crossing(r, c)

            acf_length_rows.append(
                {
                    "specimen_id": specimen_id,
                    "load_mpa": rec["load_mpa"],
                    "sample_type": rec["sample_type"],
                    "sample": rec["sample"],
                    "time_h": time_h,
                    "bulk_z_strain_percent": strain,
                    "xi_radial_um": xi_r,
                    "crossed_1e": crossed,
                }
            )

            for i in range(len(r)):
                acf_change_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "load_mpa": rec["load_mpa"],
                        "sample_type": rec["sample_type"],
                        "sample": rec["sample"],
                        "time_h": time_h,
                        "bulk_z_strain_percent": strain,
                        "r_um": r[i],
                        "delta_acf": delta_c[i],
                        "acf_current": c[i],
                        "acf_initial": c0[i],
                    }
                )

        except Exception as exc:
            warnings.warn(f"Failed specimen {specimen_id}, time={rec['time_h']}: {exc}")

psd_df = pd.DataFrame(psd_rows)
acf_change_df = pd.DataFrame(acf_change_rows)
acf_length_df = pd.DataFrame(acf_length_rows)

if psd_df.empty:
    raise RuntimeError("No PSD gain rows computed.")

if acf_change_df.empty:
    raise RuntimeError("No ACF change rows computed.")

if acf_length_df.empty:
    raise RuntimeError("No ACF length rows computed.")

psd_df.to_csv(OUTPUT_DIR / "combined_psd_bracket_gain_long.csv", index=False)
acf_change_df.to_csv(OUTPUT_DIR / "combined_radial_acf_change_long.csv", index=False)
acf_length_df.to_csv(OUTPUT_DIR / "combined_radial_acf_lengths.csv", index=False)

print()
print(f"Saved: {OUTPUT_DIR / 'combined_psd_bracket_gain_long.csv'}")
print(f"Saved: {OUTPUT_DIR / 'combined_radial_acf_change_long.csv'}")
print(f"Saved: {OUTPUT_DIR / 'combined_radial_acf_lengths.csv'}")


# =============================================================================
# Assign same strain groups to PSD and ACF rows
# =============================================================================

base_strain_df = (
    acf_length_df[
        [
            "specimen_id",
            "load_mpa",
            "sample_type",
            "sample",
            "time_h",
            "bulk_z_strain_percent",
        ]
    ]
    .drop_duplicates()
    .copy()
)

grouped_base, strain_group_labels = assign_strain_groups(
    base_strain_df,
    "bulk_z_strain_percent",
)

group_key_cols = ["specimen_id", "load_mpa", "sample_type", "sample", "time_h"]

psd_grouped = psd_df.merge(
    grouped_base[group_key_cols + ["strain_group"]],
    on=group_key_cols,
    how="inner",
)

acf_grouped = acf_change_df.merge(
    grouped_base[group_key_cols + ["strain_group"]],
    on=group_key_cols,
    how="inner",
)

acf_length_grouped = acf_length_df.merge(
    grouped_base[group_key_cols + ["strain_group"]],
    on=group_key_cols,
    how="left",
)


# =============================================================================
# Summarize PSD gain by band and strain group
# =============================================================================

psd_summary_rows = []

for (strain_group, band), g in psd_grouped.groupby(
    ["strain_group", "band"], observed=False
):
    y = g["log10_gain"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        continue

    first = g.iloc[0]

    psd_summary_rows.append(
        {
            "strain_group": strain_group,
            "band": band,
            "lambda_center_um": first["lambda_center_um"],
            "lambda_min_um": first["lambda_min_um"],
            "lambda_max_um": first["lambda_max_um"],
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "err": error_from_values(y, ERROR_BAR),
            "n": int(y.size),
        }
    )

psd_summary = pd.DataFrame(psd_summary_rows)
psd_summary["strain_group"] = pd.Categorical(
    psd_summary["strain_group"],
    categories=strain_group_labels,
    ordered=True,
)
psd_summary["band"] = pd.Categorical(
    psd_summary["band"],
    categories=psd_band_names,
    ordered=True,
)
psd_summary = psd_summary.sort_values(["strain_group", "lambda_center_um"])

psd_summary.to_csv(OUTPUT_DIR / "combined_psd_bracket_gain_summary.csv", index=False)


# =============================================================================
# Summarize ACF change by radial lag and strain group
# =============================================================================

acf_summary_rows = []

for (strain_group, r_um), g in acf_grouped.groupby(
    ["strain_group", "r_um"], observed=False
):
    y = g["delta_acf"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        continue

    acf_summary_rows.append(
        {
            "strain_group": strain_group,
            "r_um": r_um,
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "err": error_from_values(y, ERROR_BAR),
            "n": int(y.size),
        }
    )

acf_summary = pd.DataFrame(acf_summary_rows)
acf_summary["strain_group"] = pd.Categorical(
    acf_summary["strain_group"],
    categories=strain_group_labels,
    ordered=True,
)
acf_summary = acf_summary.sort_values(["strain_group", "r_um"])

acf_summary.to_csv(OUTPUT_DIR / "combined_radial_acf_change_summary.csv", index=False)


# =============================================================================
# Two-panel figure
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
ax_psd, ax_acf = axes

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.18, 0.9, len(strain_group_labels)))

# Left: bracketed PSD gain curves by strain level.
for j, strain_group in enumerate(strain_group_labels):
    s = psd_summary[psd_summary["strain_group"] == strain_group].copy()
    s = s.sort_values("lambda_center_um")

    if s.empty:
        continue

    ax_psd.errorbar(
        s["lambda_center_um"],
        s["mean"],
        yerr=s["err"],
        fmt="o-",
        lw=1.8,
        ms=4.5,
        capsize=3,
        color=colors[j],
        label=rf"$\epsilon_{{zz}}$ = {strain_group}",
    )

ax_psd.axhline(0.0, color="0.45", lw=1.0, ls="--")
ax_psd.set_xlabel(r"wavelength bracket center, $\lambda$ [$\mu$m]")
ax_psd.set_ylabel(r"mean $\log_{10}$ PSD gain")
ax_psd.set_title("A. Bracketed PSD gain")
ax_psd.grid(True, alpha=0.25)
ax_psd.legend(fontsize=7)

# Right: ACF change curves by strain level.
for j, strain_group in enumerate(strain_group_labels):
    s = acf_summary[acf_summary["strain_group"] == strain_group].copy()
    s = s.sort_values("r_um")

    if s.empty:
        continue

    ax_acf.plot(
        s["r_um"],
        s["mean"],
        lw=2.0,
        color=colors[j],
        label=rf"$\epsilon_{{zz}}$ = {strain_group}",
    )

    ax_acf.fill_between(
        s["r_um"],
        s["mean"] - s["err"],
        s["mean"] + s["err"],
        color=colors[j],
        alpha=0.15,
        linewidth=0,
    )

ax_acf.axhline(0.0, color="0.45", lw=1.0, ls="--")
ax_acf.set_xlim(0.0, ACF_MAX_LAG_UM)
ax_acf.set_xlabel(r"radial lag, $r$ [$\mu$m]")
ax_acf.set_ylabel(r"mean ACF change, $\Delta C(r)=C(r;\epsilon)-C(r;0)$")
ax_acf.set_title("B. Radial ACF change")
ax_acf.grid(True, alpha=0.25)
ax_acf.legend(fontsize=7)

fig.tight_layout()

outpath = OUTPUT_DIR / "two_panel_psd_gain_and_acf_change_by_strain.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()


# =============================================================================
# Separate figure: scatter of all radial ACF lengths vs strain
# =============================================================================

fig, ax = plt.subplots(figsize=(6.8, 4.6))

plot_len = acf_length_df.copy()
plot_len["xi_radial_um"] = pd.to_numeric(plot_len["xi_radial_um"], errors="coerce")
plot_len["bulk_z_strain_percent"] = pd.to_numeric(
    plot_len["bulk_z_strain_percent"],
    errors="coerce",
)

# Plot only actual crossings as filled points.
crossed = plot_len[
    plot_len["crossed_1e"].astype(bool)
    & np.isfinite(plot_len["xi_radial_um"])
    & np.isfinite(plot_len["bulk_z_strain_percent"])
].copy()

# Plot no-crossing cases as open triangles at MAX_LAG_FOR_XI_UM if desired.
not_crossed = plot_len[
    (~plot_len["crossed_1e"].astype(bool))
    & np.isfinite(plot_len["bulk_z_strain_percent"])
].copy()

if COLOR_SCATTER_BY_LOAD:
    for load, g in crossed.groupby("load_mpa"):
        ax.scatter(
            g["bulk_z_strain_percent"],
            g["xi_radial_um"],
            s=32,
            alpha=0.75,
            color=LOAD_COLORS.get(load, "0.35"),
            edgecolor="none",
            label=f"{int(load)} MPa",
        )

    if not not_crossed.empty:
        for load, g in not_crossed.groupby("load_mpa"):
            ax.scatter(
                g["bulk_z_strain_percent"],
                np.full(len(g), MAX_LAG_FOR_XI_UM),
                s=44,
                alpha=0.65,
                facecolors="none",
                edgecolors=LOAD_COLORS.get(load, "0.35"),
                marker="^",
                linewidths=1.0,
            )
else:
    ax.scatter(
        crossed["bulk_z_strain_percent"],
        crossed["xi_radial_um"],
        s=32,
        alpha=0.75,
        color="black",
        edgecolor="none",
        label="1/e crossing",
    )

    if not not_crossed.empty:
        ax.scatter(
            not_crossed["bulk_z_strain_percent"],
            np.full(len(not_crossed), MAX_LAG_FOR_XI_UM),
            s=44,
            alpha=0.65,
            facecolors="none",
            edgecolors="black",
            marker="^",
            linewidths=1.0,
            label="no 1/e crossing",
        )

if not not_crossed.empty:
    ax.text(
        0.02,
        0.95,
        "open triangles: no 1/e crossing\nplotted at search limit",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.25",
    )

ax.set_xlabel(r"$\epsilon_{zz}$ [%]")
ax.set_ylabel(r"radial ACF length, $\xi_r$ [$\mu$m]")
ax.set_title("Radial ACF length vs strain")
ax.grid(True, alpha=0.25)

if COLOR_SCATTER_BY_LOAD:
    ax.legend(fontsize=7, frameon=False, ncol=2)
else:
    ax.legend(fontsize=7, frameon=False)

fig.tight_layout()

outpath = OUTPUT_DIR / "radial_acf_length_scatter_vs_strain.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Four-panel endpoint spatial-configuration comparison:
#
#   Top-left:    normalized radial PSD, interrupted experiments + simulations
#   Bottom-left: normalized radial ACF, interrupted experiments + simulations
#   Top-right:   normalized radial PSD, uninterrupted experiments + simulations
#   Bottom-right: normalized radial ACF, uninterrupted experiments + simulations
#
# Endpoint comparison only:
#   - Experiment: final profilometry scan for each sample.
#   - Simulation: final simulated height field for each microstructure/face.
#
# PSD wavelength range:
#   - from twice experimental Nyquist shortest wavelength to 128 um
#   - lambda_min = 2 * (2 * exp_spacing) = 4 * exp_spacing
#
# Normalization:
#   - PSD curves are normalized by their area over log(lambda), so the plotted
#     curve shows spectral shape/configuration rather than height magnitude.
#   - ACF curves are already normalized to C(0)=1.
#
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.ndimage import distance_transform_edt

from utils.data_utils import SimResults

# =============================================================================
# User settings
# =============================================================================

DATA_ROOT = Path("/Users/gtdebru/mimosa/data")

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

# Experimental profilometry settings.
EXP_SPACING_UM = 1.379951
EXP_CROP = (slice(50, -50), slice(50, 750))
EXP_LEVEL = True
EXP_DETREND_ORDER = 1

# Simulation settings.
SIM_SPACING_UM = 1.0
SIM_LEVEL = True
SIM_DETREND_ORDER = 1

strain_col = "epav33"

micro_runs = [
    {
        "micro_id": "micro1",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro1_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro1_production.dat"
        ),
    },
    {
        "micro_id": "micro2",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro2_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro2_production.dat"
        ),
    },
    {
        "micro_id": "micro3",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro3_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro3_production.dat"
        ),
    },
]

interrupted_cases = [
    {"load": 475, "sample_type": "int"},
    {"load": 525, "sample_type": "int"},
    {"load": 575, "sample_type": "int"},
]

uninterrupted_cases = [
    {"load": 500, "sample_type": "unint"},
    {"load": 530, "sample_type": "unint"},
    {"load": 588, "sample_type": "unint"},
]

all_cases = interrupted_cases + uninterrupted_cases

LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

# PSD wavelength range.
LAMBDA_MIN_UM = 4.0 * EXP_SPACING_UM
LAMBDA_MAX_UM = 128.0
N_PSD_WAVELENGTH_BINS = 70
MIN_MODES_PER_PSD_BIN = 1

# ACF range.
ACF_MAX_LAG_UM = 128.0
ACF_RADIAL_BIN_WIDTH_UM = 2.0

# Plotting choices.
PLOT_INDIVIDUAL_CURVES = True
INDIVIDUAL_ALPHA = 0.10
PSD_YLOG = False

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load experimental point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)
if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

exp_df = point_df.copy()
exp_df["load_mpa"] = pd.to_numeric(exp_df["load_mpa"], errors="coerce")
exp_df["time_h"] = pd.to_numeric(exp_df["time_h"], errors="coerce")
exp_df["bulk_z_strain_percent"] = pd.to_numeric(
    exp_df["bulk_z_strain_percent"],
    errors="coerce",
)

exp_df = exp_df[
    np.isfinite(exp_df["load_mpa"])
    & np.isfinite(exp_df["time_h"])
    & np.isfinite(exp_df["bulk_z_strain_percent"])
].copy()

exp_df = exp_df.sort_values(
    ["sample_type", "load_mpa", "sample", "time_h"]
).reset_index(drop=True)

print("Experimental records available:")
print(exp_df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height processing
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, col = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = col.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend([x * x, x * y, y * y])
    elif order != 1:
        raise ValueError("detrend order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape, float(spacing_um_value), int(order)
    )
    coeff = inverse @ values.ravel()
    trend = (design @ coeff).reshape(values.shape)

    return values - trend


def raw_exp_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def load_exp_height(path: str | Path) -> np.ndarray:
    H = raw_exp_height(path)

    missing = ~np.isfinite(H)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        H = H[tuple(nearest)]

    if EXP_LEVEL:
        H = detrend_surface(H, EXP_SPACING_UM, order=EXP_DETREND_ORDER)

    H = H[EXP_CROP]
    H = H - np.nanmean(H)

    return H


def preprocess_sim_height(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)

    if H.ndim != 2:
        raise ValueError(f"Simulation height must be 2D; got shape {H.shape}")

    if not np.all(np.isfinite(H)):
        H = H.copy()
        H[~np.isfinite(H)] = np.nanmean(H)

    if SIM_LEVEL:
        H = detrend_surface(H, SIM_SPACING_UM, order=SIM_DETREND_ORDER)

    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD utilities
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))

    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um: np.ndarray, spacing_um_value: float):
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    return f0, f1, PSD


def radial_psd_wavelength_binned(
    height_um: np.ndarray,
    spacing_um_value: float,
    lambda_edges_um: np.ndarray,
    min_modes: int = 1,
):
    f0, f1, PSD = psd2d_height(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / FR

    nbins = len(lambda_edges_um) - 1
    lambda_center = np.sqrt(lambda_edges_um[:-1] * lambda_edges_um[1:])

    psd_radial = np.full(nbins, np.nan)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        lo = lambda_edges_um[i]
        hi = lambda_edges_um[i + 1]

        mask = (
            np.isfinite(wavelength)
            & np.isfinite(PSD)
            & (FR > 0)
            & (wavelength >= lo)
            & (wavelength < hi)
        )

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            psd_radial[i] = np.nanmean(PSD[mask])

    return lambda_center, psd_radial, modes


def integrate_trapezoid(y, x):
    """
    Compatibility helper for NumPy versions where np.trapz is unavailable.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x)
    elif hasattr(np, "trapz"):
        return np.trapz(y, x=x)
    else:
        # Manual trapezoidal integration fallback
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


def normalize_psd_shape(lambda_um: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Normalize PSD curve by area over log(lambda), so only spectral shape remains.
    """
    lam = np.asarray(lambda_um, dtype=float)
    y = np.asarray(psd, dtype=float)

    out = np.full_like(y, np.nan, dtype=float)

    valid = np.isfinite(lam) & np.isfinite(y) & (lam > 0) & (y > 0)

    if np.count_nonzero(valid) < 2:
        return out

    log_lam = np.log(lam[valid])
    area = integrate_trapezoid(y[valid], x=log_lam)

    if not np.isfinite(area) or area <= 0:
        return out

    out[valid] = y[valid] / area

    return out


lambda_edges = np.logspace(
    np.log10(LAMBDA_MIN_UM),
    np.log10(LAMBDA_MAX_UM),
    N_PSD_WAVELENGTH_BINS + 1,
)


# =============================================================================
# ACF utilities
# =============================================================================


def normalized_autocorrelation_2d(A: np.ndarray, spacing_um_value: float):
    A = np.asarray(A, dtype=float)
    A = A - np.nanmean(A)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    F = np.fft.fft2(A)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)

    center0, center1 = np.array(ac.shape) // 2
    center_value = ac[center0, center1]

    if not np.isfinite(center_value) or center_value == 0:
        ac[:] = np.nan
    else:
        ac = ac / center_value

    n0, n1 = A.shape
    z_lags_um = (np.arange(n0) - center0) * spacing_um_value
    y_lags_um = (np.arange(n1) - center1) * spacing_um_value

    return z_lags_um, y_lags_um, ac


def radial_average_acf(
    z_lags_um: np.ndarray,
    y_lags_um: np.ndarray,
    ac: np.ndarray,
    r_edges_um: np.ndarray,
):
    Y, Z = np.meshgrid(y_lags_um, z_lags_um)
    R = np.sqrt(Z**2 + Y**2)

    valid = np.isfinite(R) & np.isfinite(ac)

    r_centers = 0.5 * (r_edges_um[:-1] + r_edges_um[1:])
    c_radial = np.full_like(r_centers, np.nan, dtype=float)
    counts = np.zeros_like(r_centers, dtype=int)

    for i in range(len(r_centers)):
        mask = valid & (R >= r_edges_um[i]) & (R < r_edges_um[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            c_radial[i] = float(np.nanmean(ac[mask]))

    return r_centers, c_radial, counts


def radial_acf(height_um: np.ndarray, spacing_um_value: float, r_edges_um: np.ndarray):
    z_lags_um, y_lags_um, ac = normalized_autocorrelation_2d(
        height_um, spacing_um_value
    )
    return radial_average_acf(z_lags_um, y_lags_um, ac, r_edges_um)


r_edges = np.arange(
    0.0, ACF_MAX_LAG_UM + ACF_RADIAL_BIN_WIDTH_UM, ACF_RADIAL_BIN_WIDTH_UM
)


# =============================================================================
# Experimental endpoint curves
# =============================================================================

exp_curve_rows = []

for case in all_cases:
    load = case["load"]
    sample_type = case["sample_type"]

    case_df = exp_df[
        (exp_df["load_mpa"] == load)
        & (exp_df["sample_type"].astype(str) == sample_type)
    ].copy()

    if case_df.empty:
        warnings.warn(f"No experimental records found for {load} MPa {sample_type}")
        continue

    for sample, g in case_df.groupby("sample"):
        g = g.sort_values("time_h").copy()

        if g.empty:
            continue

        final = g.iloc[-1]

        try:
            H = load_exp_height(final["height_path"])

            lam, psd, modes = radial_psd_wavelength_binned(
                H,
                EXP_SPACING_UM,
                lambda_edges,
                min_modes=MIN_MODES_PER_PSD_BIN,
            )
            psd_norm = normalize_psd_shape(lam, psd)

            r, acf, counts = radial_acf(H, EXP_SPACING_UM, r_edges)

            for i in range(len(lam)):
                exp_curve_rows.append(
                    {
                        "source": "exp",
                        "load_mpa": load,
                        "sample_type": sample_type,
                        "sample_id": sample,
                        "replicate_id": str(sample),
                        "bulk_z_strain_percent": float(final["bulk_z_strain_percent"]),
                        "curve_type": "psd",
                        "x_um": lam[i],
                        "y": psd_norm[i],
                        "raw_y": psd[i],
                        "modes_or_counts": modes[i],
                    }
                )

            for i in range(len(r)):
                exp_curve_rows.append(
                    {
                        "source": "exp",
                        "load_mpa": load,
                        "sample_type": sample_type,
                        "sample_id": sample,
                        "replicate_id": str(sample),
                        "bulk_z_strain_percent": float(final["bulk_z_strain_percent"]),
                        "curve_type": "acf",
                        "x_um": r[i],
                        "y": acf[i],
                        "raw_y": acf[i],
                        "modes_or_counts": counts[i],
                    }
                )

        except Exception as exc:
            warnings.warn(
                f"Experimental endpoint failed for {load} {sample_type} sample {sample}: {exc}"
            )

exp_curves_df = pd.DataFrame(exp_curve_rows)


# =============================================================================
# Simulation endpoint curves
# =============================================================================

sim_curve_rows = []

for micro_run in micro_runs:
    micro_id = micro_run["micro_id"]
    sim_root = micro_run["sim_root"]
    microstructure = micro_run["microstructure"]

    for case in all_cases:
        load = case["load"]
        sample_type = case["sample_type"]

        run_dir = sim_root / f"{load}mpa_{sample_type}"

        print(f"Loading simulation endpoint: {micro_id}, {load} MPa {sample_type}")

        try:
            sim_i = SimResults.load(run_dir, microstructure=microstructure)
        except Exception as exc:
            warnings.warn(f"Failed loading simulation {run_dir}: {exc}")
            continue

        H_all = np.asarray(sim_i.height, dtype=float)
        n_faces, n_times, nz, n_width = H_all.shape

        vtk_time = np.asarray(sim_i.vtk_time, dtype=float)

        strain_time = np.asarray(sim_i.sim_time, dtype=float)
        bulk_strain_all = np.asarray(getattr(sim_i, strain_col), dtype=float)

        valid = np.isfinite(strain_time) & np.isfinite(bulk_strain_all)

        strain_time_valid = strain_time[valid]
        bulk_strain_valid = bulk_strain_all[valid]

        order = np.argsort(strain_time_valid)
        strain_time_valid = strain_time_valid[order]
        bulk_strain_valid = bulk_strain_valid[order]

        bulk_strain_vtk = np.interp(vtk_time, strain_time_valid, bulk_strain_valid)

        n = min(n_times, len(vtk_time), len(bulk_strain_vtk))

        H_all = H_all[:, :n, :, :]
        bulk_strain_vtk = bulk_strain_vtk[:n]

        final_idx = n - 1
        final_strain_percent = 100.0 * float(bulk_strain_vtk[final_idx])

        for face_idx in range(n_faces):
            replicate_id = f"{micro_id}_face{face_idx}"

            try:
                H = preprocess_sim_height(H_all[face_idx, final_idx])

                lam, psd, modes = radial_psd_wavelength_binned(
                    H,
                    SIM_SPACING_UM,
                    lambda_edges,
                    min_modes=MIN_MODES_PER_PSD_BIN,
                )
                psd_norm = normalize_psd_shape(lam, psd)

                r, acf, counts = radial_acf(H, SIM_SPACING_UM, r_edges)

                for i in range(len(lam)):
                    sim_curve_rows.append(
                        {
                            "source": "sim",
                            "load_mpa": load,
                            "sample_type": sample_type,
                            "sample_id": micro_id,
                            "replicate_id": replicate_id,
                            "bulk_z_strain_percent": final_strain_percent,
                            "curve_type": "psd",
                            "x_um": lam[i],
                            "y": psd_norm[i],
                            "raw_y": psd[i],
                            "modes_or_counts": modes[i],
                        }
                    )

                for i in range(len(r)):
                    sim_curve_rows.append(
                        {
                            "source": "sim",
                            "load_mpa": load,
                            "sample_type": sample_type,
                            "sample_id": micro_id,
                            "replicate_id": replicate_id,
                            "bulk_z_strain_percent": final_strain_percent,
                            "curve_type": "acf",
                            "x_um": r[i],
                            "y": acf[i],
                            "raw_y": acf[i],
                            "modes_or_counts": counts[i],
                        }
                    )

            except Exception as exc:
                warnings.warn(
                    f"Simulation endpoint failed for {micro_id} {load} {sample_type} face {face_idx}: {exc}"
                )

sim_curves_df = pd.DataFrame(sim_curve_rows)


# =============================================================================
# Combine and aggregate
# =============================================================================

curves_df = pd.concat([exp_curves_df, sim_curves_df], ignore_index=True)

if curves_df.empty:
    raise RuntimeError("No endpoint curves were computed.")

curves_df.to_csv(OUTPUT_DIR / "endpoint_exp_sim_psd_acf_curves_long.csv", index=False)

summary_rows = []

for (source, load, sample_type, curve_type, x_um), g in curves_df.groupby(
    ["source", "load_mpa", "sample_type", "curve_type", "x_um"]
):
    y = pd.to_numeric(g["y"], errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        continue

    summary_rows.append(
        {
            "source": source,
            "load_mpa": load,
            "sample_type": sample_type,
            "curve_type": curve_type,
            "x_um": x_um,
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
            "n": int(y.size),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    OUTPUT_DIR / "endpoint_exp_sim_psd_acf_curves_summary.csv", index=False
)

print()
print(f"Saved: {OUTPUT_DIR / 'endpoint_exp_sim_psd_acf_curves_long.csv'}")
print(f"Saved: {OUTPUT_DIR / 'endpoint_exp_sim_psd_acf_curves_summary.csv'}")


# =============================================================================
# Plot four-panel figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=False)

ax_psd_int = axes[0, 0]
ax_psd_unint = axes[0, 1]
ax_acf_int = axes[1, 0]
ax_acf_unint = axes[1, 1]


def plot_group(ax, cases, curve_type, title, ylabel):
    for case in cases:
        load = case["load"]
        sample_type = case["sample_type"]
        color = LOAD_COLORS.get(load, "0.3")

        for source, linestyle, marker, alpha, label_suffix in [
            ("exp", "-", None, 1.0, "Exp."),
            ("sim", "--", None, 0.95, "Sim."),
        ]:
            s = summary_df[
                (summary_df["source"] == source)
                & (summary_df["load_mpa"] == load)
                & (summary_df["sample_type"] == sample_type)
                & (summary_df["curve_type"] == curve_type)
            ].copy()

            if s.empty:
                continue

            s = s.sort_values("x_um")

            ax.plot(
                s["x_um"],
                s["mean"],
                color=color,
                linestyle=linestyle,
                lw=2.0 if source == "exp" else 1.8,
                alpha=alpha,
                label=f"{load} MPa {label_suffix}",
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


plot_group(
    ax_psd_int,
    interrupted_cases,
    "psd",
    "A. Interrupted: normalized radial PSD",
    r"normalized radial PSD",
)

plot_group(
    ax_psd_unint,
    uninterrupted_cases,
    "psd",
    "B. Uninterrupted: normalized radial PSD",
    r"normalized radial PSD",
)

plot_group(
    ax_acf_int,
    interrupted_cases,
    "acf",
    "C. Interrupted: radial ACF",
    r"radial ACF, $C(r)$",
)

plot_group(
    ax_acf_unint,
    uninterrupted_cases,
    "acf",
    "D. Uninterrupted: radial ACF",
    r"radial ACF, $C(r)$",
)

for ax in [ax_psd_int, ax_psd_unint]:
    ax.set_xscale("log")
    ax.set_xlim(LAMBDA_MIN_UM, LAMBDA_MAX_UM)
    ax.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")
    if PSD_YLOG:
        ax.set_yscale("log")
    ax.legend(fontsize=6, ncol=2)

for ax in [ax_acf_int, ax_acf_unint]:
    ax.set_xlim(0.0, ACF_MAX_LAG_UM)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axhline(np.exp(-1.0), color="0.5", lw=0.8, ls=":")
    ax.legend(fontsize=6, ncol=2)

# Numeric x labels for PSD log axes.
psd_ticks = np.array([6, 10, 20, 50, 100, 128], dtype=float)
psd_ticks = psd_ticks[(psd_ticks >= LAMBDA_MIN_UM) & (psd_ticks <= LAMBDA_MAX_UM)]

for ax in [ax_psd_int, ax_psd_unint]:
    ax.xaxis.set_major_locator(FixedLocator(psd_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{int(t)}" for t in psd_ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())

fig.suptitle(
    "Endpoint spatial configuration: experiment vs simulation\n"
    "PSD normalized by spectral area; ACF normalized to unity at zero lag",
    y=1.02,
)

fig.tight_layout()

outpath = OUTPUT_DIR / "four_panel_endpoint_exp_vs_sim_normalized_psd_acf.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Experimental two-panel figure + radial ACF length scatter
# =============================================================================
#
# Uses corrected ACF method:
#   - crop first
#   - plane-level cropped field
#   - mean-remove
#   - use full field, no patches
#   - overlap-normalized linear ACF via FFT convolution
#   - radial ACF bin width = Nyquist shortest wavelength = 2 * pixel spacing
#
# Outputs:
#   1. two_panel_exp_psd_gain_and_corrected_acf_change_by_strain.png
#      Left:  bracketed PSD log10 gain curves by strain level
#      Right: corrected radial ACF change curves by strain level
#
#   2. corrected_radial_acf_length_scatter_vs_strain.png
#      Scatter of radial e-folding ACF length vs strain
#
#   3. CSV outputs for PSD gains, ACF changes, and ACF lengths
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

# Exclude sub-yield load from this comparison if desired.
EXCLUDED_LOADS = {475}
# EXCLUDED_LOADS = set()

spacing_um = 1.379951
crop = (slice(50, -50), slice(50, 750))

DETREND_ORDER = 1

INITIAL_TIME_TOL_H = 1.0e-8
INITIAL_STRAIN_TOL_PERCENT = 0.05

N_STRAIN_GROUPS = 4

# PSD bracket settings.
# Same style as earlier: 20 um brackets from 2*Nyquist shortest wavelength to 300 um.
PSD_BRACKET_WIDTH_UM = 20.0
PSD_MAX_WAVELENGTH_UM = 300.0

# Corrected ACF settings.
# "Bin sizes of the Nyquist frequency" interpreted in distance units as the
# Nyquist shortest wavelength = 2 * pixel spacing.
ACF_RADIAL_BIN_WIDTH_UM = 2.0 * spacing_um
ACF_MAX_LAG_UM = 128.0
ACF_TARGET = np.exp(-1.0)
ZERO_TARGET = 0.0

ERROR_BAR = "sem"  # "std", "sem", or "ci95"

COLOR_SCATTER_BY_LOAD = True
SHOW_NO_CROSSING_MARKERS = True

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}


# =============================================================================
# Load point table
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)
if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

df = point_df.copy()

df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
df["bulk_z_strain_percent"] = pd.to_numeric(
    df["bulk_z_strain_percent"],
    errors="coerce",
)

if "delta_sa_um" in df.columns:
    df["delta_sa_um"] = pd.to_numeric(df["delta_sa_um"], errors="coerce")

df = df[
    np.isfinite(df["load_mpa"])
    & np.isfinite(df["time_h"])
    & np.isfinite(df["bulk_z_strain_percent"])
].copy()

df = df[~df["load_mpa"].isin(EXCLUDED_LOADS)].copy()
df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
    drop=True
)

if df.empty:
    raise ValueError("No valid rows remain after filtering.")

df["specimen_id"] = (
    df["sample_type"].astype(str)
    + "_"
    + df["load_mpa"].astype(int).astype(str)
    + "MPa_"
    + df["sample"].astype(str)
)

print("Records used:")
print(df.groupby(["sample_type", "load_mpa"]).size())
print(f"Number of specimens: {df['specimen_id'].nunique()}")


# =============================================================================
# Height loading: crop first, then plane-level cropped field
# =============================================================================


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, col = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = col.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend([x * x, x * y, y * y])
    elif order != 1:
        raise ValueError("DETREND_ORDER must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape,
        float(spacing_um_value),
        int(order),
    )

    coeff = inverse @ values.ravel()
    trend = (design @ coeff).reshape(values.shape)

    return values - trend


def raw_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def fill_missing_nearest(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    missing = ~np.isfinite(H)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        H = H[tuple(nearest)]

    return H


def load_cropped_then_leveled_height(path: str | Path) -> np.ndarray:
    """
    Correct processing:
      1. read raw height
      2. fill missing
      3. crop
      4. plane-level cropped field
      5. mean-remove
    """
    H = raw_height(path)
    H = fill_missing_nearest(H)
    H = H[crop]

    H = detrend_surface(H, spacing_um, order=DETREND_ORDER)
    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD functions
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))
    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um: np.ndarray, spacing_um_value: float):
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    df0 = 1.0 / (n0 * spacing_um_value)
    df1 = 1.0 / (n1 * spacing_um_value)

    return f0, f1, PSD, df0, df1


def make_psd_wavelength_bands():
    nyquist_shortest_wavelength_um = 2.0 * spacing_um
    twice_nyquist_um = 2.0 * nyquist_shortest_wavelength_um

    lambda_start_um = twice_nyquist_um
    lambda_end_um = PSD_MAX_WAVELENGTH_UM

    if lambda_end_um <= lambda_start_um:
        raise ValueError(
            f"Invalid PSD wavelength range: start={lambda_start_um}, end={lambda_end_um}"
        )

    edges = [lambda_start_um]

    while edges[-1] < lambda_end_um:
        edges.append(min(edges[-1] + PSD_BRACKET_WIDTH_UM, lambda_end_um))

    edges = np.asarray(edges, dtype=float)

    bands = {}

    for i in range(len(edges) - 1):
        name = f"{edges[i]:.1f}-{edges[i+1]:.1f} µm"
        bands[name] = (edges[i], edges[i + 1])

    return bands


def psd_band_powers(height_um: np.ndarray, bands: dict[str, tuple[float, float]]):
    f0, f1, PSD, df0, df1 = psd2d_height(height_um, spacing_um)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength_um = 1.0 / FR

    out = {}

    for name, (lo, hi) in bands.items():
        mask = (
            np.isfinite(wavelength_um)
            & (FR > 0)
            & (wavelength_um >= lo)
            & (wavelength_um < hi)
        )

        power = float(np.nansum(PSD[mask]) * df0 * df1)
        out[name] = power

    return out


psd_bands_um = make_psd_wavelength_bands()
psd_band_names = list(psd_bands_um.keys())

print()
print("PSD wavelength brackets:")
for name, (lo, hi) in psd_bands_um.items():
    print(f"  {name}: {lo:.3f} to {hi:.3f} um")


# =============================================================================
# Corrected overlap-normalized radial ACF
# =============================================================================


def overlap_normalized_acf_2d(z: np.ndarray):
    """
    Linear overlap-normalized ACF:

        rho(dx,dy) =
            <z(x,y) z(x+dx,y+dy)> / <z^2>

    computed by FFT convolution and normalized pointwise by overlap count.
    """
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)

    if np.count_nonzero(finite) < 10:
        raise ValueError("Too few finite pixels for ACF.")

    z0 = np.zeros_like(z, dtype=float)
    z_mean = np.nanmean(z)
    z0[finite] = z[finite] - z_mean

    mask = finite.astype(float)

    numerator = fftconvolve(z0, z0[::-1, ::-1], mode="full")
    overlap_counts = fftconvolve(mask, mask[::-1, ::-1], mode="full")

    covariance = np.full_like(numerator, np.nan, dtype=float)

    valid_overlap = overlap_counts > 0
    covariance[valid_overlap] = numerator[valid_overlap] / overlap_counts[valid_overlap]

    center = tuple(np.array(covariance.shape) // 2)
    zero_lag_cov = covariance[center]

    if not np.isfinite(zero_lag_cov) or zero_lag_cov <= 0:
        rho = np.full_like(covariance, np.nan, dtype=float)
    else:
        rho = covariance / zero_lag_cov

    n0, n1 = z.shape

    lag0_pix = np.arange(-(n0 - 1), n0)
    lag1_pix = np.arange(-(n1 - 1), n1)

    lag0_um = lag0_pix * spacing_um
    lag1_um = lag1_pix * spacing_um

    return lag0_um, lag1_um, rho, overlap_counts


def radial_average_acf(
    lag0_um: np.ndarray,
    lag1_um: np.ndarray,
    rho: np.ndarray,
    *,
    dr_um: float = ACF_RADIAL_BIN_WIDTH_UM,
    r_max_um: float = ACF_MAX_LAG_UM,
):
    L1, L0 = np.meshgrid(lag1_um, lag0_um)
    R = np.sqrt(L0**2 + L1**2)

    edges = np.arange(0.0, r_max_um + dr_um, dr_um)
    centers = 0.5 * (edges[:-1] + edges[1:])

    radial = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)

    valid = np.isfinite(R) & np.isfinite(rho)

    for i in range(len(centers)):
        mask = valid & (R >= edges[i]) & (R < edges[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            radial[i] = float(np.nanmean(rho[mask]))

    keep = centers < r_max_um

    return centers[keep], radial[keep], counts[keep]


def radial_acf(height_um: np.ndarray):
    lag0_um, lag1_um, rho2d, overlap_counts = overlap_normalized_acf_2d(height_um)

    r_um, rho_r, counts = radial_average_acf(
        lag0_um,
        lag1_um,
        rho2d,
        dr_um=ACF_RADIAL_BIN_WIDTH_UM,
        r_max_um=ACF_MAX_LAG_UM,
    )

    return r_um, rho_r, counts


def first_crossing_linear_interp(r_um: np.ndarray, y: np.ndarray, target: float):
    r = np.asarray(r_um, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(r) & np.isfinite(y)
    r = r[valid]
    y = y[valid]

    if len(r) < 2:
        return np.nan, False

    for k in range(1, len(r)):
        if y[k] <= target:
            r0, r1 = r[k - 1], r[k]
            y0, y1 = y[k - 1], y[k]

            if y1 == y0:
                return float(r1), True

            rcross = r0 + (target - y0) * (r1 - r0) / (y1 - y0)
            return float(rcross), True

    return np.nan, False


# =============================================================================
# Helpers
# =============================================================================


def error_from_values(y: np.ndarray, mode: str):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if y.size <= 1:
        return 0.0

    std = float(np.std(y, ddof=1))
    sem = std / np.sqrt(y.size)

    if mode == "std":
        return std
    if mode == "sem":
        return sem
    if mode == "ci95":
        return 1.96 * sem

    raise ValueError("ERROR_BAR must be 'std', 'sem', or 'ci95'.")


def assign_strain_groups(input_df: pd.DataFrame, strain_col: str):
    d = input_df[input_df[strain_col] > INITIAL_STRAIN_TOL_PERCENT].copy()

    if d.empty:
        raise ValueError("No non-initial rows available for strain grouping.")

    q = np.unique(
        np.nanquantile(
            d[strain_col].to_numpy(dtype=float),
            np.linspace(0.0, 1.0, N_STRAIN_GROUPS + 1),
        )
    )

    d["strain_group"] = None
    labels = []

    for i in range(len(q) - 1):
        lo = q[i]
        hi = q[i + 1]

        if i == len(q) - 2:
            mask = (d[strain_col] >= lo) & (d[strain_col] <= hi)
        else:
            mask = (d[strain_col] >= lo) & (d[strain_col] < hi)

        label = rf"{lo:.2g}--{hi:.2g}%"
        d.loc[mask, "strain_group"] = label
        labels.append(label)

    d = d[d["strain_group"].notna()].copy()

    d["strain_group"] = pd.Categorical(
        d["strain_group"],
        categories=labels,
        ordered=True,
    )

    return d, labels


# =============================================================================
# Compute PSD gain, ACF change, and ACF length
# =============================================================================

psd_rows = []
acf_change_rows = []
acf_length_rows = []

for specimen_id, g in df.groupby("specimen_id"):
    g = g.sort_values("time_h").copy()

    if len(g) < 2:
        continue

    initial = g.iloc[0]

    if float(initial["time_h"]) > INITIAL_TIME_TOL_H:
        warnings.warn(
            f"Skipping specimen {specimen_id}: earliest time is "
            f"{initial['time_h']}, not initial."
        )
        continue

    try:
        H0 = load_cropped_then_leveled_height(initial["height_path"])
        p0 = psd_band_powers(H0, psd_bands_um)
        r0, c0, _ = radial_acf(H0)
    except Exception as exc:
        warnings.warn(f"Initial reference failed for specimen {specimen_id}: {exc}")
        continue

    for _, rec in g.iterrows():
        try:
            H = load_cropped_then_leveled_height(rec["height_path"])

            strain = float(rec["bulk_z_strain_percent"])
            time_h = float(rec["time_h"])

            # PSD gain.
            p = psd_band_powers(H, psd_bands_um)

            for band_name in psd_band_names:
                initial_power = p0[band_name]
                current_power = p[band_name]

                gain = (
                    current_power / initial_power
                    if np.isfinite(initial_power) and initial_power > 0
                    else np.nan
                )

                log10_gain = (
                    np.log10(gain) if np.isfinite(gain) and gain > 0 else np.nan
                )

                lo, hi = psd_bands_um[band_name]

                psd_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "load_mpa": rec["load_mpa"],
                        "sample_type": rec["sample_type"],
                        "sample": rec["sample"],
                        "time_h": time_h,
                        "bulk_z_strain_percent": strain,
                        "band": band_name,
                        "lambda_min_um": lo,
                        "lambda_max_um": hi,
                        "lambda_center_um": 0.5 * (lo + hi),
                        "log10_gain": log10_gain,
                        "gain": gain,
                    }
                )

            # ACF change and length.
            r, c, _ = radial_acf(H)

            if not np.allclose(r, r0, equal_nan=True):
                raise ValueError("ACF radial grids do not match.")

            delta_c = c - c0

            xi_e, crossed_e = first_crossing_linear_interp(r, c, ACF_TARGET)
            xi_zero, crossed_zero = first_crossing_linear_interp(r, c, ZERO_TARGET)

            acf_length_rows.append(
                {
                    "specimen_id": specimen_id,
                    "load_mpa": rec["load_mpa"],
                    "sample_type": rec["sample_type"],
                    "sample": rec["sample"],
                    "time_h": time_h,
                    "bulk_z_strain_percent": strain,
                    "xi_e_um": xi_e,
                    "crossed_e": crossed_e,
                    "xi_zero_um": xi_zero,
                    "crossed_zero": crossed_zero,
                }
            )

            for i in range(len(r)):
                acf_change_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "load_mpa": rec["load_mpa"],
                        "sample_type": rec["sample_type"],
                        "sample": rec["sample"],
                        "time_h": time_h,
                        "bulk_z_strain_percent": strain,
                        "r_um": r[i],
                        "delta_acf": delta_c[i],
                        "acf_current": c[i],
                        "acf_initial": c0[i],
                    }
                )

        except Exception as exc:
            warnings.warn(f"Failed specimen {specimen_id}, time={rec['time_h']}: {exc}")

psd_df = pd.DataFrame(psd_rows)
acf_change_df = pd.DataFrame(acf_change_rows)
acf_length_df = pd.DataFrame(acf_length_rows)

if psd_df.empty:
    raise RuntimeError("No PSD gain rows computed.")

if acf_change_df.empty:
    raise RuntimeError("No ACF change rows computed.")

if acf_length_df.empty:
    raise RuntimeError("No ACF length rows computed.")

psd_df.to_csv(OUTPUT_DIR / "corrected_psd_bracket_gain_long.csv", index=False)
acf_change_df.to_csv(OUTPUT_DIR / "corrected_radial_acf_change_long.csv", index=False)
acf_length_df.to_csv(OUTPUT_DIR / "corrected_radial_acf_lengths.csv", index=False)

print()
print(f"Saved: {OUTPUT_DIR / 'corrected_psd_bracket_gain_long.csv'}")
print(f"Saved: {OUTPUT_DIR / 'corrected_radial_acf_change_long.csv'}")
print(f"Saved: {OUTPUT_DIR / 'corrected_radial_acf_lengths.csv'}")


# =============================================================================
# Assign same strain groups to PSD and ACF rows
# =============================================================================

base_strain_df = (
    acf_length_df[
        [
            "specimen_id",
            "load_mpa",
            "sample_type",
            "sample",
            "time_h",
            "bulk_z_strain_percent",
        ]
    ]
    .drop_duplicates()
    .copy()
)

grouped_base, strain_group_labels = assign_strain_groups(
    base_strain_df,
    "bulk_z_strain_percent",
)

group_key_cols = ["specimen_id", "load_mpa", "sample_type", "sample", "time_h"]

psd_grouped = psd_df.merge(
    grouped_base[group_key_cols + ["strain_group"]],
    on=group_key_cols,
    how="inner",
)

acf_grouped = acf_change_df.merge(
    grouped_base[group_key_cols + ["strain_group"]],
    on=group_key_cols,
    how="inner",
)

acf_length_grouped = acf_length_df.merge(
    grouped_base[group_key_cols + ["strain_group"]],
    on=group_key_cols,
    how="left",
)


# =============================================================================
# Summarize PSD gain by band and strain group
# =============================================================================

psd_summary_rows = []

for (strain_group, band), g in psd_grouped.groupby(
    ["strain_group", "band"], observed=False
):
    y = g["log10_gain"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        continue

    first = g.iloc[0]

    psd_summary_rows.append(
        {
            "strain_group": strain_group,
            "band": band,
            "lambda_center_um": first["lambda_center_um"],
            "lambda_min_um": first["lambda_min_um"],
            "lambda_max_um": first["lambda_max_um"],
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "err": error_from_values(y, ERROR_BAR),
            "n": int(y.size),
        }
    )

psd_summary = pd.DataFrame(psd_summary_rows)
psd_summary["strain_group"] = pd.Categorical(
    psd_summary["strain_group"],
    categories=strain_group_labels,
    ordered=True,
)
psd_summary["band"] = pd.Categorical(
    psd_summary["band"],
    categories=psd_band_names,
    ordered=True,
)
psd_summary = psd_summary.sort_values(["strain_group", "lambda_center_um"])

psd_summary.to_csv(OUTPUT_DIR / "corrected_psd_bracket_gain_summary.csv", index=False)


# =============================================================================
# Summarize ACF change by radial lag and strain group
# =============================================================================

acf_summary_rows = []

for (strain_group, r_um), g in acf_grouped.groupby(
    ["strain_group", "r_um"], observed=False
):
    y = g["delta_acf"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        continue

    acf_summary_rows.append(
        {
            "strain_group": strain_group,
            "r_um": r_um,
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "err": error_from_values(y, ERROR_BAR),
            "n": int(y.size),
        }
    )

acf_summary = pd.DataFrame(acf_summary_rows)
acf_summary["strain_group"] = pd.Categorical(
    acf_summary["strain_group"],
    categories=strain_group_labels,
    ordered=True,
)
acf_summary = acf_summary.sort_values(["strain_group", "r_um"])

acf_summary.to_csv(OUTPUT_DIR / "corrected_radial_acf_change_summary.csv", index=False)


# =============================================================================
# Two-panel figure
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
ax_psd, ax_acf = axes

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.18, 0.9, len(strain_group_labels)))

# Left: bracketed PSD gain curves by strain level.
for j, strain_group in enumerate(strain_group_labels):
    s = psd_summary[psd_summary["strain_group"] == strain_group].copy()
    s = s.sort_values("lambda_center_um")

    if s.empty:
        continue

    ax_psd.errorbar(
        s["lambda_center_um"],
        s["mean"],
        yerr=s["err"],
        fmt="o-",
        lw=1.8,
        ms=4.5,
        capsize=3,
        color=colors[j],
        label=rf"$\epsilon_{{zz}}$ = {strain_group}",
    )

ax_psd.axhline(0.0, color="0.45", lw=1.0, ls="--")
ax_psd.set_xlabel(r"wavelength bracket center, $\lambda$ [$\mu$m]")
ax_psd.set_ylabel(r"mean $\log_{10}$ PSD gain")
ax_psd.set_title("A. Bracketed PSD gain")
ax_psd.grid(True, alpha=0.25)
ax_psd.legend(fontsize=7)

# Right: corrected radial ACF change curves by strain level.
for j, strain_group in enumerate(strain_group_labels):
    s = acf_summary[acf_summary["strain_group"] == strain_group].copy()
    s = s.sort_values("r_um")

    if s.empty:
        continue

    ax_acf.plot(
        s["r_um"],
        s["mean"],
        lw=2.0,
        color=colors[j],
        label=rf"$\epsilon_{{zz}}$ = {strain_group}",
    )

    ax_acf.fill_between(
        s["r_um"],
        s["mean"] - s["err"],
        s["mean"] + s["err"],
        color=colors[j],
        alpha=0.15,
        linewidth=0,
    )

ax_acf.axhline(0.0, color="0.45", lw=1.0, ls="--")
ax_acf.set_xlim(0.0, ACF_MAX_LAG_UM)
ax_acf.set_xlabel(r"radial lag, $r$ [$\mu$m]")
ax_acf.set_ylabel(r"mean ACF change, $\Delta C(r)=C(r;\epsilon)-C(r;0)$")
ax_acf.set_title("B. Corrected radial ACF change")
ax_acf.grid(True, alpha=0.25)
ax_acf.legend(fontsize=7)

fig.tight_layout()

outpath = OUTPUT_DIR / "two_panel_corrected_psd_gain_and_acf_change_by_strain.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()


# =============================================================================
# Separate figure: scatter of all corrected radial ACF lengths vs strain
# =============================================================================

fig, ax = plt.subplots(figsize=(6.8, 4.6))

plot_len = acf_length_df.copy()

plot_len["xi_e_um"] = pd.to_numeric(plot_len["xi_e_um"], errors="coerce")
plot_len["xi_zero_um"] = pd.to_numeric(plot_len["xi_zero_um"], errors="coerce")
plot_len["bulk_z_strain_percent"] = pd.to_numeric(
    plot_len["bulk_z_strain_percent"],
    errors="coerce",
)

crossed = plot_len[
    plot_len["crossed_e"].astype(bool)
    & np.isfinite(plot_len["xi_e_um"])
    & np.isfinite(plot_len["bulk_z_strain_percent"])
].copy()

not_crossed = plot_len[
    (~plot_len["crossed_e"].astype(bool))
    & np.isfinite(plot_len["bulk_z_strain_percent"])
].copy()

if COLOR_SCATTER_BY_LOAD:
    for load, g in crossed.groupby("load_mpa"):
        ax.scatter(
            g["bulk_z_strain_percent"],
            g["xi_e_um"],
            s=32,
            alpha=0.75,
            color=LOAD_COLORS.get(load, "0.35"),
            edgecolor="none",
            label=f"{int(load)} MPa",
        )

    if SHOW_NO_CROSSING_MARKERS and not not_crossed.empty:
        for load, g in not_crossed.groupby("load_mpa"):
            ax.scatter(
                g["bulk_z_strain_percent"],
                np.full(len(g), ACF_MAX_LAG_UM),
                s=44,
                alpha=0.65,
                facecolors="none",
                edgecolors=LOAD_COLORS.get(load, "0.35"),
                marker="^",
                linewidths=1.0,
            )
else:
    ax.scatter(
        crossed["bulk_z_strain_percent"],
        crossed["xi_e_um"],
        s=32,
        alpha=0.75,
        color="black",
        edgecolor="none",
        label=r"$1/e$ crossing",
    )

    if SHOW_NO_CROSSING_MARKERS and not not_crossed.empty:
        ax.scatter(
            not_crossed["bulk_z_strain_percent"],
            np.full(len(not_crossed), ACF_MAX_LAG_UM),
            s=44,
            alpha=0.65,
            facecolors="none",
            edgecolors="black",
            marker="^",
            linewidths=1.0,
            label="no crossing",
        )

if SHOW_NO_CROSSING_MARKERS and not not_crossed.empty:
    ax.text(
        0.02,
        0.95,
        "open triangles: no 1/e crossing\nplotted at lag limit",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.25",
    )

ax.set_xlabel(r"$\epsilon_{zz}$ [%]")
ax.set_ylabel(r"radial ACF e-folding length, $\xi_r$ [$\mu$m]")
ax.set_title("Corrected radial ACF length vs strain")
ax.grid(True, alpha=0.25)

if COLOR_SCATTER_BY_LOAD:
    ax.legend(fontsize=7, frameon=False, ncol=2)
else:
    ax.legend(fontsize=7, frameon=False)

fig.tight_layout()

outpath = OUTPUT_DIR / "corrected_radial_acf_length_scatter_vs_strain.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Four-panel endpoint simulation-experiment spatial-configuration comparison
# =============================================================================
#
# Figure layout:
#
#   Top-left:     Interrupted experiments + corresponding simulations
#                 normalized radial PSD vs wavelength
#
#   Bottom-left:  Interrupted experiments + corresponding simulations
#                 normalized radial ACF vs radial lag
#
#   Top-right:    Uninterrupted experiments + corresponding simulations
#                 normalized radial PSD vs wavelength
#
#   Bottom-right: Uninterrupted experiments + corresponding simulations
#                 normalized radial ACF vs radial lag
#
# Endpoint only:
#   - Experiment: final 10x profilometry scan for each specimen
#   - Simulation: final SimResults.height field for each microstructure/face
#
# Processing:
#   - Experiment height fields: read raw, fill missing, crop, plane-level cropped
#     field, mean-remove.
#   - Simulation height fields: plane-level, mean-remove.
#
# PSD:
#   - Signed height field.
#   - Radial PSD from twice experimental Nyquist shortest wavelength to 128 um.
#   - lambda_min = 2 * (2 * experimental pixel spacing) = 4 * dx_exp.
#   - PSD curves are area-normalized over log(lambda), so the plotted PSDs
#     compare spectral shape rather than height magnitude.
#
# ACF:
#   - Corrected overlap-normalized linear ACF, not circular ACF.
#   - Radial ACF out to 128 um.
#   - ACF is already normalized to C(0)=1.
#
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from utils.data_utils import SimResults

# =============================================================================
# User settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

# Experimental profilometry settings.
EXP_SPACING_UM = 1.379951
EXP_CROP = (slice(50, -50), slice(50, 750))
EXP_DETREND_ORDER = 1

# Simulation settings.
# If simulation pixel spacing is not 1 um, change this.
SIM_SPACING_UM = 1.0
SIM_DETREND_ORDER = 1

# Simulation loading settings.
strain_col = "epav33"

micro_runs = [
    {
        "micro_id": "micro1",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro1_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro1_production.dat"
        ),
    },
    {
        "micro_id": "micro2",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro2_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro2_production.dat"
        ),
    },
    {
        "micro_id": "micro3",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro3_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro3_production.dat"
        ),
    },
]

interrupted_cases = [
    {"load": 475, "sample_type": "int"},
    {"load": 525, "sample_type": "int"},
    {"load": 575, "sample_type": "int"},
]

uninterrupted_cases = [
    {"load": 500, "sample_type": "unint"},
    {"load": 530, "sample_type": "unint"},
    {"load": 588, "sample_type": "unint"},
]

all_cases = interrupted_cases + uninterrupted_cases

LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

# PSD range.
LAMBDA_MIN_UM = 4.0 * EXP_SPACING_UM
LAMBDA_MAX_UM = 128.0
N_PSD_WAVELENGTH_BINS = 70
MIN_MODES_PER_PSD_BIN = 1

# ACF range.
ACF_MAX_LAG_UM = 128.0

# Use experimental Nyquist shortest wavelength as radial ACF bin width.
ACF_RADIAL_BIN_WIDTH_UM = 2.0 * EXP_SPACING_UM

# Plot.
PLOT_INDIVIDUAL_CURVES = True
INDIVIDUAL_ALPHA = 0.10
PSD_YLOG = False

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Utility functions
# =============================================================================


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x)
    if hasattr(np, "trapz"):
        return np.trapz(y, x=x)

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, col = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = col.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend([x * x, x * y, y * y])
    elif order != 1:
        raise ValueError("detrend order must be 1 or 2.")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape, float(spacing_um_value), int(order)
    )
    coeff = inverse @ values.ravel()
    trend = (design @ coeff).reshape(values.shape)

    return values - trend


def fill_missing_nearest(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    missing = ~np.isfinite(H)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        H = H[tuple(nearest)]

    return H


# =============================================================================
# Load experiment data
# =============================================================================

if "point_df" not in globals():
    if not POINT_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
        )
    point_df = pd.read_csv(POINT_TABLE_PATH)

required_columns = {
    "load_mpa",
    "sample_type",
    "sample",
    "time_h",
    "bulk_z_strain_percent",
    "height_path",
}

missing = required_columns - set(point_df.columns)
if missing:
    raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

exp_df = point_df.copy()
exp_df["load_mpa"] = pd.to_numeric(exp_df["load_mpa"], errors="coerce")
exp_df["time_h"] = pd.to_numeric(exp_df["time_h"], errors="coerce")
exp_df["bulk_z_strain_percent"] = pd.to_numeric(
    exp_df["bulk_z_strain_percent"],
    errors="coerce",
)

exp_df = exp_df[
    np.isfinite(exp_df["load_mpa"])
    & np.isfinite(exp_df["time_h"])
    & np.isfinite(exp_df["bulk_z_strain_percent"])
].copy()

exp_df = exp_df.sort_values(
    ["sample_type", "load_mpa", "sample", "time_h"]
).reset_index(drop=True)

print("Experimental records available:")
print(exp_df.groupby(["sample_type", "load_mpa"]).size())


# =============================================================================
# Height loading
# =============================================================================


def raw_exp_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def load_exp_height(path: str | Path) -> np.ndarray:
    """
    Experiment processing:
      1. read raw height
      2. fill missing
      3. crop
      4. plane-level cropped field
      5. mean-remove
    """
    H = raw_exp_height(path)
    H = fill_missing_nearest(H)
    H = H[EXP_CROP]
    H = detrend_surface(H, EXP_SPACING_UM, order=EXP_DETREND_ORDER)
    H = H - np.nanmean(H)

    return H


def preprocess_sim_height(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)

    if H.ndim != 2:
        raise ValueError(f"Simulation height must be 2D; got shape {H.shape}")

    if not np.all(np.isfinite(H)):
        H = H.copy()
        H[~np.isfinite(H)] = np.nanmean(H)

    H = detrend_surface(H, SIM_SPACING_UM, order=SIM_DETREND_ORDER)
    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD functions
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    W = w0[:, None] * w1[None, :]

    rms = np.sqrt(np.mean(W**2))

    if rms > 0:
        W = W / rms

    return W


def psd2d_height(height_um: np.ndarray, spacing_um_value: float):
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    n0, n1 = H.shape
    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)

    pixel_area = spacing_um_value * spacing_um_value
    n_pixels = n0 * n1

    PSD = pixel_area / n_pixels * np.abs(F) ** 2

    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    return f0, f1, PSD


def radial_psd_wavelength_binned(
    height_um: np.ndarray,
    spacing_um_value: float,
    lambda_edges_um: np.ndarray,
    min_modes: int = 1,
):
    f0, f1, PSD = psd2d_height(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = 1.0 / FR

    nbins = len(lambda_edges_um) - 1
    lambda_center = np.sqrt(lambda_edges_um[:-1] * lambda_edges_um[1:])

    psd_radial = np.full(nbins, np.nan)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        lo = lambda_edges_um[i]
        hi = lambda_edges_um[i + 1]

        mask = (
            np.isfinite(wavelength)
            & np.isfinite(PSD)
            & (FR > 0)
            & (wavelength >= lo)
            & (wavelength < hi)
        )

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            psd_radial[i] = np.nanmean(PSD[mask])

    return lambda_center, psd_radial, modes


def normalize_psd_shape(lambda_um: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Normalize PSD curve by area over log(lambda), so curves compare spectral
    shape rather than height magnitude.
    """
    lam = np.asarray(lambda_um, dtype=float)
    y = np.asarray(psd, dtype=float)

    out = np.full_like(y, np.nan, dtype=float)

    valid = np.isfinite(lam) & np.isfinite(y) & (lam > 0) & (y > 0)

    if np.count_nonzero(valid) < 2:
        return out

    log_lam = np.log(lam[valid])
    area = integrate_trapezoid(y[valid], x=log_lam)

    if not np.isfinite(area) or area <= 0:
        return out

    out[valid] = y[valid] / area

    return out


lambda_edges = np.logspace(
    np.log10(LAMBDA_MIN_UM),
    np.log10(LAMBDA_MAX_UM),
    N_PSD_WAVELENGTH_BINS + 1,
)


# =============================================================================
# Corrected overlap-normalized radial ACF
# =============================================================================


def overlap_normalized_acf_2d(z: np.ndarray, spacing_um_value: float):
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)

    if np.count_nonzero(finite) < 10:
        raise ValueError("Too few finite pixels for ACF.")

    z0 = np.zeros_like(z, dtype=float)
    z_mean = np.nanmean(z)
    z0[finite] = z[finite] - z_mean

    mask = finite.astype(float)

    numerator = fftconvolve(z0, z0[::-1, ::-1], mode="full")
    overlap_counts = fftconvolve(mask, mask[::-1, ::-1], mode="full")

    covariance = np.full_like(numerator, np.nan, dtype=float)

    valid_overlap = overlap_counts > 0
    covariance[valid_overlap] = numerator[valid_overlap] / overlap_counts[valid_overlap]

    center = tuple(np.array(covariance.shape) // 2)
    zero_lag_cov = covariance[center]

    if not np.isfinite(zero_lag_cov) or zero_lag_cov <= 0:
        rho = np.full_like(covariance, np.nan, dtype=float)
    else:
        rho = covariance / zero_lag_cov

    n0, n1 = z.shape

    lag0_pix = np.arange(-(n0 - 1), n0)
    lag1_pix = np.arange(-(n1 - 1), n1)

    lag0_um = lag0_pix * spacing_um_value
    lag1_um = lag1_pix * spacing_um_value

    return lag0_um, lag1_um, rho


def radial_average_acf(
    lag0_um: np.ndarray,
    lag1_um: np.ndarray,
    rho: np.ndarray,
    r_edges_um: np.ndarray,
):
    L1, L0 = np.meshgrid(lag1_um, lag0_um)
    R = np.sqrt(L0**2 + L1**2)

    centers = 0.5 * (r_edges_um[:-1] + r_edges_um[1:])
    radial = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)

    valid = np.isfinite(R) & np.isfinite(rho)

    for i in range(len(centers)):
        mask = valid & (R >= r_edges_um[i]) & (R < r_edges_um[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            radial[i] = float(np.nanmean(rho[mask]))

    return centers, radial, counts


def radial_acf(height_um: np.ndarray, spacing_um_value: float, r_edges_um: np.ndarray):
    lag0_um, lag1_um, rho = overlap_normalized_acf_2d(height_um, spacing_um_value)

    return radial_average_acf(
        lag0_um,
        lag1_um,
        rho,
        r_edges_um,
    )


r_edges = np.arange(
    0.0,
    ACF_MAX_LAG_UM + ACF_RADIAL_BIN_WIDTH_UM,
    ACF_RADIAL_BIN_WIDTH_UM,
)


# =============================================================================
# Experimental endpoint curves
# =============================================================================

# exp_curve_rows = []

# for case in all_cases:
#     load = case["load"]
#     sample_type = case["sample_type"]

#     case_df = exp_df[
#         (exp_df["load_mpa"] == load)
#         & (exp_df["sample_type"].astype(str) == sample_type)
#     ].copy()

#     if case_df.empty:
#         warnings.warn(f"No experimental records found for {load} MPa {sample_type}")
#         continue

#     for sample, g in case_df.groupby("sample"):
#         g = g.sort_values("time_h").copy()

#         if g.empty:
#             continue

#         final = g.iloc[-1]

#         try:
#             H = load_exp_height(final["height_path"])

#             lam, psd, modes = radial_psd_wavelength_binned(
#                 H,
#                 EXP_SPACING_UM,
#                 lambda_edges,
#                 min_modes=MIN_MODES_PER_PSD_BIN,
#             )
#             psd_norm = normalize_psd_shape(lam, psd)

#             r, acf, counts = radial_acf(H, EXP_SPACING_UM, r_edges)

#             for i in range(len(lam)):
#                 exp_curve_rows.append(
#                     {
#                         "source": "exp",
#                         "load_mpa": load,
#                         "sample_type": sample_type,
#                         "sample_id": sample,
#                         "replicate_id": str(sample),
#                         "bulk_z_strain_percent": float(final["bulk_z_strain_percent"]),
#                         "curve_type": "psd",
#                         "x_um": lam[i],
#                         "y": psd_norm[i],
#                         "raw_y": psd[i],
#                         "modes_or_counts": modes[i],
#                     }
#                 )

#             for i in range(len(r)):
#                 exp_curve_rows.append(
#                     {
#                         "source": "exp",
#                         "load_mpa": load,
#                         "sample_type": sample_type,
#                         "sample_id": sample,
#                         "replicate_id": str(sample),
#                         "bulk_z_strain_percent": float(final["bulk_z_strain_percent"]),
#                         "curve_type": "acf",
#                         "x_um": r[i],
#                         "y": acf[i],
#                         "raw_y": acf[i],
#                         "modes_or_counts": counts[i],
#                     }
#                 )

#         except Exception as exc:
#             warnings.warn(
#                 f"Experimental endpoint failed for {load} {sample_type} sample {sample}: {exc}"
#             )

# exp_curves_df = pd.DataFrame(exp_curve_rows)


# =============================================================================
# Simulation endpoint curves
# =============================================================================

sim_curve_rows = []

for micro_run in micro_runs:
    micro_id = micro_run["micro_id"]
    sim_root = micro_run["sim_root"]
    microstructure = micro_run["microstructure"]

    for case in all_cases:
        load = case["load"]
        sample_type = case["sample_type"]

        run_dir = sim_root / f"{load}mpa_{sample_type}"

        print(f"Loading simulation endpoint: {micro_id}, {load} MPa {sample_type}")

        try:
            sim_i = SimResults.load(run_dir, microstructure=microstructure)
        except Exception as exc:
            warnings.warn(f"Failed loading simulation {run_dir}: {exc}")
            continue

        H_all = np.asarray(sim_i.height, dtype=float)
        n_faces, n_times, nz, n_width = H_all.shape

        vtk_time = np.asarray(sim_i.vtk_time, dtype=float)

        strain_time = np.asarray(sim_i.sim_time, dtype=float)
        bulk_strain_all = np.asarray(getattr(sim_i, strain_col), dtype=float)

        valid = np.isfinite(strain_time) & np.isfinite(bulk_strain_all)

        strain_time_valid = strain_time[valid]
        bulk_strain_valid = bulk_strain_all[valid]

        order = np.argsort(strain_time_valid)
        strain_time_valid = strain_time_valid[order]
        bulk_strain_valid = bulk_strain_valid[order]

        bulk_strain_vtk = np.interp(
            vtk_time,
            strain_time_valid,
            bulk_strain_valid,
        )

        n = min(n_times, len(vtk_time), len(bulk_strain_vtk))

        H_all = H_all[:, :n, :, :]
        bulk_strain_vtk = bulk_strain_vtk[:n]

        final_idx = n - 1
        final_strain_percent = 100.0 * float(bulk_strain_vtk[final_idx])

        for face_idx in range(n_faces):
            replicate_id = f"{micro_id}_face{face_idx}"

            try:
                H = preprocess_sim_height(H_all[face_idx, final_idx])

                lam, psd, modes = radial_psd_wavelength_binned(
                    H,
                    SIM_SPACING_UM,
                    lambda_edges,
                    min_modes=MIN_MODES_PER_PSD_BIN,
                )
                psd_norm = normalize_psd_shape(lam, psd)

                r, acf, counts = radial_acf(H, SIM_SPACING_UM, r_edges)

                for i in range(len(lam)):
                    sim_curve_rows.append(
                        {
                            "source": "sim",
                            "load_mpa": load,
                            "sample_type": sample_type,
                            "sample_id": micro_id,
                            "replicate_id": replicate_id,
                            "bulk_z_strain_percent": final_strain_percent,
                            "curve_type": "psd",
                            "x_um": lam[i],
                            "y": psd_norm[i],
                            "raw_y": psd[i],
                            "modes_or_counts": modes[i],
                        }
                    )

                for i in range(len(r)):
                    sim_curve_rows.append(
                        {
                            "source": "sim",
                            "load_mpa": load,
                            "sample_type": sample_type,
                            "sample_id": micro_id,
                            "replicate_id": replicate_id,
                            "bulk_z_strain_percent": final_strain_percent,
                            "curve_type": "acf",
                            "x_um": r[i],
                            "y": acf[i],
                            "raw_y": acf[i],
                            "modes_or_counts": counts[i],
                        }
                    )

            except Exception as exc:
                warnings.warn(
                    f"Simulation endpoint failed for {micro_id} {load} {sample_type} "
                    f"face {face_idx}: {exc}"
                )

sim_curves_df = pd.DataFrame(sim_curve_rows)


# =============================================================================
# Combine and aggregate
# =============================================================================

curves_df = pd.concat([exp_curves_df, sim_curves_df], ignore_index=True)

if curves_df.empty:
    raise RuntimeError("No endpoint curves were computed.")

curves_df.to_csv(OUTPUT_DIR / "endpoint_exp_sim_psd_acf_curves_long.csv", index=False)

summary_rows = []

for (source, load, sample_type, curve_type, x_um), g in curves_df.groupby(
    ["source", "load_mpa", "sample_type", "curve_type", "x_um"]
):
    y = pd.to_numeric(g["y"], errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        continue

    summary_rows.append(
        {
            "source": source,
            "load_mpa": load,
            "sample_type": sample_type,
            "curve_type": curve_type,
            "x_um": x_um,
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
            "sem": float(np.std(y, ddof=1) / np.sqrt(y.size)) if y.size > 1 else 0.0,
            "n": int(y.size),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    OUTPUT_DIR / "endpoint_exp_sim_psd_acf_curves_summary.csv", index=False
)

print()
print(f"Saved: {OUTPUT_DIR / 'endpoint_exp_sim_psd_acf_curves_long.csv'}")
print(f"Saved: {OUTPUT_DIR / 'endpoint_exp_sim_psd_acf_curves_summary.csv'}")


# =============================================================================
# Plot four-panel figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=False)

ax_psd_int = axes[0, 0]
ax_psd_unint = axes[0, 1]
ax_acf_int = axes[1, 0]
ax_acf_unint = axes[1, 1]


def plot_group(ax, cases, curve_type, title, ylabel):
    for case in cases:
        load = case["load"]
        sample_type = case["sample_type"]
        color = LOAD_COLORS.get(load, "0.3")

        for source, linestyle, alpha, label_suffix in [
            ("exp", "-", 1.0, "Exp."),
            ("sim", "--", 0.95, "Sim."),
        ]:
            s = summary_df[
                (summary_df["source"] == source)
                & (summary_df["load_mpa"] == load)
                & (summary_df["sample_type"] == sample_type)
                & (summary_df["curve_type"] == curve_type)
            ].copy()

            if s.empty:
                continue

            s = s.sort_values("x_um")

            ax.plot(
                s["x_um"],
                s["mean"],
                color=color,
                linestyle=linestyle,
                lw=2.0 if source == "exp" else 1.8,
                alpha=alpha,
                label=f"{load} MPa {label_suffix}",
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


plot_group(
    ax_psd_int,
    interrupted_cases,
    "psd",
    "A. Interrupted: normalized radial PSD",
    r"normalized radial PSD",
)

plot_group(
    ax_psd_unint,
    uninterrupted_cases,
    "psd",
    "B. Uninterrupted: normalized radial PSD",
    r"normalized radial PSD",
)

plot_group(
    ax_acf_int,
    interrupted_cases,
    "acf",
    "C. Interrupted: radial ACF",
    r"radial ACF, $C(r)$",
)

plot_group(
    ax_acf_unint,
    uninterrupted_cases,
    "acf",
    "D. Uninterrupted: radial ACF",
    r"radial ACF, $C(r)$",
)

for ax in [ax_psd_int, ax_psd_unint]:
    ax.set_xscale("log")
    ax.set_xlim(LAMBDA_MIN_UM, LAMBDA_MAX_UM)
    ax.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")

    if PSD_YLOG:
        ax.set_yscale("log")

    ax.legend(fontsize=6, ncol=2)

for ax in [ax_acf_int, ax_acf_unint]:
    ax.set_xlim(0.0, ACF_MAX_LAG_UM)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axhline(np.exp(-1.0), color="0.5", lw=0.8, ls=":")
    ax.legend(fontsize=6, ncol=2)

# Numeric x labels for PSD log axes.
psd_ticks = np.array([6, 10, 20, 50, 100, 128], dtype=float)
psd_ticks = psd_ticks[(psd_ticks >= LAMBDA_MIN_UM) & (psd_ticks <= LAMBDA_MAX_UM)]

for ax in [ax_psd_int, ax_psd_unint]:
    ax.xaxis.set_major_locator(FixedLocator(psd_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{int(t)}" for t in psd_ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())

fig.suptitle(
    "Endpoint spatial configuration: experiment vs simulation\n"
    "PSD normalized by spectral area; ACF normalized to unity at zero lag",
    y=1.02,
)

fig.tight_layout()

outpath = OUTPUT_DIR / "four_panel_endpoint_exp_vs_sim_normalized_psd_acf.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Four-panel endpoint exp-vs-sim comparison, corrected operator + caching
# =============================================================================
#
# Figure:
#   Top-left:     interrupted endpoint normalized radial PSD: exp vs sim
#   Bottom-left:  interrupted endpoint radial ACF: exp vs sim
#   Top-right:    uninterrupted endpoint normalized radial PSD: exp vs sim
#   Bottom-right: uninterrupted endpoint radial ACF: exp vs sim
#
# Operator:
#   EXP:
#     raw -> fill missing -> crop -> plane-level cropped field -> mean-remove
#
#   SIM:
#     final height field -> plane-level full field -> mean-remove
#
# PSD:
#   C(fx,fy) = dx^2 * |FFT(h*w)|^2 / sum(w^2)
#   radial PSD = arithmetic mean in physical-frequency annuli
#   plotted vs wavelength center lambda = 1/sqrt(f_left*f_right)
#   normalized by trapezoidal integral over frequency
#
# ACF:
#   linear overlap-corrected covariance by FFT convolution
#   divided pointwise by overlap count
#   normalized by zero-lag covariance
#   radially averaged to 128 um
#
# IMPORTANT:
#   Set RECOMPUTE_ENDPOINT_CURVES = True only when changing processing settings
#   or source data. Otherwise this reads cached CSVs and does not reload sims.
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from utils.data_utils import SimResults

# =============================================================================
# User settings
# =============================================================================

RECOMPUTE_ENDPOINT_CURVES = False

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURVES_CACHE = (
    OUTPUT_DIR / "endpoint_exp_sim_fullfield_cropfirst_psd_acf_curves_long.csv"
)
SUMMARY_CACHE = (
    OUTPUT_DIR / "endpoint_exp_sim_fullfield_cropfirst_psd_acf_curves_summary.csv"
)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

# Experimental profilometry settings
EXP_SPACING_UM = 1.379951
EXP_CROP = (slice(50, -50), slice(50, 750))
EXP_DETREND_ORDER = 1

# Simulation settings
SIM_SPACING_UM = 1.0
SIM_DETREND_ORDER = 1
strain_col = "epav33"

micro_runs = [
    {
        "micro_id": "micro1",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro1_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro1_production.dat"
        ),
    },
    {
        "micro_id": "micro2",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro2_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro2_production.dat"
        ),
    },
    {
        "micro_id": "micro3",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro3_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro3_production.dat"
        ),
    },
]

interrupted_cases = [
    {"load": 475, "sample_type": "int"},
    {"load": 525, "sample_type": "int"},
    {"load": 575, "sample_type": "int"},
]

uninterrupted_cases = [
    {"load": 500, "sample_type": "unint"},
    {"load": 530, "sample_type": "unint"},
    {"load": 588, "sample_type": "unint"},
]

all_cases = interrupted_cases + uninterrupted_cases

LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

# PSD comparison range.
# Lower bound = twice experimental Nyquist shortest wavelength = 4*dx_exp.
LAMBDA_MIN_UM = 4.0 * EXP_SPACING_UM
LAMBDA_MAX_UM = 128.0

# Frequency-annulus binning.
# Use log-frequency annuli because the earlier bandwidth analysis reported
# logarithmic annuli. Increase/decrease as needed.
N_PSD_ANNULI = 22
MIN_MODES_PER_PSD_BIN = 1

# ACF settings.
ACF_MAX_LAG_UM = 128.0

# Per your latest instruction: radial ACF bin size based on Nyquist shortest
# wavelength. If you actually want native-pixel annuli, set this to EXP_SPACING_UM.
ACF_RADIAL_BIN_WIDTH_UM = 2.0 * EXP_SPACING_UM

PSD_YLOG = False

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Numerical helpers
# =============================================================================


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x)
    if hasattr(np, "trapz"):
        return np.trapz(y, x=x)

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, col = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = col.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend([x * x, x * y, y * y])
    elif order != 1:
        raise ValueError("detrend order must be 1 or 2")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    design, inverse = _detrend_geometry(
        values.shape, float(spacing_um_value), int(order)
    )
    coeff = inverse @ values.ravel()
    trend = (design @ coeff).reshape(values.shape)
    return values - trend


def fill_missing_nearest(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    missing = ~np.isfinite(H)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        H = H[tuple(nearest)]

    return H


# =============================================================================
# Experimental height loading: crop first, then plane-level cropped field
# =============================================================================


def raw_exp_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def load_exp_height(path: str | Path) -> np.ndarray:
    H = raw_exp_height(path)
    H = fill_missing_nearest(H)
    H = H[EXP_CROP]
    H = detrend_surface(H, EXP_SPACING_UM, order=EXP_DETREND_ORDER)
    H = H - np.nanmean(H)
    return H


def preprocess_sim_height(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)

    if H.ndim != 2:
        raise ValueError(f"Simulation height must be 2D; got shape {H.shape}")

    if not np.all(np.isfinite(H)):
        H = H.copy()
        H[~np.isfinite(H)] = np.nanmean(H)

    H = detrend_surface(H, SIM_SPACING_UM, order=SIM_DETREND_ORDER)
    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD: protocol-style 2D PSD and frequency-annulus averaging
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    return w0[:, None] * w1[None, :]


def psd2d_height_protocol(height_um: np.ndarray, spacing_um_value: float):
    """
    Protocol PSD:

        C(fx,fy) = dx^2 * |FFT(h*w)|^2 / sum(w^2)

    Units: um^4.
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)
    PSD = (spacing_um_value**2) * np.abs(F) ** 2 / np.sum(W**2)

    n0, n1 = H.shape
    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    return f0, f1, PSD


def make_frequency_edges():
    """
    Frequency edges corresponding to wavelength band:
        LAMBDA_MIN_UM <= lambda <= LAMBDA_MAX_UM

    Since lambda = 1/f:
        f_min = 1/lambda_max
        f_max = 1/lambda_min
    """
    f_min = 1.0 / LAMBDA_MAX_UM
    f_max = 1.0 / LAMBDA_MIN_UM

    return np.logspace(np.log10(f_min), np.log10(f_max), N_PSD_ANNULI + 1)


f_edges = make_frequency_edges()


def radial_psd_frequency_binned(
    height_um: np.ndarray,
    spacing_um_value: float,
    f_edges_um_inv: np.ndarray,
    min_modes: int = 1,
):
    f0, f1, PSD = psd2d_height_protocol(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    nbins = len(f_edges_um_inv) - 1
    f_center = np.sqrt(f_edges_um_inv[:-1] * f_edges_um_inv[1:])
    lambda_center = 1.0 / f_center

    radial = np.full(nbins, np.nan, dtype=float)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        lo = f_edges_um_inv[i]
        hi = f_edges_um_inv[i + 1]

        mask = np.isfinite(FR) & np.isfinite(PSD) & (FR >= lo) & (FR < hi)

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            radial[i] = float(np.nanmean(PSD[mask]))

    return f_center, lambda_center, radial, modes


def normalize_psd_over_frequency(f_center, psd):
    """
    Normalize radial PSD by trapezoidal integral over frequency.
    """
    f = np.asarray(f_center, dtype=float)
    y = np.asarray(psd, dtype=float)

    out = np.full_like(y, np.nan, dtype=float)

    valid = np.isfinite(f) & np.isfinite(y) & (f > 0) & (y > 0)

    if np.count_nonzero(valid) < 2:
        return out

    order = np.argsort(f[valid])
    f_valid = f[valid][order]
    y_valid = y[valid][order]

    area = integrate_trapezoid(y_valid, x=f_valid)

    if not np.isfinite(area) or area <= 0:
        return out

    out[valid] = y[valid] / area

    return out


# =============================================================================
# ACF: overlap-corrected linear autocorrelation
# =============================================================================


def overlap_normalized_acf_2d(z: np.ndarray, spacing_um_value: float):
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)

    if np.count_nonzero(finite) < 10:
        raise ValueError("Too few finite pixels for ACF")

    z0 = np.zeros_like(z, dtype=float)
    z_mean = np.nanmean(z)
    z0[finite] = z[finite] - z_mean

    mask = finite.astype(float)

    numerator = fftconvolve(z0, z0[::-1, ::-1], mode="full")
    overlap_counts = fftconvolve(mask, mask[::-1, ::-1], mode="full")

    covariance = np.full_like(numerator, np.nan, dtype=float)

    valid_overlap = overlap_counts > 0
    covariance[valid_overlap] = numerator[valid_overlap] / overlap_counts[valid_overlap]

    center = tuple(np.array(covariance.shape) // 2)
    zero_lag_cov = covariance[center]

    if not np.isfinite(zero_lag_cov) or zero_lag_cov <= 0:
        rho = np.full_like(covariance, np.nan, dtype=float)
    else:
        rho = covariance / zero_lag_cov

    n0, n1 = z.shape
    lag0_um = np.arange(-(n0 - 1), n0) * spacing_um_value
    lag1_um = np.arange(-(n1 - 1), n1) * spacing_um_value

    return lag0_um, lag1_um, rho


r_edges = np.arange(
    0.0,
    ACF_MAX_LAG_UM + ACF_RADIAL_BIN_WIDTH_UM,
    ACF_RADIAL_BIN_WIDTH_UM,
)


def radial_average_acf(lag0_um, lag1_um, rho, r_edges_um):
    L1, L0 = np.meshgrid(lag1_um, lag0_um)
    R = np.sqrt(L0**2 + L1**2)

    centers = 0.5 * (r_edges_um[:-1] + r_edges_um[1:])
    radial = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)

    valid = np.isfinite(R) & np.isfinite(rho)

    for i in range(len(centers)):
        mask = valid & (R >= r_edges_um[i]) & (R < r_edges_um[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            radial[i] = float(np.nanmean(rho[mask]))

    return centers, radial, counts


def radial_acf(height_um, spacing_um_value, r_edges_um):
    lag0_um, lag1_um, rho = overlap_normalized_acf_2d(height_um, spacing_um_value)
    return radial_average_acf(lag0_um, lag1_um, rho, r_edges_um)


# =============================================================================
# Compute and cache endpoint curves
# =============================================================================

if (
    RECOMPUTE_ENDPOINT_CURVES
    or (not CURVES_CACHE.exists())
    or (not SUMMARY_CACHE.exists())
):
    print("Recomputing endpoint curves. This reloads simulation data.")

    if "point_df" not in globals():
        if not POINT_TABLE_PATH.exists():
            raise FileNotFoundError(
                f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
            )
        point_df = pd.read_csv(POINT_TABLE_PATH)

    exp_df = point_df.copy()
    exp_df["load_mpa"] = pd.to_numeric(exp_df["load_mpa"], errors="coerce")
    exp_df["time_h"] = pd.to_numeric(exp_df["time_h"], errors="coerce")
    exp_df["bulk_z_strain_percent"] = pd.to_numeric(
        exp_df["bulk_z_strain_percent"],
        errors="coerce",
    )

    exp_df = exp_df[
        np.isfinite(exp_df["load_mpa"])
        & np.isfinite(exp_df["time_h"])
        & np.isfinite(exp_df["bulk_z_strain_percent"])
    ].copy()

    exp_df = exp_df.sort_values(
        ["sample_type", "load_mpa", "sample", "time_h"]
    ).reset_index(drop=True)

    exp_curve_rows = []

    for case in all_cases:
        load = case["load"]
        sample_type = case["sample_type"]

        case_df = exp_df[
            (exp_df["load_mpa"] == load)
            & (exp_df["sample_type"].astype(str) == sample_type)
        ].copy()

        for sample, g in case_df.groupby("sample"):
            final = g.sort_values("time_h").iloc[-1]

            try:
                H = load_exp_height(final["height_path"])

                f_center, lam, psd, modes = radial_psd_frequency_binned(
                    H,
                    EXP_SPACING_UM,
                    f_edges,
                    min_modes=MIN_MODES_PER_PSD_BIN,
                )
                psd_norm = normalize_psd_over_frequency(f_center, psd)

                r, acf, counts = radial_acf(H, EXP_SPACING_UM, r_edges)

                for i in range(len(lam)):
                    exp_curve_rows.append(
                        {
                            "source": "exp",
                            "load_mpa": load,
                            "sample_type": sample_type,
                            "sample_id": sample,
                            "replicate_id": str(sample),
                            "bulk_z_strain_percent": float(
                                final["bulk_z_strain_percent"]
                            ),
                            "curve_type": "psd",
                            "x_um": lam[i],
                            "frequency_um_inv": f_center[i],
                            "y": psd_norm[i],
                            "raw_y": psd[i],
                            "modes_or_counts": modes[i],
                        }
                    )

                for i in range(len(r)):
                    exp_curve_rows.append(
                        {
                            "source": "exp",
                            "load_mpa": load,
                            "sample_type": sample_type,
                            "sample_id": sample,
                            "replicate_id": str(sample),
                            "bulk_z_strain_percent": float(
                                final["bulk_z_strain_percent"]
                            ),
                            "curve_type": "acf",
                            "x_um": r[i],
                            "frequency_um_inv": np.nan,
                            "y": acf[i],
                            "raw_y": acf[i],
                            "modes_or_counts": counts[i],
                        }
                    )

            except Exception as exc:
                warnings.warn(
                    f"Experimental endpoint failed for {load} {sample_type} sample {sample}: {exc}"
                )

    exp_curves_df = pd.DataFrame(exp_curve_rows)

    sim_curve_rows = []

    for micro_run in micro_runs:
        micro_id = micro_run["micro_id"]
        sim_root = micro_run["sim_root"]
        microstructure = micro_run["microstructure"]

        for case in all_cases:
            load = case["load"]
            sample_type = case["sample_type"]

            run_dir = sim_root / f"{load}mpa_{sample_type}"

            print(f"Loading simulation endpoint: {micro_id}, {load} MPa {sample_type}")

            try:
                sim_i = SimResults.load(run_dir, microstructure=microstructure)
            except Exception as exc:
                warnings.warn(f"Failed loading simulation {run_dir}: {exc}")
                continue

            H_all = np.asarray(sim_i.height, dtype=float)
            n_faces, n_times, nz, n_width = H_all.shape

            vtk_time = np.asarray(sim_i.vtk_time, dtype=float)

            strain_time = np.asarray(sim_i.sim_time, dtype=float)
            bulk_strain_all = np.asarray(getattr(sim_i, strain_col), dtype=float)

            valid = np.isfinite(strain_time) & np.isfinite(bulk_strain_all)

            strain_time_valid = strain_time[valid]
            bulk_strain_valid = bulk_strain_all[valid]

            order = np.argsort(strain_time_valid)
            strain_time_valid = strain_time_valid[order]
            bulk_strain_valid = bulk_strain_valid[order]

            bulk_strain_vtk = np.interp(
                vtk_time,
                strain_time_valid,
                bulk_strain_valid,
            )

            n = min(n_times, len(vtk_time), len(bulk_strain_vtk))

            H_all = H_all[:, :n, :, :]
            bulk_strain_vtk = bulk_strain_vtk[:n]

            final_idx = n - 1
            final_strain_percent = 100.0 * float(bulk_strain_vtk[final_idx])

            for face_idx in range(n_faces):
                replicate_id = f"{micro_id}_face{face_idx}"

                try:
                    H = preprocess_sim_height(H_all[face_idx, final_idx])

                    f_center, lam, psd, modes = radial_psd_frequency_binned(
                        H,
                        SIM_SPACING_UM,
                        f_edges,
                        min_modes=MIN_MODES_PER_PSD_BIN,
                    )
                    psd_norm = normalize_psd_over_frequency(f_center, psd)

                    r, acf, counts = radial_acf(H, SIM_SPACING_UM, r_edges)

                    for i in range(len(lam)):
                        sim_curve_rows.append(
                            {
                                "source": "sim",
                                "load_mpa": load,
                                "sample_type": sample_type,
                                "sample_id": micro_id,
                                "replicate_id": replicate_id,
                                "bulk_z_strain_percent": final_strain_percent,
                                "curve_type": "psd",
                                "x_um": lam[i],
                                "frequency_um_inv": f_center[i],
                                "y": psd_norm[i],
                                "raw_y": psd[i],
                                "modes_or_counts": modes[i],
                            }
                        )

                    for i in range(len(r)):
                        sim_curve_rows.append(
                            {
                                "source": "sim",
                                "load_mpa": load,
                                "sample_type": sample_type,
                                "sample_id": micro_id,
                                "replicate_id": replicate_id,
                                "bulk_z_strain_percent": final_strain_percent,
                                "curve_type": "acf",
                                "x_um": r[i],
                                "frequency_um_inv": np.nan,
                                "y": acf[i],
                                "raw_y": acf[i],
                                "modes_or_counts": counts[i],
                            }
                        )

                except Exception as exc:
                    warnings.warn(
                        f"Simulation endpoint failed for {micro_id} {load} {sample_type} "
                        f"face {face_idx}: {exc}"
                    )

    sim_curves_df = pd.DataFrame(sim_curve_rows)

    curves_df = pd.concat([exp_curves_df, sim_curves_df], ignore_index=True)

    if curves_df.empty:
        raise RuntimeError("No endpoint curves were computed")

    curves_df.to_csv(CURVES_CACHE, index=False)

    summary_rows = []

    for (source, load, sample_type, curve_type, x_um), g in curves_df.groupby(
        ["source", "load_mpa", "sample_type", "curve_type", "x_um"]
    ):
        y = pd.to_numeric(g["y"], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]

        if y.size == 0:
            continue

        summary_rows.append(
            {
                "source": source,
                "load_mpa": load,
                "sample_type": sample_type,
                "curve_type": curve_type,
                "x_um": x_um,
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
                "sem": (
                    float(np.std(y, ddof=1) / np.sqrt(y.size)) if y.size > 1 else 0.0
                ),
                "n": int(y.size),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CACHE, index=False)

    print(f"Saved: {CURVES_CACHE}")
    print(f"Saved: {SUMMARY_CACHE}")

else:
    print("Using cached endpoint curves. Simulation data will not be reloaded.")
    curves_df = pd.read_csv(CURVES_CACHE)
    summary_df = pd.read_csv(SUMMARY_CACHE)


# =============================================================================
# Plot four-panel figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=False)

ax_psd_int = axes[0, 0]
ax_psd_unint = axes[0, 1]
ax_acf_int = axes[1, 0]
ax_acf_unint = axes[1, 1]


def plot_group(ax, cases, curve_type, title, ylabel):
    for case in cases:
        load = case["load"]
        sample_type = case["sample_type"]
        color = LOAD_COLORS.get(load, "0.3")

        for source, linestyle, alpha, label_suffix in [
            ("exp", "-", 1.0, "Exp."),
            ("sim", "--", 0.95, "Sim."),
        ]:
            s = summary_df[
                (summary_df["source"] == source)
                & (summary_df["load_mpa"] == load)
                & (summary_df["sample_type"] == sample_type)
                & (summary_df["curve_type"] == curve_type)
            ].copy()

            if s.empty:
                continue

            s = s.sort_values("x_um")

            if curve_type == "psd":
                ax.semilogy(
                    s["x_um"],
                    s["mean"],
                    color=color,
                    linestyle=linestyle,
                    lw=2.0 if source == "exp" else 1.8,
                    alpha=alpha,
                    label=f"{load} MPa {label_suffix}",
                )
            else:
                ax.plot(
                    s["x_um"],
                    s["mean"],
                    color=color,
                    linestyle=linestyle,
                    lw=2.0 if source == "exp" else 1.8,
                    alpha=alpha,
                    label=f"{load} MPa {label_suffix}",
                )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


plot_group(
    ax_psd_int,
    interrupted_cases,
    "psd",
    "A. Interrupted: normalized radial PSD",
    r"normalized radial PSD",
)

plot_group(
    ax_psd_unint,
    uninterrupted_cases,
    "psd",
    "B. Uninterrupted: normalized radial PSD",
    r"normalized radial PSD",
)

plot_group(
    ax_acf_int,
    interrupted_cases,
    "acf",
    "C. Interrupted: radial ACF",
    r"radial ACF, $C(r)$",
)

plot_group(
    ax_acf_unint,
    uninterrupted_cases,
    "acf",
    "D. Uninterrupted: radial ACF",
    r"radial ACF, $C(r)$",
)

for ax in [ax_psd_int, ax_psd_unint]:
    ax.set_xscale("log")
    ax.set_xlim(LAMBDA_MIN_UM, LAMBDA_MAX_UM)
    ax.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")

    if PSD_YLOG:
        ax.set_yscale("log")

    ax.legend(fontsize=6, ncol=2)

for ax in [ax_acf_int, ax_acf_unint]:
    ax.set_xlim(0.0, ACF_MAX_LAG_UM)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axhline(np.exp(-1.0), color="0.5", lw=0.8, ls=":")
    ax.legend(fontsize=6, ncol=2)

# Numeric wavelength tick labels for log-x PSD axes.
psd_ticks = np.array([6, 10, 20, 50, 100, 128], dtype=float)
psd_ticks = psd_ticks[(psd_ticks >= LAMBDA_MIN_UM) & (psd_ticks <= LAMBDA_MAX_UM)]

for ax in [ax_psd_int, ax_psd_unint]:
    ax.xaxis.set_major_locator(FixedLocator(psd_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{int(t)}" for t in psd_ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())

fig.suptitle(
    "Endpoint spatial configuration: experiment vs simulation\n"
    "Crop-first leveling; frequency-annulus PSD; overlap-corrected ACF",
    y=1.02,
)

fig.tight_layout()

outpath = OUTPUT_DIR / "four_panel_endpoint_exp_vs_sim_normalized_psd_acf_corrected.png"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Four-panel endpoint experiment-vs-simulation comparison
# Aligned with writeup calculation details EXCEPT no sectioning.
#
# Key choices:
#   - Experimental maps: raw -> fill missing -> crop -> plane-level cropped field -> mean-remove
#   - Simulation maps: final surface -> plane-level -> mean-remove
#   - PSD: protocol PSD formula, frequency annuli, radial mean in annuli
#   - PSD aggregation: median across specimens / sim faces
#   - PSD plotted as normalized radial PSD shape
#   - ACF: linear overlap-corrected covariance, not circular FFT ACF
#   - ACF radial averaging: annuli of one native 10x pixel spacing
#   - ACF aggregation: median across specimens / sim faces
#   - No sectioning, no patches
#   - Simulation data are cached; set RECOMPUTE_ENDPOINT_CURVES=True only when needed
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

# =============================================================================
# Settings
# =============================================================================

RECOMPUTE_ENDPOINT_CURVES = False  # Set True only when changing source data/settings.

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURVES_CACHE = (
    OUTPUT_DIR / "endpoint_exp_sim_fullfield_cropfirst_writeup_psd_acf_curves_long.csv"
)
SUMMARY_CACHE = (
    OUTPUT_DIR
    / "endpoint_exp_sim_fullfield_cropfirst_writeup_psd_acf_curves_summary.csv"
)

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

# Experimental profilometry settings
EXP_SPACING_UM = 1.379951
EXP_CROP = (slice(50, -50), slice(50, 750))
EXP_DETREND_ORDER = 1

# Simulation settings
SIM_SPACING_UM = 1.0
SIM_DETREND_ORDER = 1
strain_col = "epav33"

micro_runs = [
    {
        "micro_id": "micro1",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro1_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro1_production.dat"
        ),
    },
    {
        "micro_id": "micro2",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro2_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro2_production.dat"
        ),
    },
    {
        "micro_id": "micro3",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro3_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro3_production.dat"
        ),
    },
]

interrupted_cases = [
    {"load": 475, "sample_type": "int"},
    {"load": 525, "sample_type": "int"},
    {"load": 575, "sample_type": "int"},
]

uninterrupted_cases = [
    {"load": 500, "sample_type": "unint"},
    {"load": 530, "sample_type": "unint"},
    {"load": 588, "sample_type": "unint"},
]

all_cases = interrupted_cases + uninterrupted_cases

LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

# -----------------------------------------------------------------------------
# PSD binning
# -----------------------------------------------------------------------------
# Writeup-style physical frequency annuli:
#   10x spectrum: 22 logarithmic annuli from 0.00283072 to 0.36 um^-1.
#
# For the endpoint sim/exp comparison, the display is restricted to wavelengths
# <= 128 um, but the binning is still frequency-annulus based.
# -----------------------------------------------------------------------------

PSD_F_MIN_UM_INV = 0.00283072
PSD_F_MAX_UM_INV = 0.36
N_PSD_ANNULI = 22
MIN_MODES_PER_PSD_BIN = 1

PLOT_LAMBDA_MIN_UM = 1.0 / PSD_F_MAX_UM_INV
PLOT_LAMBDA_MAX_UM = 128.0

# Use log y for PSD. This prevents small short-wavelength values from appearing
# like zero when long-wavelength power dominates.
PSD_YLOG = True

# -----------------------------------------------------------------------------
# ACF binning
# -----------------------------------------------------------------------------
# Writeup ACF radial bin width is one native 10x pixel.
# If you insist on twice native-pixel binning, change this to 2.0*EXP_SPACING_UM.
# -----------------------------------------------------------------------------

ACF_RADIAL_BIN_WIDTH_UM = EXP_SPACING_UM
ACF_MAX_LAG_UM = 128.0

# Plot individual faint replicate curves? Median curves are always plotted.
PLOT_INDIVIDUAL_CURVES = False
INDIVIDUAL_ALPHA = 0.08

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Utility functions
# =============================================================================


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x)
    if hasattr(np, "trapz"):
        return np.trapz(y, x=x)

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, col = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = col.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend([x * x, x * y, y * y])
    elif order != 1:
        raise ValueError("detrend order must be 1 or 2")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    design, inverse = _detrend_geometry(
        values.shape, float(spacing_um_value), int(order)
    )
    coeff = inverse @ values.ravel()
    trend = (design @ coeff).reshape(values.shape)

    return values - trend


def fill_missing_nearest(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    missing = ~np.isfinite(H)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        H = H[tuple(nearest)]

    return H


# =============================================================================
# Experimental and simulation height processing
# =============================================================================


def raw_exp_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def load_exp_height(path: str | Path) -> np.ndarray:
    """
    Experiment endpoint field:
      raw -> fill missing -> crop -> plane-level cropped field -> mean-remove
    """
    H = raw_exp_height(path)
    H = fill_missing_nearest(H)
    H = H[EXP_CROP]
    H = detrend_surface(H, EXP_SPACING_UM, order=EXP_DETREND_ORDER)
    H = H - np.nanmean(H)
    return H


def preprocess_sim_height(H: np.ndarray) -> np.ndarray:
    """
    Simulation endpoint field:
      final height -> plane-level full field -> mean-remove
    """
    H = np.asarray(H, dtype=float)

    if H.ndim != 2:
        raise ValueError(f"Simulation height must be 2D; got shape {H.shape}")

    if not np.all(np.isfinite(H)):
        H = H.copy()
        H[~np.isfinite(H)] = np.nanmean(H)

    H = detrend_surface(H, SIM_SPACING_UM, order=SIM_DETREND_ORDER)
    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD: writeup-style formula and frequency-annulus radial averaging
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    return w0[:, None] * w1[None, :]


def psd2d_height_protocol(height_um: np.ndarray, spacing_um_value: float):
    """
    Protocol PSD:

        C(fx,fy) = dx^2 * |FFT(h*w)|^2 / sum(w^2)

    where h is mean-removed and w is separable Hann.
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)
    PSD = (spacing_um_value**2) * np.abs(F) ** 2 / np.sum(W**2)

    n0, n1 = H.shape
    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    return f0, f1, PSD


def make_frequency_edges():
    return np.logspace(
        np.log10(PSD_F_MIN_UM_INV),
        np.log10(PSD_F_MAX_UM_INV),
        N_PSD_ANNULI + 1,
    )


f_edges = make_frequency_edges()


def radial_psd_frequency_binned(
    height_um: np.ndarray,
    spacing_um_value: float,
    f_edges_um_inv: np.ndarray,
    min_modes: int = 1,
):
    f0, f1, PSD = psd2d_height_protocol(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    nbins = len(f_edges_um_inv) - 1

    f_center = np.sqrt(f_edges_um_inv[:-1] * f_edges_um_inv[1:])
    lambda_center = 1.0 / f_center

    radial = np.full(nbins, np.nan, dtype=float)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        lo = f_edges_um_inv[i]
        hi = f_edges_um_inv[i + 1]

        mask = np.isfinite(FR) & np.isfinite(PSD) & (FR >= lo) & (FR < hi)

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            radial[i] = float(np.nanmean(PSD[mask]))

    return f_center, lambda_center, radial, modes


def normalize_psd_over_frequency(f_center, psd):
    """
    Morphology-style normalized PSD:
      radial PSD divided by trapezoidal integral over frequency.
    """
    f = np.asarray(f_center, dtype=float)
    y = np.asarray(psd, dtype=float)

    out = np.full_like(y, np.nan, dtype=float)

    valid = np.isfinite(f) & np.isfinite(y) & (f > 0) & (y > 0)

    if np.count_nonzero(valid) < 2:
        return out

    order = np.argsort(f[valid])
    f_valid = f[valid][order]
    y_valid = y[valid][order]

    area = integrate_trapezoid(y_valid, x=f_valid)

    if not np.isfinite(area) or area <= 0:
        return out

    out[valid] = y[valid] / area

    return out


# =============================================================================
# ACF: linear overlap-corrected covariance and radial averaging
# =============================================================================


def overlap_normalized_acf_2d(z: np.ndarray, spacing_um_value: float):
    """
    Linear overlap-corrected normalized ACF:

        rho(dx,dy) =
        [sum z(x,y) z(x+dx,y+dy) / N(dx,dy)] / [sum z^2 / N(0,0)]

    This is not the circular FFT ACF.
    """
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)

    if np.count_nonzero(finite) < 10:
        raise ValueError("Too few finite pixels for ACF")

    z0 = np.zeros_like(z, dtype=float)
    z_mean = np.nanmean(z)
    z0[finite] = z[finite] - z_mean

    mask = finite.astype(float)

    numerator = fftconvolve(z0, z0[::-1, ::-1], mode="full")
    overlap_counts = fftconvolve(mask, mask[::-1, ::-1], mode="full")

    covariance = np.full_like(numerator, np.nan, dtype=float)

    valid_overlap = overlap_counts > 0
    covariance[valid_overlap] = numerator[valid_overlap] / overlap_counts[valid_overlap]

    center = tuple(np.array(covariance.shape) // 2)
    zero_lag_cov = covariance[center]

    if not np.isfinite(zero_lag_cov) or zero_lag_cov <= 0:
        rho = np.full_like(covariance, np.nan, dtype=float)
    else:
        rho = covariance / zero_lag_cov

    n0, n1 = z.shape

    lag0_um = np.arange(-(n0 - 1), n0) * spacing_um_value
    lag1_um = np.arange(-(n1 - 1), n1) * spacing_um_value

    return lag0_um, lag1_um, rho


r_edges = np.arange(
    0.0,
    ACF_MAX_LAG_UM + ACF_RADIAL_BIN_WIDTH_UM,
    ACF_RADIAL_BIN_WIDTH_UM,
)


def radial_average_acf(lag0_um, lag1_um, rho, r_edges_um):
    L1, L0 = np.meshgrid(lag1_um, lag0_um)
    R = np.sqrt(L0**2 + L1**2)

    centers = 0.5 * (r_edges_um[:-1] + r_edges_um[1:])
    radial = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)

    valid = np.isfinite(R) & np.isfinite(rho)

    for i in range(len(centers)):
        mask = valid & (R >= r_edges_um[i]) & (R < r_edges_um[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            radial[i] = float(np.nanmean(rho[mask]))

    keep = centers < ACF_MAX_LAG_UM

    return centers[keep], radial[keep], counts[keep]


def radial_acf(height_um, spacing_um_value, r_edges_um):
    lag0_um, lag1_um, rho = overlap_normalized_acf_2d(height_um, spacing_um_value)
    return radial_average_acf(lag0_um, lag1_um, rho, r_edges_um)


# =============================================================================
# Compute endpoint curves only if needed
# =============================================================================

if (
    RECOMPUTE_ENDPOINT_CURVES
    or (not CURVES_CACHE.exists())
    or (not SUMMARY_CACHE.exists())
):
    print("Recomputing endpoint curves. Simulation data will be reloaded.")

    from utils.data_utils import SimResults

    if "point_df" not in globals():
        if not POINT_TABLE_PATH.exists():
            raise FileNotFoundError(
                f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
            )
        point_df = pd.read_csv(POINT_TABLE_PATH)

    exp_df = point_df.copy()

    required_columns = {
        "load_mpa",
        "sample_type",
        "sample",
        "time_h",
        "bulk_z_strain_percent",
        "height_path",
    }

    missing = required_columns - set(exp_df.columns)
    if missing:
        raise ValueError(f"point_df missing required columns: {sorted(missing)}")

    exp_df["load_mpa"] = pd.to_numeric(exp_df["load_mpa"], errors="coerce")
    exp_df["time_h"] = pd.to_numeric(exp_df["time_h"], errors="coerce")
    exp_df["bulk_z_strain_percent"] = pd.to_numeric(
        exp_df["bulk_z_strain_percent"],
        errors="coerce",
    )

    exp_df = exp_df[
        np.isfinite(exp_df["load_mpa"])
        & np.isfinite(exp_df["time_h"])
        & np.isfinite(exp_df["bulk_z_strain_percent"])
    ].copy()

    exp_df = exp_df.sort_values(
        ["sample_type", "load_mpa", "sample", "time_h"]
    ).reset_index(drop=True)

    exp_curve_rows = []

    for case in all_cases:
        load = case["load"]
        sample_type = case["sample_type"]

        case_df = exp_df[
            (exp_df["load_mpa"] == load)
            & (exp_df["sample_type"].astype(str) == sample_type)
        ].copy()

        if case_df.empty:
            warnings.warn(f"No experimental rows for {load} MPa {sample_type}")
            continue

        for sample, g in case_df.groupby("sample"):
            g = g.sort_values("time_h")
            final = g.iloc[-1]

            try:
                H = load_exp_height(final["height_path"])

                f_center, lam, psd_raw, modes = radial_psd_frequency_binned(
                    H,
                    EXP_SPACING_UM,
                    f_edges,
                    min_modes=MIN_MODES_PER_PSD_BIN,
                )

                psd_norm = normalize_psd_over_frequency(f_center, psd_raw)

                r, acf, counts = radial_acf(H, EXP_SPACING_UM, r_edges)

                for i in range(len(lam)):
                    exp_curve_rows.append(
                        {
                            "source": "exp",
                            "load_mpa": load,
                            "sample_type": sample_type,
                            "sample_id": sample,
                            "replicate_id": str(sample),
                            "bulk_z_strain_percent": float(
                                final["bulk_z_strain_percent"]
                            ),
                            "curve_type": "psd",
                            "x_um": lam[i],
                            "frequency_um_inv": f_center[i],
                            "y": psd_norm[i],
                            "raw_y": psd_raw[i],
                            "modes_or_counts": modes[i],
                        }
                    )

                for i in range(len(r)):
                    exp_curve_rows.append(
                        {
                            "source": "exp",
                            "load_mpa": load,
                            "sample_type": sample_type,
                            "sample_id": sample,
                            "replicate_id": str(sample),
                            "bulk_z_strain_percent": float(
                                final["bulk_z_strain_percent"]
                            ),
                            "curve_type": "acf",
                            "x_um": r[i],
                            "frequency_um_inv": np.nan,
                            "y": acf[i],
                            "raw_y": acf[i],
                            "modes_or_counts": counts[i],
                        }
                    )

            except Exception as exc:
                warnings.warn(
                    f"Experimental endpoint failed for {load} {sample_type} sample {sample}: {exc}"
                )

    exp_curves_df = pd.DataFrame(exp_curve_rows)

    sim_curve_rows = []

    for micro_run in micro_runs:
        micro_id = micro_run["micro_id"]
        sim_root = micro_run["sim_root"]
        microstructure = micro_run["microstructure"]

        for case in all_cases:
            load = case["load"]
            sample_type = case["sample_type"]

            run_dir = sim_root / f"{load}mpa_{sample_type}"

            print(f"Loading simulation endpoint: {micro_id}, {load} MPa {sample_type}")

            try:
                sim_i = SimResults.load(run_dir, microstructure=microstructure)
            except Exception as exc:
                warnings.warn(f"Failed loading simulation {run_dir}: {exc}")
                continue

            H_all = np.asarray(sim_i.height, dtype=float)
            n_faces, n_times, nz, n_width = H_all.shape

            vtk_time = np.asarray(sim_i.vtk_time, dtype=float)

            strain_time = np.asarray(sim_i.sim_time, dtype=float)
            bulk_strain_all = np.asarray(getattr(sim_i, strain_col), dtype=float)

            valid = np.isfinite(strain_time) & np.isfinite(bulk_strain_all)

            strain_time_valid = strain_time[valid]
            bulk_strain_valid = bulk_strain_all[valid]

            order = np.argsort(strain_time_valid)
            strain_time_valid = strain_time_valid[order]
            bulk_strain_valid = bulk_strain_valid[order]

            bulk_strain_vtk = np.interp(
                vtk_time,
                strain_time_valid,
                bulk_strain_valid,
            )

            n = min(n_times, len(vtk_time), len(bulk_strain_vtk))

            H_all = H_all[:, :n, :, :]
            bulk_strain_vtk = bulk_strain_vtk[:n]

            final_idx = n - 1
            final_strain_percent = 100.0 * float(bulk_strain_vtk[final_idx])

            for face_idx in range(n_faces):
                replicate_id = f"{micro_id}_face{face_idx}"

                try:
                    H = preprocess_sim_height(H_all[face_idx, final_idx])

                    f_center, lam, psd_raw, modes = radial_psd_frequency_binned(
                        H,
                        SIM_SPACING_UM,
                        f_edges,
                        min_modes=MIN_MODES_PER_PSD_BIN,
                    )

                    psd_norm = normalize_psd_over_frequency(f_center, psd_raw)

                    r, acf, counts = radial_acf(H, SIM_SPACING_UM, r_edges)

                    for i in range(len(lam)):
                        sim_curve_rows.append(
                            {
                                "source": "sim",
                                "load_mpa": load,
                                "sample_type": sample_type,
                                "sample_id": micro_id,
                                "replicate_id": replicate_id,
                                "bulk_z_strain_percent": final_strain_percent,
                                "curve_type": "psd",
                                "x_um": lam[i],
                                "frequency_um_inv": f_center[i],
                                "y": psd_norm[i],
                                "raw_y": psd_raw[i],
                                "modes_or_counts": modes[i],
                            }
                        )

                    for i in range(len(r)):
                        sim_curve_rows.append(
                            {
                                "source": "sim",
                                "load_mpa": load,
                                "sample_type": sample_type,
                                "sample_id": micro_id,
                                "replicate_id": replicate_id,
                                "bulk_z_strain_percent": final_strain_percent,
                                "curve_type": "acf",
                                "x_um": r[i],
                                "frequency_um_inv": np.nan,
                                "y": acf[i],
                                "raw_y": acf[i],
                                "modes_or_counts": counts[i],
                            }
                        )

                except Exception as exc:
                    warnings.warn(
                        f"Simulation endpoint failed for {micro_id} {load} {sample_type} "
                        f"face {face_idx}: {exc}"
                    )

    sim_curves_df = pd.DataFrame(sim_curve_rows)

    curves_df = pd.concat([exp_curves_df, sim_curves_df], ignore_index=True)

    if curves_df.empty:
        raise RuntimeError("No endpoint curves were computed")

    curves_df.to_csv(CURVES_CACHE, index=False)

    summary_rows = []

    for (source, load, sample_type, curve_type, x_um), g in curves_df.groupby(
        ["source", "load_mpa", "sample_type", "curve_type", "x_um"]
    ):
        y = pd.to_numeric(g["y"], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]

        if y.size == 0:
            continue

        summary_rows.append(
            {
                "source": source,
                "load_mpa": load,
                "sample_type": sample_type,
                "curve_type": curve_type,
                "x_um": x_um,
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
                "sem": (
                    float(np.std(y, ddof=1) / np.sqrt(y.size)) if y.size > 1 else 0.0
                ),
                "n": int(y.size),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CACHE, index=False)

    print(f"Saved: {CURVES_CACHE}")
    print(f"Saved: {SUMMARY_CACHE}")

else:
    print("Using cached endpoint curves. Simulation data will not be reloaded.")
    curves_df = pd.read_csv(CURVES_CACHE)
    summary_df = pd.read_csv(SUMMARY_CACHE)


# =============================================================================
# Plot four-panel figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=False)

ax_psd_int = axes[0, 0]
ax_psd_unint = axes[0, 1]
ax_acf_int = axes[1, 0]
ax_acf_unint = axes[1, 1]


def plot_group(ax, cases, curve_type, title, ylabel):
    for case in cases:
        load = case["load"]
        sample_type = case["sample_type"]
        color = LOAD_COLORS.get(load, "0.3")

        for source, linestyle, alpha, label_suffix in [
            ("exp", "-", 1.0, "Exp."),
            ("sim", "--", 0.95, "Sim."),
        ]:
            s = summary_df[
                (summary_df["source"] == source)
                & (summary_df["load_mpa"] == load)
                & (summary_df["sample_type"] == sample_type)
                & (summary_df["curve_type"] == curve_type)
            ].copy()

            if s.empty:
                continue

            s = s.sort_values("x_um")

            ax.plot(
                s["x_um"],
                s["median"],
                color=color,
                linestyle=linestyle,
                lw=2.0 if source == "exp" else 1.8,
                alpha=alpha,
                label=f"{load} MPa {label_suffix}",
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


plot_group(
    ax_psd_int,
    interrupted_cases,
    "psd",
    "A. Interrupted: normalized radial PSD",
    r"median normalized radial PSD",
)

plot_group(
    ax_psd_unint,
    uninterrupted_cases,
    "psd",
    "B. Uninterrupted: normalized radial PSD",
    r"median normalized radial PSD",
)

plot_group(
    ax_acf_int,
    interrupted_cases,
    "acf",
    "C. Interrupted: radial ACF",
    r"median radial ACF, $C(r)$",
)

plot_group(
    ax_acf_unint,
    uninterrupted_cases,
    "acf",
    "D. Uninterrupted: radial ACF",
    r"median radial ACF, $C(r)$",
)

for ax in [ax_psd_int, ax_psd_unint]:
    ax.set_xscale("log")
    ax.set_xlim(PLOT_LAMBDA_MIN_UM, PLOT_LAMBDA_MAX_UM)
    ax.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")

    if PSD_YLOG:
        ax.set_yscale("log")

    ax.legend(fontsize=6, ncol=2)

for ax in [ax_acf_int, ax_acf_unint]:
    ax.set_xlim(0.0, ACF_MAX_LAG_UM)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axhline(np.exp(-1.0), color="0.5", lw=0.8, ls=":")
    ax.legend(fontsize=6, ncol=2)

# Numeric wavelength tick labels on log x-axis.
psd_ticks = np.array([3, 4, 5, 6, 10, 20, 50, 100, 128], dtype=float)
psd_ticks = psd_ticks[
    (psd_ticks >= PLOT_LAMBDA_MIN_UM) & (psd_ticks <= PLOT_LAMBDA_MAX_UM)
]

for ax in [ax_psd_int, ax_psd_unint]:
    ax.xaxis.set_major_locator(FixedLocator(psd_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in psd_ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())

fig.suptitle(
    "Endpoint spatial configuration: experiment vs simulation\n"
    "Full cropped fields; crop-first leveling; frequency-annulus PSD; overlap-corrected ACF",
    y=1.02,
)

fig.tight_layout()

outpath = (
    OUTPUT_DIR
    / "four_panel_endpoint_exp_vs_sim_fullfield_cropfirst_writeup_psd_acf.png"
)
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Endpoint height-field cache for experiment + simulation
# =============================================================================
#
# This caches endpoint height fields, NOT PSD/ACF curves.
#
# Use:
#   BUILD_ENDPOINT_HEIGHT_CACHE = True   # once, to create cache
#   BUILD_ENDPOINT_HEIGHT_CACHE = False  # thereafter, to load cache only
#
# Cached files:
#   endpoint_height_fields_metadata.csv
#   endpoint_height_fields_arrays.npz
#
# Contents:
#   - experimental final endpoint cropped raw height fields
#   - simulation final endpoint height fields
#   - metadata including source, load, type, sample/face, spacing, strain
#
# The cached arrays are intentionally not plane-leveled. Plane leveling is done
# after loading from cache so you can change detrending/order/PSD/ACF parameters
# without reloading simulation data.
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from utils.data_utils import SimResults

# =============================================================================
# User settings
# =============================================================================

BUILD_ENDPOINT_HEIGHT_CACHE = True  # set True once to build; False to load only

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT_CACHE_METADATA = OUTPUT_DIR / "endpoint_height_fields_metadata.csv"
HEIGHT_CACHE_ARRAYS = OUTPUT_DIR / "endpoint_height_fields_arrays.npz"

POINT_TABLE_PATH = OUTPUT_DIR / "roughness_plastic_strain_point_table.csv"

# Experimental profilometry
EXP_SPACING_UM = 1.379951
EXP_CROP = (slice(50, -50), slice(50, 750))

# Simulation
SIM_SPACING_UM = 1.0
strain_col = "epav33"

micro_runs = [
    {
        "micro_id": "micro1",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro1_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro1_production.dat"
        ),
    },
    {
        "micro_id": "micro2",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro2_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro2_production.dat"
        ),
    },
    {
        "micro_id": "micro3",
        "sim_root": Path(
            "/Users/gtdebru/mimosa/hpc_downloads/gtdebru/micro3_production"
        ),
        "microstructure": Path(
            "/Users/gtdebru/mimosa/microstructures/production/micro3_production.dat"
        ),
    },
]

load_cases = [
    {"load": 475, "sample_type": "int"},
    {"load": 500, "sample_type": "unint"},
    {"load": 525, "sample_type": "int"},
    {"load": 530, "sample_type": "unint"},
    {"load": 575, "sample_type": "int"},
    {"load": 588, "sample_type": "unint"},
]


# =============================================================================
# Experimental raw height I/O
# =============================================================================


def raw_exp_height(path: str | Path) -> np.ndarray:
    return (
        pd.read_csv(path, skiprows=19, header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )


def fill_missing_nearest(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    missing = ~np.isfinite(H)

    if np.any(missing):
        nearest = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        H = H[tuple(nearest)]

    return H


def load_exp_endpoint_raw_cropped(path: str | Path) -> np.ndarray:
    """
    Cache experimental endpoint as raw cropped height field, after missing fill.
    Do NOT plane-level here. Plane-level later during analysis.
    """
    H = raw_exp_height(path)
    H = fill_missing_nearest(H)
    H = H[EXP_CROP]
    return np.asarray(H, dtype=np.float64)


# =============================================================================
# Load experiment point table
# =============================================================================


def load_point_table() -> pd.DataFrame:
    if "point_df" in globals():
        df = point_df.copy()
    else:
        if not POINT_TABLE_PATH.exists():
            raise FileNotFoundError(
                f"point_df is not in memory and {POINT_TABLE_PATH} does not exist."
            )
        df = pd.read_csv(POINT_TABLE_PATH)

    required_columns = {
        "load_mpa",
        "sample_type",
        "sample",
        "time_h",
        "bulk_z_strain_percent",
        "height_path",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"point_df is missing required columns: {sorted(missing)}")

    df["load_mpa"] = pd.to_numeric(df["load_mpa"], errors="coerce")
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    df["bulk_z_strain_percent"] = pd.to_numeric(
        df["bulk_z_strain_percent"],
        errors="coerce",
    )

    df = df[
        np.isfinite(df["load_mpa"])
        & np.isfinite(df["time_h"])
        & np.isfinite(df["bulk_z_strain_percent"])
    ].copy()

    df = df.sort_values(["sample_type", "load_mpa", "sample", "time_h"]).reset_index(
        drop=True
    )

    return df


# =============================================================================
# Build endpoint height cache
# =============================================================================


def build_endpoint_height_cache():
    metadata_rows = []
    arrays = {}
    arr_idx = 0

    # -------------------------------------------------------------------------
    # Experimental final endpoint fields
    # -------------------------------------------------------------------------

    exp_df = load_point_table()

    print("Caching experimental endpoint height fields...")

    for (sample_type, load, sample), g in exp_df.groupby(
        ["sample_type", "load_mpa", "sample"]
    ):
        g = g.sort_values("time_h").copy()

        if g.empty:
            continue

        final = g.iloc[-1]

        try:
            H = load_exp_endpoint_raw_cropped(final["height_path"])

            key = f"arr_{arr_idx:06d}"
            arrays[key] = H.astype(np.float32)

            metadata_rows.append(
                {
                    "array_key": key,
                    "source": "exp",
                    "load_mpa": int(load),
                    "sample_type": str(sample_type),
                    "sample_id": str(sample),
                    "replicate_id": str(sample),
                    "micro_id": "",
                    "face_idx": np.nan,
                    "time": np.nan,
                    "time_h": float(final["time_h"]),
                    "bulk_z_strain_percent": float(final["bulk_z_strain_percent"]),
                    "spacing_um": EXP_SPACING_UM,
                    "height_shape_0": H.shape[0],
                    "height_shape_1": H.shape[1],
                    "height_path": str(final["height_path"]),
                    "run_dir": "",
                    "microstructure": "",
                }
            )

            arr_idx += 1

        except Exception as exc:
            warnings.warn(
                f"Failed caching experimental endpoint: "
                f"{load} {sample_type} sample {sample}: {exc}"
            )

    # -------------------------------------------------------------------------
    # Simulation final endpoint fields
    # -------------------------------------------------------------------------

    print("Caching simulation endpoint height fields...")

    for micro_run in micro_runs:
        micro_id = micro_run["micro_id"]
        sim_root = micro_run["sim_root"]
        microstructure = micro_run["microstructure"]

        for case in load_cases:
            load = case["load"]
            sample_type = case["sample_type"]

            run_dir = sim_root / f"{load}mpa_{sample_type}"

            print(f"  Loading {micro_id}, {load} MPa {sample_type}")

            try:
                sim_i = SimResults.load(
                    run_dir,
                    microstructure=microstructure,
                )
            except Exception as exc:
                warnings.warn(f"Failed loading simulation {run_dir}: {exc}")
                continue

            H_all = np.asarray(sim_i.height, dtype=float)
            n_faces, n_times, nz, n_width = H_all.shape

            vtk_time = np.asarray(sim_i.vtk_time, dtype=float)

            strain_time = np.asarray(sim_i.sim_time, dtype=float)
            bulk_strain_all = np.asarray(getattr(sim_i, strain_col), dtype=float)

            valid = np.isfinite(strain_time) & np.isfinite(bulk_strain_all)

            strain_time_valid = strain_time[valid]
            bulk_strain_valid = bulk_strain_all[valid]

            order = np.argsort(strain_time_valid)
            strain_time_valid = strain_time_valid[order]
            bulk_strain_valid = bulk_strain_valid[order]

            bulk_strain_vtk = np.interp(
                vtk_time,
                strain_time_valid,
                bulk_strain_valid,
            )

            n = min(n_times, len(vtk_time), len(bulk_strain_vtk))

            if n == 0:
                warnings.warn(f"No simulation time points for {run_dir}")
                continue

            final_idx = n - 1
            final_time = float(vtk_time[final_idx])
            final_strain_percent = 100.0 * float(bulk_strain_vtk[final_idx])

            for face_idx in range(n_faces):
                try:
                    H = np.asarray(H_all[face_idx, final_idx], dtype=np.float64)

                    if not np.all(np.isfinite(H)):
                        H = H.copy()
                        H[~np.isfinite(H)] = np.nanmean(H)

                    key = f"arr_{arr_idx:06d}"
                    arrays[key] = H.astype(np.float32)

                    metadata_rows.append(
                        {
                            "array_key": key,
                            "source": "sim",
                            "load_mpa": int(load),
                            "sample_type": str(sample_type),
                            "sample_id": str(micro_id),
                            "replicate_id": f"{micro_id}_face{face_idx}",
                            "micro_id": str(micro_id),
                            "face_idx": int(face_idx),
                            "time": final_time,
                            "time_h": final_time / 3600.0,
                            "bulk_z_strain_percent": final_strain_percent,
                            "spacing_um": SIM_SPACING_UM,
                            "height_shape_0": H.shape[0],
                            "height_shape_1": H.shape[1],
                            "height_path": "",
                            "run_dir": str(run_dir),
                            "microstructure": str(microstructure),
                        }
                    )

                    arr_idx += 1

                except Exception as exc:
                    warnings.warn(
                        f"Failed caching simulation endpoint: "
                        f"{micro_id}, {load} {sample_type}, face {face_idx}: {exc}"
                    )

    metadata = pd.DataFrame(metadata_rows)

    if metadata.empty:
        raise RuntimeError("No endpoint height fields were cached.")

    np.savez_compressed(HEIGHT_CACHE_ARRAYS, **arrays)
    metadata.to_csv(HEIGHT_CACHE_METADATA, index=False)

    print()
    print(f"Saved height array cache: {HEIGHT_CACHE_ARRAYS.resolve()}")
    print(f"Saved metadata cache:     {HEIGHT_CACHE_METADATA.resolve()}")
    print(f"Cached arrays: {len(arrays)}")

    return metadata, arrays


# =============================================================================
# Load endpoint height cache
# =============================================================================


def load_endpoint_height_cache():
    if not HEIGHT_CACHE_METADATA.exists():
        raise FileNotFoundError(
            f"Metadata cache missing:\n{HEIGHT_CACHE_METADATA.resolve()}\n"
            f"Set BUILD_ENDPOINT_HEIGHT_CACHE=True once."
        )

    if not HEIGHT_CACHE_ARRAYS.exists():
        raise FileNotFoundError(
            f"Array cache missing:\n{HEIGHT_CACHE_ARRAYS.resolve()}\n"
            f"Set BUILD_ENDPOINT_HEIGHT_CACHE=True once."
        )

    metadata = pd.read_csv(HEIGHT_CACHE_METADATA)
    arrays_npz = np.load(HEIGHT_CACHE_ARRAYS)

    print()
    print("Loaded endpoint height cache:")
    print(f"  metadata: {HEIGHT_CACHE_METADATA.resolve()}")
    print(f"  arrays:   {HEIGHT_CACHE_ARRAYS.resolve()}")
    print(f"  n rows:   {len(metadata)}")

    return metadata, arrays_npz


# =============================================================================
# Main cache control
# =============================================================================

print("BUILD_ENDPOINT_HEIGHT_CACHE =", BUILD_ENDPOINT_HEIGHT_CACHE)
print("HEIGHT_CACHE_METADATA =", HEIGHT_CACHE_METADATA.resolve())
print("HEIGHT_CACHE_METADATA exists =", HEIGHT_CACHE_METADATA.exists())
print("HEIGHT_CACHE_ARRAYS =", HEIGHT_CACHE_ARRAYS.resolve())
print("HEIGHT_CACHE_ARRAYS exists =", HEIGHT_CACHE_ARRAYS.exists())

if BUILD_ENDPOINT_HEIGHT_CACHE:
    endpoint_metadata, endpoint_arrays = build_endpoint_height_cache()
else:
    endpoint_metadata, endpoint_arrays = load_endpoint_height_cache()


# =============================================================================
# Example: use cached arrays for analysis
# =============================================================================
#
# Use this in downstream PSD/ACF code:
#
#   for _, rec in endpoint_metadata.iterrows():
#       H_raw = endpoint_arrays[rec["array_key"]]
#       spacing = rec["spacing_um"]
#       if rec["source"] == "exp":
#           H = detrend_surface(H_raw, spacing, order=EXP_DETREND_ORDER)
#       else:
#           H = detrend_surface(H_raw, spacing, order=SIM_DETREND_ORDER)
#       H = H - np.nanmean(H)
#
# This avoids reading experimental CSVs and avoids SimResults.load entirely.
# =============================================================================

# %%
# =============================================================================
# Four-panel endpoint experiment-vs-simulation comparison
# USING CACHED ENDPOINT HEIGHT MAPS
# =============================================================================
#
# Requires the height cache created earlier:
#
#   endpoint_height_fields_metadata.csv
#   endpoint_height_fields_arrays.npz
#
# This code DOES NOT call SimResults.load(...)
# This code DOES NOT reread experimental profilometry CSVs
#
# It recomputes PSD/ACF from cached endpoint height arrays.
#
# Operator:
#   - Exp cache stores cropped raw endpoint maps.
#   - Sim cache stores final endpoint height maps.
#   - Both are plane-leveled after loading from cache.
#   - Both are mean-removed after plane leveling.
#
# PSD:
#   - Protocol PSD:
#       C(fx,fy) = dx^2 * |FFT(h*w)|^2 / sum(w^2)
#   - radial mean in physical-frequency annuli
#   - aggregation uses median across specimens / sim faces
#   - plotted PSD is normalized by trapezoidal integral over frequency
#
# ACF:
#   - linear overlap-corrected covariance
#   - normalized by zero-lag covariance
#   - radial average in annuli
#   - aggregation uses median across specimens / sim faces
#
# Layout:
#   Top-left:     interrupted endpoint normalized radial PSD
#   Bottom-left:  interrupted endpoint radial ACF
#   Top-right:    uninterrupted endpoint normalized radial PSD
#   Bottom-right: uninterrupted endpoint radial ACF
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.signal import fftconvolve

# =============================================================================
# Settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT_CACHE_METADATA = OUTPUT_DIR / "endpoint_height_fields_metadata.csv"
HEIGHT_CACHE_ARRAYS = OUTPUT_DIR / "endpoint_height_fields_arrays.npz"

# Derived curve caches computed from the height-field cache.
# These are cheap to recompute compared with reloading SimResults, but keeping
# them avoids recomputing PSD/ACF if only plotting changes.
RECOMPUTE_DERIVED_CURVES_FROM_HEIGHT_CACHE = True

CURVES_CACHE = (
    OUTPUT_DIR / "endpoint_exp_sim_cached_height_fullfield_psd_acf_curves_long.csv"
)
SUMMARY_CACHE = (
    OUTPUT_DIR / "endpoint_exp_sim_cached_height_fullfield_psd_acf_curves_summary.csv"
)

# Detrending
EXP_DETREND_ORDER = 1
SIM_DETREND_ORDER = 1

# Cases
interrupted_cases = [
    {"load": 475, "sample_type": "int"},
    {"load": 525, "sample_type": "int"},
    {"load": 575, "sample_type": "int"},
]

uninterrupted_cases = [
    {"load": 500, "sample_type": "unint"},
    {"load": 530, "sample_type": "unint"},
    {"load": 588, "sample_type": "unint"},
]

all_cases = interrupted_cases + uninterrupted_cases

LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

# PSD binning.
# Writeup-style physical frequency annuli for 10x:
# 22 logarithmic annuli from 0.00283072 to 0.36 inverse micrometers.
PSD_F_MIN_UM_INV = 0.00283072
PSD_F_MAX_UM_INV = 0.36
N_PSD_ANNULI = 22
MIN_MODES_PER_PSD_BIN = 1

# Plot display range.
PLOT_LAMBDA_MIN_UM = 1.0 / PSD_F_MAX_UM_INV
PLOT_LAMBDA_MAX_UM = 128.0

# PSD y-axis.
PSD_YLOG = True

# ACF.
# Writeup ACF radial bin width is one native 10x pixel.
# If you specifically want twice native spacing instead, change to 2.0 * 1.379951.
EXP_NATIVE_SPACING_UM = 1.379951
ACF_RADIAL_BIN_WIDTH_UM = EXP_NATIVE_SPACING_UM
ACF_MAX_LAG_UM = 128.0

# Plot median aggregate curves.
PLOT_INDIVIDUAL_CURVES = False
INDIVIDUAL_ALPHA = 0.08

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load cached height fields
# =============================================================================

if not HEIGHT_CACHE_METADATA.exists():
    raise FileNotFoundError(
        f"Missing height metadata cache:\n{HEIGHT_CACHE_METADATA.resolve()}\n"
        "Run the endpoint height-field cache builder once first."
    )

if not HEIGHT_CACHE_ARRAYS.exists():
    raise FileNotFoundError(
        f"Missing height array cache:\n{HEIGHT_CACHE_ARRAYS.resolve()}\n"
        "Run the endpoint height-field cache builder once first."
    )

endpoint_metadata = pd.read_csv(HEIGHT_CACHE_METADATA)
endpoint_arrays = np.load(HEIGHT_CACHE_ARRAYS)

required_metadata_cols = {
    "array_key",
    "source",
    "load_mpa",
    "sample_type",
    "sample_id",
    "replicate_id",
    "bulk_z_strain_percent",
    "spacing_um",
}

missing = required_metadata_cols - set(endpoint_metadata.columns)
if missing:
    raise ValueError(f"Endpoint metadata missing columns: {sorted(missing)}")

endpoint_metadata["load_mpa"] = pd.to_numeric(
    endpoint_metadata["load_mpa"], errors="coerce"
)
endpoint_metadata["bulk_z_strain_percent"] = pd.to_numeric(
    endpoint_metadata["bulk_z_strain_percent"],
    errors="coerce",
)
endpoint_metadata["spacing_um"] = pd.to_numeric(
    endpoint_metadata["spacing_um"], errors="coerce"
)

endpoint_metadata = endpoint_metadata[
    np.isfinite(endpoint_metadata["load_mpa"])
    & np.isfinite(endpoint_metadata["bulk_z_strain_percent"])
    & np.isfinite(endpoint_metadata["spacing_um"])
].copy()

print("Loaded endpoint height cache:")
print(f"  metadata: {HEIGHT_CACHE_METADATA.resolve()}")
print(f"  arrays:   {HEIGHT_CACHE_ARRAYS.resolve()}")
print(f"  rows:     {len(endpoint_metadata)}")
print(endpoint_metadata.groupby(["source", "sample_type", "load_mpa"]).size())


# =============================================================================
# Numerical helpers
# =============================================================================


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x)
    if hasattr(np, "trapz"):
        return np.trapz(y, x=x)

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, col = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = col.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend([x * x, x * y, y * y])
    elif order != 1:
        raise ValueError("detrend order must be 1 or 2")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    design, inverse = _detrend_geometry(
        values.shape, float(spacing_um_value), int(order)
    )
    coeff = inverse @ values.ravel()
    trend = (design @ coeff).reshape(values.shape)

    return values - trend


def preprocess_cached_height(
    H_raw: np.ndarray, spacing_um: float, source: str
) -> np.ndarray:
    H = np.asarray(H_raw, dtype=float)

    if not np.all(np.isfinite(H)):
        H = H.copy()
        H[~np.isfinite(H)] = np.nanmean(H)

    if source == "exp":
        order = EXP_DETREND_ORDER
    elif source == "sim":
        order = SIM_DETREND_ORDER
    else:
        raise ValueError(f"Unknown source: {source}")

    H = detrend_surface(H, spacing_um, order=order)
    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD: protocol-style formula and frequency-annulus radial averaging
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    return w0[:, None] * w1[None, :]


def psd2d_height_protocol(height_um: np.ndarray, spacing_um_value: float):
    """
    Protocol PSD:
        C(fx,fy) = dx^2 * |FFT(h*w)|^2 / sum(w^2)
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)
    PSD = (spacing_um_value**2) * np.abs(F) ** 2 / np.sum(W**2)

    n0, n1 = H.shape
    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    return f0, f1, PSD


def make_frequency_edges():
    return np.logspace(
        np.log10(PSD_F_MIN_UM_INV),
        np.log10(PSD_F_MAX_UM_INV),
        N_PSD_ANNULI + 1,
    )


f_edges = make_frequency_edges()


def radial_psd_frequency_binned(
    height_um: np.ndarray,
    spacing_um_value: float,
    f_edges_um_inv: np.ndarray,
    min_modes: int = 1,
):
    f0, f1, PSD = psd2d_height_protocol(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    nbins = len(f_edges_um_inv) - 1

    f_center = np.sqrt(f_edges_um_inv[:-1] * f_edges_um_inv[1:])
    lambda_center = 1.0 / f_center

    radial = np.full(nbins, np.nan, dtype=float)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        lo = f_edges_um_inv[i]
        hi = f_edges_um_inv[i + 1]

        mask = np.isfinite(FR) & np.isfinite(PSD) & (FR >= lo) & (FR < hi)

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            radial[i] = float(np.nanmean(PSD[mask]))

    return f_center, lambda_center, radial, modes


def normalize_psd_over_frequency(f_center, psd):
    """
    Morphology-style normalized PSD:
      radial PSD divided by trapezoidal integral over frequency.
    """
    f = np.asarray(f_center, dtype=float)
    y = np.asarray(psd, dtype=float)

    out = np.full_like(y, np.nan, dtype=float)

    valid = np.isfinite(f) & np.isfinite(y) & (f > 0) & (y > 0)

    if np.count_nonzero(valid) < 2:
        return out

    order = np.argsort(f[valid])
    f_valid = f[valid][order]
    y_valid = y[valid][order]

    area = integrate_trapezoid(y_valid, x=f_valid)

    if not np.isfinite(area) or area <= 0:
        return out

    out[valid] = y[valid] / area

    return out


# =============================================================================
# ACF: linear overlap-corrected covariance and radial averaging
# =============================================================================


def overlap_normalized_acf_2d(z: np.ndarray, spacing_um_value: float):
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)

    if np.count_nonzero(finite) < 10:
        raise ValueError("Too few finite pixels for ACF")

    z0 = np.zeros_like(z, dtype=float)
    z_mean = np.nanmean(z)
    z0[finite] = z[finite] - z_mean

    mask = finite.astype(float)

    numerator = fftconvolve(z0, z0[::-1, ::-1], mode="full")
    overlap_counts = fftconvolve(mask, mask[::-1, ::-1], mode="full")

    covariance = np.full_like(numerator, np.nan, dtype=float)

    valid_overlap = overlap_counts > 0
    covariance[valid_overlap] = numerator[valid_overlap] / overlap_counts[valid_overlap]

    center = tuple(np.array(covariance.shape) // 2)
    zero_lag_cov = covariance[center]

    if not np.isfinite(zero_lag_cov) or zero_lag_cov <= 0:
        rho = np.full_like(covariance, np.nan, dtype=float)
    else:
        rho = covariance / zero_lag_cov

    n0, n1 = z.shape

    lag0_um = np.arange(-(n0 - 1), n0) * spacing_um_value
    lag1_um = np.arange(-(n1 - 1), n1) * spacing_um_value

    return lag0_um, lag1_um, rho


r_edges = np.arange(
    0.0,
    ACF_MAX_LAG_UM + ACF_RADIAL_BIN_WIDTH_UM,
    ACF_RADIAL_BIN_WIDTH_UM,
)


def radial_average_acf(lag0_um, lag1_um, rho, r_edges_um):
    L1, L0 = np.meshgrid(lag1_um, lag0_um)
    R = np.sqrt(L0**2 + L1**2)

    centers = 0.5 * (r_edges_um[:-1] + r_edges_um[1:])
    radial = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)

    valid = np.isfinite(R) & np.isfinite(rho)

    for i in range(len(centers)):
        mask = valid & (R >= r_edges_um[i]) & (R < r_edges_um[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            radial[i] = float(np.nanmean(rho[mask]))

    keep = centers < ACF_MAX_LAG_UM

    return centers[keep], radial[keep], counts[keep]


def radial_acf(height_um, spacing_um_value, r_edges_um):
    lag0_um, lag1_um, rho = overlap_normalized_acf_2d(height_um, spacing_um_value)
    return radial_average_acf(lag0_um, lag1_um, rho, r_edges_um)


# =============================================================================
# Compute endpoint PSD/ACF curves from cached height arrays
# =============================================================================

if (
    RECOMPUTE_DERIVED_CURVES_FROM_HEIGHT_CACHE
    or (not CURVES_CACHE.exists())
    or (not SUMMARY_CACHE.exists())
):
    print("Computing PSD/ACF curves from cached endpoint height arrays.")
    print("No SimResults.load() calls will be made.")

    curve_rows = []

    for _, rec in endpoint_metadata.iterrows():
        try:
            array_key = rec["array_key"]
            H_raw = np.asarray(endpoint_arrays[array_key], dtype=float)

            source = str(rec["source"])
            spacing = float(rec["spacing_um"])

            H = preprocess_cached_height(H_raw, spacing, source)

            f_center, lam, psd_raw, modes = radial_psd_frequency_binned(
                H,
                spacing,
                f_edges,
                min_modes=MIN_MODES_PER_PSD_BIN,
            )

            psd_norm = normalize_psd_over_frequency(f_center, psd_raw)

            r, acf, counts = radial_acf(H, spacing, r_edges)

            metadata_common = {
                "source": source,
                "load_mpa": int(rec["load_mpa"]),
                "sample_type": str(rec["sample_type"]),
                "sample_id": str(rec["sample_id"]),
                "replicate_id": str(rec["replicate_id"]),
                "bulk_z_strain_percent": float(rec["bulk_z_strain_percent"]),
            }

            for i in range(len(lam)):
                curve_rows.append(
                    {
                        **metadata_common,
                        "curve_type": "psd",
                        "x_um": float(lam[i]),
                        "frequency_um_inv": float(f_center[i]),
                        "y": float(psd_norm[i]) if np.isfinite(psd_norm[i]) else np.nan,
                        "raw_y": (
                            float(psd_raw[i]) if np.isfinite(psd_raw[i]) else np.nan
                        ),
                        "modes_or_counts": int(modes[i]),
                    }
                )

            for i in range(len(r)):
                curve_rows.append(
                    {
                        **metadata_common,
                        "curve_type": "acf",
                        "x_um": float(r[i]),
                        "frequency_um_inv": np.nan,
                        "y": float(acf[i]) if np.isfinite(acf[i]) else np.nan,
                        "raw_y": float(acf[i]) if np.isfinite(acf[i]) else np.nan,
                        "modes_or_counts": int(counts[i]),
                    }
                )

        except Exception as exc:
            warnings.warn(
                f"Failed endpoint curve computation for array_key={rec.get('array_key', '?')}, "
                f"source={rec.get('source', '?')}, load={rec.get('load_mpa', '?')}, "
                f"sample={rec.get('sample_id', '?')}: {exc}"
            )

    curves_df = pd.DataFrame(curve_rows)

    if curves_df.empty:
        raise RuntimeError(
            "No endpoint curves were computed from cached height arrays."
        )

    curves_df.to_csv(CURVES_CACHE, index=False)

    # Median aggregation across specimens/faces.
    summary_rows = []

    for (source, load, sample_type, curve_type, x_um), g in curves_df.groupby(
        ["source", "load_mpa", "sample_type", "curve_type", "x_um"]
    ):
        y = pd.to_numeric(g["y"], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]

        if y.size == 0:
            continue

        summary_rows.append(
            {
                "source": source,
                "load_mpa": load,
                "sample_type": sample_type,
                "curve_type": curve_type,
                "x_um": x_um,
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
                "sem": (
                    float(np.std(y, ddof=1) / np.sqrt(y.size)) if y.size > 1 else 0.0
                ),
                "n": int(y.size),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CACHE, index=False)

    print(f"Saved: {CURVES_CACHE.resolve()}")
    print(f"Saved: {SUMMARY_CACHE.resolve()}")

else:
    print("Using cached derived PSD/ACF curves.")
    print(f"Curves cache:  {CURVES_CACHE.resolve()}")
    print(f"Summary cache: {SUMMARY_CACHE.resolve()}")

    curves_df = pd.read_csv(CURVES_CACHE)
    summary_df = pd.read_csv(SUMMARY_CACHE)


# =============================================================================
# Plot four-panel figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=False)

ax_psd_int = axes[0, 0]
ax_psd_unint = axes[0, 1]
ax_acf_int = axes[1, 0]
ax_acf_unint = axes[1, 1]


def plot_group(ax, cases, curve_type, title, ylabel):
    for case in cases:
        load = case["load"]
        sample_type = case["sample_type"]
        color = LOAD_COLORS.get(load, "0.3")

        for source, linestyle, alpha, label_suffix in [
            ("exp", "-", 1.0, "Exp."),
            ("sim", "--", 0.95, "Sim."),
        ]:
            s = summary_df[
                (summary_df["source"] == source)
                & (summary_df["load_mpa"] == load)
                & (summary_df["sample_type"] == sample_type)
                & (summary_df["curve_type"] == curve_type)
            ].copy()

            if s.empty:
                continue

            s = s.sort_values("x_um")

            if PLOT_INDIVIDUAL_CURVES:
                indiv = curves_df[
                    (curves_df["source"] == source)
                    & (curves_df["load_mpa"] == load)
                    & (curves_df["sample_type"] == sample_type)
                    & (curves_df["curve_type"] == curve_type)
                ].copy()

                for _, gi in indiv.groupby("replicate_id"):
                    gi = gi.sort_values("x_um")
                    ax.plot(
                        gi["x_um"],
                        gi["y"],
                        color=color,
                        linestyle=linestyle,
                        lw=0.7,
                        alpha=INDIVIDUAL_ALPHA,
                    )

            ax.plot(
                s["x_um"],
                s["median"],
                color=color,
                linestyle=linestyle,
                lw=2.0 if source == "exp" else 1.8,
                alpha=alpha,
                label=f"{load} MPa {label_suffix}",
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


plot_group(
    ax_psd_int,
    interrupted_cases,
    "psd",
    "A. Interrupted: normalized radial PSD",
    r"median normalized radial PSD",
)

plot_group(
    ax_psd_unint,
    uninterrupted_cases,
    "psd",
    "B. Uninterrupted: normalized radial PSD",
    r"median normalized radial PSD",
)

plot_group(
    ax_acf_int,
    interrupted_cases,
    "acf",
    "C. Interrupted: radial ACF",
    r"median radial ACF, $C(r)$",
)

plot_group(
    ax_acf_unint,
    uninterrupted_cases,
    "acf",
    "D. Uninterrupted: radial ACF",
    r"median radial ACF, $C(r)$",
)

for ax in [ax_psd_int, ax_psd_unint]:
    ax.set_xscale("log")
    ax.set_xlim(PLOT_LAMBDA_MIN_UM, PLOT_LAMBDA_MAX_UM)
    ax.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")

    if PSD_YLOG:
        ax.set_yscale("log")

    ax.legend(fontsize=6, ncol=2)

for ax in [ax_acf_int, ax_acf_unint]:
    ax.set_xlim(0.0, ACF_MAX_LAG_UM)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axhline(np.exp(-1.0), color="0.5", lw=0.8, ls=":")
    ax.legend(fontsize=6, ncol=2)

# Numeric wavelength tick labels on log x-axis.
psd_ticks = np.array([3, 4, 5, 6, 10, 20, 50, 100, 128], dtype=float)
psd_ticks = psd_ticks[
    (psd_ticks >= PLOT_LAMBDA_MIN_UM) & (psd_ticks <= PLOT_LAMBDA_MAX_UM)
]

for ax in [ax_psd_int, ax_psd_unint]:
    ax.xaxis.set_major_locator(FixedLocator(psd_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in psd_ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())

fig.suptitle(
    "Endpoint spatial configuration: experiment vs simulation\n"
    "Cached endpoint heights; full-field crop-first leveling; writeup-style PSD and ACF operators",
    y=1.02,
)

fig.tight_layout()

outpath = (
    OUTPUT_DIR / "four_panel_endpoint_exp_vs_sim_cached_heights_writeup_operators.png"
)
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Four-panel endpoint experiment-vs-simulation comparison
# USING CACHED ENDPOINT HEIGHT MAPS ONLY
# =============================================================================
#
# No SimResults.load().
# No experimental CSV rereading.
#
# Requires cached height fields:
#   endpoint_height_fields_metadata.csv
#   endpoint_height_fields_arrays.npz
#
# Calculation alignment with writeup, except NO sectioning:
#   EXP:
#     cached raw cropped endpoint field -> plane-level -> mean-remove
#   SIM:
#     cached endpoint field -> plane-level -> mean-remove
#
#   PSD:
#     C(fx,fy) = dx^2 * |FFT(h*w)|^2 / sum(w^2)
#     radial arithmetic mean in physical-frequency annuli
#     22 logarithmic frequency annuli from 0.00283072 to 0.36 um^-1
#     plot only retained writeup band:
#         4.1264 <= lambda <= 60.8714 um
#     PSD normalized by trapezoidal integral over retained frequency band
#     aggregation: median across specimen endpoint fields / sim faces
#
#   ACF:
#     linear overlap-corrected covariance, not circular FFT ACF
#     normalized by zero-lag covariance
#     radial annuli of one native 10x pixel = 1.379951 um
#     radial lags from 0 to <128 um
#     aggregation: median across specimen endpoint fields / sim faces
#
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.signal import fftconvolve

# =============================================================================
# Settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT_CACHE_METADATA = OUTPUT_DIR / "endpoint_height_fields_metadata.csv"
HEIGHT_CACHE_ARRAYS = OUTPUT_DIR / "endpoint_height_fields_arrays.npz"

# Recompute PSD/ACF curves from cached height arrays.
# This does NOT reload simulation data.
RECOMPUTE_DERIVED_CURVES_FROM_HEIGHT_CACHE = True

CURVES_CACHE = (
    OUTPUT_DIR
    / "endpoint_exp_sim_cached_heights_writeup_bins_fullfield_curves_long.csv"
)
SUMMARY_CACHE = (
    OUTPUT_DIR / "endpoint_exp_sim_cached_heights_writeup_bins_fullfield_summary.csv"
)

# Detrending
EXP_DETREND_ORDER = 1
SIM_DETREND_ORDER = 1

# Native 10x spacing from writeup
EXP_NATIVE_SPACING_UM = 1.379951

# Cases
interrupted_cases = [
    {"load": 475, "sample_type": "int"},
    {"load": 525, "sample_type": "int"},
    {"load": 575, "sample_type": "int"},
]

uninterrupted_cases = [
    {"load": 500, "sample_type": "unint"},
    {"load": 530, "sample_type": "unint"},
    {"load": 588, "sample_type": "unint"},
]

all_cases = interrupted_cases + uninterrupted_cases

LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

# -----------------------------------------------------------------------------
# Writeup-matched PSD settings
# -----------------------------------------------------------------------------
PSD_F_MIN_UM_INV = 0.00283072
PSD_F_MAX_UM_INV = 0.36
N_PSD_ANNULI = 22
MIN_MODES_PER_PSD_BIN = 1

# Retained writeup comparison band
PSD_RETAINED_LAMBDA_MIN_UM = 4.1264
PSD_RETAINED_LAMBDA_MAX_UM = 60.8714

PLOT_LAMBDA_MIN_UM = PSD_RETAINED_LAMBDA_MIN_UM
PLOT_LAMBDA_MAX_UM = PSD_RETAINED_LAMBDA_MAX_UM
PSD_YLOG = True

# -----------------------------------------------------------------------------
# Writeup-matched ACF settings
# -----------------------------------------------------------------------------
ACF_RADIAL_BIN_WIDTH_UM = EXP_NATIVE_SPACING_UM
ACF_MAX_LAG_UM = PSD_RETAINED_LAMBDA_MAX_UM

PLOT_INDIVIDUAL_CURVES = False
INDIVIDUAL_ALPHA = 0.08

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load cached endpoint height arrays
# =============================================================================

if not HEIGHT_CACHE_METADATA.exists():
    raise FileNotFoundError(
        f"Missing cached height metadata:\n{HEIGHT_CACHE_METADATA.resolve()}\n"
        "Build endpoint height cache first."
    )

if not HEIGHT_CACHE_ARRAYS.exists():
    raise FileNotFoundError(
        f"Missing cached height arrays:\n{HEIGHT_CACHE_ARRAYS.resolve()}\n"
        "Build endpoint height cache first."
    )

endpoint_metadata = pd.read_csv(HEIGHT_CACHE_METADATA)
endpoint_arrays = np.load(HEIGHT_CACHE_ARRAYS)

required_metadata_cols = {
    "array_key",
    "source",
    "load_mpa",
    "sample_type",
    "sample_id",
    "replicate_id",
    "bulk_z_strain_percent",
    "spacing_um",
}

missing = required_metadata_cols - set(endpoint_metadata.columns)
if missing:
    raise ValueError(f"Endpoint metadata missing columns: {sorted(missing)}")

endpoint_metadata = endpoint_metadata.copy()
endpoint_metadata["load_mpa"] = pd.to_numeric(
    endpoint_metadata["load_mpa"], errors="coerce"
)
endpoint_metadata["bulk_z_strain_percent"] = pd.to_numeric(
    endpoint_metadata["bulk_z_strain_percent"],
    errors="coerce",
)
endpoint_metadata["spacing_um"] = pd.to_numeric(
    endpoint_metadata["spacing_um"], errors="coerce"
)

endpoint_metadata = endpoint_metadata[
    np.isfinite(endpoint_metadata["load_mpa"])
    & np.isfinite(endpoint_metadata["bulk_z_strain_percent"])
    & np.isfinite(endpoint_metadata["spacing_um"])
].copy()

# Restrict to requested six cases.
case_keys = {(case["load"], case["sample_type"]) for case in all_cases}
endpoint_metadata = endpoint_metadata[
    endpoint_metadata.apply(
        lambda r: (int(r["load_mpa"]), str(r["sample_type"])) in case_keys,
        axis=1,
    )
].copy()

print("Loaded endpoint height cache:")
print(f"  metadata: {HEIGHT_CACHE_METADATA.resolve()}")
print(f"  arrays:   {HEIGHT_CACHE_ARRAYS.resolve()}")
print(f"  rows:     {len(endpoint_metadata)}")
print(endpoint_metadata.groupby(["source", "sample_type", "load_mpa"]).size())


# =============================================================================
# Numerical helpers
# =============================================================================


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x)
    if hasattr(np, "trapz"):
        return np.trapz(y, x=x)

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, col = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = col.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend([x * x, x * y, y * y])
    elif order != 1:
        raise ValueError("detrend order must be 1 or 2")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    design, inverse = _detrend_geometry(
        values.shape, float(spacing_um_value), int(order)
    )
    coeff = inverse @ values.ravel()
    trend = (design @ coeff).reshape(values.shape)

    return values - trend


def preprocess_cached_height(
    H_raw: np.ndarray, spacing_um: float, source: str
) -> np.ndarray:
    """
    Cached experimental arrays are already cropped raw endpoint fields.
    Cached simulation arrays are raw endpoint simulated surfaces.
    Detrending is intentionally done here.
    """
    H = np.asarray(H_raw, dtype=float)

    if not np.all(np.isfinite(H)):
        H = H.copy()
        H[~np.isfinite(H)] = np.nanmean(H)

    source = str(source)

    if source == "exp":
        order = EXP_DETREND_ORDER
    elif source == "sim":
        order = SIM_DETREND_ORDER
    else:
        raise ValueError(f"Unknown source: {source}")

    H = detrend_surface(H, spacing_um, order=order)
    H = H - np.nanmean(H)

    return H


# =============================================================================
# PSD functions: writeup formula and frequency annuli
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    return w0[:, None] * w1[None, :]


def psd2d_height_protocol(height_um: np.ndarray, spacing_um_value: float):
    """
    Protocol PSD:
        C(fx,fy) = dx^2 * |FFT(h*w)|^2 / sum(w^2)
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)
    PSD = (spacing_um_value**2) * np.abs(F) ** 2 / np.sum(W**2)

    n0, n1 = H.shape
    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    return f0, f1, PSD


def make_frequency_edges():
    return np.logspace(
        np.log10(PSD_F_MIN_UM_INV),
        np.log10(PSD_F_MAX_UM_INV),
        N_PSD_ANNULI + 1,
    )


f_edges = make_frequency_edges()


def radial_psd_frequency_binned(
    height_um: np.ndarray,
    spacing_um_value: float,
    f_edges_um_inv: np.ndarray,
    min_modes: int = 1,
):
    f0, f1, PSD = psd2d_height_protocol(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    nbins = len(f_edges_um_inv) - 1
    f_center = np.sqrt(f_edges_um_inv[:-1] * f_edges_um_inv[1:])
    lambda_center = 1.0 / f_center

    radial = np.full(nbins, np.nan, dtype=float)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        lo = f_edges_um_inv[i]
        hi = f_edges_um_inv[i + 1]

        mask = np.isfinite(FR) & np.isfinite(PSD) & (FR >= lo) & (FR < hi)

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            radial[i] = float(np.nanmean(PSD[mask]))

    return f_center, lambda_center, radial, modes


def normalize_psd_over_retained_frequency_band(f_center, lambda_center, psd):
    """
    Normalize by trapezoidal integral over retained writeup band:
        4.1264 <= lambda <= 60.8714 um
    Integration is over frequency.
    """
    f = np.asarray(f_center, dtype=float)
    lam = np.asarray(lambda_center, dtype=float)
    y = np.asarray(psd, dtype=float)

    out = np.full_like(y, np.nan, dtype=float)

    retained = (
        np.isfinite(f)
        & np.isfinite(lam)
        & np.isfinite(y)
        & (f > 0)
        & (lam > 0)
        & (y > 0)
        & (lam >= PSD_RETAINED_LAMBDA_MIN_UM)
        & (lam <= PSD_RETAINED_LAMBDA_MAX_UM)
    )

    if np.count_nonzero(retained) < 2:
        return out

    order = np.argsort(f[retained])
    f_ret = f[retained][order]
    y_ret = y[retained][order]

    area = integrate_trapezoid(y_ret, x=f_ret)

    if not np.isfinite(area) or area <= 0:
        return out

    finite_positive = np.isfinite(y) & (y > 0)
    out[finite_positive] = y[finite_positive] / area

    return out


# =============================================================================
# ACF: overlap-corrected linear covariance and radial averaging
# =============================================================================


def overlap_normalized_acf_2d(z: np.ndarray, spacing_um_value: float):
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)

    if np.count_nonzero(finite) < 10:
        raise ValueError("Too few finite pixels for ACF")

    z0 = np.zeros_like(z, dtype=float)
    z_mean = np.nanmean(z)
    z0[finite] = z[finite] - z_mean

    mask = finite.astype(float)

    numerator = fftconvolve(z0, z0[::-1, ::-1], mode="full")
    overlap_counts = fftconvolve(mask, mask[::-1, ::-1], mode="full")

    covariance = np.full_like(numerator, np.nan, dtype=float)

    valid_overlap = overlap_counts > 0
    covariance[valid_overlap] = numerator[valid_overlap] / overlap_counts[valid_overlap]

    center = tuple(np.array(covariance.shape) // 2)
    zero_lag_cov = covariance[center]

    if not np.isfinite(zero_lag_cov) or zero_lag_cov <= 0:
        rho = np.full_like(covariance, np.nan, dtype=float)
    else:
        rho = covariance / zero_lag_cov

    n0, n1 = z.shape

    lag0_um = np.arange(-(n0 - 1), n0) * spacing_um_value
    lag1_um = np.arange(-(n1 - 1), n1) * spacing_um_value

    return lag0_um, lag1_um, rho


r_edges = np.arange(
    0.0,
    ACF_MAX_LAG_UM + ACF_RADIAL_BIN_WIDTH_UM,
    ACF_RADIAL_BIN_WIDTH_UM,
)


def radial_average_acf(lag0_um, lag1_um, rho, r_edges_um):
    L1, L0 = np.meshgrid(lag1_um, lag0_um)
    R = np.sqrt(L0**2 + L1**2)

    centers = 0.5 * (r_edges_um[:-1] + r_edges_um[1:])
    radial = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)

    valid = np.isfinite(R) & np.isfinite(rho)

    for i in range(len(centers)):
        mask = valid & (R >= r_edges_um[i]) & (R < r_edges_um[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            radial[i] = float(np.nanmean(rho[mask]))

    keep = centers < ACF_MAX_LAG_UM

    return centers[keep], radial[keep], counts[keep]


def radial_acf(height_um, spacing_um_value, r_edges_um):
    lag0_um, lag1_um, rho = overlap_normalized_acf_2d(height_um, spacing_um_value)
    return radial_average_acf(lag0_um, lag1_um, rho, r_edges_um)


# =============================================================================
# Compute derived curves from cached height arrays
# =============================================================================

if (
    RECOMPUTE_DERIVED_CURVES_FROM_HEIGHT_CACHE
    or (not CURVES_CACHE.exists())
    or (not SUMMARY_CACHE.exists())
):
    print("Computing PSD/ACF curves from cached endpoint height arrays.")
    print("No SimResults.load() calls will be made.")

    curve_rows = []

    for _, rec in endpoint_metadata.iterrows():
        try:
            array_key = rec["array_key"]
            H_raw = np.asarray(endpoint_arrays[array_key], dtype=float)

            source = str(rec["source"])
            spacing = float(rec["spacing_um"])

            H = preprocess_cached_height(H_raw, spacing, source)

            f_center, lam, psd_raw, modes = radial_psd_frequency_binned(
                H,
                spacing,
                f_edges,
                min_modes=MIN_MODES_PER_PSD_BIN,
            )

            psd_norm = normalize_psd_over_retained_frequency_band(
                f_center,
                lam,
                psd_raw,
            )

            r, acf, counts = radial_acf(H, spacing, r_edges)

            metadata_common = {
                "source": source,
                "load_mpa": int(rec["load_mpa"]),
                "sample_type": str(rec["sample_type"]),
                "sample_id": str(rec["sample_id"]),
                "replicate_id": str(rec["replicate_id"]),
                "bulk_z_strain_percent": float(rec["bulk_z_strain_percent"]),
            }

            for i in range(len(lam)):
                # Save only retained writeup PSD band.
                if not (
                    np.isfinite(lam[i])
                    and PSD_RETAINED_LAMBDA_MIN_UM
                    <= lam[i]
                    <= PSD_RETAINED_LAMBDA_MAX_UM
                ):
                    continue

                curve_rows.append(
                    {
                        **metadata_common,
                        "curve_type": "psd",
                        "x_um": float(lam[i]),
                        "frequency_um_inv": float(f_center[i]),
                        "y": float(psd_norm[i]) if np.isfinite(psd_norm[i]) else np.nan,
                        "raw_y": (
                            float(psd_raw[i]) if np.isfinite(psd_raw[i]) else np.nan
                        ),
                        "modes_or_counts": int(modes[i]),
                    }
                )

            for i in range(len(r)):
                curve_rows.append(
                    {
                        **metadata_common,
                        "curve_type": "acf",
                        "x_um": float(r[i]),
                        "frequency_um_inv": np.nan,
                        "y": float(acf[i]) if np.isfinite(acf[i]) else np.nan,
                        "raw_y": float(acf[i]) if np.isfinite(acf[i]) else np.nan,
                        "modes_or_counts": int(counts[i]),
                    }
                )

        except Exception as exc:
            warnings.warn(
                f"Failed endpoint curve computation for "
                f"array_key={rec.get('array_key', '?')}, "
                f"source={rec.get('source', '?')}, "
                f"load={rec.get('load_mpa', '?')}, "
                f"sample={rec.get('sample_id', '?')}: {exc}"
            )

    curves_df = pd.DataFrame(curve_rows)

    if curves_df.empty:
        raise RuntimeError(
            "No endpoint curves were computed from cached height arrays."
        )

    curves_df.to_csv(CURVES_CACHE, index=False)

    # Median aggregation across specimens/faces.
    summary_rows = []

    for (source, load, sample_type, curve_type, x_um), g in curves_df.groupby(
        ["source", "load_mpa", "sample_type", "curve_type", "x_um"]
    ):
        y = pd.to_numeric(g["y"], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]

        if y.size == 0:
            continue

        summary_rows.append(
            {
                "source": source,
                "load_mpa": load,
                "sample_type": sample_type,
                "curve_type": curve_type,
                "x_um": x_um,
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
                "sem": (
                    float(np.std(y, ddof=1) / np.sqrt(y.size)) if y.size > 1 else 0.0
                ),
                "n": int(y.size),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CACHE, index=False)

    print(f"Saved: {CURVES_CACHE.resolve()}")
    print(f"Saved: {SUMMARY_CACHE.resolve()}")

else:
    print("Using cached derived PSD/ACF curves.")
    print(f"Curves cache:  {CURVES_CACHE.resolve()}")
    print(f"Summary cache: {SUMMARY_CACHE.resolve()}")

    curves_df = pd.read_csv(CURVES_CACHE)
    summary_df = pd.read_csv(SUMMARY_CACHE)


# =============================================================================
# Plot four-panel figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=False)

ax_psd_int = axes[0, 0]
ax_psd_unint = axes[0, 1]
ax_acf_int = axes[1, 0]
ax_acf_unint = axes[1, 1]


def plot_group(ax, cases, curve_type, title, ylabel):
    for case in cases:
        load = case["load"]
        sample_type = case["sample_type"]
        color = LOAD_COLORS.get(load, "0.3")

        for source, linestyle, alpha, label_suffix in [
            ("exp", "-", 1.0, "Exp."),
            ("sim", "--", 0.95, "Sim."),
        ]:
            s = summary_df[
                (summary_df["source"] == source)
                & (summary_df["load_mpa"] == load)
                & (summary_df["sample_type"] == sample_type)
                & (summary_df["curve_type"] == curve_type)
            ].copy()

            if s.empty:
                continue

            s = s.sort_values("x_um")

            if PLOT_INDIVIDUAL_CURVES:
                indiv = curves_df[
                    (curves_df["source"] == source)
                    & (curves_df["load_mpa"] == load)
                    & (curves_df["sample_type"] == sample_type)
                    & (curves_df["curve_type"] == curve_type)
                ].copy()

                for _, gi in indiv.groupby("replicate_id"):
                    gi = gi.sort_values("x_um")
                    ax.plot(
                        gi["x_um"],
                        gi["y"],
                        color=color,
                        linestyle=linestyle,
                        lw=0.7,
                        alpha=INDIVIDUAL_ALPHA,
                    )

            ax.plot(
                s["x_um"],
                s["median"],
                color=color,
                linestyle=linestyle,
                lw=2.0 if source == "exp" else 1.8,
                alpha=alpha,
                label=f"{load} MPa {label_suffix}",
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


plot_group(
    ax_psd_int,
    interrupted_cases,
    "psd",
    "A. Interrupted: normalized radial PSD",
    r"median normalized radial PSD",
)

plot_group(
    ax_psd_unint,
    uninterrupted_cases,
    "psd",
    "B. Uninterrupted: normalized radial PSD",
    r"median normalized radial PSD",
)

plot_group(
    ax_acf_int,
    interrupted_cases,
    "acf",
    "C. Interrupted: radial ACF",
    r"median radial ACF, $C(r)$",
)

plot_group(
    ax_acf_unint,
    uninterrupted_cases,
    "acf",
    "D. Uninterrupted: radial ACF",
    r"median radial ACF, $C(r)$",
)

for ax in [ax_psd_int, ax_psd_unint]:
    ax.set_xscale("log")
    ax.set_xlim(PLOT_LAMBDA_MIN_UM, PLOT_LAMBDA_MAX_UM)
    ax.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")

    if PSD_YLOG:
        ax.set_yscale("log")

    ax.legend(fontsize=6, ncol=2)

for ax in [ax_acf_int, ax_acf_unint]:
    ax.set_xlim(0.0, ACF_MAX_LAG_UM)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axhline(np.exp(-1.0), color="0.5", lw=0.8, ls=":")
    ax.legend(fontsize=6, ncol=2)

# Numeric wavelength tick labels on log x-axis.
psd_ticks = np.array([4.1264, 5, 10, 20, 30, 40, 50, 60.8714], dtype=float)
psd_ticks = psd_ticks[
    (psd_ticks >= PLOT_LAMBDA_MIN_UM) & (psd_ticks <= PLOT_LAMBDA_MAX_UM)
]

for ax in [ax_psd_int, ax_psd_unint]:
    ax.xaxis.set_major_locator(FixedLocator(psd_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in psd_ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())

fig.suptitle(
    "Endpoint spatial configuration: experiment vs simulation\n"
    "Cached endpoint heights; crop-first leveling; writeup-matched PSD/ACF bins; full-field operator",
    y=1.02,
)

fig.tight_layout()

outpath = (
    OUTPUT_DIR
    / "four_panel_endpoint_exp_vs_sim_cached_heights_writeup_bins_fullfield.png"
)
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()

# %%
# =============================================================================
# Four-panel endpoint experiment-vs-simulation comparison
# USING CACHED ENDPOINT HEIGHT MAPS ONLY
# WITH WRITEUP-STYLE SECTIONING
# =============================================================================
#
# No SimResults.load().
# No experimental CSV rereading.
#
# Requires cached height fields:
#   endpoint_height_fields_metadata.csv
#   endpoint_height_fields_arrays.npz
#
# Experimental operator:
#   cached raw cropped endpoint field
#   -> split into 2x4 sections
#   -> plane-level each section independently
#   -> mean-remove each section
#   -> compute PSD/ACF per section
#   -> specimen curve = median across sections
#
# Simulation operator:
#   cached endpoint simulation field
#   -> split into non-overlapping 128x128 um sections if possible
#   -> plane-level each section independently
#   -> mean-remove each section
#   -> compute PSD/ACF per section
#   -> sim-face curve = median across sections
#
# PSD:
#   C(fx,fy) = dx^2 * |FFT(h*w)|^2 / sum(w^2)
#   radial arithmetic mean in physical-frequency annuli
#   writeup 10x primary annuli:
#       22 logarithmic frequency annuli from 0.00283072 to 0.36 um^-1
#   plot only retained writeup band:
#       4.1264 <= lambda <= 60.8714 um
#   PSD normalized by trapezoidal integral over retained frequency band
#   aggregation: median across specimens / sim faces
#
# ACF:
#   linear overlap-corrected covariance, not circular FFT ACF
#   normalized by zero-lag covariance
#   radial annuli of one native 10x pixel = 1.379951 um
#   radial lags from 0 to <60.8714 um, matching retained upper scale
#   aggregation: median across specimens / sim faces
# =============================================================================

from pathlib import Path
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.signal import fftconvolve

# =============================================================================
# Settings
# =============================================================================

OUTPUT_DIR = Path("roughness_strain_publication_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT_CACHE_METADATA = OUTPUT_DIR / "endpoint_height_fields_metadata.csv"
HEIGHT_CACHE_ARRAYS = OUTPUT_DIR / "endpoint_height_fields_arrays.npz"

# Recompute PSD/ACF curves from cached height arrays.
# This does NOT reload simulation data.
RECOMPUTE_DERIVED_CURVES_FROM_HEIGHT_CACHE = True

CURVES_CACHE = (
    OUTPUT_DIR / "endpoint_exp_sim_cached_heights_writeup_sectioned_curves_long.csv"
)
SUMMARY_CACHE = (
    OUTPUT_DIR / "endpoint_exp_sim_cached_heights_writeup_sectioned_summary.csv"
)

# Native 10x spacing from writeup
EXP_NATIVE_SPACING_UM = 1.379951

# Sectioning
EXP_SECTION_ROWS = 2
EXP_SECTION_COLS = 4

# Simulation section size: 128 um. At 1 um spacing this is 128 pixels.
SIM_SECTION_SIZE_UM = 128.0

# Detrending
EXP_DETREND_ORDER = 1
SIM_DETREND_ORDER = 1

# Cases
interrupted_cases = [
    {"load": 475, "sample_type": "int"},
    {"load": 525, "sample_type": "int"},
    {"load": 575, "sample_type": "int"},
]

uninterrupted_cases = [
    {"load": 500, "sample_type": "unint"},
    {"load": 530, "sample_type": "unint"},
    {"load": 588, "sample_type": "unint"},
]

all_cases = interrupted_cases + uninterrupted_cases

LOAD_COLORS = {
    475: "tab:blue",
    500: "tab:orange",
    525: "tab:green",
    530: "tab:red",
    575: "tab:purple",
    588: "tab:brown",
}

# -----------------------------------------------------------------------------
# Writeup-matched PSD settings
# -----------------------------------------------------------------------------

PSD_F_MIN_UM_INV = 0.00283072
PSD_F_MAX_UM_INV = 0.36
N_PSD_ANNULI = 22
MIN_MODES_PER_PSD_BIN = 1

PSD_RETAINED_LAMBDA_MIN_UM = 4.1264
PSD_RETAINED_LAMBDA_MAX_UM = 60.8714

PLOT_LAMBDA_MIN_UM = PSD_RETAINED_LAMBDA_MIN_UM
PLOT_LAMBDA_MAX_UM = PSD_RETAINED_LAMBDA_MAX_UM
PSD_YLOG = True

# -----------------------------------------------------------------------------
# Writeup-matched ACF settings
# -----------------------------------------------------------------------------

ACF_RADIAL_BIN_WIDTH_UM = EXP_NATIVE_SPACING_UM
ACF_MAX_LAG_UM = PSD_RETAINED_LAMBDA_MAX_UM

PLOT_INDIVIDUAL_CURVES = False
INDIVIDUAL_ALPHA = 0.08

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# =============================================================================
# Load cached endpoint height arrays
# =============================================================================

if not HEIGHT_CACHE_METADATA.exists():
    raise FileNotFoundError(
        f"Missing cached height metadata:\n{HEIGHT_CACHE_METADATA.resolve()}\n"
        "Build endpoint height cache first."
    )

if not HEIGHT_CACHE_ARRAYS.exists():
    raise FileNotFoundError(
        f"Missing cached height arrays:\n{HEIGHT_CACHE_ARRAYS.resolve()}\n"
        "Build endpoint height cache first."
    )

endpoint_metadata = pd.read_csv(HEIGHT_CACHE_METADATA)
endpoint_arrays = np.load(HEIGHT_CACHE_ARRAYS)

required_metadata_cols = {
    "array_key",
    "source",
    "load_mpa",
    "sample_type",
    "sample_id",
    "replicate_id",
    "bulk_z_strain_percent",
    "spacing_um",
}

missing = required_metadata_cols - set(endpoint_metadata.columns)
if missing:
    raise ValueError(f"Endpoint metadata missing columns: {sorted(missing)}")

endpoint_metadata = endpoint_metadata.copy()
endpoint_metadata["load_mpa"] = pd.to_numeric(
    endpoint_metadata["load_mpa"], errors="coerce"
)
endpoint_metadata["bulk_z_strain_percent"] = pd.to_numeric(
    endpoint_metadata["bulk_z_strain_percent"],
    errors="coerce",
)
endpoint_metadata["spacing_um"] = pd.to_numeric(
    endpoint_metadata["spacing_um"], errors="coerce"
)

endpoint_metadata = endpoint_metadata[
    np.isfinite(endpoint_metadata["load_mpa"])
    & np.isfinite(endpoint_metadata["bulk_z_strain_percent"])
    & np.isfinite(endpoint_metadata["spacing_um"])
].copy()

case_keys = {(case["load"], case["sample_type"]) for case in all_cases}

endpoint_metadata = endpoint_metadata[
    endpoint_metadata.apply(
        lambda r: (int(r["load_mpa"]), str(r["sample_type"])) in case_keys,
        axis=1,
    )
].copy()

print("Loaded endpoint height cache:")
print(f"  metadata: {HEIGHT_CACHE_METADATA.resolve()}")
print(f"  arrays:   {HEIGHT_CACHE_ARRAYS.resolve()}")
print(f"  rows:     {len(endpoint_metadata)}")
print(endpoint_metadata.groupby(["source", "sample_type", "load_mpa"]).size())


# =============================================================================
# Numerical helpers
# =============================================================================


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x)
    if hasattr(np, "trapz"):
        return np.trapz(y, x=x)

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


@lru_cache(maxsize=256)
def _detrend_geometry(shape: tuple[int, int], spacing_um_value: float, order: int):
    row, col = np.indices(shape, dtype=np.float64)

    x = row.ravel() * float(spacing_um_value)
    y = col.ravel() * float(spacing_um_value)

    terms = [np.ones(row.size), x, y]

    if order == 2:
        terms.extend([x * x, x * y, y * y])
    elif order != 1:
        raise ValueError("detrend order must be 1 or 2")

    design = np.column_stack(terms)
    inverse = np.linalg.pinv(design)

    return design, inverse


def detrend_surface(
    values: np.ndarray, spacing_um_value: float, order: int = 1
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    design, inverse = _detrend_geometry(
        values.shape, float(spacing_um_value), int(order)
    )
    coeff = inverse @ values.ravel()
    trend = (design @ coeff).reshape(values.shape)

    return values - trend


def prepare_section(
    section_raw: np.ndarray, spacing_um: float, order: int
) -> np.ndarray:
    H = np.asarray(section_raw, dtype=float)

    if not np.all(np.isfinite(H)):
        H = H.copy()
        H[~np.isfinite(H)] = np.nanmean(H)

    H = detrend_surface(H, spacing_um, order=order)
    H = H - np.nanmean(H)

    return H


def split_exp_2x4_sections(H_raw: np.ndarray):
    """
    Experimental writeup sectioning:
      cached experimental height field is already cropped.
      split into 2x4 grid.
    """
    H = np.asarray(H_raw, dtype=float)
    n0, n1 = H.shape

    row_edges = np.linspace(0, n0, EXP_SECTION_ROWS + 1, dtype=int)
    col_edges = np.linspace(0, n1, EXP_SECTION_COLS + 1, dtype=int)

    sections = []

    for i in range(EXP_SECTION_ROWS):
        for j in range(EXP_SECTION_COLS):
            sec = H[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]]
            sections.append((f"sec{i}_{j}", sec))

    return sections


def split_sim_sections(H_raw: np.ndarray, spacing_um: float):
    """
    Simulation comparison sectioning:
      split into non-overlapping sections with physical size approximately
      SIM_SECTION_SIZE_UM x SIM_SECTION_SIZE_UM.

    If the simulation surface is already <= requested section size in either
    dimension, use the full surface as one section.
    """
    H = np.asarray(H_raw, dtype=float)
    n0, n1 = H.shape

    section_px = int(round(SIM_SECTION_SIZE_UM / spacing_um))
    section_px = max(section_px, 1)

    if n0 < section_px or n1 < section_px:
        return [("sec0_0", H)]

    nrow = n0 // section_px
    ncol = n1 // section_px

    if nrow < 1 or ncol < 1:
        return [("sec0_0", H)]

    used0 = nrow * section_px
    used1 = ncol * section_px

    start0 = (n0 - used0) // 2
    start1 = (n1 - used1) // 2

    sections = []

    for i in range(nrow):
        for j in range(ncol):
            r0 = start0 + i * section_px
            r1 = r0 + section_px
            c0 = start1 + j * section_px
            c1 = c0 + section_px

            sec = H[r0:r1, c0:c1]
            sections.append((f"sec{i}_{j}", sec))

    return sections


def get_sections_for_record(H_raw: np.ndarray, spacing_um: float, source: str):
    source = str(source)

    if source == "exp":
        raw_sections = split_exp_2x4_sections(H_raw)
        order = EXP_DETREND_ORDER
    elif source == "sim":
        raw_sections = split_sim_sections(H_raw, spacing_um)
        order = SIM_DETREND_ORDER
    else:
        raise ValueError(f"Unknown source: {source}")

    sections = []

    for section_id, sec_raw in raw_sections:
        sec = prepare_section(sec_raw, spacing_um, order=order)
        sections.append((section_id, sec))

    return sections


# =============================================================================
# PSD functions
# =============================================================================


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    n0, n1 = shape
    w0 = np.hanning(n0)
    w1 = np.hanning(n1)
    return w0[:, None] * w1[None, :]


def psd2d_height_protocol(height_um: np.ndarray, spacing_um_value: float):
    """
    Protocol PSD:
        C(fx,fy) = dx^2 * |FFT(h*w)|^2 / sum(w^2)
    """
    H = np.asarray(height_um, dtype=float)
    H = H - np.nanmean(H)

    W = hann2d(H.shape)
    Hw = H * W

    F = np.fft.fft2(Hw)
    PSD = (spacing_um_value**2) * np.abs(F) ** 2 / np.sum(W**2)

    n0, n1 = H.shape
    f0 = np.fft.fftfreq(n0, d=spacing_um_value)
    f1 = np.fft.fftfreq(n1, d=spacing_um_value)

    return f0, f1, PSD


def make_frequency_edges():
    return np.logspace(
        np.log10(PSD_F_MIN_UM_INV),
        np.log10(PSD_F_MAX_UM_INV),
        N_PSD_ANNULI + 1,
    )


f_edges = make_frequency_edges()


def radial_psd_frequency_binned(
    height_um: np.ndarray,
    spacing_um_value: float,
    f_edges_um_inv: np.ndarray,
    min_modes: int = 1,
):
    f0, f1, PSD = psd2d_height_protocol(height_um, spacing_um_value)

    F1, F0 = np.meshgrid(f1, f0)
    FR = np.sqrt(F0**2 + F1**2)

    nbins = len(f_edges_um_inv) - 1
    f_center = np.sqrt(f_edges_um_inv[:-1] * f_edges_um_inv[1:])
    lambda_center = 1.0 / f_center

    radial = np.full(nbins, np.nan, dtype=float)
    modes = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        lo = f_edges_um_inv[i]
        hi = f_edges_um_inv[i + 1]

        mask = np.isfinite(FR) & np.isfinite(PSD) & (FR >= lo) & (FR < hi)

        modes[i] = int(np.count_nonzero(mask))

        if modes[i] >= min_modes:
            radial[i] = float(np.nanmean(PSD[mask]))

    return f_center, lambda_center, radial, modes


def normalize_psd_over_retained_frequency_band(f_center, lambda_center, psd):
    """
    Normalize by trapezoidal integral over retained writeup band:
        4.1264 <= lambda <= 60.8714 um
    Integration is over frequency.
    """
    f = np.asarray(f_center, dtype=float)
    lam = np.asarray(lambda_center, dtype=float)
    y = np.asarray(psd, dtype=float)

    out = np.full_like(y, np.nan, dtype=float)

    retained = (
        np.isfinite(f)
        & np.isfinite(lam)
        & np.isfinite(y)
        & (f > 0)
        & (lam > 0)
        & (y > 0)
        & (lam >= PSD_RETAINED_LAMBDA_MIN_UM)
        & (lam <= PSD_RETAINED_LAMBDA_MAX_UM)
    )

    if np.count_nonzero(retained) < 2:
        return out

    order = np.argsort(f[retained])
    f_ret = f[retained][order]
    y_ret = y[retained][order]

    area = integrate_trapezoid(y_ret, x=f_ret)

    if not np.isfinite(area) or area <= 0:
        return out

    finite_positive = np.isfinite(y) & (y > 0)
    out[finite_positive] = y[finite_positive] / area

    return out


# =============================================================================
# ACF functions
# =============================================================================


def overlap_normalized_acf_2d(z: np.ndarray, spacing_um_value: float):
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)

    if np.count_nonzero(finite) < 10:
        raise ValueError("Too few finite pixels for ACF")

    z0 = np.zeros_like(z, dtype=float)
    z_mean = np.nanmean(z)
    z0[finite] = z[finite] - z_mean

    mask = finite.astype(float)

    numerator = fftconvolve(z0, z0[::-1, ::-1], mode="full")
    overlap_counts = fftconvolve(mask, mask[::-1, ::-1], mode="full")

    covariance = np.full_like(numerator, np.nan, dtype=float)

    valid_overlap = overlap_counts > 0
    covariance[valid_overlap] = numerator[valid_overlap] / overlap_counts[valid_overlap]

    center = tuple(np.array(covariance.shape) // 2)
    zero_lag_cov = covariance[center]

    if not np.isfinite(zero_lag_cov) or zero_lag_cov <= 0:
        rho = np.full_like(covariance, np.nan, dtype=float)
    else:
        rho = covariance / zero_lag_cov

    n0, n1 = z.shape
    lag0_um = np.arange(-(n0 - 1), n0) * spacing_um_value
    lag1_um = np.arange(-(n1 - 1), n1) * spacing_um_value

    return lag0_um, lag1_um, rho


r_edges = np.arange(
    0.0,
    ACF_MAX_LAG_UM + ACF_RADIAL_BIN_WIDTH_UM,
    ACF_RADIAL_BIN_WIDTH_UM,
)


def radial_average_acf(lag0_um, lag1_um, rho, r_edges_um):
    L1, L0 = np.meshgrid(lag1_um, lag0_um)
    R = np.sqrt(L0**2 + L1**2)

    centers = 0.5 * (r_edges_um[:-1] + r_edges_um[1:])
    radial = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)

    valid = np.isfinite(R) & np.isfinite(rho)

    for i in range(len(centers)):
        mask = valid & (R >= r_edges_um[i]) & (R < r_edges_um[i + 1])
        counts[i] = int(np.count_nonzero(mask))

        if counts[i] > 0:
            radial[i] = float(np.nanmean(rho[mask]))

    keep = centers < ACF_MAX_LAG_UM

    return centers[keep], radial[keep], counts[keep]


def radial_acf(height_um, spacing_um_value, r_edges_um):
    lag0_um, lag1_um, rho = overlap_normalized_acf_2d(height_um, spacing_um_value)
    return radial_average_acf(lag0_um, lag1_um, rho, r_edges_um)


# =============================================================================
# Compute sectioned endpoint curves from cached height arrays
# =============================================================================

if (
    RECOMPUTE_DERIVED_CURVES_FROM_HEIGHT_CACHE
    or (not CURVES_CACHE.exists())
    or (not SUMMARY_CACHE.exists())
):
    print("Computing sectioned PSD/ACF curves from cached endpoint height arrays.")
    print("No SimResults.load() calls will be made.")

    curve_rows = []

    for _, rec in endpoint_metadata.iterrows():
        try:
            array_key = rec["array_key"]
            H_raw = np.asarray(endpoint_arrays[array_key], dtype=float)

            source = str(rec["source"])
            spacing = float(rec["spacing_um"])

            sections = get_sections_for_record(H_raw, spacing, source)

            psd_section_curves = []
            acf_section_curves = []
            psd_meta = None
            acf_meta = None

            for section_id, H_sec in sections:
                f_center, lam, psd_raw, modes = radial_psd_frequency_binned(
                    H_sec,
                    spacing,
                    f_edges,
                    min_modes=MIN_MODES_PER_PSD_BIN,
                )

                psd_norm = normalize_psd_over_retained_frequency_band(
                    f_center,
                    lam,
                    psd_raw,
                )

                psd_section_curves.append(psd_norm)

                if psd_meta is None:
                    psd_meta = {
                        "f_center": f_center,
                        "lambda_center": lam,
                        "modes": modes,
                    }

                r, acf, counts = radial_acf(H_sec, spacing, r_edges)
                acf_section_curves.append(acf)

                if acf_meta is None:
                    acf_meta = {
                        "r": r,
                        "counts": counts,
                    }

            if len(psd_section_curves) == 0 or len(acf_section_curves) == 0:
                continue

            psd_section_curves = np.vstack(psd_section_curves)
            acf_section_curves = np.vstack(acf_section_curves)

            # Writeup aggregation: median across fields/sections.
            psd_specimen_curve = np.nanmedian(psd_section_curves, axis=0)
            acf_specimen_curve = np.nanmedian(acf_section_curves, axis=0)

            metadata_common = {
                "source": source,
                "load_mpa": int(rec["load_mpa"]),
                "sample_type": str(rec["sample_type"]),
                "sample_id": str(rec["sample_id"]),
                "replicate_id": str(rec["replicate_id"]),
                "bulk_z_strain_percent": float(rec["bulk_z_strain_percent"]),
                "n_sections": int(len(sections)),
            }

            f_center = psd_meta["f_center"]
            lam = psd_meta["lambda_center"]
            modes = psd_meta["modes"]

            for i in range(len(lam)):
                if not (
                    np.isfinite(lam[i])
                    and PSD_RETAINED_LAMBDA_MIN_UM
                    <= lam[i]
                    <= PSD_RETAINED_LAMBDA_MAX_UM
                ):
                    continue

                curve_rows.append(
                    {
                        **metadata_common,
                        "curve_type": "psd",
                        "x_um": float(lam[i]),
                        "frequency_um_inv": float(f_center[i]),
                        "y": (
                            float(psd_specimen_curve[i])
                            if np.isfinite(psd_specimen_curve[i])
                            else np.nan
                        ),
                        "raw_y": np.nan,
                        "modes_or_counts": int(modes[i]),
                    }
                )

            r = acf_meta["r"]
            counts = acf_meta["counts"]

            for i in range(len(r)):
                curve_rows.append(
                    {
                        **metadata_common,
                        "curve_type": "acf",
                        "x_um": float(r[i]),
                        "frequency_um_inv": np.nan,
                        "y": (
                            float(acf_specimen_curve[i])
                            if np.isfinite(acf_specimen_curve[i])
                            else np.nan
                        ),
                        "raw_y": (
                            float(acf_specimen_curve[i])
                            if np.isfinite(acf_specimen_curve[i])
                            else np.nan
                        ),
                        "modes_or_counts": int(counts[i]),
                    }
                )

        except Exception as exc:
            warnings.warn(
                f"Failed sectioned curve computation for "
                f"array_key={rec.get('array_key', '?')}, "
                f"source={rec.get('source', '?')}, "
                f"load={rec.get('load_mpa', '?')}, "
                f"sample={rec.get('sample_id', '?')}: {exc}"
            )

    curves_df = pd.DataFrame(curve_rows)

    if curves_df.empty:
        raise RuntimeError(
            "No endpoint curves were computed from cached height arrays."
        )

    curves_df.to_csv(CURVES_CACHE, index=False)

    summary_rows = []

    for (source, load, sample_type, curve_type, x_um), g in curves_df.groupby(
        ["source", "load_mpa", "sample_type", "curve_type", "x_um"]
    ):
        y = pd.to_numeric(g["y"], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]

        if y.size == 0:
            continue

        summary_rows.append(
            {
                "source": source,
                "load_mpa": load,
                "sample_type": sample_type,
                "curve_type": curve_type,
                "x_um": x_um,
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
                "sem": (
                    float(np.std(y, ddof=1) / np.sqrt(y.size)) if y.size > 1 else 0.0
                ),
                "n": int(y.size),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CACHE, index=False)

    print(f"Saved: {CURVES_CACHE.resolve()}")
    print(f"Saved: {SUMMARY_CACHE.resolve()}")

else:
    print("Using cached derived PSD/ACF curves.")
    print(f"Curves cache:  {CURVES_CACHE.resolve()}")
    print(f"Summary cache: {SUMMARY_CACHE.resolve()}")

    curves_df = pd.read_csv(CURVES_CACHE)
    summary_df = pd.read_csv(SUMMARY_CACHE)


# =============================================================================
# Plot four-panel figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=False)

ax_psd_int = axes[0, 0]
ax_psd_unint = axes[0, 1]
ax_acf_int = axes[1, 0]
ax_acf_unint = axes[1, 1]


def plot_group(ax, cases, curve_type, title, ylabel):
    for case in cases:
        load = case["load"]
        sample_type = case["sample_type"]
        color = LOAD_COLORS.get(load, "0.3")

        for source, linestyle, alpha, label_suffix in [
            ("exp", "-", 1.0, "Exp."),
            ("sim", "--", 0.95, "Sim."),
        ]:
            s = summary_df[
                (summary_df["source"] == source)
                & (summary_df["load_mpa"] == load)
                & (summary_df["sample_type"] == sample_type)
                & (summary_df["curve_type"] == curve_type)
            ].copy()

            if s.empty:
                continue

            s = s.sort_values("x_um")

            if PLOT_INDIVIDUAL_CURVES:
                indiv = curves_df[
                    (curves_df["source"] == source)
                    & (curves_df["load_mpa"] == load)
                    & (curves_df["sample_type"] == sample_type)
                    & (curves_df["curve_type"] == curve_type)
                ].copy()

                for _, gi in indiv.groupby("replicate_id"):
                    gi = gi.sort_values("x_um")
                    ax.plot(
                        gi["x_um"],
                        gi["y"],
                        color=color,
                        linestyle=linestyle,
                        lw=0.7,
                        alpha=INDIVIDUAL_ALPHA,
                    )

            ax.plot(
                s["x_um"],
                s["median"],
                color=color,
                linestyle=linestyle,
                lw=2.0 if source == "exp" else 1.8,
                alpha=alpha,
                label=f"{load} MPa {label_suffix}",
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


plot_group(
    ax_psd_int,
    interrupted_cases,
    "psd",
    "A. Interrupted: normalized radial PSD",
    r"median normalized radial PSD",
)

plot_group(
    ax_psd_unint,
    uninterrupted_cases,
    "psd",
    "B. Uninterrupted: normalized radial PSD",
    r"median normalized radial PSD",
)

plot_group(
    ax_acf_int,
    interrupted_cases,
    "acf",
    "C. Interrupted: radial ACF",
    r"median radial ACF, $C(r)$",
)

plot_group(
    ax_acf_unint,
    uninterrupted_cases,
    "acf",
    "D. Uninterrupted: radial ACF",
    r"median radial ACF, $C(r)$",
)

for ax in [ax_psd_int, ax_psd_unint]:
    ax.set_xscale("log")
    ax.set_xlim(PLOT_LAMBDA_MIN_UM, PLOT_LAMBDA_MAX_UM)
    ax.set_xlabel(r"wavelength, $\lambda$ [$\mu$m]")

    if PSD_YLOG:
        ax.set_yscale("log")

    ax.legend(fontsize=6, ncol=2)

for ax in [ax_acf_int, ax_acf_unint]:
    ax.set_xlim(0.0, ACF_MAX_LAG_UM)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axhline(np.exp(-1.0), color="0.5", lw=0.8, ls=":")
    ax.legend(fontsize=6, ncol=2)

psd_ticks = np.array([4.1264, 5, 10, 20, 30, 40, 50, 60.8714], dtype=float)
psd_ticks = psd_ticks[
    (psd_ticks >= PLOT_LAMBDA_MIN_UM) & (psd_ticks <= PLOT_LAMBDA_MAX_UM)
]

for ax in [ax_psd_int, ax_psd_unint]:
    ax.xaxis.set_major_locator(FixedLocator(psd_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in psd_ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())

fig.suptitle(
    "Endpoint spatial configuration: experiment vs simulation\n"
    "Cached endpoint heights; writeup-style sectioning; PSD/ACF bins matched to retained band",
    y=1.02,
)

fig.tight_layout()

outpath = (
    OUTPUT_DIR / "four_panel_endpoint_exp_vs_sim_cached_heights_writeup_sectioned.png"
)
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath.resolve()}")

plt.show()


