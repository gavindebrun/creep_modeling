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
                        "bulk_z_strain_percent": float(record.bulk_z_strain_percent),
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
                        "bulk_z_strain_percent": float(record.bulk_z_strain_percent),
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
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, np.ndarray]]:
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
    grouped, labels, colors = positive_strain_groups(observations)
    keys = ["specimen_id", "load_mpa", "sample_type", "sample", "time_h"]
    psd_grouped = psd_long.merge(grouped[keys + ["strain_group"]], on=keys, how="inner")
    acf_grouped = acf_long.merge(grouped[keys + ["strain_group"]], on=keys, how="inner")

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
            acf_rows.append(
                {
                    "strain_group": strain_group,
                    "r_um": lag,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "err": error_from_values(values, "sem"),
                    "n": int(values.size),
                }
            )
    psd_summary = pd.DataFrame(psd_rows)
    acf_summary = pd.DataFrame(acf_rows)
    for frame in (psd_summary, acf_summary):
        frame["strain_group"] = pd.Categorical(
            frame["strain_group"], categories=labels, ordered=True
        )
    psd_summary = psd_summary.sort_values(["strain_group", "lambda_center_um"])
    acf_summary = acf_summary.sort_values(["strain_group", "r_um"])
    return psd_summary, acf_summary, labels, colors


def plot_bracketed_psd_gain_and_acf_change(
    psd_summary: pd.DataFrame,
    acf_summary: pd.DataFrame,
    labels: Sequence[str],
    colors: Mapping[str, np.ndarray],
    *,
    save: bool = True,
) -> plt.Figure:
    """Exact corrected A/B presentation from figures.py:23737--23807."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax_psd, ax_acf = axes
    for label in labels:
        selected = psd_summary[psd_summary["strain_group"] == label].sort_values(
            "lambda_center_um"
        )
        ax_psd.errorbar(
            selected["lambda_center_um"],
            selected["mean"],
            yerr=selected["err"],
            fmt="o-",
            lw=1.8,
            ms=4.5,
            capsize=3,
            color=colors[label],
            label=rf"$\epsilon_{{zz}}$ = {label}",
        )
    ax_psd.axhline(0.0, color="0.45", lw=1.0, ls="--")
    ax_psd.set_xlabel(r"wavelength bracket center, $\lambda$ [$\mu$m]")
    ax_psd.set_ylabel(r"mean $\log_{10}$ PSD gain")
    ax_psd.set_title("A. Bracketed PSD gain")
    ax_psd.grid(True, alpha=0.25)
    ax_psd.legend(fontsize=7)

    for label in labels:
        selected = acf_summary[acf_summary["strain_group"] == label].sort_values("r_um")
        ax_acf.plot(
            selected["r_um"],
            selected["mean"],
            lw=2.0,
            color=colors[label],
            label=rf"$\epsilon_{{zz}}$ = {label}",
        )
        ax_acf.fill_between(
            selected["r_um"],
            selected["mean"] - selected["err"],
            selected["mean"] + selected["err"],
            color=colors[label],
            alpha=0.15,
            linewidth=0,
        )
    ax_acf.axhline(0.0, color="0.45", lw=1.0, ls="--")
    ax_acf.set_xlim(0.0, ACF_CHANGE_MAX_LAG_UM)
    ax_acf.set_xlabel(r"radial lag, $r$ [$\mu$m]")
    ax_acf.set_ylabel(r"mean ACF change, $\Delta C(r)=C(r;\epsilon)-C(r;0)$")
    ax_acf.set_title("B. Corrected radial ACF change")
    ax_acf.grid(True, alpha=0.25)
    ax_acf.legend(fontsize=7)
    fig.tight_layout()
    if save:
        fig.savefig(
            OUTPUT_DIR / "two_panel_corrected_psd_gain_and_acf_change_by_strain.png",
            bbox_inches="tight",
        )
    return fig
