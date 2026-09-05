"""Figures for the meta-benchmark sweeps in `msmetrics.meta`.

One plot per summary, so the picture and the number can never disagree: `response` for
`sensitivity` and `response_shape`, `null` for `null_control`, `specificity` for `specificity`, and
`scorecard` for reading a whole panel of metrics at once.

Every function takes the tidy frame from `msmetrics.meta.sweep`, accepts an existing axes, returns
what it drew, and never calls `show`, so the figures compose into a notebook or a report without
fighting it.

.. code-block:: python

    from msmetrics import plotting as pl

    pl.response(curve)
    pl.null(curve, dose=1.0)
    pl.scorecard(curve)
    pl.specificity(curve, signal="dilute", nuisance="loading")
"""

import matplotlib.pyplot as plt
import numpy as np

from msmetrics import meta

__all__ = ["null", "response", "scorecard", "specificity"]


def _panels(n: int, axes, width: float = 4.0, height: float = 3.2):
    """Return `n` axes, creating a one-row figure if none were supplied."""
    if axes is not None:
        axes = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
        if axes.size < n:
            raise ValueError(f"need {n} axes, got {axes.size}.")
        return axes[:n]
    _, created = plt.subplots(1, n, figsize=(width * n, height), squeeze=False)
    return created.ravel()


def response(curve, *, metrics=None, perturbations=None, n_sd=2.0, axes=None):
    """Dose-response curves, one panel per perturbation and one line per metric.

    The reference-dose noise band and the detection dose are drawn onto the same panel as the curve,
    so sensitivity, monotonicity, dynamic range and saturation are all read off one figure rather
    than cross-checked between four tables. A metric that saturates shows it by flattening; a metric
    with no dynamic range shows it by never leaving the band.

    Parameters
    ----------
    curve
        Frame returned by `msmetrics.meta.sweep`.
    metrics, perturbations
        Optional subsets to draw, in the order given. Defaults to everything in the frame.
    n_sd
        Half-width of the shaded reference band, in reference standard deviations. Matches the
        `n_sd` of `msmetrics.meta.sensitivity`, whose detection dose is marked.
    axes
        Existing axes to draw into, one per perturbation.

    Returns
    -------
    numpy.ndarray
        The axes drawn into, one per perturbation.
    """
    metrics = list(metrics) if metrics is not None else list(dict.fromkeys(curve["metric"]))
    perturbations = list(perturbations) if perturbations is not None else list(dict.fromkeys(curve["perturbation"]))

    noise = meta.reference_noise(curve).set_index("metric")
    thresholds = meta.sensitivity(curve, n_sd=n_sd).groupby(["perturbation", "metric"])["detection_dose"].first()
    drawn = _panels(len(perturbations), axes)

    for ax, perturbation in zip(drawn, perturbations, strict=False):
        panel = curve[curve["perturbation"] == perturbation]
        for metric in metrics:
            series = panel[panel["metric"] == metric]
            if series.empty:
                continue
            grouped = series.groupby("dose", sort=True)["value"]
            doses, mean, sd = grouped.mean().index.to_numpy(float), grouped.mean().to_numpy(), grouped.std(ddof=1)

            (line,) = ax.plot(doses, mean, marker="o", label=metric, zorder=3)
            colour = line.get_color()
            ax.fill_between(doses, mean - sd, mean + sd, color=colour, alpha=0.2, lw=0, zorder=2)
            ax.scatter(series["dose"], series["value"], s=6, color=colour, alpha=0.35, zorder=1)

            if metric in noise.index:
                centre, spread = float(noise.loc[metric, "mean_0"]), float(noise.loc[metric, "sd_0"])
                if np.isfinite(spread):
                    ax.axhspan(centre - n_sd * spread, centre + n_sd * spread, color=colour, alpha=0.07, lw=0)

            detected = thresholds.get((perturbation, metric), float("nan"))
            if np.isfinite(detected):
                ax.axvline(detected, color=colour, ls=":", lw=1)

        ax.set_title(perturbation)
        ax.set_xlabel(_dose_label(curve, perturbation))
        ax.set_ylabel("metric value")

    drawn[0].legend(frameon=False, fontsize="small")
    return drawn


def _dose_label(curve, perturbation) -> str:
    units = curve.loc[curve["perturbation"] == perturbation, "dose_unit"]
    unit = units.iloc[0] if len(units) else ""
    return f"dose ({unit})" if unit else "dose"


def null(curve, *, dose=1.0, perturbation=None, bins=20, axes=None):
    """Each metric's null distribution, with its real-label value marked.

    This is the figure that answers whether a metric is doing anything at all: if the observed value
    sits inside the histogram of values obtained after the labels were destroyed, the metric is not
    reading the structure it claims to read.

    Parameters
    ----------
    curve
        Frame returned by `msmetrics.meta.sweep`, normally over `PermuteLabels`.
    dose
        Dose treated as the null.
    perturbation
        Which perturbation to draw. Defaults to the first in the frame.
    bins
        Histogram bins.
    axes
        Existing axes to draw into, one per metric.

    Returns
    -------
    numpy.ndarray
        The axes drawn into, one per metric.
    """
    perturbation = perturbation if perturbation is not None else curve["perturbation"].iloc[0]
    panel = curve[curve["perturbation"] == perturbation]
    summary = meta.null_control(panel, dose=dose).set_index("metric")

    metrics = list(dict.fromkeys(panel["metric"]))
    drawn = _panels(len(metrics), axes)

    for ax, metric in zip(drawn, metrics, strict=False):
        values = panel.loc[np.isclose(panel["dose"], dose) & (panel["metric"] == metric), "value"].dropna()
        ax.hist(values, bins=bins, color="0.7", edgecolor="white")

        if metric in summary.index:
            row = summary.loc[metric]
            ax.axvline(row["observed"], color="crimson", lw=2)
            ax.set_title(f"{metric}\np = {row['p_value']:.3g}, z = {row['z']:.2f}", fontsize="medium")
        else:
            ax.set_title(metric)

        ax.set_xlabel("metric value")
        ax.set_ylabel("null replicates")

    return drawn


def scorecard(curve, *, statistic="range_over_noise", ax=None, cmap=None):
    """Heatmap of one summary statistic across every metric and perturbation.

    Parameters
    ----------
    curve
        Frame returned by `msmetrics.meta.sweep`.
    statistic
        Column of `msmetrics.meta.response_shape` (for example `"range_over_noise"`, `"spearman"`,
        `"saturation_dose"`) or `"detection_dose"` from `msmetrics.meta.sensitivity`.
    ax
        Existing axes to draw into.
    cmap
        Colormap. Defaults to a diverging map centred at zero for signed statistics and a sequential
        one otherwise.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.

    Raises
    ------
    KeyError
        If `statistic` is not produced by either summary.
    """
    if statistic == "detection_dose":
        summary = meta.sensitivity(curve).groupby(["perturbation", "metric"], as_index=False)[statistic].first()
    else:
        summary = meta.response_shape(curve)
        if statistic not in summary:
            raise KeyError(f"`{statistic}` is not a column of `response_shape`; it has {sorted(summary.columns)}.")

    table = summary.pivot(index="metric", columns="perturbation", values=statistic)
    signed = bool(np.nanmin(table.to_numpy(float)) < 0) if table.notna().any().any() else False
    limit = np.nanmax(np.abs(table.to_numpy(float))) if table.notna().any().any() else 1.0

    ax = ax if ax is not None else plt.subplots(figsize=(1.4 * table.shape[1] + 2, 0.6 * table.shape[0] + 2))[1]
    image = ax.imshow(
        table.to_numpy(float),
        cmap=cmap or ("RdBu_r" if signed else "viridis"),
        vmin=-limit if signed else None,
        vmax=limit if signed else None,
        aspect="auto",
    )

    ax.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_title(statistic)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            cell = table.to_numpy(float)[i, j]
            if np.isfinite(cell):
                ax.text(j, i, f"{cell:.2g}", ha="center", va="center", fontsize="small")
    ax.figure.colorbar(image, ax=ax, label=statistic)
    return ax


def specificity(curve, *, signal, nuisance, ax=None):
    """Signal response against nuisance response, one point per metric.

    Both slopes are plotted rather than their contrast, because a metric lands on the diagonal either
    by responding to everything or by responding to nothing, and only seeing both coordinates tells
    those two apart. Points above the diagonal track biology more strongly than the confound.

    Parameters
    ----------
    curve
        Frame returned by `msmetrics.meta.sweep`, containing both perturbations.
    signal, nuisance
        Names of the biological and the technical perturbation.
    ax
        Existing axes to draw into.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    summary = meta.specificity(curve, signal=signal, nuisance=nuisance)
    ax = ax if ax is not None else plt.subplots(figsize=(4.5, 4.5))[1]

    x = np.abs(summary["nuisance_slope"].to_numpy(float))
    y = np.abs(summary["signal_slope"].to_numpy(float))
    ax.scatter(x, y, s=40, zorder=3)
    for name, xi, yi in zip(summary["metric"], x, y, strict=False):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize="small")

    finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    top = float(finite.max()) * 1.15 if finite.size else 1.0
    ax.plot([0, top], [0, top], color="0.6", ls="--", lw=1, zorder=1)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel(f"|{nuisance}| response (reference SD per unit dose)")
    ax.set_ylabel(f"|{signal}| response (reference SD per unit dose)")
    ax.set_title("above the diagonal: tracks biology")
    return ax
