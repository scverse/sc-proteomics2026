"""Meta-benchmarking: measuring whether a metric behaves, rather than measuring data.

A metric that reports a number for every dataset is not thereby useful. Before trusting one, we want
evidence that it collapses when the signal it claims to measure is destroyed, that it responds
monotonically as that signal is titrated back, that it has dynamic range worth reading rather than
saturating immediately, and that it tracks biology rather than cell size or imputation strength.

Those are four readings of one experiment: perturb the data at increasing dose, recompute the metric,
and look at the resulting curve. `sweep` runs that experiment and returns it as a tidy frame; the
summary functions read the frame four ways.

.. code-block:: python

    from msmetrics import meta
    from msmetrics import perturbations as pert

    curve = meta.sweep(
        adata,
        metrics={"ari": my_ari, "ilisi": my_ilisi},
        perturbations={
            "permute_ct": pert.PermuteLabels("cell_type", stratify_by="batch"),
            "dilute": pert.DiluteSignal("cell_type"),
            "loading": pert.InjectLoadingOffset(),
        },
        n_replicates=20,
        prepare=my_pca,
    )

    meta.null_control(curve)
    meta.sensitivity(curve)
    meta.response_shape(curve)
    meta.specificity(curve, signal="dilute", nuisance="loading")

A metric is any callable taking an `AnnData` and returning a float; the mapping key names it. Paired
metrics read the pre-perturbation matrix from `adata.layers["truth"]`, which `InjectMissing` stashes,
so `msmetrics.variance_preservation` wraps in one lambda.

Not built, and worth adding when the need appears: separate metric-level replicates and an ICC, to
tell whether a metric's own seed noise exceeds its dose response; a confusion summary, asking whether
a metric can distinguish *which* perturbation happened at matched damage; cell and feature subsampling
as explicit coverage axes, which need their own framing because their trend is largely small-sample
estimator bias rather than sensitivity; bootstrap intervals on the detection dose.
"""

import random

import numpy as np
import pandas as pd
from anndata import AnnData

__all__ = [
    "null_control",
    "reference_noise",
    "response_shape",
    "sensitivity",
    "specificity",
    "sweep",
]

COLUMNS = ["perturbation", "dose", "dose_unit", "replicate", "metric", "value", "realized", "error"]

_EPS = 1e-12


def _seed_globals(*key: int) -> None:
    """Seed the global numpy and stdlib RNGs from an integer key.

    Metrics that wrap scanpy or leidenalg draw from the global RNG, so handing a `Generator` to the
    perturbation is not enough to make a clustering metric reproducible.
    """
    seed = int(np.random.SeedSequence(list(key)).generate_state(1)[0])
    np.random.seed(seed)
    random.seed(seed)


def _subsample(adata: AnnData, fraction: float, rng: np.random.Generator) -> AnnData:
    """Draw a fraction of the cells, without replacement."""
    if not 0 < fraction <= 1:
        raise ValueError(f"`subsample_frac` must be in (0, 1], got {fraction}.")
    n = max(2, round(fraction * adata.n_obs))
    if n >= adata.n_obs:
        return adata.copy()
    return adata[np.sort(rng.choice(adata.n_obs, size=n, replace=False))].copy()


def sweep(
    adata: AnnData,
    metrics: dict,
    perturbations: dict,
    *,
    doses=(0.0, 0.25, 0.5, 0.75, 1.0),
    n_replicates: int = 20,
    prepare=None,
    subsample_frac: float = 0.9,
    seed: int = 0,
    on_error: str = "nan",
) -> pd.DataFrame:
    """Evaluate every metric across a dose series of every perturbation.

    Parameters
    ----------
    adata
        Dataset to perturb. Never modified.
    metrics
        Mapping of name to a callable taking an `AnnData` and returning a float.
    perturbations
        Mapping of name to a `msmetrics.perturbations.Perturbation`. The key names the perturbation
        everywhere downstream, so two differently configured instances of the same class coexist.
    doses
        Dose grid, in each perturbation's own unit. Must include the reference dose, normally 0.
    n_replicates
        Independent replicates per dose. The replicates at the lowest dose are the reference
        distribution every summary standardises against, so this is also the resolution of the
        permutation p-value in `null_control`.
    prepare
        Optional `f(adata, rng) -> AnnData` run once per replicate after the perturbation and before
        any metric. Embeddings and the missing-value policy belong here: if each metric computed its
        own PCA, a difference in measured sensitivity between two metrics would partly be a
        difference in preprocessing, and the comparison would not mean what it appears to.
    subsample_frac
        Fraction of cells drawn per replicate. This is what gives the reference dose a non-degenerate
        spread — with a pure identity at dose 0 and a deterministic metric, every standardised effect
        below would divide by zero. Subsampling rather than bootstrapping, because resampling with
        replacement produces duplicate cells and breaks nearest-neighbour metrics.
    seed
        Base seed. Runs with the same seed and arguments are identical.
    on_error
        `"nan"` records a failing metric as a `nan` row carrying the exception text; `"raise"`
        propagates it.

    Returns
    -------
    pandas.DataFrame
        Long frame with one row per perturbation x dose x replicate x metric, columns
        `perturbation, dose, dose_unit, replicate, metric, value, realized, error`. The run
        configuration is recorded in `.attrs["config"]`.

    Raises
    ------
    ValueError
        If `on_error` is not one of the two accepted values, or `subsample_frac` is out of range.

    Examples
    --------
    .. code-block:: python

        curve = meta.sweep(adata, {"ari": my_ari}, {"dilute": pert.DiluteSignal("cell_type")})
        meta.response_shape(curve)

    Notes
    -----
    Every dose of a replicate is handed a *freshly constructed* generator seeded on
    `(seed, perturbation, replicate)` and deliberately not on the dose. A perturbation therefore draws
    the same random offset at every dose and merely scales it, which is what makes the dose-response
    curve a curve. Reseeding per dose instead is the most damaging mistake available here: nothing
    crashes, the curve simply acquires noise and loses monotonicity.
    """
    if on_error not in {"nan", "raise"}:
        raise ValueError(f"`on_error` must be 'nan' or 'raise', got {on_error!r}.")

    rows = []
    for perturbation_index, (perturbation_name, perturbation) in enumerate(perturbations.items()):
        for replicate in range(n_replicates):
            subsample_rng = np.random.default_rng([seed, perturbation_index, replicate, 0])
            reference = _subsample(adata, subsample_frac, subsample_rng)

            for dose_index, dose in enumerate(doses):
                rng = np.random.default_rng([seed, perturbation_index, replicate])
                perturbed = perturbation(reference, dose, rng)

                if prepare is not None:
                    prepare_rng = np.random.default_rng([seed, perturbation_index, replicate, 1])
                    perturbed = prepare(perturbed, prepare_rng)

                realized = _realized(perturbed, perturbation)

                for metric_index, (metric_name, metric) in enumerate(metrics.items()):
                    _seed_globals(seed, perturbation_index, replicate, dose_index, metric_index)
                    try:
                        value, error = float(metric(perturbed)), ""
                    except Exception as exc:
                        if on_error == "raise":
                            raise
                        value, error = float("nan"), f"{type(exc).__name__}: {exc}"

                    rows.append(
                        (
                            perturbation_name,
                            float(dose),
                            getattr(perturbation, "dose_unit", ""),
                            replicate,
                            metric_name,
                            value,
                            realized,
                            error,
                        )
                    )

    curve = pd.DataFrame(rows, columns=COLUMNS)
    curve.attrs["config"] = {
        "doses": tuple(float(d) for d in doses),
        "n_replicates": n_replicates,
        "subsample_frac": subsample_frac,
        "seed": seed,
        "metrics": tuple(metrics),
        "perturbations": {name: repr(p) for name, p in perturbations.items()},
        "prepare": getattr(prepare, "__name__", None) if prepare is not None else None,
    }
    return curve


def _realized(adata: AnnData, perturbation) -> float:
    """Pull the perturbation's headline realized-damage value out of `.uns`."""
    key = getattr(perturbation, "realized_key", None)
    if key is None:
        return float("nan")
    recorded = adata.uns.get("perturbation", {}).get("realized", {})
    return float(recorded.get(key, float("nan")))


def _reference_dose(curve: pd.DataFrame) -> float:
    return float(curve["dose"].min())


def reference_noise(curve: pd.DataFrame) -> pd.DataFrame:
    """Spread of each metric at the reference dose: the denominator every other summary uses.

    Parameters
    ----------
    curve
        Frame returned by `sweep`.

    Returns
    -------
    pandas.DataFrame
        One row per metric, with `mean_0`, `sd_0` and `n_0`. Replicates are pooled across
        perturbations, since at the reference dose every perturbation is the identity while the cell
        subsamples still differ, so pooling adds genuinely independent replicates rather than copies.

    Notes
    -----
    An `sd_0` of zero means the metric is deterministic given the data *and* the subsampling did not
    move it — usually `subsample_frac=1`. Every standardised effect downstream is then undefined and
    is returned as `nan` rather than as infinity.
    """
    reference = curve[np.isclose(curve["dose"], _reference_dose(curve))]
    grouped = reference.groupby("metric", sort=False)["value"]
    return (
        pd.DataFrame({"mean_0": grouped.mean(), "sd_0": grouped.std(ddof=1), "n_0": grouped.count()})
        .reset_index()
        .astype({"n_0": int})
    )


def null_control(curve: pd.DataFrame, *, dose: float = 1.0) -> pd.DataFrame:
    """Compare each metric's real-label value against its distribution under a destroyed signal.

    Parameters
    ----------
    curve
        Frame returned by `sweep`, normally over `PermuteLabels`.
    dose
        Dose treated as the null, i.e. where the structure has been fully destroyed.

    Returns
    -------
    pandas.DataFrame
        One row per perturbation x metric, with `observed` (mean at the reference dose), `null_mean`,
        `null_sd`, `n_null`, a two-sided empirical `p_value` and a `z` score.

    Notes
    -----
    Deliberately direction-free: nothing here needs to know whether the metric scores high or low on
    good data, so it works on any metric without registering an expected value. Read `z` for the sign.

    The p-value cannot fall below `1 / (n_null + 1)`, so its resolution is set by `n_replicates`. For
    a null control specifically, run `sweep` with `doses=(0.0, 1.0)` and 100 or more replicates rather
    than the default.
    """
    reference_dose = _reference_dose(curve)
    rows = []

    for (perturbation, metric), group in curve.groupby(["perturbation", "metric"], sort=False):
        null = group.loc[np.isclose(group["dose"], dose), "value"].dropna().to_numpy()
        observed = group.loc[np.isclose(group["dose"], reference_dose), "value"].dropna().to_numpy()
        if null.size == 0 or observed.size == 0:
            continue

        value = float(observed.mean())
        centre = float(null.mean())
        spread = float(null.std(ddof=1)) if null.size > 1 else float("nan")
        extreme = int(np.sum(np.abs(null - centre) >= abs(value - centre)))

        rows.append(
            {
                "perturbation": perturbation,
                "metric": metric,
                "observed": value,
                "null_mean": centre,
                "null_sd": spread,
                "n_null": int(null.size),
                "p_value": (1 + extreme) / (null.size + 1),
                "z": (value - centre) / spread if np.isfinite(spread) and spread > _EPS else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def sensitivity(curve: pd.DataFrame, *, n_sd: float = 2.0, x: str = "dose") -> pd.DataFrame:
    """How large a perturbation has to be before a metric notices it.

    Parameters
    ----------
    curve
        Frame returned by `sweep`.
    n_sd
        How many reference standard deviations the metric has to move to count as having noticed.
    x
        `"dose"` reports the threshold in the perturbation's own dose unit; `"realized"` reports it in
        terms of the damage the perturbation recorded actually having done, which is the comparable
        axis when two perturbations parameterise their dose differently.

    Returns
    -------
    pandas.DataFrame
        One row per perturbation x metric x dose with the standardised `effect`
        `|mean(d) - mean(0)| / sd_0`, alongside a `detection_dose` constant within each group.

    Notes
    -----
    `detection_dose` is linearly interpolated between the two grid points that bracket the crossing,
    so it is not quantised to the dose grid and does not silently improve when the grid is refined.
    It is `nan` when the curve never crosses, which is the honest answer for a metric that does not
    respond — as opposed to the smallest grid dose, which a threshold-free reading would report.
    """
    if x not in {"dose", "realized"}:
        raise ValueError(f"`x` must be 'dose' or 'realized', got {x!r}.")

    noise = reference_noise(curve).set_index("metric")
    reference_dose = _reference_dose(curve)
    frames = []

    for (perturbation, metric), group in curve.groupby(["perturbation", "metric"], sort=False):
        sd = float(noise.loc[metric, "sd_0"]) if metric in noise.index else float("nan")
        means = group.groupby("dose", sort=True)[["value", "realized"]].mean()
        baseline = float(means.loc[reference_dose, "value"])

        effect = (means["value"] - baseline).abs() / (sd if np.isfinite(sd) and sd > _EPS else np.nan)
        axis = means.index.to_numpy(dtype=float) if x == "dose" else means["realized"].to_numpy(dtype=float)

        frames.append(
            pd.DataFrame(
                {
                    "perturbation": perturbation,
                    "metric": metric,
                    "dose": means.index.to_numpy(dtype=float),
                    "realized": means["realized"].to_numpy(dtype=float),
                    "effect": effect.to_numpy(dtype=float),
                    "detection_dose": _crossing(axis, effect.to_numpy(dtype=float), n_sd),
                }
            )
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """First `x` at which `y` rises through `threshold`, linearly interpolated."""
    for i in range(1, len(y)):
        if np.isfinite(y[i]) and y[i] >= threshold:
            previous = y[i - 1]
            if not np.isfinite(previous) or previous >= threshold or y[i] == previous:
                return float(x[i])
            span = (threshold - previous) / (y[i] - previous)
            return float(x[i - 1] + span * (x[i] - x[i - 1]))
    return float("nan")


def response_shape(curve: pd.DataFrame, *, saturation: float = 0.9) -> pd.DataFrame:
    """Whether each metric's dose response is monotone, and how much of it is usable.

    Parameters
    ----------
    curve
        Frame returned by `sweep`.
    saturation
        Fraction of the total change that counts as saturated.

    Returns
    -------
    pandas.DataFrame
        One row per perturbation x metric, with signed `spearman` against dose, the
        `monotone_fraction` of dose steps moving in the dominant direction, the `floor` and `ceiling`
        of the mean response, the `dynamic_range` from the reference dose to the highest dose,
        `range_over_noise`, and the `saturation_dose` at which the response first reaches
        `saturation` of its total change.

    Notes
    -----
    `range_over_noise` is the column to read, not `dynamic_range`. Raw ranges are not comparable
    across metrics on different scales — an ARI in `[0, 1]` against an iLISI in `[1, n_batches]` —
    whereas dividing each metric's response by its own noise puts them in the same units of
    detectability.

    A `saturation_dose` well below the top of the grid means the metric stopped distinguishing more
    damage from less: the sweep should be rerun on a finer grid below that point, since everything
    above it carries no information.
    """
    noise = reference_noise(curve).set_index("metric")
    reference_dose = _reference_dose(curve)
    rows = []

    for (perturbation, metric), group in curve.groupby(["perturbation", "metric"], sort=False):
        usable = group.dropna(subset=["value"])
        rho = usable[["dose", "value"]].corr(method="spearman").iloc[0, 1] if len(usable) > 2 else float("nan")

        means = group.groupby("dose", sort=True)["value"].mean()
        doses = means.index.to_numpy(dtype=float)
        values = means.to_numpy(dtype=float)
        steps = np.diff(values)
        total = float(values[-1] - values[0])
        sd = float(noise.loc[metric, "sd_0"]) if metric in noise.index else float("nan")

        if steps.size and np.any(steps != 0):
            direction = np.sign(total) if total != 0 else np.sign(steps[np.argmax(np.abs(steps))])
            monotone = float(np.mean(np.sign(steps) == direction))
        else:
            monotone = float("nan")

        rows.append(
            {
                "perturbation": perturbation,
                "metric": metric,
                "spearman": float(rho),
                "monotone_fraction": monotone,
                "floor": float(np.nanmin(values)),
                "ceiling": float(np.nanmax(values)),
                "dynamic_range": abs(total),
                "range_over_noise": abs(total) / sd if np.isfinite(sd) and sd > _EPS else float("nan"),
                "saturation_dose": _crossing(doses, np.abs(values - values[0]), saturation * abs(total))
                if abs(total) > _EPS
                else float("nan"),
                "reference_dose": reference_dose,
            }
        )

    return pd.DataFrame(rows)


def specificity(curve: pd.DataFrame, *, signal: str, nuisance: str) -> pd.DataFrame:
    """Whether a metric moves with biology or with a technical confound.

    Parameters
    ----------
    curve
        Frame returned by `sweep`, containing both named perturbations.
    signal
        Name of the biological perturbation, normally a `DiluteSignal` sweep.
    nuisance
        Name of the technical perturbation — a loading offset for the cell-size question, an
        `InjectMissing` sweep with an imputer in `prepare` for the imputation-strength question.

    Returns
    -------
    pandas.DataFrame
        One row per metric, with `signal_slope` and `nuisance_slope` in reference standard deviations
        per unit dose, and a `contrast` in `[-1, 1]`. Positive contrast means the metric tracks
        biology more strongly than the confound.

    Raises
    ------
    KeyError
        If either perturbation is absent from the frame.

    Notes
    -----
    A bounded contrast rather than a slope ratio. A ratio blows up whenever the denominator is near
    zero — exactly the case of a well-behaved metric, which is where the number matters most — and it
    inherits whatever dose parameterisation each perturbation happened to choose.

    Read the two slopes as well as the contrast. A contrast near zero is produced both by a metric
    that responds to everything and by one that responds to nothing, and only the slopes tell those
    apart.
    """
    present = set(curve["perturbation"])
    missing = {signal, nuisance} - present
    if missing:
        raise KeyError(f"perturbation(s) {sorted(missing)} not in the sweep; it holds {sorted(present)}.")

    noise = reference_noise(curve).set_index("metric")
    rows = []

    for metric, group in curve.groupby("metric", sort=False):
        sd = float(noise.loc[metric, "sd_0"]) if metric in noise.index else float("nan")
        slopes = {
            role: _slope(group[group["perturbation"] == name], sd)
            for role, name in (("signal", signal), ("nuisance", nuisance))
        }
        magnitude = abs(slopes["signal"]) + abs(slopes["nuisance"])
        rows.append(
            {
                "metric": metric,
                "signal": signal,
                "nuisance": nuisance,
                "signal_slope": slopes["signal"],
                "nuisance_slope": slopes["nuisance"],
                "contrast": (abs(slopes["signal"]) - abs(slopes["nuisance"])) / magnitude
                if magnitude > _EPS
                else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _slope(group: pd.DataFrame, sd: float) -> float:
    """Least-squares slope of value against dose, in units of the metric's reference noise."""
    usable = group.dropna(subset=["value"])
    if len(usable) < 2 or usable["dose"].nunique() < 2 or not (np.isfinite(sd) and sd > _EPS):
        return float("nan")
    return float(np.polyfit(usable["dose"].to_numpy(float), usable["value"].to_numpy(float), 1)[0] / sd)
