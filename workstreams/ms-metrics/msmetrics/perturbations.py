"""Controlled perturbations of a dataset, for meta-benchmarking metrics.

A perturbation takes a dataset and a `dose`, and returns a copy of that dataset with a known amount
of a specific kind of damage applied. Sweeping the dose and recomputing a metric at every step is
what `msmetrics.meta` does; this module supplies the damage.

Each perturbation is a class configured in `__init__` and applied in `__call__`, so that binding a
label key to a perturbation reads as `DiluteSignal("cell_type")` rather than as a `functools.partial`.
Every one of them:

- returns a copy and never mutates its input,
- is the identity at `dose = 0`,
- takes its dose in the native unit named by its `dose_unit` attribute, since the injectors have no
  natural maximum and rescaling them to `[0, 1]` would smuggle an invented constant into every
  downstream comparison.

The three injectors (`InjectBatchShift`, `InjectLoadingOffset`, `DiluteSignal`) leave the missingness
pattern bit-identical. If they also changed which values were observed, their effect on a metric
would be confounded with `InjectMissing`'s, and comparing the two would mean nothing.

Values are assumed to be log-intensities, so batch effects, loading differences and fold changes are
all additive.
"""

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

_EPS = 1e-9

__all__ = [
    "DiluteSignal",
    "InjectBatchShift",
    "InjectLoadingOffset",
    "InjectMissing",
    "PermuteLabels",
    "Perturbation",
]


def _dense(adata: AnnData) -> np.ndarray:
    """Return `adata.X` as a float array, refusing sparse input."""
    X = adata.X
    if not isinstance(X, np.ndarray):
        raise TypeError(
            f"`adata.X` must be a dense numpy array, got {type(X).__name__}. Proteomics matrices encode "
            "missingness as `nan`, which a sparse matrix cannot represent."
        )
    return np.asarray(X, dtype=float)


def _nan_mad(X: np.ndarray) -> np.ndarray:
    """Robust per-feature scale, `1.4826 * MAD`, floored away from zero.

    The MAD rather than the standard deviation, because a protein observed in a handful of cells has
    a meaningless SD, and scaling an injected shift by it would make that shift enormous or invisible
    for reasons that have nothing to do with the metric under test.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
        median = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - median), axis=0)
    sigma = 1.4826 * mad
    return np.where(np.isfinite(sigma) & (sigma > _EPS), sigma, _EPS)


def _between_group_variance_fraction(X: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of total variance that lies between groups, averaged over features.

    Used as the `realized` damage of the perturbations whose dose is not itself directly
    interpretable, so that summaries can plot against how far the data actually moved rather than
    against the size of the knob that was turned.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        grand = np.nanmean(X, axis=0)
        between = np.zeros(X.shape[1])
        for level in pd.unique(groups):
            rows = groups == level
            centroid = np.nanmean(X[rows], axis=0)
            between += rows.sum() * np.square(centroid - grand)
        total = np.nansum(np.square(X - grand), axis=0)

    usable = np.isfinite(between) & np.isfinite(total) & (total > _EPS)
    if not usable.any():
        return float("nan")
    return float(np.mean(between[usable] / total[usable]))


class Perturbation:
    """Base class holding the copy-and-record boilerplate shared by every perturbation.

    Subclasses implement `_apply`, which mutates the already-copied `AnnData` in place and returns a
    dict of what it actually did.

    Note that this base deliberately does *not* short-circuit at `dose = 0`. Identity at zero dose
    falls out of each subclass's own arithmetic — a zero-scaled offset, a zero-length shuffle, a
    zero-sized mask — so it is worth verifying by test rather than guaranteeing by a wrapper that
    would also skip `InjectMissing`'s bookkeeping of the pre-perturbation matrix.
    """

    #: Unit the dose is expressed in, recorded into the sweep frame.
    dose_unit: str = "fraction"
    #: Key of `realized` that `msmetrics.meta.sweep` records as the realized-damage column.
    realized_key: str | None = None

    def __call__(self, adata: AnnData, dose: float, rng: np.random.Generator) -> AnnData:
        """Apply the perturbation at `dose`, returning a new `AnnData`."""
        out = adata.copy()
        realized = self._apply(out, float(dose), rng) or {}
        out.uns["perturbation"] = {"kind": type(self).__name__, "dose": float(dose), "realized": realized}
        return out

    def _apply(self, adata: AnnData, dose: float, rng: np.random.Generator) -> dict[str, float] | None:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class PermuteLabels(Perturbation):
    """Shuffle a fraction of an `.obs` column, destroying the structure it encodes.

    At `dose = 1` this is the classical label-permutation null: whatever a metric reports afterwards
    is what it reports on data with no relationship between the labels and the values. Intermediate
    doses give a graded null, so the same sweep also shows how quickly a metric notices.

    Parameters
    ----------
    key
        Column of `.obs` to shuffle.
    stratify_by
        If given, permute within each level of this column instead of globally. In plate-based
        proteomics the batch is often correlated with the cell type, and a global permutation then
        also destroys the batch x cell-type contingency table — which moves batch metrics as well and
        makes the perturbation something other than a clean biological null.

    Notes
    -----
    Shuffling `k` labels is not a derangement: some of them land back where they started, so
    `realized["fraction_changed"]` runs below the dose, and further below it when one label dominates.
    """

    dose_unit = "fraction_labels_shuffled"
    realized_key = "fraction_changed"

    def __init__(self, key: str, stratify_by: str | None = None):
        self.key = key
        self.stratify_by = stratify_by

    def _apply(self, adata, dose, rng):
        if self.key not in adata.obs:
            raise KeyError(f"`{self.key}` is not a column of `.obs`.")

        column = adata.obs[self.key]
        labels = column.to_numpy()
        shuffled = labels.copy()

        if self.stratify_by is None:
            groups = [np.arange(adata.n_obs)]
        else:
            strata = adata.obs[self.stratify_by].to_numpy()
            groups = [np.flatnonzero(strata == level) for level in pd.unique(strata)]

        for index in groups:
            k = round(dose * index.size)
            if k < 2:
                continue
            chosen = rng.choice(index, size=k, replace=False)
            shuffled[chosen] = labels[rng.permutation(chosen)]

        adata.obs[self.key] = pd.Series(shuffled, index=adata.obs.index).astype(column.dtype)
        return {"fraction_changed": float(np.mean(shuffled != labels))}


class InjectBatchShift(Perturbation):
    """Add a per-batch, per-feature offset, growing the between-batch variance.

    For batch `b` and feature `j`, `X'[i, j] = X[i, j] + dose * sigma_j * delta[b, j]`, where
    `sigma_j` is the feature's robust scale and `delta` is a rank-`rank` matrix, centred across
    batches by batch size and rescaled to unit RMS. The dose is therefore the RMS size of the
    injected shift in units of the feature's own MAD.

    Parameters
    ----------
    batch_key
        Column of `.obs` holding the batch.
    rank
        Number of shared directions the batch effect spans. The default of 1 reflects real
        LC-MS batch effects, which are low-rank — gradient drift, column age, ionisation efficiency —
        rather than independent per-protein noise. Rank matters: a rank-1 shift is much harder to
        correct than an i.i.d. one, and metrics respond to the two differently, so it is worth
        sweeping in its own right.

    Notes
    -----
    Centring the offsets across batches keeps the global feature means fixed, so only the
    *between-batch* variance grows. Without it the injection would also move global location, and any
    metric with a location dependence would respond for the wrong reason.

    A per-batch shift that is constant across features is cancelled exactly by per-cell median
    normalisation. If such a normalisation runs in the sweep's `prepare` step, a metric can then score
    as perfectly insensitive to this perturbation purely because of the preprocessing.
    """

    dose_unit = "feature_mad"
    realized_key = "batch_variance_fraction"

    def __init__(self, batch_key: str, rank: int = 1):
        if rank < 1:
            raise ValueError(f"`rank` must be at least 1, got {rank}.")
        self.batch_key = batch_key
        self.rank = rank

    def _apply(self, adata, dose, rng):
        X = _dense(adata)
        batches = adata.obs[self.batch_key].to_numpy()
        levels = pd.unique(batches)
        if levels.size < 2:
            raise ValueError(f"`{self.batch_key}` has {levels.size} level(s); a batch shift needs at least 2.")

        loadings = rng.standard_normal((levels.size, self.rank))
        directions = rng.standard_normal((self.rank, X.shape[1]))
        delta = loadings @ directions

        # Centre per feature, weighted by how many values of that feature each batch actually
        # observed. Weighting by batch size instead leaves the global means drifting whenever
        # missingness differs between batches, which it always does.
        observed = np.stack([np.isfinite(X[batches == level]).sum(axis=0) for level in levels]).astype(float)
        totals = observed.sum(axis=0)
        weights = np.divide(observed, totals, out=np.full_like(observed, 1.0 / levels.size), where=totals > 0)
        delta -= (delta * weights).sum(axis=0, keepdims=True)

        rms = float(np.sqrt(np.mean(np.square(delta))))
        if rms > _EPS:
            delta /= rms

        position = {level: i for i, level in enumerate(levels)}
        rows = np.array([position[level] for level in batches])
        adata.X = X + dose * _nan_mad(X)[None, :] * delta[rows, :]

        return {"batch_variance_fraction": _between_group_variance_fraction(adata.X, batches)}


class InjectMissing(Perturbation):
    """Mask a fraction of the observed values, either at random or intensity-dependently.

    A single latent detectability score is drawn per replicate, and the `dose` fraction of observed
    entries with the lowest score is masked. That gives an exact target fraction and, crucially,
    *nested* masks across doses: whatever is missing at dose 0.3 is still missing at dose 0.5, so the
    dose-response curve reflects increasing missingness rather than a fresh random draw per point.

    Parameters
    ----------
    mechanism
        `"mcar"` masks uniformly at random. `"mnar"` scores each entry by its intensity plus logistic
        noise, so low-abundance values drop out first — the realistic proteomics case.
    steepness
        Width of the logistic noise for `"mnar"`, in units of the observed value spread. Small values
        approach deterministic MNAR (the very lowest intensities go first), large values approach
        MCAR. Ignored for `"mcar"`.
    truth_layer
        Layer the pre-masking matrix is stashed in, so that paired metrics such as
        `msmetrics.variance_preservation` can compare against the values that were removed.

    Notes
    -----
    Unlike the injectors, this changes the missingness pattern, which is the point. It is therefore
    the perturbation to use as the imputation axis: put an imputer in the sweep's `prepare` step and
    this becomes a titration of imputation strength.
    """

    dose_unit = "fraction_observed_masked"
    realized_key = "fraction_missing"

    def __init__(self, mechanism: str = "mcar", steepness: float = 1.0, truth_layer: str = "truth"):
        if mechanism not in {"mcar", "mnar"}:
            raise ValueError(f"`mechanism` must be 'mcar' or 'mnar', got {mechanism!r}.")
        self.mechanism = mechanism
        self.steepness = steepness
        self.truth_layer = truth_layer

    def _apply(self, adata, dose, rng):
        X = _dense(adata)
        adata.layers[self.truth_layer] = X.copy()

        flat = X.reshape(-1)
        observed = np.flatnonzero(np.isfinite(flat))
        k = round(dose * observed.size)

        if k > 0 and observed.size > 0:
            values = flat[observed]
            if self.mechanism == "mcar":
                score = rng.random(observed.size)
            else:
                spread = float(np.std(values)) or 1.0
                score = values + self.steepness * spread * rng.logistic(size=observed.size)
            flat = flat.copy()
            flat[observed[np.argsort(score, kind="stable")[:k]]] = np.nan

        adata.X = flat.reshape(X.shape)
        return {
            "fraction_missing": float(np.mean(~np.isfinite(adata.X))),
            "fraction_observed_masked": float(k / observed.size) if observed.size else float("nan"),
        }


class InjectLoadingOffset(Perturbation):
    """Add a random per-cell offset, the log-space form of a cell size or loading difference.

    `X'[i, j] = X[i, j] + s_i` with `s_i ~ N(0, (dose * sigma)^2)`, where `sigma` is the standard
    deviation of the per-cell means in the real data. The dose is therefore expressed in units of the
    dataset's own cell-to-cell spread: `dose = 1` roughly doubles it.

    This is the nuisance axis of the specificity analysis — a well-behaved biological metric should
    move much less under it than under `DiluteSignal`.

    Notes
    -----
    Named "offset" rather than "scale" because on log-intensities a loading difference is additive.
    The offsets are centred, so the global mean is unchanged and only the cell-to-cell spread grows.
    """

    dose_unit = "cell_mean_sd"

    def _apply(self, adata, dose, rng):
        X = _dense(adata)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            cell_means = np.nanmean(X, axis=1)
        sigma = float(np.nanstd(cell_means))
        offset = rng.standard_normal(adata.n_obs) * dose * (sigma if sigma > _EPS else 1.0)
        adata.X = X + (offset - offset.mean())[:, None]
        return {"offset_sd": float(np.std(offset))}


class DiluteSignal(Perturbation):
    """Shrink the class centroids toward the global mean, removing biological signal.

    `X'[i, j] = X[i, j] - dose * (mu[c(i), j] - mu[j])`, so between-class mean differences scale by
    `(1 - dose)` while within-class variance is untouched. The dose is therefore exactly the fraction
    of between-class signal removed — natively an effect size, which is what makes it the clean
    counterpart to the nuisance perturbations in `specificity`.

    Parameters
    ----------
    label_key
        Column of `.obs` holding the biological grouping.
    min_obs
        Features observed in fewer than this many cells of a class are left alone for that class,
        since their centroid is not estimable.

    Notes
    -----
    Only the discrete class mean structure is removed. Continuous biological variation and the
    residual within-class covariance survive at any dose, so `dose = 1` does not make the classes
    inseparable — it makes their nanmean centroids equal. Under MNAR those centroids are themselves
    biased, and biased differently per class when detection rate varies with cell type, so the
    empirical separation reported in `realized` falls short of `1 - dose`. That gap is informative
    rather than a bug.
    """

    dose_unit = "fraction_signal_removed"
    realized_key = "separation_empirical"

    def __init__(self, label_key: str, min_obs: int = 3):
        self.label_key = label_key
        self.min_obs = min_obs

    def _apply(self, adata, dose, rng):
        X = _dense(adata)
        labels = adata.obs[self.label_key].to_numpy()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            grand = np.nanmean(X, axis=0)

            shift = np.zeros_like(X)
            skipped = 0
            levels = pd.unique(labels)
            for level in levels:
                rows = labels == level
                centroid = np.nanmean(X[rows], axis=0)
                estimable = (np.isfinite(X[rows]).sum(axis=0) >= self.min_obs) & np.isfinite(centroid)
                estimable &= np.isfinite(grand)
                shift[rows] = np.where(estimable, centroid - grand, 0.0)
                skipped += int((~estimable).sum())

        adata.X = X - dose * shift
        return {
            "separation_empirical": _between_group_variance_fraction(adata.X, labels),
            "fraction_features_skipped": float(skipped / (levels.size * X.shape[1])),
        }
