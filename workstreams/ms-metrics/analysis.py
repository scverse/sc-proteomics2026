# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: alphapept
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## ms-metrics workstream

# %% [markdown]
# The aim of this notebook is to run and test the individual modules on our example datasets.
#
# We represent the steps as a nested list of steps. Each step consists of a metric + method, where the metric gives some diagnostic information and the method applies some kind of change to the dataset. Broadly, steps fall in either two categories: those that change the shape of the dataset, and those that change the distribution of the values within the dataset:
#
# - Changing data shape (Filtering): mask features or samples
# - Changing data distribution (Normalization/Batch Correction): move & rescale samples or features
#
# Each step consists of a metric to answer a given question (e.g. computing data completeness, outliers, distributions, clustering) and a method that changes the data. For example, checking the metric data completeness and seeing that 12 % of features are less than 10 % complete across all samples could inform using a method for filtering for data completeness. Checking for completeness and removing incomplete features would then constitute one step in the analysis of a dataset. An ideal analysis would then be a sequence of steps, each of which consists of metric + method.
#

# %% [markdown]
# ### References:
#
# - https://www.nature.com/articles/s41467-025-64718-y main reference for batch correction on different proteomics data levels

# %%
import alphapepttools as apt
import anndata as ad
import numpy as np
import pandas as pd

from msmetrics.datasets import wu2025
from msmetrics.utils import draw_missingness

# %% [markdown]
# ### Analysis workflow:

# %%
# ### 1. Load data (I/O)
adata = ad.read_h5ad(wu2025())
adata

# %%
# ### 2. Log-transform (Preprocessing)
# Data is already log-transformed!
# adata_log = apt.pp.nanlog(adata, copy = True)

# %%
# ### 3. Visualize data completeness (QC-inspection)
draw_missingness(
    X=adata.X,
    xlabel="Features",
    ylabel="Samples",
    title="Missingness Heatmap",
)

# %%
# ### 4. Normalization

# %%
# ### 5. Imputation

# %%
# ### 6. Batch correction

# %%
# ### 7. Differential expression (out of scope)

# %% [markdown]
# ## Meta-benchmarking: do the metrics themselves behave?
#
# A metric that returns a number for every dataset is not thereby useful. `msmetrics.meta` perturbs
# the data by a known amount, recomputes each metric, and reads the resulting dose-response curve
# four ways: does the metric collapse when the signal is destroyed (`null_control`), how much damage
# does it take to notice (`sensitivity`), is the response monotone and wide enough to read
# (`response_shape`), and does it move with biology or with a technical confound (`specificity`).
#
# Below we run every metric the package ships against every perturbation, on wu2025.

# %% [markdown]
# ### Working set
#
# Two thirds of the proteins are observed in under 5 % of cells, and 1094 of the 2599 cells carry no
# `celltype` or `sample` label at all. Both have to go before any of this means anything.

# %%
import warnings

from sklearn.utils.extmath import randomized_svd

from msmetrics import compute_neighborhood_preservation, meta, variance_preservation
from msmetrics import perturbations as pert
from msmetrics import plotting as pl
from msmetrics.utils import perform_leiden_clustering, point_cluster_distance

MIN_COMPLETENESS = 0.2
N_COMPONENTS = 15

labelled = adata[adata.obs["celltype"].notna() & adata.obs["sample"].notna()].copy()
complete = np.isfinite(np.asarray(labelled.X, float)).mean(axis=0) >= MIN_COMPLETENESS
working = labelled[:, complete].copy()
working.obs["celltype"] = working.obs["celltype"].cat.remove_unused_categories()
working.obs["sample"] = working.obs["sample"].cat.remove_unused_categories()

print(working.shape, f"{np.mean(~np.isfinite(np.asarray(working.X, float))):.1%} missing")
pd.crosstab(working.obs["celltype"], working.obs["sample"])

# %% [markdown]
# ### One embedding, shared by every metric
#
# If each metric computed its own PCA, a difference in their measured sensitivity would partly be a
# difference in their preprocessing. `prepare` runs once per replicate and hands every metric the
# same imputed matrix and the same embedding.
#
# It also carries the three references the paired metrics need, all keyed by cell name so they
# survive the per-replicate subsampling:
#
# - the matrix as it stood before imputation, for `variance_preservation`;
# - the embedding of the *untouched* dataset, for `neighborhood_preservation`;
# - the untouched imputed matrix and its Leiden labels, for `point_cluster_distance`. Clusters are
#   defined on the *before* space by construction, so Leiden runs once here rather than 1800 times
#   inside the sweep.


# %%
def embed(X, n_components=N_COMPONENTS, seed=0):
    """Mean-impute the missing values, then take the leading left singular vectors."""
    with warnings.catch_warnings():
        # A protein can end up fully masked at a high missingness dose, which has no mean. Those
        # columns fall back to zero below, so the warning carries nothing.
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        column = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    Y = np.where(np.isfinite(X), X, np.nan_to_num(column)[None, :])
    Y = Y - Y.mean(axis=0)
    U, S, _ = randomized_svd(Y, n_components=n_components, random_state=seed)
    return Y, U * S


reference_matrix, reference_coordinates = embed(np.asarray(working.X, float))
reference_matrix = pd.DataFrame(reference_matrix, index=working.obs_names)
reference_embedding = pd.DataFrame(reference_coordinates, index=working.obs_names)

_clustered = working.copy()
_clustered.X = reference_matrix.to_numpy()
perform_leiden_clustering(_clustered)
reference_clusters = _clustered.obs["_leiden"]
print(f"{reference_clusters.nunique()} Leiden clusters on the untouched data")


def prepare(a, rng):
    observed = np.asarray(a.X, float)
    imputed, embedding = embed(observed)
    a.layers["pre_imputation"] = observed
    a.layers["reference"] = reference_matrix.loc[a.obs_names].to_numpy()
    a.layers["perturbed"] = imputed
    a.X = imputed
    a.obsm["X_pca"] = embedding
    a.obsm["X_pca_reference"] = reference_embedding.loc[a.obs_names].to_numpy()
    a.obs["_leiden"] = reference_clusters.loc[a.obs_names].cat.remove_unused_categories()
    return a


def _point_cluster_distance(a):
    with warnings.catch_warnings():
        # Masking every observed value leaves a constant matrix, whose cell-to-centroid distances
        # are all equal and whose Spearman correlation is therefore undefined. That top dose is
        # legitimately unscoreable; `response_shape` drops it and reports `max_usable_dose`.
        warnings.filterwarnings("ignore", message="An input array is constant")
        return point_cluster_distance(a, "reference", "perturbed")


metrics = {
    "neighborhood_preservation": lambda a: compute_neighborhood_preservation(a, "X_pca_reference", "X_pca")[
        "adjusted_overlap"
    ],
    "point_cluster_distance": _point_cluster_distance,
    "variance_preservation": lambda a: variance_preservation(a.layers["pre_imputation"], a.X)["median_ratio"],
}

perturbations = {
    "dilute_celltype": pert.DiluteSignal("celltype"),  # biology
    "loading_offset": pert.InjectLoadingOffset(),  # cell size
    "batch_shift": pert.InjectBatchShift("sample"),  # batch
    "missing_mnar": pert.InjectMissing(mechanism="mnar"),  # missingness, hence imputation strength
    # Structure held fixed, only the cell count falls -- stratified on the full celltype x sample
    # table, not each margin, because the two are correlated here.
    "subsample": pert.SubsampleCells(["celltype", "sample"]),
    # `sample` is confounded with `celltype` here -- GW13 is mostly IN-CGE and oRG -- so a global
    # permutation would destroy the batch x celltype table too. Permute within sample instead.
    "permute_celltype": pert.PermuteLabels("celltype", stratify_by="sample"),
}

# %%
curve = meta.sweep(
    working,
    metrics,
    perturbations,
    doses=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    n_replicates=30,
    prepare=prepare,
    seed=0,
)
curve.head()

# %% [markdown]
# ### How to read the figures
#
# | Figure | What is on it | How to read it | Why it is built that way |
# | --- | --- | --- | --- |
# | `pl.response` | Mean ± sd of each metric against dose, one panel per perturbation. The pale horizontal band is the dose-0 noise, `mean₀ ± n_sd·sd₀`; the dotted vertical line is the interpolated detection dose. | Steepness near the origin is sensitivity. Flattening is saturation — everything to the right of it carries no information. A curve that never leaves the band means the metric is blind to that perturbation. | Sensitivity, monotonicity, range and saturation are four properties of one curve; splitting them across four tables loses the shape that makes them legible. Drawing the noise band turns "is this move real?" into something you can see rather than something you have to look up. |
# | `pl.null` | Histogram of the metric after the labels are destroyed, with the real-label value as a red rule. | Rule inside the histogram: the metric is not reading the labels. Rule far outside: it is, and `z` says by how much. | Direction-free. Nothing here needs to know whether high or low is "good" for a given metric, so it works on any metric without registering an expected value. |
# | `pl.scorecard` | Metric × perturbation heatmap of one summary column. | Read a **row** for one metric's response profile — what it notices and what it ignores. Read a **column** to pick the metric for a given failure mode. | The one-glance answer to "which metric for which job". Defaults to `range_over_noise` rather than `dynamic_range` for the reason in the next table. |
# | `pl.specificity` | \|biology slope\| against \|nuisance slope\|, one point per metric, with the `y = x` diagonal. | Above the diagonal: tracks biology. Below: tracks the confound. Near the origin: responds to neither, whatever the contrast says. | It plots both coordinates instead of their ratio. A point lands on the diagonal either by responding to everything or by responding to nothing, and a single contrast number cannot separate those two. |
#
# ### Comparing across metrics
#
# The trap: every metric lives on its own scale. `neighborhood_preservation` is an adjusted overlap
# in `[0, 1]`, `point_cluster_distance` is a Spearman correlation in `[-1, 1]`, a variance ratio is
# unbounded above, an iLISI runs from 1 to the number of batches. A drop of 0.4 is catastrophic for
# one and unremarkable for another, so **no comparison may be made on raw metric values**.
#
# Everything the harness reports is therefore divided by that metric's *own* reference noise `sd₀`,
# the spread it shows across replicates when nothing has been done to the data. The result is in
# units of "how many detectable steps did the metric move", which is the same currency for every
# metric on the page.
#
# | Column | Comparable across metrics? | Why |
# | --- | --- | --- |
# | `value`, `floor`, `ceiling`, `dynamic_range` | **No** | Raw metric units. Kept for reading one metric's curve, never for ranking two. |
# | `response_profile` shares | **Yes** | A ratio taken *within* one row, so `sd₀` cancels out of it entirely. This is what `summarize` shows. |
# | `contrast` | **Yes** | Same reason: both slopes carry the same `sd₀`, which cancels. Bounded in `[-1, 1]`, so a near-zero denominator cannot inflate it. |
# | `spearman`, `monotone_fraction` | **Yes** | Rank-based and unit-free, computed on raw values with no denominator. |
# | `saturation_dose` | **Yes, within one perturbation** | A fraction of the metric's own total change. Each perturbation has its own `dose_unit`, so do not compare a dose across columns. |
# | `p_value`, `z` | **Yes** | Positions within the metric's own null distribution. |
# | `range_over_noise`, `detection_dose`, `signal_slope`, `nuisance_slope` | **Only when `sd₀` is sound** | All divide by `sd₀`, so all inherit a collapsed denominator. `detection_dose` is the crossing of `2·sd₀`, so it fails in exactly the same way and for the same metrics. |
#
# So the ranking order is `contrast` and `spearman` first, the profile shares to see *what* a metric
# measures, and the `sd₀`-dependent columns last and only after checking `reference_noise`.
#
# Three cautions that follow from this:
#
# - A share says nothing about magnitude. A metric that barely moves at all still has a 100 % column,
#   namely whichever perturbation moved it least little. `peak` is what separates that case from a
#   metric with real dynamic range, which is why the two are shown side by side.
# - `range_over_noise` rewards a *precise* metric as much as a *responsive* one, since `sd₀` is in
#   the denominator. A metric that is very reproducible while measuring the wrong thing scores well
#   on it. Always read it next to `contrast`, which is what says whether the thing being measured is
#   the thing you wanted.
# - A response to `subsample` is **not** sensitivity. Nearest-neighbour and clustering estimators are
#   biased by the number of cells, so they drift when the dataset shrinks even though its structure
#   is untouched -- that is exactly what holding the composition fixed isolates. Read that column as
#   "are this metric's values comparable between datasets of different size", which is a useful
#   thing to know and a different question from the one the other columns answer.
# - **That caveat bites here.** `point_cluster_distance` returns exactly 1.0000 at dose 0 -- the
#   cluster centroids are defined on the reference space, so the two distance matrices agree to
#   numerical precision -- which leaves `sd₀` around `1e-5` and sends its `range_over_noise` into the
#   tens of thousands, against roughly 130 for `neighborhood_preservation`. That ratio is a collapsed
#   denominator, not a real advantage. `summarize` flags it with `⚠ sd₀≈0` in the `peak` column and
#   still shows that metric's shares, because those survive what the peak does not. The fix on the
#   harness side would be to recompute the reference clustering per replicate so the metric has a
#   real reference distribution; until then, rank it on `contrast` and `spearman`.
# - `sd₀` is estimated from `n_replicates` values at dose 0, so every standardised column inherits
#   that estimate's uncertainty. With the 30 replicates used here it is stable enough to rank
#   metrics; with 5 it would not be.


# %%
def summarize(curve, *, signal, nuisance, ax=None):
    """One row per metric: what it responds to, how strongly, and whether it tracks biology.

    Each response cell is that perturbation's share of the metric's *own* largest response, so a row
    reads as a sentence -- "100 % missingness, blind to everything else" -- and every row is on the
    same 0 to 100 scale whatever units the metric itself uses. `peak` carries the absolute size that
    the shares deliberately throw away, in units of the metric's reference noise.
    """
    import matplotlib.pyplot as plt
    from plottable import ColumnDefinition, Table

    profile = meta.response_profile(curve).set_index("metric")
    contrast = meta.specificity(curve, signal=signal, nuisance=nuisance).set_index("metric")["contrast"]

    responses = [column for column in profile.columns if column in set(curve["perturbation"])]
    table = profile[responses].copy()

    # `peak` is suppressed where the reference noise collapsed, since dividing by it produced the
    # number rather than measuring anything. The shares stay: they are a ratio within the row, so
    # `sd_0` cancels out of them and they remain readable for exactly the metric whose peak does not.
    table["peak"] = [
        "⚠ sd₀≈0" if degenerate else f"{value:,.0f}×"
        for value, degenerate in zip(profile["peak"], profile["sd_0_degenerate"])
    ]
    # Written out as text rather than left numeric, because plottable skips a `nan` before the
    # formatter runs and the cell would come out blank -- indistinguishable from missing data, when
    # what it means is that the metric responded to neither side and the contrast is undefined.
    table["tracks_bio"] = [
        "n/a" if not np.isfinite(value) else f"{'yes' if value > 0 else 'no'}  ({value:+.2f})"
        for value in contrast.reindex(profile.index)
    ]
    table = table.reset_index()

    ax = ax if ax is not None else plt.subplots(figsize=(1.5 * len(table.columns) + 3, 1.2 + 0.6 * len(table)))[1]

    def share_colour(value):
        if not np.isfinite(value) or value <= 0:
            return "#f4f4f4"
        return plt.get_cmap("Blues")(0.06 + 0.52 * value)

    def contrast_colour(worded):
        if worded == "n/a":
            return "#999999"
        magnitude = min(abs(float(worded.split("(")[1].rstrip(")"))), 0.5) / 0.5
        return plt.get_cmap("RdBu")(0.5 + (0.45 if worded.startswith("yes") else -0.45) * magnitude)

    definitions = [
        ColumnDefinition("metric", width=2.6, textprops={"ha": "left", "weight": "bold"}),
        *[
            ColumnDefinition(
                name,
                title=name.replace("_", "\n"),
                width=1.0,
                group="responds to  (% of its own peak)",
                formatter=lambda v: "-" if not np.isfinite(v) else f"{v:.0%}",
                cmap=share_colour,
            )
            for name in responses
        ],
        ColumnDefinition("peak", title="peak\n(× noise)", width=1.1, group="how strongly"),
        ColumnDefinition(
            "tracks_bio",
            title=f"vs {nuisance}",
            width=1.4,
            group="tracks biology?",
            text_cmap=contrast_colour,
        ),
    ]

    Table(
        table,
        ax=ax,
        index_col="metric",
        column_definitions=definitions,
        textprops={"fontsize": 10, "ha": "center"},
        row_dividers=True,
        col_label_divider=True,
    )
    return ax


# %% [markdown]
# ### Is the response monotone, and how much of it is usable?
#
# `max_usable_dose` shows where a sweep ran out of computable doses -- masking every observed value
# leaves `variance_preservation` nothing to score, so its curve stops at 0.8.

# %%
meta.reference_noise(curve)

# %%
meta.response_shape(curve)

# %%
meta.sensitivity(curve).groupby(["perturbation", "metric"])["detection_dose"].first().unstack()

# %%
pl.response(curve)

# %% [markdown]
# ### Does the metric track biology, or the confound?
#
# `contrast` above 0 means the metric responds more strongly to diluted biology than to the
# nuisance. Read both slopes as well: a contrast near zero is produced both by a metric that
# responds to everything and by one that responds to nothing.
#
# Note that `prepare` here does no per-cell normalisation; median-normalising would cancel a scalar
# loading offset outright, and re-running this with normalisation in `prepare` is the obvious next
# experiment.

# %%
pd.concat(
    meta.specificity(curve, signal="dilute_celltype", nuisance=nuisance)
    for nuisance in ("loading_offset", "batch_shift", "missing_mnar")
)

# %%
pl.specificity(curve, signal="dilute_celltype", nuisance="loading_offset")

# %%
pl.scorecard(curve, statistic="range_over_noise")

# %% [markdown]
# ### Null control
#
# The null control gets its own sweep: its p-value cannot fall below `1 / (n_replicates + 1)`, so
# the 30 replicates above could never reach significance no matter how clearly a metric responded.

# %%
null_curve = meta.sweep(
    working,
    metrics,
    {"permute_celltype": pert.PermuteLabels("celltype", stratify_by="sample")},
    doses=(0.0, 1.0),
    n_replicates=100,
    prepare=prepare,
    seed=0,
)
meta.null_control(null_curve)

# %%
pl.null(null_curve, dose=1.0)

# %% [markdown]
# ### The scorecard
#
# One row per metric, standardised so the rows can be compared directly.

# %%
summarize(curve, signal="dilute_celltype", nuisance="loading_offset")
