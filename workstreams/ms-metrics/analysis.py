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
import anndata as ad
import numpy as np
import pandas as pd

from msmetrics.datasets import wu2025

# %%
adata = ad.read_h5ad(wu2025())

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
import os
import warnings

import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

from msmetrics import (
    bras,
    clisi_knn,
    compute_neighborhood_preservation,
    graph_connectivity,
    ilisi_knn,
    isolated_labels,
    kbet,
    kbet_per_label,
    lisi_knn,
    meta,
    nmi_ari_cluster_labels_kmeans,
    nmi_ari_cluster_labels_leiden,
    pcr_comparison,
    sbee,
    silhouette_batch,
    silhouette_label,
    variance_preservation,
)
from msmetrics import perturbations as pert
from msmetrics import plotting as pl
from msmetrics.utils import perform_leiden_clustering, point_cluster_distance

MIN_COMPLETENESS = 0.2
N_COMPONENTS = 15

# The sweep is the expensive part: 7 perturbations x 5 doses x 19 metrics, with no parallelism.
# These defaults are sized to finish in about ten minutes, which is enough to show what the harness
# reports and not enough for numbers worth quoting. Raise them through the environment for a real
# run -- `MSMETRICS_N_REPLICATES=30 MSMETRICS_NULL_REPLICATES=100` is roughly an hour.
#
# Replicate count is cheaper than it looks for the one thing that matters most. `reference_noise`
# pools the dose-0 replicates of *every* perturbation, so `sd_0` is estimated from 7 x N values, not
# N -- 21 even here. What thins out at this size is the per-dose means behind each curve, so read
# the response figures for shape and the scorecard for direction, not either for precision.
N_REPLICATES = int(os.environ.get("MSMETRICS_N_REPLICATES", "3"))
NULL_REPLICATES = int(os.environ.get("MSMETRICS_NULL_REPLICATES", "15"))
# Sorted and deduplicated, because the rest of the notebook reads DOSES[0] as the reference dose and
# DOSES[-1] as the strongest, and this is unvalidated environment input.
DOSES = tuple(sorted({float(dose) for dose in os.environ.get("MSMETRICS_DOSES", "0,0.25,0.5,0.75,1.0").split(",")}))
if len(DOSES) < 2:
    raise ValueError(f"MSMETRICS_DOSES needs at least two distinct doses, got {DOSES}.")

print(f"sweep: {len(DOSES)} doses {DOSES}, {N_REPLICATES} replicates, {NULL_REPLICATES} for the null control")

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


def scoreable(metric):
    """Refuse to score an embedding that has collapsed to a single point.

    At `missing_mnar` dose 1.0 every observed value is masked, `embed` mean-imputes a matrix that has
    no means left to take, falls back to zero, and hands the metrics an all-zero embedding. That is
    *finite*, so `scib_wrapper._embedding` passes it straight through -- and on identical points
    `silhouette_batch` and `bras` return 1.0, their perfect score, with `kbet_per_label` at 0.88.
    Nothing is separated when nothing is anywhere, so total destruction reads as flawless
    integration. Left in, that one dose supplies most of the apparent dynamic range of the batch
    metrics and points the wrong way.

    The rule is uniform rather than per-metric: a zero-variance embedding is not a valid input to
    anything defined on distances between cells, whether or not that particular metric happens to
    return a wrong-looking number. `variance_preservation` already reports `nan` at that dose, and
    `response_shape` already drops non-finite doses and reports `max_usable_dose`, so this makes the
    embedding metrics agree with machinery that is already there.
    """

    def guarded(a):
        embedding = np.asarray(a.obsm["X_pca"], float)
        if not np.isfinite(embedding).all() or float(embedding.std()) <= 1e-8:
            return float("nan")
        return metric(a)

    return guarded


def _shared(function):
    """Serve one key of a multi-output metric, computing it once per prepared dataset.

    `sweep` calls every entry of `metrics` independently, so the two keys of an NMI/ARI pair would
    otherwise cluster the same embedding twice. The cache is keyed on object *identity* and keeps a
    reference to the dataset it holds; keying on `id()` alone would be a correctness bug, because
    CPython reuses an id once the previous replicate is freed and the next one would then be served
    a stale score.
    """
    held = {"dataset": None, "value": None}

    def read(a, key):
        if held["dataset"] is not a:
            held["dataset"], held["value"] = a, function(a)
        return float(held["value"][key])

    return read


# Cluster once per replicate at a fixed resolution. `nmi_ari_cluster_labels_leiden` otherwise
# defaults to `optimize_resolution=True` and sweeps ten resolutions, which is both ten times the
# cost and a metric that re-tunes itself against every perturbed dataset -- an adaptive metric has
# no clean dose response, which is the one thing this notebook is trying to measure.
_kmeans = _shared(lambda a: nmi_ari_cluster_labels_kmeans(a, label_key="celltype"))
_leiden = _shared(
    lambda a: nmi_ari_cluster_labels_leiden(a, label_key="celltype", optimize_resolution=False, resolution=1.0)
)

# Grouped by what the metric claims to measure, which is also the order the scorecard reads in and
# the grouping `pl.response` is drawn by -- nineteen lines on one panel is not a figure.
FAMILIES = {
    "paired, label-blind": [
        "neighborhood_preservation",
        "point_cluster_distance",
        "variance_preservation",
    ],
    "biological signal conserved": [
        "silhouette_label",
        "clisi_knn",
        "lisi_knn",
        "nmi_kmeans",
        "ari_kmeans",
        "nmi_leiden",
        "ari_leiden",
        "isolated_labels",
        "graph_connectivity",
    ],
    "batch signal removed": [
        "silhouette_batch",
        "ilisi_knn",
        "kbet",
        "kbet_per_label",
        "bras",
        "sbee",
        "pcr_comparison",
    ],
}

metrics = {
    "neighborhood_preservation": scoreable(
        lambda a: compute_neighborhood_preservation(a, "X_pca_reference", "X_pca")["adjusted_overlap"]
    ),
    "point_cluster_distance": _point_cluster_distance,
    "variance_preservation": lambda a: variance_preservation(a.layers["pre_imputation"], a.X)["median_ratio"],
    "silhouette_label": scoreable(lambda a: silhouette_label(a, label_key="celltype")),
    "clisi_knn": scoreable(lambda a: clisi_knn(a, label_key="celltype")),
    # The unscaled twin of `clisi_knn`, summarised the way `ilisi_knn` summarises its own per-cell
    # array. `clisi_knn` is `(n_labels - median(lisi)) / (n_labels - 1)` and `n_labels` is 8 at every
    # dose here, so this row is an affine image of that one -- and every summary in this notebook is
    # either a within-row ratio or a rank, both of which cancel an affine map. The two rows come out
    # *identical* to the printed precision, which is the point of carrying it: it is the control that
    # says how much of a scorecard row is the metric and how much is the rescaling.
    "lisi_knn": scoreable(lambda a: float(np.median(lisi_knn(a, label_key="celltype")))),
    "nmi_kmeans": scoreable(lambda a: _kmeans(a, "nmi")),
    "ari_kmeans": scoreable(lambda a: _kmeans(a, "ari")),
    "nmi_leiden": scoreable(lambda a: _leiden(a, "nmi")),
    "ari_leiden": scoreable(lambda a: _leiden(a, "ari")),
    "isolated_labels": scoreable(lambda a: isolated_labels(a, label_key="celltype", batch_key="sample")),
    "graph_connectivity": scoreable(lambda a: graph_connectivity(a, label_key="celltype")),
    "silhouette_batch": scoreable(lambda a: silhouette_batch(a, label_key="celltype", batch_key="sample")),
    "ilisi_knn": scoreable(lambda a: ilisi_knn(a, batch_key="sample")),
    # Annotated `-> float` upstream but returns `(acceptance_rate, chi2, p_values)`; the rate is the
    # score and the other two are per-observation diagnostics.
    "kbet": scoreable(lambda a: kbet(a, batch_key="sample")[0]),
    "kbet_per_label": scoreable(lambda a: kbet_per_label(a, label_key="celltype", batch_key="sample")),
    "bras": scoreable(lambda a: bras(a, label_key="celltype", batch_key="sample")),
    "sbee": scoreable(lambda a: sbee(a, label_key="celltype", batch_key="sample")),
    # The only wrapper that is itself a before/after comparison, so it gets the same reference
    # embedding the paired metrics above use. The covariate is the batch: this asks how much of the
    # variance the batch explained changed.
    #
    # `scale=False` is not a preference here. The default rescaling clamps a negative difference to
    # zero, and every perturbation in this sweep *adds* batch variance rather than removing it, so
    # the scaled score is a flat 0.000 at every dose while the signed one runs 0.000, 0.089, 0.263.
    # Clamping would make the metric look blind when it is only pointed the other way.
    "pcr_comparison": scoreable(
        lambda a: pcr_comparison(a, "X_pca_reference", "X_pca", covariate_key="sample", scale=False)
    ),
}
assert set(metrics) == {name for family in FAMILIES.values() for name in family}

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
    # The same null in mirror image, and the one the batch metrics need: shuffling cell types says
    # nothing about whether `ilisi_knn` or `kbet` read their *batch* labels. Stratified for the same
    # reason, and it has room to move -- every cell type spans four or five of the five samples.
    "permute_sample": pert.PermuteLabels("sample", stratify_by="celltype"),
}

# The header should read as English, not as the dictionary keys the perturbations happen to use.
DAMAGE = {
    "dilute_celltype": "biology\nerased",
    "loading_offset": "cell size\nvaried",
    "batch_shift": "batch effect\nadded",
    "missing_mnar": "values\nmasked",
    "subsample": "fewer\ncells",
    "permute_celltype": "cell types\nshuffled",
    "permute_sample": "batches\nshuffled",
}

# %%
curve = meta.sweep(
    working,
    metrics,
    perturbations,
    doses=DOSES,
    n_replicates=N_REPLICATES,
    prepare=prepare,
    seed=0,
)
curve.head()

# %% [markdown]
# ### What failed, and whether that is a bug or a result
#
# `sweep` records a metric that raised as a `nan` carrying the exception text, which means a broken
# metric and a blind one look identical in every figure below. Read this cell before any of them.
#
# The expected entries are the four `n_neighbors = 90` metrics at `subsample` dose 1.0: stratifying
# on the full `celltype x sample` table with a floor of two cells per stratum leaves 77 cells, and a
# 90-neighbour graph over 77 points is not defined. That is the honest outcome and **not** something
# to fix by clamping `k` to the cell count -- a metric whose definition changes with the dose has no
# dose response left to measure, which is the confound this column exists to isolate.
# `response_shape` reports `max_usable_dose` for those rows instead.

# %%
failures = curve[curve["error"].fillna("") != ""]
print(f"{len(failures)} of {len(curve)} evaluations raised")
failures.groupby(["perturbation", "dose", "metric"])["error"].agg(["size", "first"])

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
# They do not even agree on where "no signal" sits. A rescaled silhouette is at chance at 0.5, an
# ARI at 0, a `variance_preservation` ratio is *ideal* at 1 and bad in both directions, and
# `kbet` is an acceptance rate whose good end is 1 while `pcr_comparison` here is a signed
# difference whose null is 0. Nothing below needs to know which: every summary in this notebook is
# either a within-row ratio, an absolute distance from the metric's own reference, or a rank, so the
# harness never has to register an expected value per metric. That is what lets a table hold
# nineteen metrics from three different lineages without a lookup of what "good" means for each.
#
# Everything the harness reports is therefore divided by that metric's *own* reference noise `sd₀`,
# the spread it shows across replicates when nothing has been done to the data. The result is in
# units of "how many detectable steps did the metric move", which is the same currency for every
# metric on the page.
#
# | Column | Comparable across metrics? | Why |
# | --- | --- | --- |
# | `value`, `floor`, `ceiling`, `dynamic_range` | **No** | Raw metric units. Kept for reading one metric's curve, never for ranking two. |
# | `range_over_noise` | **Yes, with care** | `dynamic_range / sd₀`. Comparable only while every `sd₀` is a real noise estimate — see the second caution below. |
# | `response_profile` shares | **Yes** | Each perturbation as a fraction of that metric's own strongest response. A ratio taken within one row, so `sd₀` cancels out of it exactly and it survives a degenerate reference dose. |
# | `signal_slope`, `nuisance_slope` | **Yes** | Already reference-SD per unit dose. |
# | `contrast` | **Yes** | Bounded in `[-1, 1]` by construction, so it cannot be inflated by a near-zero denominator. |
# | `spearman`, `monotone_fraction` | **Yes** | Rank-based and unit-free already. |
# | `detection_dose`, `saturation_dose` | **Yes, within one perturbation** | Expressed on the dose axis, which belongs to the perturbation rather than to the metric — but each perturbation has its own `dose_unit`, so do not compare a dose across columns. |
# | `p_value`, `z` | **Yes** | Both are positions within the metric's own null distribution. |
#
# One addition now that the wrapped metrics are in: `p_value` is only comparable **within the
# matching null**. A metric that never reads `.obs["celltype"]` returns `p = 1.000` under
# `permute_celltype` because it is paired, not because it failed a test it was sitting for. Compare
# a p-value down its own column, never across the two.
#
# Two cautions that follow from this:
#
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
#   cluster centroids are defined on the reference space, so at dose 0 it compares that space against
#   itself -- which leaves `sd₀` around `1e-5` and sends its `range_over_noise` into the tens of
#   thousands, against roughly 350 for `neighborhood_preservation`. That ratio is a collapsed
#   denominator, not a real advantage.
#
#   This is why the scorecard below reports *shares of each metric's own peak* rather than
#   `range_over_noise`: the shares are a within-row ratio, so `sd₀` cancels and they stay meaningful
#   for a degenerate metric even though its peak does not. `response_profile` flags such a metric in
#   `sd_0_degenerate` and the table blanks only the affected column.
# - `sd₀` is estimated from `n_replicates` values at dose 0, so every standardised column inherits
#   that estimate's uncertainty. With the 30 replicates used here it is stable enough to rank
#   metrics; with 5 it would not be.


# %%
def summarize(curve, *, signal, nuisance, null_curve=None, labels=None, ax=None):
    """One row per metric: what it responds to, how strongly, and whether it tracks biology.

    The response columns are each perturbation's share of that metric's own strongest response, so a
    row reads as a profile -- what this metric is actually measuring -- rather than as six numbers on
    an unbounded scale. Being a ratio taken within one row, the shares do not depend on `sd_0`, which
    is what makes them comparable between metrics whose reference noise differs by orders of
    magnitude. The absolute scale survives as the single `peak` column.

    `labels` maps perturbation names to the wording used in the header. The names are dictionary keys
    chosen for code, and a reader of the figure should not have to decode `dilute_celltype` to work
    out that the column means "the biological signal was removed".
    """
    import matplotlib.pyplot as plt
    from plottable import ColumnDefinition, Table
    from plottable.cmap import centered_cmap

    profile = meta.response_profile(curve).set_index("metric")
    detection = (
        meta.sensitivity(curve)
        .groupby(["perturbation", "metric"])["detection_dose"]
        .first()
        .unstack(level="perturbation")
    )
    contrast = meta.specificity(curve, signal=signal, nuisance=nuisance).set_index("metric")["contrast"]

    responses = [name for name in profile.columns if name in set(curve["perturbation"])]
    table = profile[responses].copy()
    # `peak` is meaningless where the reference dose is degenerate, but the shares beside it are not,
    # because `sd_0` cancels out of a within-row ratio. Suppress only the column that is affected.
    table["peak"] = pd.Series(
        [
            "no noise\nfloor" if degenerate else f"{value:,.0f}x"
            for value, degenerate in zip(profile["peak"], profile["sd_0_degenerate"])
        ],
        index=profile.index,
    )
    table["detects"] = detection[signal]
    table["contrast"] = contrast

    def worded(value):
        if not np.isfinite(value):
            return "moves for\nneither"
        return f"{'biology' if value > 0 else 'the confound'}\n({value:+.2f})"

    # Mapped off the column rather than zipped over `contrast`. A list assigns by position while
    # every other assignment here aligns by index, so a reordered `contrast` would silently have
    # attached each metric's wording to a different row while the number beside it stayed right.
    table["tracks_bio"] = table["contrast"].map(worded)
    nulls = []
    if null_curve is not None:
        # One column per permutation, not one pooled column. `null_control` returns a row per
        # (perturbation, metric), so collapsing on `metric` alone would assign from a duplicate
        # index and raise -- and the two nulls answer different questions anyway. Shuffling cell
        # types says nothing about whether `ilisi_knn` reads its batch labels, and vice versa.
        control = meta.null_control(null_curve)
        for name, group in control.groupby("perturbation", sort=False):
            column = f"null_p_{name}"
            table[column] = group.set_index("metric")["p_value"]
            nulls.append((column, name))
    table = table.reset_index()

    ax = ax if ax is not None else plt.subplots(figsize=(1.8 * len(table.columns), 1.2 + 0.7 * len(table)))[1]
    shares = plt.get_cmap("Blues")
    labels = labels or {}

    definitions = [
        ColumnDefinition("metric", width=2.4, textprops={"ha": "left", "weight": "bold"}),
        *[
            ColumnDefinition(
                name,
                title=labels.get(name, name.replace("_", "\n")),
                width=1.0,
                group="what it notices  —  each damage as a % of this metric's own strongest response",
                formatter=lambda v: f"{v:.0%}",
                # Fixed [0, 1] domain, so a cell means the same thing in every row.
                cmap=lambda v: shares(0.06 + 0.5 * v) if np.isfinite(v) else "#f2f2f2",
            )
            for name in responses
        ],
        # Only the response block gets a group header. A one-column group whose label is wider than
        # its column just collides with its neighbour, so these say everything in the title instead.
        ColumnDefinition("peak", title="strongest response\n(noise widths)", width=1.5),
        ColumnDefinition(
            "detects",
            title=f"smallest '{labels.get(signal, signal)}'\nit can see",
            width=1.5,
            formatter=lambda v: "never" if not np.isfinite(v) else f"{v:.3f}",
        ),
        ColumnDefinition(
            "tracks_bio",
            title=f"moves more for\nbiology or {labels.get(nuisance, nuisance)}?",
            width=1.7,
            text_cmap=centered_cmap(table["contrast"].fillna(0), cmap=plt.get_cmap("RdBu"), center=0),
        ),
        *[
            ColumnDefinition(
                column,
                title=f"p when\n{labels.get(name, name)}",
                width=1.4,
                formatter="{:.3f}",
                group="does it read the labels at all?",
            )
            for column, name in nulls
        ],
    ]

    Table(
        table,
        ax=ax,
        index_col="metric",
        # `contrast` is carried only to colour `tracks_bio`; `metric` becomes the index.
        columns=[name for name in table.columns if name not in {"metric", "contrast"}],
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
def grid(n, width=4.0, height=3.2, ncols=4):
    """A wrapped grid of `n` axes, for the plotting helpers that default to a single row.

    `pl.response` and `pl.null` build `plt.subplots(1, n)` when no axes are given, which at nineteen
    metrics is a figure six feet wide. Both accept `axes=`, so the wrapping belongs here rather than
    in the package.
    """
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    for extra in axes.ravel()[n:]:
        extra.set_visible(False)
    figure.tight_layout()
    return axes.ravel()[:n]


for family, names in FAMILIES.items():
    axes = grid(len(perturbations))
    pl.response(curve, metrics=names, axes=axes)
    axes[0].figure.suptitle(family, y=1.0, fontsize=13, weight="bold")
    axes[0].figure.tight_layout()
    plt.show()

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
# `spearman` rather than `range_over_noise`. The heatmap's colour limit is a global maximum, and
# `point_cluster_distance` has no usable noise floor -- its `range_over_noise` reaches five figures
# where its neighbours reach three -- so that statistic flattens all nineteen rows to one shade.
# Spearman is rank-based, bounded in [-1, 1], and carries the response *direction*, which is the one
# thing the percentage scorecard below drops when it takes an absolute dynamic range.
pl.scorecard(curve, statistic="spearman")

# %% [markdown]
# ### Null control
#
# The null control gets its own sweep, at more replicates than the dose series above: its p-value
# cannot fall below `1 / (n_replicates + 1)`, so a metric could never reach significance on the
# replicate count used for the curves no matter how clearly it responded.
#
# **Two nulls, because there are two kinds of label.** Shuffling cell types asks whether a metric
# reads the biology; shuffling batches asks whether it reads the batch. A metric is only tested by
# the null that matches the annotation it consumes -- `permute_celltype` says nothing about
# `ilisi_knn`, which never looks at `celltype`, and `permute_sample` says nothing about
# `silhouette_label`. Both columns appear in the scorecard so that pairing stays visible instead of
# being averaged away.
#
# This is also the part of the notebook that changed most when the `scib-metrics` wrappers landed.
# The three original metrics are all *paired*: they compare a before matrix or embedding against an
# after one and never touch `.obs`, so they moved by exactly zero under permutation and the null
# control had nothing to report. The wrapped metrics take `label_key` and `batch_key` directly, so
# this is the first run where a p-value here can come back meaning something.
#
# A reduced render -- see the parameters printed above -- estimates `sd_0` from few replicates, and
# every standardised column inherits that. Read a cheap run for *whether* something moved, and the
# full run for numbers worth quoting.

# %%
null_curve = meta.sweep(
    working,
    metrics,
    {name: perturbations[name] for name in ("permute_celltype", "permute_sample")},
    doses=(DOSES[0], DOSES[-1]),
    n_replicates=NULL_REPLICATES,
    prepare=prepare,
    seed=0,
)
meta.null_control(null_curve)

# %%
for name in ("permute_celltype", "permute_sample"):
    axes = grid(len(metrics), width=3.4, height=2.6, ncols=5)
    pl.null(null_curve, dose=DOSES[-1], perturbation=name, axes=axes)
    axes[0].figure.suptitle(DAMAGE[name].replace("\n", " "), y=1.0, fontsize=13, weight="bold")
    axes[0].figure.tight_layout()
    plt.show()

# %% [markdown]
# ### The scorecard
#
# One row per metric. Each response cell is that perturbation's share of the metric's *own* strongest
# response, so a row reads as a profile -- what is this metric actually measuring? -- instead of six
# numbers on an unbounded scale. The absolute strength survives as the single `peak` column, blanked
# where the reference dose is degenerate and the number would be meaningless.
#
# Read a row across: a metric whose peak sits on `missing_mnar` with near-zero elsewhere is a
# missingness detector, whatever else it is named; one that responds to everything at roughly equal
# strength is a generic "something changed" alarm and cannot tell you *what* changed.
#
# Read a column down to pick a metric for a failure mode you actually care about. That is what the
# nineteen rows buy over the original three: the question stops being "does this metric work" and
# becomes "which of these is the right instrument", and those are answered by different columns.
#
# Two traps specific to a table this wide:
#
# - **A high share is not a good score.** The shares say what a metric is sensitive *to*, not
#   whether that is what you wanted. A batch metric with a large `fewer cells` share is reporting
#   sample size, and the fact that it also moves under `batch effect added` does not redeem it --
#   read `tracks_bio` and the two null columns next to every share.
# - **Metrics that share a computation share their failures.** Every row here reads the same
#   fifteen-component SVD that `prepare` builds, so a row of near-identical profiles is evidence
#   about the embedding, not nineteen independent confirmations. `lisi_knn` and `clisi_knn` are the
#   deliberate example: same quantity, one rescaled, carried so that the size of the gap between
#   them is visible rather than assumed.

# %%
scorecard = summarize(
    curve, signal="dilute_celltype", nuisance="loading_offset", null_curve=null_curve, labels=DAMAGE
)

# Written out so the copy in `docs/` is a build product of this notebook rather than a hand export
# that silently goes stale. It records whatever sweep the run used, so re-commit it from a local run
# rather than from CI, which renders a smaller one still.
scorecard.figure.savefig("docs/meta-benchmark-wu2025.png", dpi=200, bbox_inches="tight")
