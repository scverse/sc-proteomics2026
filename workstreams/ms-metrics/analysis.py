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
from msmetrics import utils

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

# Visualize the missing values (diagnose)
utils.draw_missingness(
    X=adata.X,
    xlabel="Features",
    ylabel="Samples",
    title="Missingness Heatmap",
)

# Flag features that are more than 60 % missing
print("--> Dropping too incomplete features", flush = True)
apt.pp.filter_data_completeness(
    adata = adata,
    max_missing_fraction = 0.6,
    action = "drop",
)

# Compute actual completeness in features
apt.metrics.fraction_complete(
    adata=adata,
)

# Compute median intensity of features
adata.obs["median_intensity"] = np.nanmedian(adata.X, axis=1)

# Flag outliers based on median absolute deviation
adata.obs["outlier"] = utils.mad_outlier(adata.obs["fraction_complete"], n_mad=3, direction="down") | utils.mad_outlier(
    adata.obs["median_intensity"], n_mad=3, direction="both"
)

# Remove outliers
print("--> Dropping too outlier samples", flush = True)
print(f"Removing {adata.obs['outlier'].sum()} outlier samples: more than 3 MADs from the median", flush = True)
adata = adata[~adata.obs["outlier"]]

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
import os
import warnings

from sklearn.utils.extmath import randomized_svd

from msmetrics import compute_neighborhood_preservation, meta, variance_preservation
from msmetrics import perturbations as pert
from msmetrics import plotting as pl
from msmetrics.utils import perform_leiden_clustering, point_cluster_distance

MIN_COMPLETENESS = 0.2
N_COMPONENTS = 15

# The sweep is the expensive part, so CI renders a cheaper version of it. The defaults are the full
# run; the environment only ever makes it smaller.
N_REPLICATES = int(os.environ.get("MSMETRICS_N_REPLICATES", "30"))
NULL_REPLICATES = int(os.environ.get("MSMETRICS_NULL_REPLICATES", "100"))
# Sorted and deduplicated, because the rest of the notebook reads DOSES[0] as the reference dose and
# DOSES[-1] as the strongest, and this is unvalidated environment input.
DOSES = tuple(sorted({float(dose) for dose in os.environ.get("MSMETRICS_DOSES", "0,0.2,0.4,0.6,0.8,1.0").split(",")}))
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
    doses=DOSES,
    n_replicates=N_REPLICATES,
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
# | `range_over_noise` | **Yes, with care** | `dynamic_range / sd₀`. Comparable only while every `sd₀` is a real noise estimate — see the second caution below. |
# | `response_profile` shares | **Yes** | Each perturbation as a fraction of that metric's own strongest response. A ratio taken within one row, so `sd₀` cancels out of it exactly and it survives a degenerate reference dose. |
# | `signal_slope`, `nuisance_slope` | **Yes** | Already reference-SD per unit dose. |
# | `contrast` | **Yes** | Bounded in `[-1, 1]` by construction, so it cannot be inflated by a near-zero denominator. |
# | `spearman`, `monotone_fraction` | **Yes** | Rank-based and unit-free already. |
# | `detection_dose`, `saturation_dose` | **Yes, within one perturbation** | Expressed on the dose axis, which belongs to the perturbation rather than to the metric — but each perturbation has its own `dose_unit`, so do not compare a dose across columns. |
# | `p_value`, `z` | **Yes** | Both are positions within the metric's own null distribution. |
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
    if null_curve is not None:
        table["null_p"] = meta.null_control(null_curve).set_index("metric")["p_value"]
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
        *(
            [ColumnDefinition("null_p", title="p when the labels\nare shuffled", width=1.4, formatter="{:.3f}")]
            if null_curve is not None
            else []
        ),
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
# The null control gets its own sweep, at more replicates than the dose series above: its p-value
# cannot fall below `1 / (n_replicates + 1)`, so a metric could never reach significance on the
# replicate count used for the curves no matter how clearly it responded.
#
# A reduced render -- see the parameters printed above -- estimates `sd_0` from few replicates, and
# every standardised column inherits that. Read a cheap run for *whether* something moved, and the
# full 30-replicate run for numbers worth quoting.

# %%
null_curve = meta.sweep(
    working,
    metrics,
    {"permute_celltype": pert.PermuteLabels("celltype", stratify_by="sample")},
    doses=(DOSES[0], DOSES[-1]),
    n_replicates=NULL_REPLICATES,
    prepare=prepare,
    seed=0,
)
meta.null_control(null_curve)

# %%
pl.null(null_curve, dose=1.0)

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

# %%
# The header should read as English, not as the dictionary keys the perturbations happen to use.
DAMAGE = {
    "dilute_celltype": "biology\nerased",
    "loading_offset": "cell size\nvaried",
    "batch_shift": "batch effect\nadded",
    "missing_mnar": "values\nmasked",
    "subsample": "fewer\ncells",
    "permute_celltype": "labels\nshuffled",
}

summarize(curve, signal="dilute_celltype", nuisance="loading_offset", null_curve=null_curve, labels=DAMAGE)
