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

# %% [markdown]
# ### Getting the main dataset

# %%
adata = ad.read_h5ad(wu2025())
adata

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
# It also carries two things the paired metrics need: the matrix as it stood before imputation, and
# the embedding of the *untouched* dataset, indexed by cell name so it survives the per-replicate
# subsampling. `neighborhood_preservation` then asks the same question at every dose — how far did
# this perturbation move the local structure away from the original data?


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


reference_embedding = pd.DataFrame(embed(np.asarray(working.X, float))[1], index=working.obs_names)


def prepare(a, rng):
    observed = np.asarray(a.X, float)
    imputed, embedding = embed(observed)
    a.layers["pre_imputation"] = observed
    a.X = imputed
    a.obsm["X_pca"] = embedding
    a.obsm["X_pca_reference"] = reference_embedding.loc[a.obs_names].to_numpy()
    return a


metrics = {
    "neighborhood_preservation": lambda a: compute_neighborhood_preservation(a, "X_pca_reference", "X_pca")[
        "adjusted_overlap"
    ],
    "variance_preservation": lambda a: variance_preservation(a.layers["pre_imputation"], a.X)["median_ratio"],
}

perturbations = {
    "dilute_celltype": pert.DiluteSignal("celltype"),  # biology
    "loading_offset": pert.InjectLoadingOffset(),  # cell size
    "batch_shift": pert.InjectBatchShift("sample"),  # batch
    "missing_mnar": pert.InjectMissing(mechanism="mnar"),  # missingness, hence imputation strength
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

# %%
meta.reference_noise(curve)

# %% [markdown]
# ### Is the response monotone, and how much of it is usable?
#
# Read `range_over_noise`, not `dynamic_range`: raw ranges are not comparable across metrics on
# different scales, whereas dividing each metric's response by its own noise puts them in the same
# units of detectability. `max_usable_dose` shows where a sweep ran out of computable doses --
# masking every observed value leaves `variance_preservation` nothing to score, so its curve stops
# at 0.8.

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
# `neighborhood_preservation` comes out **negative against the loading offset** -- it responds
# harder to cell size than to cell type, so a drop in it does not by itself mean biology was lost.
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
#
# Both metrics come out at exactly p = 1, z = 0 -- neither of them reads `.obs` at all, so shuffling
# the labels cannot move them. That is the finding, not a failure: msmetrics currently ships no
# label-aware metric for this control to bite on.

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
