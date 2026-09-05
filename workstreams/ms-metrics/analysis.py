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
# the data at increasing dose, recomputes each metric, and reads the resulting curve four ways:
# does the metric collapse when the signal is destroyed (`null_control`), how much damage does it
# take to notice (`sensitivity`), is the response monotone and wide enough to read
# (`response_shape`), and does it move with biology or with a technical confound (`specificity`).

# %%
from msmetrics import meta, variance_preservation
from msmetrics import perturbations as pert
from msmetrics import plotting as pl


def prepare(adata, rng):
    """One embedding, shared by every metric.

    Every metric in a sweep must see the same preprocessing, or a difference in their measured
    sensitivity is partly a difference in their PCA rather than in the metrics.
    """
    X = np.asarray(adata.X, dtype=float)
    column_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    X = np.where(np.isfinite(X), X, np.nan_to_num(column_mean)[None, :])
    X = X - X.mean(axis=0)
    adata.obsm["X_pca"] = X @ np.linalg.svd(X, full_matrices=False)[2][:10].T
    return adata


def group_separation(adata, key):
    """Toy stand-in for a real batch/bio metric: between-group variance fraction on the embedding."""
    embedding, groups = adata.obsm["X_pca"], adata.obs[key].to_numpy()
    grand = embedding.mean(axis=0)
    between = sum(
        (groups == level).sum() * np.square(embedding[groups == level].mean(axis=0) - grand)
        for level in pd.unique(groups)
    )
    total = np.square(embedding - grand).sum(axis=0)
    usable = total > 0
    return float(np.mean(between[usable] / total[usable]))


metrics = {
    "bio": lambda a: group_separation(a, "cell_type"),
    "batch": lambda a: group_separation(a, "batch"),
}

perturbations = {
    "dilute": pert.DiluteSignal("cell_type"),  # biology
    "loading": pert.InjectLoadingOffset(),  # cell size
    "batch_shift": pert.InjectBatchShift("batch"),  # batch
    "missing": pert.InjectMissing(mechanism="mnar"),  # imputation strength, with the imputer above
}

curve = meta.sweep(adata, metrics, perturbations, n_replicates=20, prepare=prepare, seed=0)
curve.head()

# %%
meta.response_shape(curve)

# %%
# Read `contrast`: positive means the metric tracks biology more strongly than the confound. Read
# both slopes too, since a contrast near zero is produced both by a metric that responds to
# everything and by one that responds to nothing.
meta.specificity(curve, signal="dilute", nuisance="loading")

# %%
pl.response(curve, metrics=["bio", "batch"])
pl.scorecard(curve)
pl.specificity(curve, signal="dilute", nuisance="loading")

# %% [markdown]
# The null control gets its own sweep. Its p-value cannot fall below `1 / (n_replicates + 1)`, so
# the 20 replicates above could never reach significance no matter how clearly a metric responds.

# %%
null_curve = meta.sweep(
    adata,
    metrics={"bio": metrics["bio"], "batch": metrics["batch"]},
    perturbations={"permute_ct": pert.PermuteLabels("cell_type", stratify_by="batch")},
    doses=(0.0, 1.0),
    n_replicates=100,
    prepare=prepare,
    seed=0,
)
pl.null(null_curve, dose=1.0)
meta.null_control(null_curve)

# %% [markdown]
# Paired metrics get their own sweep, over the perturbation that stashes the values it removed.
# `variance_preservation` needs the matrix from before the masking, which only `InjectMissing`
# records; asking for it after any other perturbation gives a `KeyError` row rather than a number.
#
# The expected reading: mean imputation piles imputed values onto each feature's centre, so the
# variance ratio falls steadily as more values are imputed. The dose-0 value already sits below 1
# because `prepare` imputes the missingness the dataset arrived with.

# %%
imputation_curve = meta.sweep(
    adata,
    metrics={"variance_preservation": lambda a: variance_preservation(a.layers["truth"], a.X)["median_ratio"]},
    perturbations={"missing": pert.InjectMissing(mechanism="mnar")},
    doses=(0.0, 0.2, 0.4, 0.6),
    n_replicates=20,
    prepare=prepare,
    seed=0,
)
pl.response(imputation_curve)
meta.response_shape(imputation_curve)
