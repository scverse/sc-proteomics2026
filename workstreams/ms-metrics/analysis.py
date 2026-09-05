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
import pandas as pd
import numpy as np
import anndata as ad
import alphapepttools as apt

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
    X = adata.X,
    xlabel = "Features",
    ylabel = "Samples",
    title = "Missingness Heatmap",
)

# Flag features that are more than 60 % missing
number_of_samples = adata.shape[0]  # number of samples
apt.pp.filter_data_completeness(
    adata = adata,
    max_missing_fraction = 0.6,
    action = "flag"
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

# %%
# ### 4. Normalization 

# %%
# ### 5. Imputation

# %%
# ### 6. Batch correction

# %%
# ### 7. Differential expression (out of scope)