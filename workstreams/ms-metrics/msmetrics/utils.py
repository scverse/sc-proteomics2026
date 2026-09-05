"""Metrics for diagnosing the effect of imputation on single-cell proteomics data."""

import warnings

import alphapepttools as apt
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def mad_outlier(values, n_mad=3.0, direction="both"):
    """Boolean mask of values further than ``n_mad`` MADs from the median.

    References
    ----------
    Heumos, L. et al. (2023). Best practices for single-cell analysis across
    modalities. Nat Rev Genet 24, 550-572. https://doi.org/10.1038/s41576-023-00586-w
    """
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = median_abs_deviation(values, nan_policy="omit")
    if mad == 0:
        return ~np.isfinite(values)

    deviation = {
        "both": np.abs(values - median),
        "up": values - median,
        "down": median - values,
    }[direction]
    return (~np.isfinite(values)) | (deviation > n_mad * mad)

def draw_missingness(
    X: np.ndarray,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (5, 2),
    return_figure: bool = False,
) -> plt.Figure:
    """Plot a matrix as a present/absent map, with rows and columns sorted by completeness.

    Parameters
    ----------
    X
        Observations x features array, missing values encoded as `nan`.
    xlabel
        Label for the x-axis. If `None` (default), the axis is left unlabelled.
    ylabel
        Label for the y-axis. If `None` (default), the axis is left unlabelled.
    title
        Title placed above the map. If `None` (default), no title is set.
    figsize
        Figure size in inches.
    return_figure
        Whether to return fig, e.g. for saving the plot


    Returns
    -------
    plt.Figure
        Figure holding the map and its legend.

    Raises
    ------
    ValueError
        If `X` is not two-dimensional.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("`X` must be a two-dimensional array of shape (observations, features).")

    colors = {
        "missing": apt.pl.BaseColors.get("lightred"),
        "measured": apt.pl.BaseColors.get("green"),
    }
    palette = np.array([colors["missing"], colors["measured"]])

    measured = np.isfinite(X).astype(int)
    measured = measured[:, np.argsort(measured.sum(axis=0))[::-1]]
    measured = measured[np.argsort(measured.sum(axis=1))[::-1], :]

    fig, axes = apt.pl.create_figure(ncols=2, figsize=figsize, width_ratios=[6, 1])

    ax_map = axes[0]
    ax_map.imshow(palette[measured], aspect="auto")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    apt.pl.label_axes(ax=ax_map, xlabel=xlabel, ylabel=ylabel, title=title)

    ax_legend = axes[1]
    ax_legend.axis("off")
    apt.pl.add_legend_to_axes(ax=ax_legend, levels=colors)

    if return_figure:
        return fig
    else:
        plt.show()


def variance_preservation(
    observed: np.ndarray,
    imputed: np.ndarray,
    *,
    min_valid: int = 3,
) -> dict[str, np.ndarray | float]:
    """Variance preservation of an imputation.

    Quantify how strongly imputation distorted the spread of each feature, by comparing the
    variance of a feature after imputation to the variance of its observed values before
    imputation:

        variance_ratio = Var(X_imputed) / Var(X_observed)

    where `X_observed` are the non-missing values of a feature before imputation and
    `X_imputed` are all values of the same feature after imputation.

    A ratio of 1 means the imputed values left the spread of the feature untouched. Ratios
    below 1 indicate variance shrinkage, as expected from mean or median imputation, which
    piles imputed values onto the centre of the distribution and thereby inflates apparent
    reproducibility. Ratios above 1 indicate variance inflation, as expected from downshifted
    gaussian imputation, which places imputed values away from the observed distribution and
    can manufacture differential abundance.

    Parameters
    ----------
    observed
        Observations x features array before imputation, missing values encoded as `nan`.
    imputed
        Observations x features array after imputation, of the same shape as `observed`.
        Remaining `nan` values are ignored, since imputing only a subset of features is a
        legitimate workflow.
    min_valid
        Minimum number of observed values a feature needs before imputation for its ratio to
        be computed. Features below the threshold are set to `nan`.

    Returns
    -------
    dict
        `per_feature`
            Variance ratio per feature, `nan` where it could not be computed.
        `median_ratio`
            Median ratio across incompletely observed features.

    Raises
    ------
    ValueError
        If the arrays are not two-dimensional or differ in shape.

    Examples
    --------
    Compare an imputed matrix against the original one:

    .. code-block:: python

        import alphapepttools as apt
        import msmetrics as msm

        adata_imputed = apt.pp.impute_gaussian(adata, copy=True)
        result = msm.variance_preservation(adata.X, adata_imputed.X)

        print(result["median_ratio"])

    A median ratio above 1 is the expected signature of downshifted gaussian imputation, while
    median imputation of the same data gives a ratio well below 1, since every imputed value is
    placed exactly at the feature centre.

    Inspect the features whose spread was distorted the most:

    .. code-block:: python

        adata.var["variance_ratio"] = result["per_feature"]
        print(adata.var["variance_ratio"].sort_values().head())

    Notes
    -----
    Since imputation only fills missing values and leaves observed ones untouched, the ratio of a
    feature cannot fall below `1 - fraction_missing`. That floor is reached exactly when every
    missing value is filled with the centre of the feature. A low ratio on a sparsely observed
    feature may therefore be that floor rather than a poor imputation, and `median_ratio` mixes
    both regimes when missingness varies widely across features. Read it together with the
    missingness of the features it summarises.
    """
    observed = np.asarray(observed)
    imputed = np.asarray(imputed)

    if observed.ndim != 2 or imputed.ndim != 2:
        raise ValueError("`observed` and `imputed` must be two-dimensional arrays of shape (observations, features).")
    if observed.shape != imputed.shape:
        raise ValueError(
            f"`observed` and `imputed` must have the same shape, got {observed.shape} and {imputed.shape}."
        )

    n_valid = np.isfinite(observed).sum(axis=0)
    fraction_missing = 1.0 - n_valid / observed.shape[0]

    ratio = np.full(observed.shape[1], np.nan)
    computable = n_valid >= max(min_valid, 2)

    # Both variances use the unbiased estimator, since the observed one is estimated from `n_valid`
    # values and the imputed one from all observations. Degenerate slices are masked out below, so
    # the warnings they raise carry no information for the caller.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        variance_observed = np.nanvar(observed[:, computable], axis=0, ddof=1)
        variance_imputed = np.nanvar(imputed[:, computable], axis=0, ddof=1)

    ratio[computable] = np.divide(
        variance_imputed,
        variance_observed,
        out=np.full_like(variance_observed, np.nan),
        where=variance_observed > 0,
    )

    # Fully observed features carry a ratio of 1 by construction and would only pull the median towards 1.
    summarised = np.isfinite(ratio) & (fraction_missing > 0)

    return {
        "per_feature": ratio,
        "median_ratio": float(np.median(ratio[summarised])) if summarised.any() else float("nan"),
    }

def perform_leiden_clustering(
    adata: AnnData,
    *,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    random_state: int = 0,
):
    """Cluster observations in AnnData using Leiden clustering.

    Parameters
    ----------
    adata
        Input with shape (n_cells, n_features).
    resolution
        Leiden resolution parameter. Higher values generally produce more
        clusters.
    n_neighbors
        Number of neighbors used to construct the neighborhood graph.
    random_state
        Random seed for reproducibility.
    """
    if adata.shape[0] < 2:
        raise ValueError("At least two observations are required.")

    sc.pp.neighbors(
        adata,
        n_neighbors=min(n_neighbors, adata.shape[0] - 1),
        use_rep="X",
    )

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        key_added="_leiden",
    )

def point_cluster_distance(adata: AnnData, before_layer: str, after_layer: str, cluster_key: str = "_leiden") -> float:
    """Point Cluster Distance (PCD).
    
    Measures preservation of global cell-cell structure between `before`
    and `after` using distances from cells to cluster centroids.

    Clusters are determined in `before`. The same cluster assignments are
    then used to compute centroids in both spaces. Cell-to-centroid distance
    matrices are flattened and compared using Spearman correlation.

    Parameters
    ----------
    before
        Shape (n_cells, n_features_before). Matrix before a pp step.
    after
        Shape (n_cells, n_features_after). Matrix after a pp step.
    cluster_labels
        Clusters assignment of the cells in before and after.

    Returns
    -------
    float
        Spearman correlation between the PCD matrices. Higher is better.
    """
    # retreive clustering labels from the AnnData object
    labels = adata.obs[cluster_key].cat.codes.to_numpy()
    return _point_cluster_distance(
        adata.layers[before_layer],
        adata.layers[after_layer],
        labels,
    )

def _point_cluster_distance(
    before: csr_matrix,
    after: csr_matrix,
    cluster_labels: np.ndarray[int],
) -> float:
    """Point Cluster Distance (PCD).

    Compute PCD given the before and after matrices and labels.
    """
    if before.shape[0] != after.shape[0]:
        raise ValueError(
            "`before` and `after` must contain the same number of cells."
        )

    # Compute centroids in before and after using the cluster labels.
    n_clusters = np.unique(cluster_labels).shape[0]
    def centroids(x: csr_matrix) -> np.ndarray:
        return np.vstack([
            np.asarray(x[cluster_labels == cluster].mean(axis=0)).ravel()
            for cluster in range(n_clusters)
        ])

    before_centroids = centroids(before)
    after_centroids = centroids(after)

    # Distance from every cell to every centroid.
    # Shapes: (n_cells, n_clusters)
    before_distances = pairwise_distances(
        before, before_centroids, metric="euclidean"
    )
    after_distances = pairwise_distances(
        after, after_centroids, metric="euclidean"
    )

    # Compare the flattened PCD matrices using Spearman correlation.
    correlation = spearmanr(
        before_distances.ravel(),
        after_distances.ravel(),
    ).statistic

    return float(correlation)

def _knn_adjacency(embedding: np.ndarray, n_neighbors: int) -> csr_matrix:
    """Binary observations x observations adjacency of the `n_neighbors` nearest neighbours, excluding self."""
    # Querying without arguments makes scikit-learn exclude every observation from its own
    # neighbourhood, which stays correct when duplicate observations make the self-match ambiguous.
    indices = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding).kneighbors(return_distance=False)

    n_obs = indices.shape[0]
    indptr = np.arange(n_obs + 1) * n_neighbors
    return csr_matrix(
        (np.ones(indices.size, dtype=bool), indices.ravel(), indptr),
        shape=(n_obs, n_obs),
    )


def neighborhood_preservation(
    before: np.ndarray,
    after: np.ndarray,
    *,
    n_neighbors: int = 20,
) -> dict[str, np.ndarray | float]:
    """Neighborhood preservation between two embeddings of the same observations.

    Quantify how strongly a processing step rearranged the local structure of the data, by
    comparing the k nearest neighbours of every observation before and after the step:

        overlap_i = |N_before(i) intersect N_after(i)| / k

    where `N(i)` is the set of the k nearest neighbours of observation `i`, excluding itself.

    Because two unrelated embeddings still share neighbours by chance, the summary is corrected
    against that floor, in the spirit of an adjusted Rand index:

        adjusted_overlap = (mean_overlap - chance_overlap) / (1 - chance_overlap)

    with `chance_overlap = k / (n_obs - 1)`, the expected overlap of two random neighbourhoods.
    An adjusted overlap of 1 means the step left every neighbourhood untouched, 0 means the
    resulting embedding is no more similar to the original one than a random one would be.

    Parameters
    ----------
    before
        Observations x dimensions embedding before the processing step, e.g. `adata.obsm["X_pca"]`.
    after
        Observations x dimensions embedding after the processing step, over the same observations
        in the same order.
    n_neighbors
        Number of nearest neighbours per observation, excluding the observation itself.

    Returns
    -------
    dict
        `per_observation`
            Raw overlap fraction per observation, between 0 and 1.
        `mean_overlap`
            Mean raw overlap across observations.
        `chance_overlap`
            Overlap expected from two unrelated embeddings, `n_neighbors / (n_obs - 1)`.
        `adjusted_overlap`
            Mean overlap rescaled so that 1 is perfect preservation and 0 is chance level.
            `nan` where the correction is undefined, i.e. when every observation neighbours
            every other one.

    Examples
    --------
    Check how far imputation moved the cells relative to each other:

    .. code-block:: python

        import alphapepttools as apt
        import msmetrics as msm

        apt.pp.bpca(adata)
        adata_imputed = apt.pp.impute_gaussian(adata, copy=True)
        apt.pp.bpca(adata_imputed)

        result = msm.neighborhood_preservation(adata.obsm["X_pca"], adata_imputed.obsm["X_pca"])
        print(result["adjusted_overlap"])

    Locate the cells whose neighbourhood was rearranged the most:

    .. code-block:: python

        adata.obs["neighborhood_overlap"] = result["per_observation"]
        sc.pl.embedding(adata, basis="X_umap", color="neighborhood_overlap")

    Notes
    -----
    The chance of random overlaps increases with neihborhood size and dataset size. 
    Thus, this metric is not comparable between different datasets with different numbers of samples.
    """
    before = np.asarray(before)
    after = np.asarray(after)

    n_obs = before.shape[0]

    # Binary nearest neighbors graphs
    before_knn_graph = _knn_adjacency(before, n_neighbors)
    after_knn_graph = _knn_adjacency(after, n_neighbors)

    # If shared: 1x1=1
    # If not shared: 1x0 = 0/0x1 = 0
    shared = before_knn_graph.multiply(after_knn_graph)

    # Row-wise/observation-wise summation
    per_observation = np.asarray(shared.sum(axis=1)).ravel() / n_neighbors

    mean_overlap = float(per_observation.mean())

    chance_overlap = n_neighbors / (n_obs - 1)

    # Adjust by chance
    # Maximum dynamic range is (random subsample to 1)
    chance_overlap = (mean_overlap - chance_overlap) / (1 - chance_overlap)

    return {
        "per_observation": per_observation,
        "mean_overlap": mean_overlap,
        "chance_overlap": chance_overlap,
        "adjusted_overlap": chance_overlap,
    }


def compute_neighborhood_preservation(
    adata: AnnData,
    embedding_before: str,
    embedding_after: str,
    *,
    n_neighbors: int = 20,
) -> dict[str, np.ndarray | float]:
    """Neighborhood preservation between two embeddings stored in the same :class:`anndata.AnnData`.

    Parameters
    ----------
    adata
        Annotated data matrix holding both embeddings in `.obsm`.
    embedding_before
        Key in `adata.obsm` of the embedding before the processing step.
    embedding_after
        Key in `adata.obsm` of the embedding after the processing step.
    n_neighbors
        Number of nearest neighbours per observation, excluding the observation itself.

    Returns
    -------
    dict
        As returned by :func:`neighborhood_preservation`.

    Raises
    ------
    KeyError
        If either key is missing from `adata.obsm`.

    Examples
    --------
    .. code-block:: python

        result = msm.compute_neighborhood_preservation(adata, "X_pca_raw", "X_pca_imputed")
        print(result["adjusted_overlap"])
    """
    for key in (embedding_before, embedding_after):
        if key not in adata.obsm:
            raise KeyError(f"`{key}` is not in `adata.obsm`, available keys are {list(adata.obsm)}.")

    return neighborhood_preservation(
        adata.obsm[embedding_before],
        adata.obsm[embedding_after],
        n_neighbors=n_neighbors,
    )
