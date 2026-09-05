"""scib-metrics wrappers

`scib_metrics` takes plain arrays and, for the graph-based metrics, a `NeighborsResults` that the
caller has to build. The wrappers below keep that plumbing in one place, so that every metric is
called the same way: an `AnnData`, an `.obsm` key for the embedding, and `.obs` columns for the
annotations. Each wrapper carries the documentation of the metric it wraps, see `_wraps_scib`.
"""

from collections.abc import Callable
from inspect import getdoc
from textwrap import indent

import numpy as np
from anndata import AnnData
from scib_metrics.nearest_neighbors import NeighborsResults
from sklearn.neighbors import NearestNeighbors

_SCIB_PARAMETER_DOCS = {
    "adata": "adata\n    Annotated data matrix, holding the embedding in `.obsm` and the annotations in `.obs`.",
    "embedding_key": (
        "embedding_key\n    Key in `adata.obsm` of the embedding to score, e.g. `'X_pca'`. Metrics are defined on a\n"
        "    low-dimensional representation, not on the expression matrix itself."
    ),
    "embedding_before": "embedding_before\n    Key in `adata.obsm` of the embedding before the processing step.",
    "embedding_after": "embedding_after\n    Key in `adata.obsm` of the embedding after the processing step.",
    "label_key": ("label_key\n    Column in `adata.obs` holding the biological grouping to conserve, e.g. cell type."),
    "batch_key": (
        "batch_key\n    Column in `adata.obs` holding the technical grouping to mix, e.g. plate, acquisition day\n"
        "    or instrument."
    ),
    "covariate_key": (
        "covariate_key\n    Column in `adata.obs` holding the covariate whose explained variance is compared before\n"
        "    and after the processing step."
    ),
    "categorical": (
        "categorical\n    Whether to one-hot encode the covariate instead of regressing on it directly. By default\n"
        "    this follows the dtype of the column, so that a discrete covariate such as the plate is\n"
        "    encoded and a continuous one such as the number of proteins per cell is not."
    ),
    "n_neighbors": (
        "n_neighbors\n    Size of the k-nearest-neighbour graph built from the embedding. The observation itself is\n"
        "    included, as `scib_metrics` expects. The default is the value the `scib_metrics`\n"
        "    `Benchmarker` uses for this metric, so that scores are comparable to published ones."
    ),
}


def _wraps_scib(metric: Callable, *parameters: str) -> Callable:
    """Append the parameter block and the verbatim `scib_metrics` documentation to a wrapper.

    Restating the documentation of every wrapped metric would leave two descriptions to keep in
    sync, so the original one is quoted instead and stays correct across `scib_metrics` versions.
    """

    def decorate(wrapper: Callable) -> Callable:
        blocks = "\n".join(_SCIB_PARAMETER_DOCS[parameter] for parameter in ("adata", *parameters))
        original = indent(getdoc(metric) or "No documentation available.", "    ")

        wrapper.__doc__ = (
            f"{getdoc(wrapper)}\n\n"
            f"Thin wrapper around :func:`scib_metrics.{metric.__name__}`, which reads the arrays it "
            f"scores from `adata`.\n\n"
            f"Parameters\n----------\n{blocks}\n"
            f"kwargs\n    Forwarded to :func:`scib_metrics.{metric.__name__}`.\n\n"
            f"Returns\n-------\nAs returned by :func:`scib_metrics.{metric.__name__}`.\n\n"
            f"Notes\n-----\nDocumentation of :func:`scib_metrics.{metric.__name__}`, verbatim:\n\n"
            f"{original}\n"
        )
        wrapper.wrapped_metric = metric
        wrapper.required_keys = tuple(p for p in parameters if p.endswith("_key") and not p.startswith("embedding"))
        return wrapper

    return decorate


def _embedding(adata: AnnData, embedding_key: str) -> np.ndarray:
    """Embedding stored under `embedding_key`, as a dense float array."""
    if embedding_key not in adata.obsm:
        raise KeyError(f"`{embedding_key}` is not in `adata.obsm`, available keys are {list(adata.obsm)}.")

    embedding = np.asarray(adata.obsm[embedding_key], dtype=float)
    if not np.isfinite(embedding).all():
        raise ValueError(f"`adata.obsm['{embedding_key}']` contains `nan` or `inf`, which no metric can score.")
    return embedding


def _annotation(adata: AnnData, key: str, parameter: str) -> np.ndarray:
    """Observation annotation stored in `adata.obs[key]`, as a plain array."""
    if key not in adata.obs:
        raise KeyError(f"`{parameter}='{key}'` is not in `adata.obs`, available columns are {list(adata.obs)}.")

    annotation = np.asarray(adata.obs[key])
    if len(np.unique(annotation)) < 2:
        raise ValueError(f"`{parameter}='{key}'` has a single level; a metric over one group is not informative.")
    return annotation


def _neighbors(embedding: np.ndarray, n_neighbors: int) -> NeighborsResults:
    """K-nearest-neighbour graph in the format `scib_metrics` expects."""
    n_obs = embedding.shape[0]
    if not 2 <= n_neighbors <= n_obs:
        raise ValueError(f"`n_neighbors` must be between 2 and {n_obs} for {n_obs} observations, got {n_neighbors}.")

    # Passing the embedding back in makes scikit-learn return each observation as its own first
    # neighbour, which is the convention `NeighborsResults` documents.
    fitted = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding)
    distances, indices = fitted.kneighbors(embedding)
    return NeighborsResults(indices=indices, distances=distances)
