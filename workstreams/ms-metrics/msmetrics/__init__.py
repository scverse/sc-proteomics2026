"""Metrics and methods for diagnosing and processing single-cell proteomics datasets."""

__version__ = "0.1.0"

from msmetrics import datasets, meta, perturbations, plotting, utils
from msmetrics.utils import (
    bras,
    clisi_knn,
    compute_neighborhood_preservation,
    diagnose_covariates,
    graph_connectivity,
    ilisi_knn,
    isolated_labels,
    kbet,
    kbet_per_label,
    lisi_knn,
    neighborhood_preservation,
    nmi_ari_cluster_labels_kmeans,
    nmi_ari_cluster_labels_leiden,
    pcr_comparison,
    sbee,
    silhouette_batch,
    silhouette_label,
    variance_preservation,
)

__all__ = [
    "bras",
    "clisi_knn",
    "compute_neighborhood_preservation",
    "datasets",
    "diagnose_covariates",
    "graph_connectivity",
    "ilisi_knn",
    "isolated_labels",
    "kbet",
    "kbet_per_label",
    "lisi_knn",
    "meta",
    "neighborhood_preservation",
    "nmi_ari_cluster_labels_kmeans",
    "nmi_ari_cluster_labels_leiden",
    "pcr_comparison",
    "perturbations",
    "plotting",
    "sbee",
    "silhouette_batch",
    "silhouette_label",
    "utils",
    "variance_preservation",
]
