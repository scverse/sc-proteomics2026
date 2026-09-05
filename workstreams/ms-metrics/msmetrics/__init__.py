"""Metrics and methods for diagnosing and processing single-cell proteomics datasets."""

__version__ = "0.1.0"

from msmetrics import utils
from msmetrics.utils import (
    compute_neighborhood_preservation,
    neighborhood_preservation,
    variance_preservation,
)

__all__ = [
    "compute_neighborhood_preservation",
    "neighborhood_preservation",
    "utils",
    "variance_preservation",
]
