"""Metrics for diagnosing the effect of imputation on single-cell proteomics data."""

import warnings

import numpy as np


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
