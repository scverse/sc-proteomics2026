"""Tests for the meta-benchmark harness.

Every assertion here is an invariant rather than a snapshot, so a failure names what broke rather
than that a number moved. The summary tests build their tidy frame by hand, which decouples them
from the simulation machinery: if `sweep` breaks, exactly one test fails.
"""

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from msmetrics import meta, plotting
from msmetrics import perturbations as pert

N_OBS, N_VARS = 60, 20


@pytest.fixture
def adata():
    """Synthetic log-intensity data: 3 separated cell types, 2 batches, 20 % missing."""
    rng = np.random.default_rng(0)
    cell_type = np.repeat(["A", "B", "C"], N_OBS // 3)
    batch = np.tile(["b1", "b2"], N_OBS // 2)

    offsets = {"A": 0.0, "B": 3.0, "C": -3.0}
    X = rng.normal(20.0, 1.0, size=(N_OBS, N_VARS))
    for level, offset in offsets.items():
        X[cell_type == level, : N_VARS // 2] += offset

    X[rng.random(X.shape) < 0.2] = np.nan
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(cell_type), "batch": pd.Categorical(batch)},
            index=[f"cell{i}" for i in range(N_OBS)],
        ),
        var=pd.DataFrame(index=[f"prot{j}" for j in range(N_VARS)]),
    )


def every_perturbation():
    return {
        "permute": pert.PermuteLabels("cell_type"),
        "permute_stratified": pert.PermuteLabels("cell_type", stratify_by="batch"),
        "batch_shift": pert.InjectBatchShift("batch"),
        "missing_mcar": pert.InjectMissing(mechanism="mcar"),
        "missing_mnar": pert.InjectMissing(mechanism="mnar"),
        "loading": pert.InjectLoadingOffset(),
        "dilute": pert.DiluteSignal("cell_type"),
    }


@pytest.mark.parametrize("name", list(every_perturbation()))
def test_dose_zero_is_identity_and_input_is_not_mutated(adata, name):
    perturbation = every_perturbation()[name]
    before = adata.X.copy()

    out = perturbation(adata, 0.0, np.random.default_rng(0))

    np.testing.assert_array_equal(out.X, before, err_msg=f"{name} changed the values at dose 0")
    pd.testing.assert_frame_equal(out.obs, adata.obs, obj=f"{name} .obs at dose 0")
    pd.testing.assert_frame_equal(out.var, adata.var, obj=f"{name} .var at dose 0")
    np.testing.assert_array_equal(adata.X, before, err_msg=f"{name} mutated its input")


def test_dose_scales_linearly_and_missing_masks_are_nested(adata):
    """The offset must be drawn once per replicate and merely scaled by dose.

    Redrawing per dose would leave every test above passing while quietly turning the dose-response
    curve into noise, so this is the assertion that catches it.
    """
    shift = pert.InjectBatchShift("batch")
    base = shift(adata, 0.0, np.random.default_rng(1)).X
    half = shift(adata, 0.5, np.random.default_rng(1)).X
    full = shift(adata, 1.0, np.random.default_rng(1)).X

    np.testing.assert_allclose(full - base, 2 * (half - base), atol=1e-9)

    missing = pert.InjectMissing(mechanism="mnar")
    light = np.isnan(missing(adata, 0.3, np.random.default_rng(2)).X)
    heavy = np.isnan(missing(adata, 0.5, np.random.default_rng(2)).X)
    assert np.all(heavy[light]), "the dose 0.3 mask is not nested inside the dose 0.5 mask"


def test_injectors_preserve_the_missingness_pattern(adata):
    observed = np.isnan(adata.X)

    for name in ("batch_shift", "loading", "dilute"):
        out = every_perturbation()[name](adata, 1.0, np.random.default_rng(3))
        np.testing.assert_array_equal(np.isnan(out.X), observed, err_msg=f"{name} changed missingness")

    masked = pert.InjectMissing()(adata, 0.4, np.random.default_rng(3))
    assert np.all(np.isnan(masked.X)[observed]), "InjectMissing un-masked an already missing value"
    assert np.isnan(masked.X).sum() > observed.sum(), "InjectMissing masked nothing"


def test_dilute_signal_equalises_centroids_and_keeps_within_class_variance(adata):
    """At dose 1 the class means collapse onto the global mean and nothing else moves.

    Asserting equal centroids, and deliberately not asserting that the classes became inseparable:
    the residual within-class covariance survives at any dose, so the stronger claim is false.
    """
    out = pert.DiluteSignal("cell_type", min_obs=1)(adata, 1.0, np.random.default_rng(4))
    labels = adata.obs["cell_type"].to_numpy()
    grand = np.nanmean(out.X, axis=0)

    for level in np.unique(labels):
        rows = labels == level
        np.testing.assert_allclose(np.nanmean(out.X[rows], axis=0), grand, atol=1e-8)
        np.testing.assert_allclose(
            np.nanvar(out.X[rows], axis=0),
            np.nanvar(adata.X[rows], axis=0),
            atol=1e-8,
            err_msg="diluting the class means also changed the within-class spread",
        )


def test_batch_shift_keeps_global_means_and_separates_batches(adata):
    out = pert.InjectBatchShift("batch")(adata, 1.0, np.random.default_rng(5))
    batch = adata.obs["batch"].to_numpy()

    np.testing.assert_allclose(np.nanmean(out.X, axis=0), np.nanmean(adata.X, axis=0), atol=1e-8)

    before = np.nanmean(adata.X[batch == "b1"], axis=0) - np.nanmean(adata.X[batch == "b2"], axis=0)
    after = np.nanmean(out.X[batch == "b1"], axis=0) - np.nanmean(out.X[batch == "b2"], axis=0)
    assert np.nanmean(np.abs(after)) > np.nanmean(np.abs(before)), "the batches did not move apart"


def test_a_scalar_batch_offset_is_cancelled_by_per_cell_median_normalisation(adata):
    """A pure per-batch scalar offset is invisible after median normalisation.

    Pinned as a test because it is a trap rather than a bug: with such a normalisation in `prepare`,
    a metric would score as perfectly insensitive to a batch effect for reasons that have nothing to
    do with the metric.
    """
    X = adata.X.copy()
    shifted = X + np.where(adata.obs["batch"].to_numpy() == "b1", 2.0, -2.0)[:, None]

    normalise = lambda values: values - np.nanmedian(values, axis=1, keepdims=True)
    np.testing.assert_allclose(normalise(shifted), normalise(X), atol=1e-8)


def batch_separation(adata):
    """Toy metric: mean absolute difference between the two batch centroids."""
    batch = adata.obs["batch"].to_numpy()
    return float(
        np.nanmean(np.abs(np.nanmean(adata.X[batch == "b1"], axis=0) - np.nanmean(adata.X[batch == "b2"], axis=0)))
    )


def test_sweep_schema_is_well_formed_and_the_run_is_reproducible(adata):
    doses = (0.0, 0.5, 1.0)
    metrics = {"separation": batch_separation, "broken": lambda a: 1 / 0}
    perturbations = {"batch_shift": pert.InjectBatchShift("batch"), "dilute": pert.DiluteSignal("cell_type")}

    curve = meta.sweep(adata, metrics, perturbations, doses=doses, n_replicates=4, seed=0)

    assert list(curve.columns) == meta.COLUMNS
    assert len(curve) == len(perturbations) * len(doses) * 4 * len(metrics)
    assert not curve.duplicated(["perturbation", "dose", "replicate", "metric"]).any()

    broken = curve[curve["metric"] == "broken"]
    assert broken["value"].isna().all() and broken["error"].str.startswith("ZeroDivisionError").all()

    noise = meta.reference_noise(curve).set_index("metric")
    assert noise.loc["separation", "sd_0"] > 0, "subsampling did not give the reference dose a spread"

    again = meta.sweep(adata, metrics, perturbations, doses=doses, n_replicates=4, seed=0)
    pd.testing.assert_frame_equal(curve, again)


def test_sweep_response_is_monotone_for_a_metric_that_should_respond(adata):
    curve = meta.sweep(
        adata,
        {"separation": batch_separation},
        {"batch_shift": pert.InjectBatchShift("batch")},
        doses=(0.0, 0.5, 1.0, 2.0),
        n_replicates=5,
        seed=1,
    )
    shape = meta.response_shape(curve).iloc[0]
    assert shape["spearman"] > 0.8
    assert shape["monotone_fraction"] == 1.0
    assert shape["range_over_noise"] > 1


def synthetic_curve(response, *, n_replicates=30, seed=0, noise=0.05, perturbation="p", metric="m"):
    """Tidy frame for a metric whose mean value at each dose is `response(dose)`."""
    rng = np.random.default_rng(seed)
    doses = (0.0, 0.25, 0.5, 0.75, 1.0)
    rows = [
        (perturbation, dose, "fraction", replicate, metric, response(dose) + rng.normal(0, noise), np.nan, "")
        for dose in doses
        for replicate in range(n_replicates)
    ]
    return pd.DataFrame(rows, columns=meta.COLUMNS)


def test_summaries_stay_quiet_on_degenerate_metrics_and_flag_a_responsive_one():
    """The meta-test: a constant and a pure-noise metric must not look sensitive.

    If the summaries call a metric that ignores the data sensitive, every number they produce above
    this point is worthless, so this is the test the module exists to pass.
    """
    constant = synthetic_curve(lambda d: 0.5, noise=0.0, metric="constant")
    noise_only = synthetic_curve(lambda d: 0.0, noise=1.0, seed=7, metric="noise")
    responsive = synthetic_curve(lambda d: 1.0 - d, metric="responsive")

    for curve in (constant, noise_only):
        name = curve["metric"].iloc[0]
        assert meta.null_control(curve)["p_value"].iloc[0] > 0.05, f"{name} was called significant"
        assert not np.isfinite(meta.sensitivity(curve)["detection_dose"].iloc[0]), f"{name} got a detection dose"
        rho = meta.response_shape(curve)["spearman"].iloc[0]
        assert not np.isfinite(rho) or abs(rho) < 0.3, f"{name} was called monotone"

    assert meta.null_control(responsive)["p_value"].iloc[0] < 0.05
    assert meta.sensitivity(responsive)["detection_dose"].iloc[0] < 0.25
    assert meta.response_shape(responsive)["spearman"].iloc[0] < -0.9


def test_detection_dose_is_interpolated_between_grid_points():
    """A step at 0.4 must not be reported as 0.5 just because 0.5 is on the grid."""
    curve = synthetic_curve(lambda d: 4 * d, noise=0.1, seed=3)
    detected = meta.sensitivity(curve, n_sd=2.0)["detection_dose"].iloc[0]
    assert 0.0 < detected < 0.25, f"expected an interpolated crossing inside the first step, got {detected}"


def test_specificity_contrast_is_signed_and_bounded():
    signal = synthetic_curve(lambda d: 2.0 * d, perturbation="dilute")
    nuisance = synthetic_curve(lambda d: 0.1 * d, perturbation="loading")
    curve = pd.concat([signal, nuisance], ignore_index=True)

    row = meta.specificity(curve, signal="dilute", nuisance="loading").iloc[0]
    assert 0 < row["contrast"] <= 1
    assert abs(row["signal_slope"]) > abs(row["nuisance_slope"])

    flipped = meta.specificity(curve, signal="loading", nuisance="dilute").iloc[0]
    assert -1 <= flipped["contrast"] < 0

    with pytest.raises(KeyError, match="not in the sweep"):
        meta.specificity(curve, signal="dilute", nuisance="absent")


def test_plots_draw_what_the_summaries_report():
    curve = pd.concat(
        [
            synthetic_curve(lambda d: 1.0 - d, perturbation="dilute"),
            synthetic_curve(lambda d: 0.05 * d, perturbation="loading"),
        ],
        ignore_index=True,
    )

    axes = plotting.response(curve, n_sd=2.0)
    assert axes.size == 2
    assert len(axes[0].get_lines()) >= 1, "no dose-response line was drawn"

    noise = meta.reference_noise(curve).iloc[0]
    band = axes[0].patches[-1]
    assert band.get_y() == pytest.approx(noise["mean_0"] - 2 * noise["sd_0"], abs=1e-6)
    assert band.get_y() + band.get_height() == pytest.approx(noise["mean_0"] + 2 * noise["sd_0"], abs=1e-6)

    assert np.atleast_1d(plotting.null(curve, dose=1.0)).size == 1

    ax = plotting.scorecard(curve, statistic="range_over_noise")
    expected = meta.response_shape(curve).pivot(index="metric", columns="perturbation", values="range_over_noise")
    np.testing.assert_allclose(ax.images[0].get_array().data, expected.to_numpy(float))

    assert plotting.specificity(curve, signal="dilute", nuisance="loading").get_xlabel().startswith("|loading|")
