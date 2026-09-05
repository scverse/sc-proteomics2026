"""Tests for the dataset accessors, without touching the network."""

import pytest

from msmetrics import datasets

VALID = datasets.HDF5_SIGNATURE + b"payload"
QUOTA_PAGE = b"<!DOCTYPE html><html><head><title>Google Drive - Quota exceeded</title>"


def fake_downloader(payload, calls):
    def download(url, output, quiet=True):
        calls.append(url)
        datasets.Path(output).write_bytes(payload)
        return output

    return download


def test_wu2025_downloads_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(datasets, "DATASETS_DIR", tmp_path / "cache")
    monkeypatch.setattr(datasets.gdown, "download", fake_downloader(VALID, calls))

    path = datasets.wu2025()
    assert path.read_bytes() == VALID

    assert datasets.wu2025() == path
    assert len(calls) == 1, "cached dataset was downloaded again"

    datasets.wu2025(force=True)
    assert len(calls) == 2, "force=True did not re-download"


def test_wu2025_raises_on_failed_download(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "DATASETS_DIR", tmp_path / "cache")
    monkeypatch.setattr(datasets.gdown, "download", lambda url, output, quiet=True: None)

    with pytest.raises(OSError, match="failed"):
        datasets.wu2025()


def test_wu2025_rejects_a_download_that_is_not_hdf5(tmp_path, monkeypatch):
    """Drive answers a throttled request with an HTML page, and `gdown` reports that as success."""
    monkeypatch.setattr(datasets, "DATASETS_DIR", tmp_path / "cache")
    monkeypatch.setattr(datasets.gdown, "download", fake_downloader(QUOTA_PAGE, []))

    with pytest.raises(OSError, match="did not return an HDF5 file"):
        datasets.wu2025()

    assert not (tmp_path / "cache" / "wu2025.h5ad").exists(), "the rejected download was left behind to be reused"


def test_wu2025_replaces_a_poisoned_cache(tmp_path, monkeypatch):
    """A bad file cached by an earlier run must not be handed back forever."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "wu2025.h5ad").write_bytes(QUOTA_PAGE)

    calls = []
    monkeypatch.setattr(datasets, "DATASETS_DIR", cache)
    monkeypatch.setattr(datasets.gdown, "download", fake_downloader(VALID, calls))

    assert datasets.wu2025().read_bytes() == VALID
    assert len(calls) == 1, "the poisoned cache was returned instead of being refetched"
