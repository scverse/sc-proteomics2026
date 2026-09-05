"""Tests for the dataset accessors, without touching the network."""

import pytest

from msmetrics import datasets


def test_wu2025_downloads_once(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, output, quiet=True):
        calls.append(url)
        datasets.Path(output).write_bytes(b"h5ad")
        return output

    monkeypatch.setattr(datasets, "DATASETS_DIR", tmp_path / "cache")
    monkeypatch.setattr(datasets.gdown, "download", fake_download)

    path = datasets.wu2025()
    assert path.read_bytes() == b"h5ad"

    assert datasets.wu2025() == path
    assert len(calls) == 1, "cached dataset was downloaded again"

    datasets.wu2025(force=True)
    assert len(calls) == 2, "force=True did not re-download"


def test_wu2025_raises_on_failed_download(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "DATASETS_DIR", tmp_path / "cache")
    monkeypatch.setattr(datasets.gdown, "download", lambda url, output, quiet=True: None)

    with pytest.raises(OSError, match="failed"):
        datasets.wu2025()
