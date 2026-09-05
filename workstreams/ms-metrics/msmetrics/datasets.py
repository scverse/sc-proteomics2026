"""Download and cache the single-cell proteomics datasets used for benchmarking."""

import os
from pathlib import Path

import gdown
import platformdirs

DATASETS_DIR = Path(os.environ.get("MSMETRICS_DATA_DIR") or platformdirs.user_cache_dir("msmetrics"))

WU2025_URL = "https://drive.google.com/uc?export=download&id=11iEGmao3XdJo6But65cxu7suy_yLul7N"

#: First bytes of every HDF5 file, and so of every `.h5ad`.
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _is_hdf5(path: Path) -> bool:
    """Whether a file begins with the HDF5 signature."""
    with path.open("rb") as handle:
        return handle.read(len(HDF5_SIGNATURE)) == HDF5_SIGNATURE


def wu2025(*, force: bool = False) -> Path:
    """Wu 2025 single-cell proteomics dataset

    Parameters
    ----------
    force
        Re-download even if the file is already cached, for example after the upstream file changed.

    Returns
    -------
    Path
        Location of the cached `wu2025.h5ad` file.

    Raises
    ------
    OSError
        If the download fails, or if what arrives is not an HDF5 file.

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        from msmetrics import datasets

        adata = ad.read_h5ad(datasets.wu2025())

    Notes
    -----
    The download is checked against the HDF5 signature before being handed back, and a cached file is
    checked before being reused. Google Drive answers with an HTML page rather than the file when a
    link's download quota is exceeded or its sharing changes, and without this check those bytes
    would be saved as `wu2025.h5ad` and reused as though they were the dataset. A cached file that
    fails the check is discarded and fetched again.
    """
    destination = DATASETS_DIR / "wu2025.h5ad"

    if destination.exists() and not force:
        if _is_hdf5(destination):
            return destination
        destination.unlink()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if gdown.download(WU2025_URL, str(destination), quiet=True) is None:
        raise OSError(f"Downloading {WU2025_URL} failed.")

    if not _is_hdf5(destination):
        preview = destination.read_bytes()[:64]
        destination.unlink()
        raise OSError(
            f"{WU2025_URL} did not return an HDF5 file. It begins with {preview!r}, which is what Google "
            "Drive serves when a link's download quota is exceeded or its sharing has changed. Retry "
            "later, or check the link."
        )

    return destination
