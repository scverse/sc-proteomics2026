"""Download and cache the single-cell proteomics datasets used for benchmarking."""

import os
from pathlib import Path

import gdown
import platformdirs

DATASETS_DIR = Path(os.environ.get("MSMETRICS_DATA_DIR") or platformdirs.user_cache_dir("msmetrics", "your_company"))

WU2025_URL = "https://drive.google.com/uc?export=download&id=11iEGmao3XdJo6But65cxu7suy_yLul7N"


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

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        from msmetrics import datasets

        adata = ad.read_h5ad(datasets.wu2025())
    """
    destination = DATASETS_DIR / "wu2025.h5ad"
    if destination.exists() and not force:
        return destination

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if gdown.download(WU2025_URL, str(destination), quiet=True) is None:
        raise OSError(f"Downloading {WU2025_URL} failed.")

    return destination
