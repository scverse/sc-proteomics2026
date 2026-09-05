"""Test-session setup.

The backend is chosen here rather than inside a test module because `msmetrics` imports
`matplotlib.pyplot` at package import time, through `plotting`. Selecting it from a test file would
depend on whether some earlier test had already imported the package, which is a function of
pytest's collection order; conftest runs before any of that.
"""

import matplotlib

matplotlib.use("Agg")
