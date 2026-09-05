# ms-metrics

Metrics and methods for diagnosing and processing single-cell proteomics datasets.

Each analysis step pairs a **metric** (a diagnostic, e.g. data completeness, outliers,
distributions, clustering) with a **method** that changes the data. Broadly, methods either
change the shape of the dataset (filtering: masking features or samples) or the distribution
of its values (normalization / batch correction: moving and rescaling samples or features).

See `analysis.py` for running and testing the individual modules on example datasets.

## Installation

From this directory:

```sh
pip install -e .
```

Then, in a notebook:

```python
import msmetrics
```
