# MOCA One-Way Cut

`moca-oneway-cut` packages the one-way MOCA estimator with cutting feedback:

1. A treatment branch learns the propensity score and a treatment token.
2. The outcome branch attends to covariate tokens and the learned treatment token.
3. Cutting feedback is built in, so outcome loss does not backpropagate into the treatment branch.

## Install

From a local checkout:

```bash
pip install -e .
```

To run the example notebooks:

```bash
pip install -e ".[examples]"
```

Open `examples/example.ipynb` to run the same synthetic-data smoke test as `examples/basic_usage.py`.

From GitHub after publishing:

```bash
pip install git+https://github.com/LW-rock/moca-oneway-cut.git
```

## Quick Start

```python
import numpy as np
from moca_oneway_cut import MOCAOneWayCuttingFeedback

X = np.random.normal(size=(500, 8))
T = np.random.binomial(1, 0.5, size=500)
Y = 1.0 + X[:, 0] + T * (1.0 + X[:, 1]) + np.random.normal(size=500)

est = MOCAOneWayCuttingFeedback(
    d_model=32,
    nhead=4,
    treat_epochs=40,
    outcome_epochs=60,
    batch_size=128,
    device="cpu",
    random_state=123,
)
est.fit(X, T, Y)

tau_hat = est.effect(X)
ate_hat = est.ate(X)
mu0_hat, mu1_hat = est.predict_potential_outcomes(X)
propensity = est.propensity(X)
```

## IHDP Smoke Test

Open `notebooks/ihdp_smoke_test.ipynb` to run an IHDP usability check across five seeds. The notebook loads one IHDP replication, fits `MOCAOneWayCuttingFeedback`, reports ATE/CATE metrics, and saves `outputs/ihdp_smoke_test_results_5seeds.csv` plus `outputs/ihdp_smoke_test_summary_5seeds.csv`.

By default it reads IHDP from the public CEVAE CSV URL. To use a local copy instead, set `IHDP_CSV_PATH` before running the notebook.

## Main Parameters

- `d_model`: token embedding dimension.
- `nhead`: number of attention heads.
- `num_layers`: number of self-attention encoder layers.
- `dropout`: transformer and attention dropout.
- `gate_temp`: softmax temperature for fusion gates.
- `treat_epochs`, `outcome_epochs`: training epochs for each stage.
- `lr_treat`, `lr_outcome`: learning rates.
- `validation_fraction`: fraction split from training data when validation data is not supplied.
- `batch_size`, `device`, `random_state`, `verbose`.

## Saving and Loading

```python
est.save("moca_model.pt")
loaded = MOCAOneWayCuttingFeedback.load("moca_model.pt", device="cpu")
```
