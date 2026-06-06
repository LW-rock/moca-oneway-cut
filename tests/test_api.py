import numpy as np

from moca_oneway_cut import MOCAOneWayCuttingFeedback


def test_api_smoke():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(80, 4))
    T = rng.binomial(1, 0.5, size=80)
    Y = X[:, 0] + T * (1 + X[:, 1]) + rng.normal(size=80)

    est = MOCAOneWayCuttingFeedback(
        d_model=8,
        nhead=2,
        treat_epochs=1,
        outcome_epochs=1,
        batch_size=32,
        validation_fraction=0.2,
    )
    est.fit(X, T, Y)

    tau = est.effect(X[:10])
    assert tau.shape == (10,)
    assert np.isfinite(est.ate(X[:10]))
