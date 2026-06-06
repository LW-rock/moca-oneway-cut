import numpy as np

from moca_oneway_cut import MOCAOneWayCuttingFeedback


def main():
    rng = np.random.default_rng(2026)
    n = 600
    p = 8
    X = rng.normal(size=(n, p))
    logits = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    T = rng.binomial(1, e)
    tau = 1.0 + X[:, 0] - 0.5 * X[:, 2]
    mu0 = 0.5 + X[:, 1] + 0.2 * X[:, 3]
    Y = mu0 + T * tau + rng.normal(scale=1.0, size=n)

    model = MOCAOneWayCuttingFeedback(
        d_model=32,
        nhead=4,
        num_layers=1,
        dropout=0.1,
        gate_temp=1.0,
        treat_epochs=40,
        outcome_epochs=60,
        lr_treat=1e-3,
        lr_outcome=3e-4,
        batch_size=128,
        validation_fraction=0.2,
        random_state=2026,
        verbose=True,
    )
    model.fit(X, T, Y)
    ate_true = float(tau.mean())
    tau_hat = model.effect(X)
    ate_hat = model.ate(X)

    print("ATE truth:", ate_true)
    print("ATE estimate:", ate_hat)
    print("ATE bias:", ate_hat - ate_true)
    print("First five CATE truths:", tau[:5])
    print("First five CATE estimates:", tau_hat[:5])
    print("First five CATE biases:", tau_hat[:5] - tau[:5])


if __name__ == "__main__":
    main()
