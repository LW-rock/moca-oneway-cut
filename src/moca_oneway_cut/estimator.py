from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CausalDataset, as_1d_float_array, as_2d_float_array, to_numpy
from .models import OutcomeFusionModel, TreatmentQueryModel

_OBSOLETE_CONFIG_KEYS = ("query_mode", "query_delta_scale", "cut_feedback")


@dataclass
class MOCAConfig:
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 1
    dropout: float = 0.1
    gate_temp: float = 1.0
    batch_size: int = 128
    treat_epochs: int = 40
    outcome_epochs: int = 60
    lr_treat: float = 1e-3
    lr_outcome: float = 3e-4
    validation_fraction: float = 0.2
    random_state: int = 123
    device: str = "cpu"
    verbose: bool = False


class MOCAOneWayCuttingFeedback:
    """One-way MOCA treatment-effect estimator with cutting feedback.

    The outcome loss does not backpropagate into the treatment branch.
    """

    def __init__(self, **kwargs):
        self.config = MOCAConfig(**kwargs)
        self.treatment_model_ = None
        self.outcome_model_ = None
        self.n_features_in_ = None

    def fit(self, X, T, Y, X_val=None, T_val=None, Y_val=None):
        X, T, Y = self._validate_training_arrays(X, T, Y)
        self._set_seed()
        train_arrays, valid_arrays = self._make_train_valid_split(X, T, Y, X_val, T_val, Y_val)
        train_loader = self._make_loader(*train_arrays, shuffle=True)
        valid_loader = self._make_loader(*valid_arrays, shuffle=False)

        self.n_features_in_ = X.shape[1]
        self.treatment_model_ = TreatmentQueryModel(
            p=self.n_features_in_,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            gate_temp=self.config.gate_temp,
        )
        self.treatment_model_ = self._train_treatment_model(self.treatment_model_, train_loader, valid_loader)

        self.outcome_model_ = OutcomeFusionModel(
            p=self.n_features_in_,
            treatment_model=self.treatment_model_,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            gate_temp=self.config.gate_temp,
        )
        self.outcome_model_ = self._train_outcome_model(self.outcome_model_, train_loader, valid_loader)
        return self

    def predict_potential_outcomes(self, X):
        out = self._predict_raw(X)
        return out["mu0"], out["mu1"]

    def effect(self, X):
        return self._predict_raw(X)["tau"]

    def ate(self, X):
        return float(np.mean(self.effect(X)))

    def propensity(self, X):
        return self._predict_raw(X)["ps"]

    def predict(self, X, treatment=None):
        mu0, mu1 = self.predict_potential_outcomes(X)
        if treatment is None:
            return mu0, mu1
        treatment = as_1d_float_array(treatment, "treatment")
        if len(treatment) != len(mu0):
            raise ValueError("treatment must have the same number of rows as X.")
        return np.where(treatment > 0.5, mu1, mu0)

    def diagnostics(self, X):
        return self._predict_raw(X)

    def save(self, path):
        self._require_fitted()
        path = Path(path)
        payload = {
            "config": asdict(self.config),
            "n_features_in": self.n_features_in_,
            "treatment_state": self.treatment_model_.state_dict(),
            "outcome_state": self.outcome_model_.state_dict(),
        }
        torch.save(payload, path)
        return self

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(path, map_location=device or "cpu")
        config = payload["config"]
        for obsolete_key in _OBSOLETE_CONFIG_KEYS:
            config.pop(obsolete_key, None)
        if device is not None:
            config["device"] = device
        est = cls(**config)
        est.n_features_in_ = payload["n_features_in"]
        est.treatment_model_ = TreatmentQueryModel(
            p=est.n_features_in_,
            d_model=est.config.d_model,
            nhead=est.config.nhead,
            num_layers=est.config.num_layers,
            dropout=est.config.dropout,
            gate_temp=est.config.gate_temp,
        )
        est.outcome_model_ = OutcomeFusionModel(
            p=est.n_features_in_,
            treatment_model=est.treatment_model_,
            d_model=est.config.d_model,
            nhead=est.config.nhead,
            num_layers=est.config.num_layers,
            dropout=est.config.dropout,
            gate_temp=est.config.gate_temp,
        )
        est.treatment_model_.load_state_dict(payload["treatment_state"])
        est.outcome_model_.load_state_dict(payload["outcome_state"])
        est.treatment_model_.to(est.config.device)
        est.outcome_model_.to(est.config.device)
        return est

    def _validate_training_arrays(self, X, T, Y):
        X = as_2d_float_array(X)
        T = as_1d_float_array(T, "T")
        Y = as_1d_float_array(Y, "Y")
        if len(X) != len(T) or len(X) != len(Y):
            raise ValueError("X, T, and Y must have the same number of rows.")
        unique_t = set(np.unique(T).astype(int).tolist())
        if not unique_t.issubset({0, 1}) or len(unique_t) < 2:
            raise ValueError("T must contain binary treatment values 0 and 1.")
        return X, T, Y

    def _make_train_valid_split(self, X, T, Y, X_val, T_val, Y_val):
        if X_val is not None or T_val is not None or Y_val is not None:
            if X_val is None or T_val is None or Y_val is None:
                raise ValueError("X_val, T_val, and Y_val must be provided together.")
            return (X, T, Y), self._validate_training_arrays(X_val, T_val, Y_val)

        frac = self.config.validation_fraction
        if frac <= 0:
            return (X, T, Y), (X, T, Y)
        if frac >= 1:
            raise ValueError("validation_fraction must be smaller than 1.")

        rng = np.random.default_rng(self.config.random_state)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        n_valid = max(1, int(round(len(X) * frac)))
        valid_idx = indices[:n_valid]
        train_idx = indices[n_valid:]
        if len(train_idx) == 0:
            raise ValueError("validation_fraction leaves no training rows.")
        return (X[train_idx], T[train_idx], Y[train_idx]), (X[valid_idx], T[valid_idx], Y[valid_idx])

    def _make_loader(self, X, T, Y, shuffle):
        generator = torch.Generator()
        generator.manual_seed(self.config.random_state)
        return DataLoader(
            CausalDataset(X, T, Y),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    def _train_treatment_model(self, model, train_loader, valid_loader):
        model.to(self.config.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.lr_treat)
        best_state = None
        best_valid = np.inf
        for epoch in range(1, self.config.treat_epochs + 1):
            model.train()
            train_losses = []
            for X, T, _ in train_loader:
                X, T = X.to(self.config.device), T.to(self.config.device)
                out = model(X)
                loss = F.binary_cross_entropy(out["ps"], T)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_losses.append(loss.item())
            valid_loss = self._treatment_valid_loss(model, valid_loader)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_state = self._state_dict_cpu(model)
            if self.config.verbose:
                print(
                    f"[MOCA-Treat] Epoch {epoch:03d} | "
                    f"train_loss={np.mean(train_losses):.4f} | valid_loss={valid_loss:.4f}"
                )
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _train_outcome_model(self, model, train_loader, valid_loader):
        model.to(self.config.device)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=self.config.lr_outcome)
        best_state = None
        best_valid = np.inf
        for epoch in range(1, self.config.outcome_epochs + 1):
            model.train()
            train_losses = []
            for X, T, Y in train_loader:
                X, T, Y = X.to(self.config.device), T.to(self.config.device), Y.to(self.config.device)
                out = model(X)
                loss = self._outcome_loss(out, T, Y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_losses.append(loss.item())
            valid_loss = self._outcome_valid_loss(model, valid_loader)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_state = self._state_dict_cpu(model)
            if self.config.verbose:
                print(
                    f"[MOCA-OneWay] Epoch {epoch:03d} | "
                    f"train_loss={np.mean(train_losses):.4f} | valid_loss={valid_loss:.4f}"
                )
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _treatment_valid_loss(self, model, valid_loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for X, T, _ in valid_loader:
                X, T = X.to(self.config.device), T.to(self.config.device)
                losses.append(F.binary_cross_entropy(model(X)["ps"], T).item())
        return float(np.mean(losses))

    def _outcome_valid_loss(self, model, valid_loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for X, T, Y in valid_loader:
                X, T, Y = X.to(self.config.device), T.to(self.config.device), Y.to(self.config.device)
                losses.append(self._outcome_loss(model(X), T, Y).item())
        return float(np.mean(losses))

    def _outcome_loss(self, out, T, Y):
        mu0, mu1 = out["mu0"], out["mu1"]
        mask0, mask1 = (T == 0), (T == 1)
        zero = torch.tensor(0.0, device=Y.device)
        loss0 = ((Y[mask0] - mu0[mask0]) ** 2).mean() if mask0.sum() > 0 else zero
        loss1 = ((Y[mask1] - mu1[mask1]) ** 2).mean() if mask1.sum() > 0 else zero
        return loss0 + loss1

    def _predict_raw(self, X):
        self._require_fitted()
        X = as_2d_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        self.outcome_model_.eval()
        self.outcome_model_.to(self.config.device)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.config.device)
        with torch.no_grad():
            out = self.outcome_model_(X_t)
        return {
            "mu0": to_numpy(out["mu0"]),
            "mu1": to_numpy(out["mu1"]),
            "tau": to_numpy(out["tau"]),
            "ps": to_numpy(out["ps"]),
            "gate0": to_numpy(out["gate0"]),
            "gate1": to_numpy(out["gate1"]),
            "treat_gate": to_numpy(out["treat_gate"]),
        }

    def _require_fitted(self):
        if self.outcome_model_ is None or self.treatment_model_ is None:
            raise RuntimeError("Estimator is not fitted. Call fit before prediction.")

    def _set_seed(self):
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        torch.cuda.manual_seed_all(self.config.random_state)

    @staticmethod
    def _state_dict_cpu(model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
