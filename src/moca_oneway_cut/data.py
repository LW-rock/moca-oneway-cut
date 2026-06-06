import numpy as np
import torch
from torch.utils.data import Dataset


class CausalDataset(Dataset):
    def __init__(self, X, T, Y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.T = torch.as_tensor(T, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]


def as_2d_float_array(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
    return X


def as_1d_float_array(values, name):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return values


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x
