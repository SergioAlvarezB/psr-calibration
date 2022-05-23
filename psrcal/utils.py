from collections.abc import Iterable

import torch
import numpy as np


def onehot_encode(X, n_classes=None):
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()

    N = X.shape[0]

    if n_classes is None:
        n_classes = X.max()+1

    onehot = np.zeros((N, n_classes))
    onehot[np.arange(N), X] = 1.

    return torch.Tensor(onehot)


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)



def check_label(label, dims):
    if not isinstance(label, Iterable):
        label = np.array([label]*dims[0])
    if label.ndim < 2:
        label = onehot_encode(label, n_classes=dims[1])

    return label
