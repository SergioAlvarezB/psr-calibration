from collections.abc import Iterable

import torch
import numpy as np

from psrcal.utils import onehot_encode


def check_label(label, dims):
    if not isinstance(label, Iterable):
        label = np.array([label]*dims[0])
    if label.ndim < 2:
        label = onehot_encode(label, n_classes=dims[1])

    return label


def logcost(q, label):
    label = check_label(label, q.shape)

    return -torch.sum(label*torch.log(q), dim=1)


def brier(q, label):

    label = check_label(label, q.shape)

    return torch.sum((label-q)**2, dim=1)


def shift(psr, offs):
    K = offs.shape[0]
    eof = torch.exp(offs)
    So = K**0.5 * eof/eof.norm()

    def shifted_psr(q, label):
        label = check_label(label, q.shape)
        return psr(q, label)*torch.sum(label*So, dim=1)

    return shifted_psr