import numpy as np
from IPython import embed
from scipy.special import softmax
import torch
from psrcal.utils import onehot_encode, check_label


def CostFunction(log_probs, labels, C, norm=False):

    # Normalize the cost matrix
    C -= C.min(axis=0)
    C = torch.tensor(C, dtype=log_probs.dtype)

    def _decisions(p):
        return (p @ C.T).argmin(axis=-1)

    probs = torch.exp(log_probs)
    costs = C[_decisions(probs),labels]
        
    if norm:
        priors = (torch.bincount(labels)/float(labels.shape[0])).type(dtype=C.dtype)
        naive_costs = C[_decisions(priors), labels]
        prior_cost = torch.mean(naive_costs)
    else:
        prior_cost = 1.0

    return torch.mean(costs)/prior_cost



def LogLoss(log_probs, labels, norm=False, _shift=False):
        
    ii = torch.arange(len(labels))

    if norm:
        priors = torch.bincount(labels)/float(labels.shape[0])
        prior_entropy = -torch.sum(priors * torch.log(priors))
    else:
        prior_entropy = 1.0

    if _shift:
        mask = torch.zeros(log_probs.shape)
        mask[ii, labels] = 1
        return -log_probs*mask/ prior_entropy

    return torch.mean(-log_probs[ii, labels])/ prior_entropy


def Brier(log_probs, labels, _shift):
        
    probs = torch.exp(log_probs)
    labels = onehot_encode(labels, n_classes=probs.shape[-1])
    losses = torch.sum((labels-probs)**2, dim=1)

    if _shift:
        return losses

    return torch.mean(losses)


def ECE(log_probs, target, M=15):
    """"Computes ECE score as defined in https://arxiv.org/abs/1706.04599"""

    probs = torch.exp(log_probs)

    N = probs.shape[0]

    if probs.ndim>1:
        confs, preds = torch.max(probs, axis=1)
    else:
        confs = probs
        preds = probs >= 0.5

    # Generate intervals
    limits = np.linspace(0, 1, num=M+1)
    lows, highs = limits[:-1], limits[1:]

    ece = 0
    for low, high in zip(lows, highs):
        ix = (low < confs) & (confs <= high)
        n = torch.sum(ix)
        if n<1:
            continue
        curr_preds = preds[ix]
        curr_confs = confs[ix]
        curr_target = target[ix]
        curr_acc = torch.mean((curr_preds == curr_target).type(dtype=probs.dtype))
        ece += n*torch.abs(torch.mean(curr_confs)-curr_acc)

    return ece * 100/N



def shift(loss, off):
    K = off.shape[0]
    eof = torch.exp(off)
    So = K**0.5 * eof/eof.norm()

    def shifted_loss(log_probs, labels):
        label = check_label(labels, log_probs.shape[-1])
        qs = torch.softmax(log_probs + off, dim=1)
        return loss(torch.log(qs), label) @ So.reshape(-1, 1)

    return shifted_loss

