import numpy as np
from IPython import embed
from scipy.special import softmax
import torch
from psrcal.utils import onehot_encode, check_label
import matplotlib.pyplot as plt

def CostFunction(log_probs, labels, C=None, norm=True):

    if C is None:
        # Use the standard 0-1 cost matrix
        C = 1 - np.eye(log_probs.shape[1])    

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


def LogLoss(log_probs, labels, norm=True, priors=None):
        
    priors, weights = _get_priors_and_weights(labels, priors)
    norm_factor = LogLoss(torch.log(priors.expand(log_probs.shape[0],-1)), labels, norm=False, priors=priors) if norm else 1.0

    # The loss on each sample is weighted by the inverse of the
    # frequency of the corresponding class in the test data
    # times the external prior
    ii = torch.arange(len(labels))
    losses = -log_probs[ii, labels]
    score  = torch.mean(weights*losses)

    return score / norm_factor


def LogLossSE(log_probs, ref_log_probs, norm=True):
        
    if norm:
        priors = torch.mean(torch.exp(ref_log_probs), axis=0)
        prior_entropy = -torch.sum(priors * torch.log(priors))
    else:
        prior_entropy = 1.0

    return torch.mean(torch.sum(-torch.exp(ref_log_probs) * log_probs, axis=1))/ prior_entropy



def Brier(log_probs, labels, norm=True, priors=None):
        
    priors, weights = _get_priors_and_weights(labels, priors)
    norm_factor = Brier(torch.log(priors.expand(log_probs.shape[0],-1)), labels, norm=False, priors=priors) if norm else 1.0


    # The loss on each sample is weighted by the inverse of the
    # frequency of the corresponding class in the test data
    # times the external prior
    probs         = torch.exp(log_probs)
    labels_onehot = onehot_encode(labels, n_classes=probs.shape[-1])
    losses        = (labels_onehot-probs)**2
    score         = torch.mean(torch.atleast_2d(weights).T*losses)

    return score / norm_factor


def CalLossLogLoss(log_probs, cal_log_probs, targets, priors=None):

    raw = LogLoss(log_probs, targets, priors)
    cal = LogLoss(cal_log_probs, targets, priors)

    return (raw-cal)/raw*100


def CalLossBrier(log_probs, cal_log_probs, targets, priors=None):

    raw = Brier(log_probs, targets, priors)
    cal = Brier(cal_log_probs, targets, priors)

    return (raw-cal)/raw*100


def _get_priors_and_weights(labels, priors):

    data_priors = torch.bincount(labels)/float(labels.shape[0])
    if priors is None:
        priors = data_priors
        weights = torch.tensor(1.0)
    else:
        weights = priors[labels]/data_priors[labels] 

    return priors, weights


def ECE(log_probs, target, M=15, return_values=False):
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
    ave_accs = []
    ave_confs = []
    counts = []
    for low, high in zip(lows, highs):
        ix = (low < confs) & (confs <= high)
        n = torch.sum(ix)
        if n==0:
            continue
        curr_preds = preds[ix]
        curr_confs = confs[ix]
        curr_target = target[ix]
        ave_acc  = torch.mean((curr_preds == curr_target).type(dtype=probs.dtype))
        ave_conf = torch.mean(curr_confs)
        ave_accs.append(ave_acc.detach().numpy())
        ave_confs.append(ave_conf.detach().numpy())
        counts.append(n.detach().numpy())
        ece += n*torch.abs(ave_conf-ave_acc)

    if return_values:
        return ece * 100/N, np.array(ave_accs), np.array(ave_confs), np.array(counts)
    else:
        return ece * 100/N


def plot_reliability_diagram(scores, targets, outfile, nbins=15, title=''):

    metric_value, accs, confs, counts = ECE(scores, targets, return_values=True, M=nbins)
    plt.figure()
    plt.plot(confs, accs, "-*", label="ave_acc")
    plt.plot(confs, np.abs(confs-accs), "-*", label="abs(ave_acc-ave_conf)")
    plt.plot(confs, counts/np.sum(counts), "-*", label="fraction_of_samples")
    plt.plot(confs, counts/np.sum(counts)*np.abs(confs-accs), "-*", label="n/N * abs(ave_acc-ave_conf)")
    #for acc, conf, count in zip(accs, confs, counts):
    #  plt.annotate(str(count),  xy=(conf, np.abs(conf-acc)))
    plt.plot([0,1],[0,1],':k')
    plt.xlabel("ave_conf")
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.savefig(outfile)




def shift(loss, off):
    K = off.shape[0]
    eof = torch.exp(off)
    So = K**0.5 * eof/eof.norm()

    def shifted_loss(log_probs, labels):
        label = check_label(labels, log_probs.shape[-1])
        qs = torch.softmax(log_probs + off, dim=1)
        return loss(torch.log(qs), label) @ So.reshape(-1, 1)

    return shifted_loss

