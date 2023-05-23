import torch
from psrcal.optim.vecmodule import Parameter, LBFGS_Objective, lbfgs
from psrcal import losses 
from IPython import embed
import numpy as np

class AffineCal(LBFGS_Objective):

    def __init__(self, scores, labels, bias=True, priors=None):    
        # If priors are provided, ignore the data priors and use those ones instead
        # In this case, the scores are taken to be log scaled likelihoods

        super().__init__()

        self.temp = Parameter(torch.tensor(1.0, dtype=torch.float64))

        if priors is not None:
            self.priors = torch.Tensor(priors)
        else:
            self.priors = None

        self.has_bias = bias
        if bias:
            if self.priors is not None:
                # If external priors are provided, initialize the bias this way
                # so that if the scores are perfectly calibrated for those priors
                # the training process does not need to do anything.
                self.bias = Parameter(-torch.log(self.priors))
            else:
                self.bias = Parameter(torch.zeros(scores.shape[1], dtype=torch.float64))
        else:
            self.bias = 0

        self.scores = scores
        self.labels = labels

    def calibrate(self, scores):
        self.cal_scores = self.temp * scores + self.bias
        if self.priors is not None:
            self.cal_scores += torch.log(self.priors)

        self.log_probs = self.cal_scores - torch.logsumexp(self.cal_scores, axis=-1, keepdim=True) 
        return self.log_probs

    def loss(self):
        pass

class AffineCalLogLoss(AffineCal):
    def loss(self):
        return losses.LogLoss(self.calibrate(self.scores), self.labels, priors=self.priors, norm=False)
        

class AffineCalECE(AffineCal):
    def loss(self):
        return losses.ECE(self.calibrate(self.scores), self.labels)

class AffineCalLogLossPlusECE(AffineCal):

    def __init__(self, scores, labels, ece_weight=0.5, bias=True):
        super().__init__(scores, labels, bias)
        self.ece_weight = ece_weight

    def loss(self):
        return (1-self.ece_weight) * losses.LogLoss(self.calibrate(self.scores), self.labels) + self.ece_weight * losses.ECE(self.calibrate(self.scores), self.labels)


class AffineCalBrier(AffineCal):
    def loss(self):
        return losses.Brier(self.calibrate(self.scores), self.labels, norm=False)


class HistogramBinningCal():

    def __init__(self, scores, labels, M=15, **kwargs):

        # Histogram binning, as implemented here, only applies to binary classification.

        if scores.ndim != 1 and scores.shape[1]!=2:
            raise Exception("Histogram binning only implemented for binary classification")

        # The method assumes the scores are log probs, but we bin the probs, so take the exp.
        scores = torch.exp(scores)
        labels = labels.double()

        # Take the second score for binning
        if scores.ndim == 2:
            scores = scores[:,1]

        # Generate intervals
        limits = np.linspace(0, 1, num=M+1)
        self.lows, self.highs = limits[:-1], limits[1:]
        prop2s = []
        post2s = []
        self.cal_transform = []
        self.ave_score_per_bin = []

        # Obtain the proportion of samples of class 2 for each bin
        # This is the calibration transform to be applied for any posterior
        # within that bin.
        for low, high in zip(self.lows, self.highs):
            ix = (low < scores) & (scores <= high)
            n = torch.sum(ix)
            self.cal_transform.append(torch.mean(labels[ix]) if n!=0 else 0.0)
            self.ave_score_per_bin.append(torch.mean(scores[ix]) if n!= 0 else 0.0)

    def calibrate(self, scores):

        scores = torch.exp(scores)
        if scores.ndim == 2:
            scores = scores[:,1]
            return_both_probs = True
        else:
            return_both_probs = False

        cal_scores = torch.zeros_like(scores)
        binned_scores = torch.zeros_like(scores)

        # Obtain the proportion of samples of class 2 for each bin
        # This is the calibration transform to be applied for any posterior
        # within that bin.
        for i, (low, high) in enumerate(zip(self.lows, self.highs)):
            ix = (low < scores) & (scores <= high)
            cal_scores[ix] = self.cal_transform[i]
            binned_scores[ix] = self.ave_score_per_bin[i]

        if return_both_probs:
            cal_scores = torch.stack([1-cal_scores, cal_scores]).T

        # Save the binned scores for when we use this calibration to compute the ECE
        self.binned_scores = binned_scores

        # Go back to log domain
        return torch.log(cal_scores)



def calibrate(trnscores, trnlabels, tstscores, calclass, quiet=True, **kwargs):

    obj = calclass(trnscores, trnlabels, **kwargs)

    if calclass == HistogramBinningCal:

        return obj.calibrate(tstscores), [obj.binned_scores, obj.lows, obj.highs, obj.cal_transform, obj.ave_score_per_bin]

    else:

        paramvec, value, curve, success = lbfgs(obj, 100, quiet=quiet)
        
        if not success:
            raise Exception("LBFGS was unable to converge")
            
        return obj.calibrate(tstscores), [obj.temp, obj.bias] if obj.has_bias else [obj.temp]



