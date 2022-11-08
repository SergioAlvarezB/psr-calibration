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


class Obj(LBFGS_Objective):
    def __init__(self):
        super().__init__()
        self.target = torch.arange(4).reshape(2,2)
        self.X = Parameter(torch.randn(2,2))  # random init

    def loss(self):
        X = self.X - self.target
        return (X.T @ X).trace()


def calibrate(trnscores, trnlabels, tstscores, calclass, quiet=True, **kwargs):

    obj = calclass(trnscores, trnlabels, **kwargs)
    
    paramvec, value, curve, success = lbfgs(obj, 100, quiet=quiet)
    
    if not success:
        raise Exception("LBFGS was unable to converge")
        
    return obj.calibrate(tstscores), [obj.temp, obj.bias] if obj.has_bias else [obj.temp]

