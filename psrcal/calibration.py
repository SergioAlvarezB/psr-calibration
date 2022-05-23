import torch
from psrcal.optim.vecmodule import Parameter, LBFGS_Objective, lbfgs
from psrcal import losses 
from IPython import embed

class AffineCal(LBFGS_Objective):

    def __init__(self, scores, labels, bias=True):
        super().__init__()

        self.temp = Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.has_bias = bias

        if bias:
            self.bias = Parameter(torch.zeros(scores.shape[1], dtype=torch.float64))
        else:
            self.bias = 0

        self.scores = scores
        self.labels = labels

    def calibrate(self, scores):
        self.cal_scores = self.temp * scores + self.bias
        self.log_probs = self.cal_scores - torch.logsumexp(self.cal_scores, axis=-1, keepdim=True) 
        return self.log_probs

    def loss(self):
        pass

class AffineCalLogLoss(AffineCal):
    def loss(self):
        return losses.LogLoss(self.calibrate(self.scores), self.labels)


class AffineCalECE(AffineCal):
    def loss(self):
        return losses.ECE(self.calibrate(self.scores), self.labels)

class AffineCalLogLossPlusECE(AffineCal):

    def __init__(self, scores, labels, ece_weight=1.0, bias=True):
        super().__init__(scores, labels, bias)
        self.ece_weight = ece_weight

    def loss(self):
        return losses.LogLoss(self.calibrate(self.scores), self.labels) + self.ece_weight * losses.ECE(self.calibrate(self.scores), self.labels)


class AffineCalBrier(AffineCal):
    def loss(self):
        return losses.Brier(self.calibrate(self.scores), self.labels)


class Obj(LBFGS_Objective):
    def __init__(self):
        super().__init__()
        self.target = torch.arange(4).reshape(2,2)
        self.X = Parameter(torch.randn(2,2))  # random init

    def loss(self):
        X = self.X - self.target
        return (X.T @ X).trace()




def calibrate(scores, labels, calclass, fit_scores=None, quiet=True, **kwargs):

    if fit_scores is not None:
        obj = calclass(fit_scores, labels, **kwargs)
    else:
        obj = calclass(scores, labels, **kwargs)
    paramvec, value, curve = lbfgs(obj, 100, quiet=quiet)
    
    return obj.calibrate(scores), [obj.temp, obj.bias] if obj.has_bias else [obj.temp]
