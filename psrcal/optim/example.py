import torch
from torch import as_tensor as ten

from psrcal.optim.vecmodule import Parameter, LBFGS_Objective, lbfgs

class Obj(LBFGS_Objective):
    def __init__(self):
        super().__init__()
        self.target = torch.arange(4).reshape(2,2)
        self.X = Parameter(torch.randn(2,2))  # random init
        
    def loss(self):
        X = self.X - self.target
        return (X.T @ X).trace()


obj = Obj()
paramvec, value, curve = lbfgs(obj,20)
print("paramvec:",paramvec)
print("\nX:",obj.X)