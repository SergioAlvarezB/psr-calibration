from torch.nn import Module, Parameter
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torch import as_tensor as ten

from psrcal.optim.fullbatch_optim import train_obj_scipy_lbfgs

class VecModule(Module):
    """
    This subclass of torch.nn.Module provides a convenient interface to a 
    vectorized view of all the trainable parameters in the module. The 
    trainable parameters are those that have requires_grad=True.
    """
    
    def __init__(self):
        super().__init__()
        self.trainmode = True
    
    def trainable_parameters(self): 
        """
        This is a generator (i.e. iterator), that returns all of the trainable
        parameters in this module, and recursively in all contained modules.
        
        If you want all parameters instead, use self.parameters(), as in any
        Module.
        """
        for p in self.parameters(): 
            if p.requires_grad: yield p

    def set_paramvec(self, paramvec, zero_grad = False):
        """
        Writes the given numpy vector into the trainable parameters
        """
        vector_to_parameters(ten(paramvec),self.trainable_parameters())
        if zero_grad: self.zero_grad()

    def get_paramvec(self):
        """
        Extracts the trainable parameters as a numpy vector
        """
        return parameters_to_vector(self.trainable_parameters()).detach().numpy()
    
    def get_gradvec(self):
        """
        Extracts the current accumulated gradient of the trainable parameters 
        as a numpy vector
        """
        grads = (p.grad for p in self.trainable_parameters())
        return parameters_to_vector(grads).detach().numpy()


    def set_evalmode(self):
        """
        Stores requires_grad for each parameter into self.flags, so that we
        can later revert to trainmode again. Then sets requires_grad = False
        for all parmeters and sets grad=None for all parameters.
        """
        if not self.trainmode: return
        self.flags = [p.requires_grad for p in self.parameters()]    
        for p in self.parameters(): 
            p.requires_grad = False
            p.grad = None

    def set_trainmode(self):
        """
        Reverts to trainmode by recovering requires_grad for each parmeter
        from the stored flags. Sets grad=None for each parameter.
        """
        if self.trainmode: return
        for p, flag in zip(self.parameters(), self.flags): 
            p.requires_grad = flag
            p.grad = None
            
            
class LBFGS_Objective(VecModule):
    """
    This is a Module and a VecModule. You should implement a subclass to serve
    as LBFGS optimization objective. You should implement a method called
    loss, with signature loss(self) that returns the objective value. Gradient 
    will be done automatically by torch. To retrieve a vector of parameters,
    use yourmodule.get_params(). The to-vbe-optimized optimized params include 
    all members wrapped as Parameter and all Params in members that are also 
    Modules.
    
    """
    def __init__(self):
        super().__init__()            
        
    def __call__(self,paramvec):
        self.set_paramvec(paramvec,True)
        loss = self.loss()
        loss.backward()
        return loss.item(), self.get_gradvec()

    def loss(self):
        raise NotImplementedError
    
    
def lbfgs(obj,maxiters,quiet=False):    
        return train_obj_scipy_lbfgs(obj,obj.get_paramvec(),maxiters,quiet)
        
