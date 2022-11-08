from scipy.optimize import minimize 



class Callback:
    """
    A small tool for tracking iterations. Not of direct concern to the user.
    
    Objects of this class can be passed as callback to scipy.optimize.minimize().
    """
    def __init__(self):
        self.iter = 0
        self.scores = []
    def __call__(self,*p):
        self.iter += 1
    def record_score(self,score):
        scores, iter = self.scores, self.iter
        if iter < len(scores):
            scores[iter] = score
        else:
            scores.append(score)




    
    
def train_obj_scipy_lbfgs(obj, paramvec0, maxiters = 20, quiet=False):
    """
    Minimizes the given objective using scipy L-BFGS-B. 
    
    Inputs:
        obj: maps paramvec --> loss, gradient  (input and outputs are numpy)
        paramvec0: a numpy vector to initialize the optimnization
    
    Returns: 
        
        - the final paramvec 
        - the final minimum objective value 
        - a list of objective values per iteration (the training curve)
        
    """
    
    cb = Callback()
    
    def obj_cb(paramvec):
        loss_np, grad_np = obj(paramvec)
        cb.record_score(loss_np)
        if not quiet: print("iter",cb.iter,": ",loss_np)
        return loss_np, grad_np

    res = minimize(obj_cb, paramvec0, method="L-BFGS-B", jac=True, callback=cb, 
                   options={'maxiter':maxiters, 'maxls':40}) 


    if not quiet:
        print("scipy lbfgs termination status:")
        print(" ",res.message)
        print("  success:",res.success)
        print("  obj:",res.fun)
        print("  niters:",res.nit)
        print("  nfun:",res.nfev)
        print("  params:", res.x)
        print()
        
    return res.x, res.fun, cb.scores, res.success

    