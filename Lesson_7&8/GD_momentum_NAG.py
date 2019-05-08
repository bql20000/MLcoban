import numpy as np

eps = 1e-3

def grad(x):
    return 2*x + 10*np.cos(x)
def cost(x):
    return x**2 + 10*np.sin(x)
def has_converged(w):
    return abs(grad(w)) < eps
def GD(w, gramma, eta):
    w_pre = w
    v_pre = 0
    for it in range(100):
        #v_new = eta * grad(w_pre)                                                          #without momentum
        #v_new = gramma * v_pre + eta * grad(w_pre)                                           #with momentum
        v_new = gramma * v_pre + eta * grad(w_pre - gramma * v_pre)                            #Nesterove momentum (NAG)
        w_new = w_pre - v_new
        if has_converged(w_new): 
            break
        (v_pre, w_pre) = (v_new, w_new)
    return (w_pre, it)

(x1, it1) = GD(5, 0.9, 0.1)
print('Minimum found: %f' %cost(x1), ' after %d iterations' %(it1+1))





























































































































