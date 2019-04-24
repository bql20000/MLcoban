import numpy as np

eps = 1e-3

def grad(x):
    return 2*x + 5*np.cos(x) 

def cost(x):
    return x**2 + 5*np.sin(x)

def stop_criteria(val_1, val_2):
    return (val_1 - val_2) < eps

def myGD1(x0, eta):
    pre_x = x0;
    for it in range(100):
        new_x = pre_x - eta * grad(pre_x)
        if (abs(grad(new_x)) < eps):
            break;
        pre_x = new_x
    return (pre_x, it)

(x1, it1) = myGD1(-5, 0.1)
(x2, it2) = myGD1(5, 0.1)
print('Solution x1 = %f, cost = %f, number of iterations = %d' %(x1, cost(x1), it1))
print('Solution x2 = %f, cost = %f, number of iterations = %d' %(x2, cost(x2), it2))