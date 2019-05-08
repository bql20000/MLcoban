import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)

means = [[5, 6], [6, 7]]
cov = [[.3, .2], [.2, .3]]
N = 100
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0, X1), axis = 0)

plt.plot(X0[:,0], X0[:,1], 'ro', color='red')
plt.plot(X1[:,0], X1[:,1], 'ro', color='blue')
#plt.show()
X = np.concatenate((X, np.ones((2*N,1))), axis = 1)
y = np.concatenate((np.ones((N, 1)), -1*np.ones((N, 1))), axis = 0)

def predict(x, w):
    return np.sign(np.dot(x, w))

def has_converged(X, y, w):
    return np.array_equal(predict(X, w), y)

def myperceptron(X, y, w_init):
    w = w_init
    wrong = []
    it = 0
    while (has_converged(X, y, w) == False):
        it += 1
        pred = predict(X, w) 
        wrong = np.where(np.equal(pred, y) == False)[0]
        suffed_id = np.random.permutation(len(wrong))
        for ii in range(len(wrong)):
            i = wrong[suffed_id[ii]]
            w += y[i] * X[i,:].reshape(w.shape[0],1)

    return (w, it)

def my2perceptron(X, y, w_init):
    w = w_init
    it = 0
    while (has_converged(X, y, w) == False):
        it += 1
        suffed_id = np.random.permutation(2*N)
        for z in range(2*N):
            i = suffed_id[z]
            xi = X[i,:].reshape(1, w.shape[0])
            yi = y[i]
            if predict(xi, w) != yi:
               w += yi * xi.T

    return (w, it)

def perceptron(X, y, w_init):
    w = w_init
    it = 0
    while True:
        it += 1
        mix_id = np.random.permutation(2*N)
        for i in range(2*N):
            xi = X[mix_id[i], :].reshape(1, w.shape[0])
            yi = y[mix_id[i]]

            if predict(xi, w) != yi: # misclassified point
                w = w + (yi * xi).T
        if has_converged(X, y, w):
            break
    return w, it

def perceptron_NonelinearlySeparatable(X, y, w_init):
    w = w_init
    best_wrong = 2*N
    best_w = w
    it = 0
    while (has_converged(X, y, w) == False):
        it += 1
        suffed_id = np.random.permutation(2*N)
        wrong = 0
        for z in range(2*N):
            i = suffed_id[z]
            xi = X[i,:].reshape(1, w.shape[0])
            yi = y[i]
            if predict(xi, w) != yi:
               wrong += 1
               w += yi * xi.T
        if (wrong < best_wrong):
            best_wrong = wrong
            best_w = w 
        if (it > 100):
            break

    return (best_w, it)

w, it = perceptron_NonelinearlySeparatable(X, y, np.random.randn(X.shape[1], 1))

print(w)
print('number of interations = %d' %it)
#print(X[3])
#print(predict(X, sol))  

xx = np.linspace(0, 10, num=10)
yy = (-w[2][0] - w[0][0] * xx) / w[1][0]
plt.plot(xx, yy) 
#lt.axis([-5, 5, -5, 5])
plt.show()


































































