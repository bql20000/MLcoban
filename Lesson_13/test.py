import numpy as np
from sklearn.metrics import accuracy_score

np.random.seed(1)

from scipy import sparse
def convert_labels(y, C):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y

def cost(X, Y, W):
    # W(d x C)  &&  X(d x N)  &&  Y(C x N)
    A = softmax(W.T.dot(X))     # A(C x N)
    return -np.sum(Y*np.log(A))

def softmax(Z):              
    Z = np.exp(Z)
    return Z / Z.sum(axis = 0)

def grad(X, Y, W):
    A = softmax(W.T.dot(X))
    E = A - Y
    return X.dot(E.T)

def numerical_grad(X, Y, W, eps = 1e-6):
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps
            W_n[i, j] -= eps
            g[i, j] = (cost(X, Y, W_p) - cost(X, Y, W_n))/(2*eps)
    return g

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0).T
X = np.concatenate((np.ones((1,3*N)), X), axis = 0)

print(X)

C = 3
y = np.asarray([0]*N + [1]*N + [2]*N).T
Y = convert_labels(y, C)

print(Y.shape)

d = X.shape[0]
W_init = np.random.randn(d, C)
g1 = grad(X, Y, W_init)
g2 = numerical_grad(X, Y, W_init)
print(np.linalg.norm(g1 - g2))






















