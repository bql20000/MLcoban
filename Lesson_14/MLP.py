import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import sparse
from sklearn.metrics import accuracy_score

np.random.seed(1)

N = 100
d0 = 2
C = 3
X = np.zeros((d0, N*C))
y = np.zeros(N*C, dtype = 'uint8')

for j in range(C):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1.0, N)
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
    X[:, ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
    y[ix] = j

plt.plot(X[0, :N], X[1, :N], 'bs', markersize=7)
plt.plot(X[0, N:2*N], X[1, N:2*N], 'g^', markersize=7)
plt.plot(X[0, 2*N:], X[1, 2*N:], 'ro', markersize=7)

plt.axis([-1.5, 1.5, -1.5, 1.5])

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.show()



def softmax(Z):         #stable
    Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return Z / np.sum(Z, axis=0)

def convert_labels(y, C):
    return sparse.coo_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(C, len(y))).toarray()

def cost(Y, Yhat):
    return -np.sum(Y * np.log(Yhat))/Y.shape[1]

#set size of layers
d0 = 2
d1 = h = 100
d2 = C = 3

#initialize parameters randomly
W1 = 0.01*np.random.randn(d0, d1)
b1 = np.zeros((d1, 1))
W2= 0.01*np.random.randn(d1, d2)
b2 = np.zeros((d2, 1))

#initialize data
#X = Mat(C x N)
Y = convert_labels(y, C)
N = X.shape[1]
eta = 1

for i in range(10000):
    #feedforward
    Z1 = W1.T.dot(X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = W2.T.dot(A1) + b2
    Yhat = softmax(Z2)

    #print loss after each 1000 interations
    if i % 1000 == 0:
        loss = cost(Y, Yhat)
        print('iter %d, loss = %.2f' %(i, loss))

    #backpropagation
    E2 = (Yhat - Y) / N
    dW2 = np.dot(A1, E2.T)      # A=(d1 x N), E2=(C x N), dW2=(d1 x C)
    db2 = np.sum(E2, axis=1, keepdims=True)
    E1 = np.dot(W2, E2)
    E1[Z1 <= 0] = 0
    dW1 = np.dot(X, E1.T)
    db1 = np.sum(E1, axis=1, keepdims=True)

    #gradient descent update
    W1 -= eta * dW1
    b1 -= eta * db1 
    W2 -= eta * dW2
    b2 -= eta * db2

Z1 = W1.T.dot(X) + b1
A1 = np.maximum(Z1, 0)
Z2 = W2.T.dot(A1) + b2
Yhat = softmax(Z2)
Y_pred = np.argmax(Yhat, axis=0)

print('Accuracy score = %.2f' %(100*accuracy_score(Y_pred, y)))
















































