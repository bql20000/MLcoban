import numpy as np
import matplotlib.pyplot as plt 

#questioning? loss function 

#generate data
np.random.seed(2)
N = 20
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

X = np.concatenate((X, np.ones((N,1))), axis=1)

#functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(w, X, y, lam):
    z = sigmoid(X.dot(w))
    return -np.mean(y*np.log(z) + (1-y)*np.log(1-z)) + lam*0.5/X.shape[0]*np.sum(w*w)

def logistic_regression(w_init, X, y, eta, lam, epoches=500, eps=1e-6):
    w = w_init
    d = X.shape[1]
    for n_epoch in range(epoches):
        w_old = w 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i].reshape(1, 2)
            yi = y[i]
            zi = sigmoid(xi.dot(w))
            w = w - eta * ((zi - yi)*xi.T + lam*w)
            #print(loss(w, X, y, lam))
        if (np.linalg.norm(w - w_old) / d < eps):
            break
        
    return w, n_epoch

eta = 0.05
lam = 0.0001
w, epoch = logistic_regression(np.random.randn(X.shape[1], 1), X, y, eta, lam) 
print(w)

#visualize solution
x0 = X[y == 0, 0]
y0 = y[y == 0]
x1 = X[y == 1, 0]
y1 = y[y == 1]
plt.plot(x0, y0, 'ro', markersize=8)
plt.plot(x1, y1, 'bs', markersize=8)
xx = np.linspace(0,6,1000)
yy = sigmoid(w[0][0] * xx + w[1][0])
plt.plot(xx, yy, color='green', linewidth=2)
plt.plot(-w[1][0]/w[0][0], 0.5, 'y^', markersize=8)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()






















































