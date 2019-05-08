import numpy as np
from sklearn.metrics import accuracy_score

np.random.seed(1)

from scipy import sparse
def convert_labels(y, C):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y

def softmax(Z):              
    Z = np.exp(Z)
    return Z / Z.sum(axis = 0)

def softmax_stable(Z):              #change to softmax_stable
    Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return Z / Z.sum(axis = 0)

def cost(X, Y, W):
    # W(d x C)  &&  X(d x N)  &&  Y(C x N)
    A = softmax(W.T.dot(X))     # A(C x N)
    return -np.sum(Y*np.log(A))

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

def softmax_regression(X, y, W_init, eta, n_epoch, eps = 1e-3):
    W = W_init
    N = X.shape[1]
    d = X.shape[0]
    C = W.shape[1]

    best = W            #pocket

    for it in range(n_epoch):
        mix_ids = np.random.permutation(N)
        W_old = W
        for i in mix_ids:
            xi = X[:, i].reshape(d, 1)       #fix lai dim cua X, y???
            yi = Y[:, i].reshape(C, 1)
            ai = softmax_stable(W.T.dot(xi))
            W = W + eta * xi.dot((yi-ai).T)
        if (cost(X, Y, W) < cost(X, Y, best)): best = W
        if np.linalg.norm(W - W_old) < eps:
            break

    return best, it

def pred(X, W):
    #W: d x C
    #X: d x N
    Y = softmax_stable(W.T.dot(X))
    return np.argmax(Y, axis=0).T

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

#visualize data
import matplotlib.pyplot as plt
plt.plot(X0[:, 0], X0[:, 1], 'b^', color='blue', markersize=4)
plt.plot(X1[:, 0], X1[:, 1], 'go', color='green', markersize=4)
plt.plot(X2[:, 0], X2[:, 1], 'rs', color='red', markersize=4)
#plt.show()

X = np.concatenate((X0, X1, X2), axis = 0)
X = np.concatenate((np.ones((3*N,1)), X), axis = 1).T

C = 3
y = np.asarray([0]*N + [1]*N + [2]*N)
Y = convert_labels(y, C)

d = X.shape[0]
W_init = np.random.randn(d, C)
g1 = grad(X, Y, W_init)
g2 = numerical_grad(X, Y, W_init)
print(np.linalg.norm(g1 - g2))

eta = 0.05
W, it = softmax_regression(X, y, W_init, eta, 100)
print(W)

#check_result
y_pred = pred(X, W)
#print(y_pred)
print('Accuracy score: %.5f, after %d epoches' %(accuracy_score(y_pred, y),it+1))










original_label = y

def display(X, label):
#     K = np.amax(label) + 1
    X0 = X[:, label == 0]
    X1 = X[:, label == 1]
    X2 = X[:, label == 2]
    
    plt.plot(X0[0, :], X0[1, :], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[0, :], X1[1, :], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[0, :], X2[1, :], 'rs', markersize = 4, alpha = .8)

#     plt.axis('equal')
    plt.axis('off')
    plt.plot()
    plt.show()
    
display(X[1:, :], original_label)

#Visualize 
# x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# y_min, y_max = X[:, 2].min() - .5, X[:, 2].max() + .5
# x_min, x_max = -4, 14
# y_min, y_max = -4, 14

xm = np.arange(-2, 11, 0.025)
xlen = len(xm)
ym = np.arange(-3, 10, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)


# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# xx.ravel(), yy.ravel()

print(np.ones((1, xx.size)).shape)
xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

# print(xx.shape, yy.shape)
XX = np.concatenate((np.ones((1, xx.size)), xx1, yy1), axis = 0)


print(XX.shape)

Z = pred(XX, W)

#Put the result into a color plot
Z = Z.reshape(xx.shape)
# plt.figure(1
# plt.pcolormesh(xx, yy, Z, cmap='jet', alpha = .35)

CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)

# Plot also the training points
# plt.scatter(X[:, 1], X[:, 2], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')

plt.xlim(-2, 11)
plt.ylim(-3, 10)
plt.xticks(())
plt.yticks(())
# plt.axis('equal')
display(X[1:, :], original_label)
plt.savefig('ex1.png', bbox_inches='tight', dpi = 300)





















