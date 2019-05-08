import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(2)

#generate data
N = 1000
x = np.random.rand(N)
y = 4 + 3 * x + 2 * np.random.randn(N)
X = np.concatenate((x.reshape(-1, 1), np.ones((N, 1))), axis = 1)
y = y.reshape(-1, 1)

#visualize data
#plt.plot(x, y, 'ro')
#plt.show()

#sklearn
model = LinearRegression()
model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

w, b = model.coef_[0][0], model.intercept_[0]
sol_sklearn = np.array([b, w])
print(sol_sklearn)

#GD
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = X[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi*a).reshape(2, 1)

def SGD(w_init, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    count = 0
    for it in range(10):
        # shuffle data 
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1 
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new                 
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:                                    
                    return w
                w_last_check = w_this_check
    return w

sol = SGD(np.array([[2], [1]]), 0.1)
print(sol[-1])
























