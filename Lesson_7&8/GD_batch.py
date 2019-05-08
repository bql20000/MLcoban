import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(11)

#generate data
N = 10
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
y_pred = model.predict(x.reshape(N, 1))
print(sol_sklearn, model.score(y_pred.reshape(-1, 1), y))

#GD
def grad(cur_id, w):
    xi = X[cur_id, :]
    yi = y[cur_id]
    return (xi * (np.dot(xi, w) - yi)).reshape(2, 1)

def SGD(w0, eta):
    w_pre = w0
    iter_check_w = 10
    w_last_check = w0
    count = 0
    for step in range(10):
        rand_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            cur_id = rand_id[i]
            w_new = w_pre - eta * grad(cur_id, w_pre)
            w_pre = w_new
            if (count%iter_check_w == 0):
                if np.linalg.norm(w_new - w_last_check)/len(w_new) < 1e-3:
                    return w_new
                w_last_check = w_new
            
            
    return (w_pre)

sol = SGD(np.array([[2], [1]]), 0.1)
print(sol)
























