from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#initalize data
N = 1000
x = np.random.rand(N)
y = 4 + 3*x + 5*np.random.randn(N)

#visualize data
#plt.plot(x, y, 'ro')
#plt.show()

#Linear Regression
model = LinearRegression()
model.fit(x.reshape(N, 1), y.reshape(N, 1))

w, b = model.coef_[0][0], model.intercept_[0]
sol_sklearn = np.array([b, w])
print(sol_sklearn)


#Gradient Descent
eps = 1e-4
def grad(w):
    return 1/N * X.dot(X_bar.dot(w) - y)

def cost(w):
    return 1/(2*N) * np.ligalg.norm(y - X_bar.dot(w))**2

def myGD(w0, eta):
    pre_w = w0

    for it in range(100):
        new_w = pre_w - eta * grad(pre_w)
        if np.linalg.norm(grad(new_w))/len(new_w) < eps:        #???
            break
        pre_w = new_w
    return (pre_w, it)

#reshape(-1, 1) means we set the 2nd dimension equal to 1 and we let numpy figure out the (-1) dimension base on the length of the original array
y = y.reshape(-1, 1)
X_bar = np.concatenate( (x.reshape(-1, 1), np.ones((N, 1))) , axis = 1)
X = X_bar.T
w_init = np.array([[2], [1]])

(w, it) = myGD(w_init, 1)

print('Solution found by GD: w = ', w.T, '\nNumber of iterations: %d' %(it+1))



































































