import numpy as np

X = np.array([[1, 2],[3, 4]])
xi = X[1].reshape(1, -1)
print(xi.shape)

w = np.array([1, 2])
print(np.linalg.norm(w))