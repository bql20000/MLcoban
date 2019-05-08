import numpy as np

y = np.array([[1], [-1]])
xi = np.array([1, 2, 3]).reshape(1,3)

yi = y[1]
w = yi * xi
print(yi.shape, xi.shape, w.shape)