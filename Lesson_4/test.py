from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(5)

N = 4
a = np.array([[1, 2], [2, 3]])
b = np.array([[5, 4], [1, 7]])

D = cdist(a, b, 'sqeuclidean')
ans = np.argmin(D, axis = 0)
print(D)
print(ans)

