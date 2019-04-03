import numpy as np

a = [6]
b = np.full_like(a, 1)
b[0]  = 0.5
print(b)