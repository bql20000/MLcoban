import matplotlib.pyplot as plt
import numpy as np

a = np.arange(-2, 2.5, 0.5)
b = np.arange(-2, 2.5, 0.5)

A, B = np.meshgrid(a, b)

tmp = A + B
C = (1 - np.exp(-2*tmp)) / (1+ np.exp(-2*tmp))

plt.contour(A, B, C, 8)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

x = np.linspace(-2, 2, 11, endpoint=True)
S = (1 - np.exp(-2*x)) / (1+ np.exp(-2*x))
plt.plot(x, S)

plt.show()
