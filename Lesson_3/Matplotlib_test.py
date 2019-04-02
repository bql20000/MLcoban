import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 11)
y = np.linspace(-5, 10, 11)

xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)

X = np.linspace(-np.pi, np.pi, 256);
C, S = np.cos(X), np.sin(X)

plt.xlim(-5, 5)
plt.ylim(-2, 2)
plt.plot(X, S, label='cosin')
plt.plot(X, C, label='sine')
plt.legend(loc='upper left')

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))


plt.show()