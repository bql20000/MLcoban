import numpy as np
import matplotlib.pyplot as plt

n = 5   

#Initing data
X = np.array([[120, 130, 150, 200, 300]]).T
y = np.array([[2, 3, 4, 7, 9]]).T

#Visualizing data
plt.plot(X, y, 'ro')
plt.axis([0, 400, 0, 12])
plt.xlabel("Pig's' Weight (kg)")
plt.ylabel("Pig's Price (million VNĐ)")
#plt.show()

#Linear Regression
X_bar = np.append(X, np.ones(n).reshape(n, 1), axis=1)
X = np.array(X_bar).T

A = np.dot(X, X_bar)
w = np.dot(np.dot(np.linalg.inv(A), X), y)

xx = np.linspace(0, 400, 401)
yy = xx * w[0][0] + w[1][0]

print(w)


plt.plot(xx, yy)
plt.show()
