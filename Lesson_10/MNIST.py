import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

np.random.seed(2)

mnist = fetch_mldata('MNIST original', data_home='/home/long/Code/MLcoban/Lesson_10')
N, d = mnist.data.shape

x_all = mnist.data
y_all = mnist.target


x0 = x_all[y_all == 0, :]
x1 = x_all[y_all == 1, :]

X = np.concatenate((x0, x1), axis = 0)
y = np.concatenate((np.zeros(x0.shape[0]), np.ones(x1.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2000)

model = LogisticRegression(C = 1e5)   # C = 1/lamda
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy %.2f' %(100*accuracy_score(y_test, y_pred)))

#print missed classification
#mis = np.where(y_pred != y_test)[0]

Xmis = X_test[y_pred != y_test, :]

from display_network import *

plt.axis('off')
A = display_network(Xmis.T, 1, Xmis.shape[0])
f2 = plt.imshow(A, interpolation='nearest')
plt.gray()
plt.show()





