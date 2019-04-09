import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

print('Number of clasess: %d' %len(np.unique(iris_y)))
print('Number of points: %d' %len(iris_x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=50)

print("Training size: %d" %len(x_train))
print("Testing size: %d" %len(x_test))

#1NN
clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy of 1NN: %.2f %%" %(100 * accuracy_score(y_test, y_pred)))

#10NN
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy of 10NN: %.2f %%" %(100 * accuracy_score(y_test, y_pred)))

#10NN with weights
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy of 10NN (weight = 1/distance): %.2f %%" %(100 * accuracy_score(y_test, y_pred)))