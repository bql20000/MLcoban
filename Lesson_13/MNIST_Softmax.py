import numpy as np
from sklearn.linear_model import LogisticRegression
from mnist import MNIST
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score
from mlxtend.data import loadlocal_mnist

path = '/home/long/Code/MLcoban/Lesson_13/MNIST/'
X_train, y_train = loadlocal_mnist(images_path= path + 'train-images.idx3-ubyte', labels_path= path + 'train-labels.idx1-ubyte')
X_test, y_test =  loadlocal_mnist(images_path= path + 't10k-images.idx3-ubyte', labels_path= path + 't10k-labels.idx1-ubyte')
#print('Dimensions: %s x %s' % (X_train.shape[0], X_train.shape[1]))
#print('Dimensions: %s x %s' % (X_test.shape[0], X_test.shape[1]))

X_train = X_train / 255.0
X_test = X_test / 255.0

model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy score : %.2f" %(100*accuracy_score(y_pred, y_test)))
















