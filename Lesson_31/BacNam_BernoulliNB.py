from __future__ import print_function
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


d1 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

train_data = np.array([d1, d2, d3, d4])
train_label = np.array(['B', 'B', 'B', 'N'])
 
d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])

model = BernoulliNB()
model.fit(train_data, train_label)

print(model.predict(d5)[0])
print(model.predict_proba(d6))
