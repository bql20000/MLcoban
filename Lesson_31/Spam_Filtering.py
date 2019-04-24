import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

nwords = 2500

path = '/home/long/Code/MLcoban/Lesson_31/ex6DataPrepared/'
train_data_fn = 'train-features.txt'
test_data_fn = 'test-features.txt'
train_label_fn = 'train-labels.txt'
test_label_fn = 'test-labels.txt'

def read_data(data_fn, label_fn):
    with open(path + label_fn) as f:
        content = f.readlines()
    label = [int(x.strip()) for x in content]           

    data = np.zeros((len(content), nwords))
    with open(path + data_fn) as f:
        content = f.readlines()
        content = [x.strip() for x in content]          #remove '\n' 
        for line in content:
            a = line.split(' ')
            data[int(a[0])-1][int(a[1])-1] = int(a[2])

    return (data, label)
        
    

read_data(train_data_fn, train_label_fn)
(train_data, train_label) = read_data(train_data_fn, train_label_fn)
(test_data, test_label) = read_data(test_data_fn, test_label_fn)

model = MultinomialNB()
model.fit(train_data, train_label)

y_pred = model.predict(test_data)
print('Training size = %d, accuracy = %.2f%%' % (train_data.shape[0], accuracy_score(test_label, y_pred) * 100))