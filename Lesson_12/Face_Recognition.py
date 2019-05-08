import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import misc
from scipy.io import loadmat

np.random.seed(1)

path = '/home/long/Code/MLcoban/Lesson_12'
train_ids = np.arange(1, 26)
test_ids = np.arange(26, 51)
view_ids = np.hstack( (np.arange(1, 8), np.arange(14, 21)) )

ARbase = loadmat('/home/long/Code/MLcoban/Lesson_12/randomfaces4ar/randomfaces4ar.mat')

def check_id():
    a, b, c = 2, 50, 26
    cur_id = 0
    pos = np.zeros((a, b+1, c+1), dtype=int)
    for i in range(a):
        for j in range(1, b+1):
            for k in range(1, c+1):
                pos[i][j][k] = cur_id
                cur_id += 1
    return pos

def build_use_ids_list(n_personType, type_ids, view_ids):
    a, b, c = 2, 50, 26
    list_id = []
    for i in range(a):
        for j in type_ids:
            for k in view_ids:
                cur_id = i*b*c + (j-1)*c + k-1
                list_id.append(cur_id)
                #if (pos[i][j][k] != cur_id): print(pos[i][j][k], cur_id)
    return list_id

#print(len(ARbase))
#for key in ARbase.items():
#    print(key)

pos = check_id()
train_list_id = build_use_ids_list(2, train_ids, view_ids)
test_list_id = build_use_ids_list(2, test_ids, view_ids)

print(ARbase['featureMat'].shape)
X_all = ARbase['featureMat'].T

X_train, X_test = X_all[train_list_id, :], X_all[test_list_id, :]
n_train = int(X_train.shape[0]/2)
y_train = np.concatenate( (np.zeros((n_train,1)), np.ones((n_train,1))), axis=0 )
y_test = y_train

model = LogisticRegression(C=1e5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(X_test.shape)
print('Accuracy score = %.2f' %accuracy_score(y_pred, y_test))


    




