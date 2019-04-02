from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

K = 3
X = np.concatenate((X0, X1, X2), axis = 0)

X0 = np.array([0]*N)
X1 = np.array([1]*N)
X2 = np.array([2]*N)
original_label = np.concatenate((X0, X1, X2), axis = 0)

def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 3, alpha = 0.9)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 3, alpha = 0.9)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 3, alpha = 0.9)

    plt.axis('equal')
    plt.plot()
    plt.show()

def dist(a, b):
    return (a[0]-b[0]) * (a[0]-b[0]) + (a[1]-b[1]) * (a[1]-b[1])

def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    labels = []
    for i in range(X.shape[0]):
        cur_label = 0
        min_dist = dist(X[i], centers[0])
        for cluster in range(len(centers)): 
            tmp = dist(X[i], centers[cluster])
            if (tmp < min_dist): 
                min_dist = tmp
                cur_label = cluster
        labels.append(cur_label)
    return np.array(labels)
        
def new_centers(X, labels):
    K = np.amax(labels) + 1
    new = []
    for i in range(K): 
        new.append([0, 0])

    for i in range(X.shape[0]):
        for dim in range(X.shape[1]):
            new[labels[i]][dim] += X[i][dim]

    for i in range(K):
        for dim in range(X.shape[1]):
            new[i][dim] /= X[labels == i, :].shape[0]

    return new

def cal_lost(X, labels, centers):
    total_lost = 0
    for i in range(X.shape[0]):
        total_lost += dist(centers[labels[i]], X[i])
    return total_lost

def kmeans(X, K):
    centers = kmeans_init_centers(X, K)
    labels = []
    pre_lost = -1
    while True:
        print(centers)
        labels = kmeans_assign_labels(X, centers)
        lost = cal_lost(X, labels, centers)
        if (lost == pre_lost): 
            return (centers, labels)
        else:
            pre_lost = lost
            centers = new_centers(X, labels)
            

#kmeans_display(X, original_label)

(centers, labels) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[:])

kmeans_display(X, labels)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label)
