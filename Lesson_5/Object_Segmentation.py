import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#read & visualize
img = mpimg.imread('/home/long/Code/MLcoban/Lesson_5/girl3.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
#plt.show() 
X = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

kmeans = KMeans(n_clusters=10).fit(X)

#Solution 1
def kmeans_assign_color(X, centers):
    D = cdist(X, centers)
    tmp = np.argmin(D, axis=1)
    ans = np.zeros_like(X)
    for i in range(tmp.shape[0]): ans[i] = centers[tmp[i]]
    return ans

#Solution 2
labels = kmeans.predict(X)
K = np.amax(labels) + 1
#zeros_like --> img2 has type same as X (int)
img2 = np.zeros_like(X)
for k in range(K): img2[labels == k] = kmeans.cluster_centers_[k] 
#here right side is float, but left side's type is int, so result is int


img2 = kmeans_assign_color(X, kmeans.cluster_centers_)

img3 = img2.reshape(img.shape[0], img.shape[1], img.shape[2])
plt.imshow(img3)
plt.show()



