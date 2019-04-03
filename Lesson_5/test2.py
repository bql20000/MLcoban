import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

img = mpimg.imread('/home/long/Code/MLcoban/Lesson_5/girl3.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
#plt.axis('off')
#plt.show() 
X = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

def kmeans_assign_color(X, centers):
    D = cdist(X, centers)
    tmp = np.argmin(D, axis=1)
    ans = np.zeros_like(X)
    return ans 

kmeans = KMeans(n_clusters=3).fit(X)

img5 = kmeans_assign_color(X, kmeans_assign_color)




