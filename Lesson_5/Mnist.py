import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from sklearn.cluster import KMeans
import os

print(os.path.abspath("Lesson_5"))
mndata = MNIST('/home/long/Code/MLcoban/Lesson_5')

mndata.load_testing()
#X = mndata.test_images

#kmeans = KMeans(n_clusters=10).fit(X)
#pred_label = kmeans.predict(X)