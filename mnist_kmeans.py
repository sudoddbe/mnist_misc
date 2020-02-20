import numpy as np
import matplotlib.pyplot as plt
from mnist_utils import *

def kmeans_clustering(data, k = 3, max_iter = 100):
    clusters = data[np.random.randint(data.shape[0], size=k),:,:].copy()
    dist2 = np.zeros((data.shape[0], k))
    for _ in range(max_iter):
        for i, cluster in enumerate(clusters):
            dist2[:,i] = np.sum((data.reshape(data.shape[0], -1) - cluster.flatten())**2, axis=-1)

        closest_cluster = np.argmin(dist2, axis = -1)
        new_clusters = np.array([np.mean(data[closest_cluster == i], axis = 0) for i in range(k)])
        maxdiff = np.max(np.abs(new_clusters - clusters))
        if maxdiff < 0.01:
            return new_clusters
        clusters = new_clusters
    return clusters

def kmeans_classification(data, cluster_dict):
    dist2 = np.zeros((data.shape[0], len(cluster_dict)))
    #Could just as well have used the keys directly in this case, but it is
    #more general to not use that the labels in this case are numbers...
    for i, clusters in enumerate(cluster_dict.values()):
        tmpdist2 = np.zeros((data.shape[0], clusters.shape[0]))
        for ii, cluster in enumerate(clusters):
            tmpdist2[:,ii] = np.sum((data.reshape(data.shape[0], -1) - cluster.flatten())**2, axis=-1)
        dist2[:,i] = np.amin(tmpdist2, axis=-1)

    classification_index = np.argmin(dist2, axis=-1)
    classification = np.array([cluster_dict.keys()[i] for i in classification_index])
    return classification

(x_train, y_train), (x_val, y_val) = load_mnist()
#x_train = x_train[0:10000,:,:]
#y_train = y_train[0:10000]
label_data_dict = split_by_label(x_train, y_train)

cluster_dict = {key: kmeans_clustering(val, k = 15) for key, val in label_data_dict.iteritems()}
images = []
for i in range(10):
    images.append(cluster_dict[i])
images = np.array(images).reshape((-1, x_train.shape[1], x_train.shape[2]))
imshow_subplots(images)

classification = kmeans_classification(x_val, cluster_dict)
print np.mean(classification == y_val)
fail_ind = classification != y_val
failures = x_val[fail_ind, :, :]
c_vec = classification[fail_ind]
for i in range(10):
    plt.figure()
    plt.imshow(failures[i])
    c = c_vec[i]
    plt.title("classified as %i"%c)

plt.show()
