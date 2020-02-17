import numpy as np
import matplotlib.pyplot as plt
from mnist_utils import *

def get_nearest_neighbour(training_data, sample):
    tmp = training_data
    if len(training_data.shape) > 2:
#basically flattening the last dimensions of the trainingdata
        tmp = training_data.reshape(training_data.shape[0], -1)
    diff = tmp - sample.flatten()
    dist2 = np.sum(diff**2, axis=-1)
    return np.argmin(dist2)

def get_k_nearest_neighbour(training_data, training_labels, sample, k = 10):
    tmp = training_data
    if len(training_data.shape) > 2:
#basically flattening the last dimensions of the trainingdata
        tmp = training_data.reshape(training_data.shape[0], -1)
    diff = tmp - sample.flatten()
    dist2 = np.sum(diff**2, axis=-1)
    sort_index = np.argsort(dist2)
    vote_index = sort_index[0:k]
    labels = training_labels[vote_index]
    return np.argmax(np.bincount(labels))

(x_train, y_train), (x_val, y_val) = load_mnist()
x_train_flat = x_train.reshape(x_train.shape[0], -1)
category = np.zeros_like(y_val)
for i in range(x_val.shape[0]):
    nn = get_nearest_neighbour(x_train_flat, x_val[i])
    category[i] = y_train[nn]
    #label = get_k_nearest_neighbour(x_train_flat, y_train, x_val[i])
    #category[i] = label 
    if i % 100 == 0:
        print i
        print np.mean(category[0:i] == y_val[0:i])

print np.mean(category == y_val)
