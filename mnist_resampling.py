import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from mnist_utils import *
import mnist_kmeans

def maximum_spread_resampling(data, starting_point=None, size=25):
    if starting_point == None:
        sample_set = np.mean(data, axis = 0)
    else:
        sample_set = starting_point
    sample_set = np.reshape(sample_set, (1, -1))
    while sample_set.shape[0] < size:
        sample_set = np.vstack((sample_set, next_max_spread_sample(data, sample_set)))
    return sample_set

def next_max_spread_sample(data, sample_set, threshold=1000000000):
    dm = scipy.spatial.distance_matrix(data, sample_set)
    min_dm = np.min(dm, axis = -1)
    sample_to_add = np.argmax(min_dm)
    return data[sample_to_add, :]

if __name__=="__main__":
    print "loading data"
    (x_train, y_train), (x_val, y_val) = load_mnist()
    #x_train = x_train[0:10000,:]
    #y_train = y_train[0:10000]
    print "splitting data"
    label_data_dict = split_by_label(x_train, y_train)

    print "resampling"
    resampling_dict = {}
    for key, val in label_data_dict.iteritems():
        resampling_dict[key] = maximum_spread_resampling(val)
        imshow_subplots(resampling_dict[key])


    classification = mnist_kmeans.kmeans_classification(x_val, resampling_dict)
    print np.mean(classification == y_val)
    fail_ind = classification != y_val
    failures = x_val[fail_ind, :]
    c_vec = classification[fail_ind]
    for i in range(10):
        plt.figure()
        mnist_sample_imshow(failures[i])
        c = c_vec[i]
        plt.title("classified as %i"%c)

    plt.show()
