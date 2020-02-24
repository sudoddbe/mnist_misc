import numpy as np
import matplotlib.pyplot as plt
from mnist_utils import *


def PCA(data, number_eigenvectors = 15):
#Building a convariance matrix in order to do PCA. We need to first remove the 
#average, this correspond to moving the center of the hyperdimensional point
#cluster to the zero - we only want the eigenvectors to model the variations 
#i.e the most common variational axes...
    m = np.mean(data, axis = 0)
    covariance_matrix = np.zeros((np.prod(m.shape), np.prod(m.shape))) 
    for sample in data:
        tmp = sample - m
        covariance_matrix += np.outer(tmp, tmp) 

#We do not really need to divide by the number of samples - this is just a 
#scaling of the eigenvalues. Since the covariance matrix is guaranteed to be 
#symmetric we can use the hermitian eigh version in np.linalg.
    w, v = np.linalg.eigh(covariance_matrix / data.shape[0])

#We want the largest eigenvalues, so sort and reverse... 
    ind = np.argsort(w)[::-1]
    v = v[:, ind]
    v = v[:, 0:number_eigenvectors]
    v = v.T
    w = w[ind]

    return m, v

def get_PCA_error(data, m, eigenvectors):
    tmp = data
#Subtract mean
    tmp = tmp - m

#Project unto the eigenvectors
    coeff = np.matmul(tmp, eigenvectors.T)
    proj = np.matmul(coeff, eigenvectors)
    
    diff = tmp - proj 
    error = np.sum(diff**2, axis = 1)
    return error



(x_train, y_train), (x_val, y_val) = load_mnist()
#x_train = x_train[0:10000,:]
#y_train = y_train[0:10000]
label_data_dict = split_by_label(x_train, y_train)
images = [np.mean(d, axis = 0) for d in label_data_dict.values()]
imshow_subplots(images)

PCA_dict = {}
errors = {}
for k,v in label_data_dict.iteritems():
    (m, eigenvectors) = PCA(v)
    images = [e for e in eigenvectors]
    images.append(m)
    imshow_subplots(images)
    PCA_dict[k] = (m, eigenvectors)
    error = get_PCA_error(x_val, m, eigenvectors)
    errors[k] = error

errors_vec = np.zeros((len(label_data_dict.keys()), len(error)))
for k, v in errors.iteritems():
    errors_vec[k] = v

category = np.argmin(errors_vec, axis = 0)
print np.mean(category == y_val)
fail_ind = category != y_val
failures = x_val[fail_ind, :]
c_vec = category[fail_ind]
for i in range(10):
    plt.figure()
    mnist_sample_imshow(failures[i])
    c = c_vec[i]
    plt.title("classified as %i"%c)
plt.show()
