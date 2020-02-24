import numpy as np
import matplotlib.pyplot as plt

#Size of a mnist sample
MNIST_IMAGE_SIZE = (28, 28)

def load_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float64")
    x_test = x_test.astype("float64")
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    return (x_train, y_train), (x_test, y_test)

def reshape_samples(data):
    return np.reshape(data, (data.shape[0], MNIST_IMAGE_SIZE[0], MNIST_IMAGE_SIZE[1]))

def split_by_label(dataset, labels):
    unique_labels = np.unique(labels)
    label_data_dict = {l : dataset[labels == l, :] for l in unique_labels}
    return label_data_dict

def mnist_sample_imshow(sample):
    sample = np.reshape(sample, MNIST_IMAGE_SIZE)
    plt.imshow(sample)

def imshow_subplots(images):
    number_subplots = len(images)
    rows = int(np.sqrt(number_subplots))
    cols = np.ceil(number_subplots * (1.0/rows))
    cols += cols*rows < number_subplots
    plt.figure()
    i = 1
    for image in images:
        plt.subplot(rows,cols,i)
        i += 1
        mnist_sample_imshow(image)
