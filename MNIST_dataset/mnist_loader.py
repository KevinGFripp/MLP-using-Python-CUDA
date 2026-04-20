import numpy as np
import struct
from array import array
from os.path import join
from pathlib import Path


# MNIST Data Loader Class
class MnistDataloader():
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_image_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_image_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_image_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)



def load_mnist_data():

    # Set file paths based on added MNIST Datasets
    input_path = Path(__file__).resolve().parent

    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # Load MNIST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath,
                                       training_labels_filepath,
                                       test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # repackage as array, normalise image data -> [0,1]
    train_data = np.asarray(x_train,dtype=np.float32).reshape(60000, 784).transpose()/255
    ydata = np.asarray(y_train,dtype=np.float32)

    #one hot format
    train_ydata = np.zeros((10,60000),dtype=np.float32)
    for n in range(60000):
        train_ydata[int(ydata[n]),n] = 1.0

    # repackage as array, normalise image data -> [0,1]
    test_data = np.asarray(x_test,dtype=np.float32).reshape(10000, 784).transpose()/255

    #one hot format
    ydata = np.asarray(y_test, dtype=np.float32)
    test_ydata = np.zeros((10, 10000), dtype=np.float32)
    for n in range(10000):
        test_ydata[int(ydata[n]), n] = 1.0

    return (train_data, train_ydata), (test_data, test_ydata)