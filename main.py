from MNIST_dataset.mnist_loader import load_mnist_data
from src.MLP.multi_layer_perceptron_gpu import MultiLayerPerceptronGPU
from cupy import asarray


if __name__ == '__main__':

    # # Load the MNIST dataset
    (train_data,y),(test_data,test) = load_mnist_data()

    # Move data to gpu
    train_data_gpu = asarray(train_data,dtype='float32',order='C')
    y_gpu = asarray(y,dtype='float32',order='C')
    test_data_gpu = asarray(test_data,dtype='float32',order='C')
    test_gpu = asarray(test,dtype='float32',order='C')

    # Setup
    mlp = MultiLayerPerceptronGPU(layer_sizes=[784, 1024, 512, 10], batch_size=256, seed=0)

    mlp.train(train_data_gpu, y_gpu, epochs=10)

    mlp.test(test_data_gpu, test_gpu)

    mlp.plot(test_data_gpu)





