import cupy as cp
import numpy as np
from src.MLP.shuffle_data_gpu import shuffle_data
from src.Optimiser.optimisers_gpu import SGDOptimiser, AdamOptimiser
from src.MLP.network_layer_gpu import Layer

from src.CUDA.MLP.Activations.activation_kernels import relu as relu_cuda
#from src.CUDA.MLP.Forwards.forward_propagation_kernels import hidden_layer, forward_propagate
from src.CUDA.MLP.Forwards.forward_propagation_kernels import hidden_layer_wmma, forward_propagate_wmma
#from src.CUDA.MLP.Backwards.backward_propagation_kernels import hidden_layer_gradient, weight_gradient
from src.CUDA.MLP.Backwards.backward_propagation_kernels import weight_gradient_wmma, hidden_layer_gradient_wmma

import matplotlib.pyplot as plt
from cupy.random import permutation

class MultiLayerPerceptronGPU(object):

    def __init__(self, layer_sizes, batch_size, seed=None):

        if seed is not None:
            cp.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.batch_size = batch_size

        self.layers = []

        for i in range(self.n_layers):

            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]

            W = self.__he_weights_initialisation((n_out, n_in))
            dW = cp.zeros_like(W, dtype=cp.float32,order='C')

            b = cp.zeros(n_out, dtype=cp.float32,order='C')
            db = cp.zeros_like(b, dtype=cp.float32,order='C')

            a = cp.zeros((n_out, batch_size), dtype=cp.float32,order='C')
            z = cp.zeros((n_out, batch_size), dtype=cp.float32,order='C')

            gradient = cp.zeros_like(a, dtype=cp.float32, order='C')

            a_train = cp.zeros_like(a, dtype=cp.float32, order='C')
            z_train = cp.zeros_like(a, dtype=cp.float32, order='C')

            a_test = cp.zeros_like(a, dtype=cp.float32,order='C')
            z_test = cp.zeros_like(a, dtype=cp.float32, order='C')


            # Adam optimiser
            mW = cp.zeros_like(W, dtype=cp.float32,order='C')
            vW = cp.zeros_like(W, dtype=cp.float32,order='C')

            mb = cp.zeros_like(b, dtype=cp.float32,order='C')
            vb = cp.zeros_like(b, dtype=cp.float32,order='C')

            layer = Layer(W=W, b=b, dW=dW, db = db, gradient=gradient, a = a, z = z,
                          mW = mW, vW = vW, mb = mb, vb = vb,
                          a_train=a_train, z_train=z_train, a_test=a_test, z_test=z_test)

            self.layers.append(layer)

    @staticmethod
    def __xavier_weights_initialisation(shape) -> cp.ndarray:
        '''
        Computes the xavier initialization values for a weight matrix of size (shape)
        :param shape: weights dimensions
        :return: weights: ndarray
        '''

        in_dim, out_dim = shape
        xavier_limits = np.sqrt(6.) / np.sqrt(np.float32(in_dim) + np.float32(out_dim))
        weights = cp.asarray(cp.random.uniform(-xavier_limits, xavier_limits, (in_dim, out_dim)),
                             dtype=cp.float32,order='C')

        return weights

    @staticmethod
    def __he_weights_initialisation(shape) -> cp.ndarray:
        '''
        Computes the He weights initialization values for a weight matrix of size (shape)
        :param shape: weights dimensions
        :return: weights: ndarray
        '''

        in_dim, out_dim = shape
        xavier_limits = np.sqrt(6.) / np.sqrt(np.float32(in_dim))
        weights = cp.asarray(cp.random.uniform(-xavier_limits, xavier_limits, (in_dim, out_dim)),
                             dtype=cp.float32, order='C')

        return weights

    # Activation functions
    @staticmethod
    def relu(z: cp.ndarray,a: cp.ndarray) -> cp.ndarray:
        relu_cuda(z,a)
        return a

    @staticmethod
    def relu_grad(x: cp.ndarray) -> cp.ndarray:
        return cp.asarray((x > 0), dtype=cp.float32,order='C')

    @staticmethod
    def softmax(x: cp.ndarray) -> cp.ndarray:
        x = x - cp.max(x, axis=0, keepdims=True)
        expx = cp.exp(x,dtype=cp.float32)

        return cp.asarray(expx / cp.sum(expx, axis=0, keepdims=True, dtype=cp.float32),order='C')

    @staticmethod
    def __accuracy(y_prediction:cp.ndarray, y:cp.ndarray):
        classification_predictions = cp.argmax(y_prediction, axis=0)
        truth = cp.argmax(y, axis=0)

        is_equal = cp.asarray(cp.equal(classification_predictions, truth), dtype=cp.float32)

        accuracy_score = (cp.mean(is_equal, dtype=cp.float32))

        return 100*accuracy_score.get()

    @staticmethod
    def cross_entropy_loss(prediction: cp.ndarray, y:cp.ndarray):
        '''
        Evaluate the cross-entropy loss function -sum{y*log(p)} over the number of logits,
        and average the loss over the number of samples.
        :param prediction: output from the forward pass of the network
        :param y: truth
        :return: loss
        '''

        eps = cp.asarray(1e-8, dtype=cp.float32)

        loss = cp.sum(y * cp.log(prediction + eps))

        return -loss.get()/y.shape[1]

    def __forward_batch_cuda(self, x) -> cp.ndarray:
        '''
           Compute the forward pass through the MLP network using a batch=batch_size of the training data.
           :param x: ndarray
           return:
        '''
        # Input
        hidden_layer_wmma(self.layers[0].W, x, self.layers[0].b, self.layers[0].z,self.layers[0].a)

        # hidden layers
        for i in range(1,self.n_layers-1):
            hidden_layer_wmma(self.layers[i].W, self.layers[i-1].a, self.layers[i].b, self.layers[i].z, self.layers[i].a)

        #Output layer
        l = self.n_layers - 1
        forward_propagate_wmma(self.layers[l].W, self.layers[l-1].a, self.layers[l].b, self.layers[l].z)
        self.layers[l].a = self.softmax(self.layers[l].z)

        return self.layers[l].a

    def __forward_train_dataset_cuda(self, x) -> cp.ndarray:
        '''
            Compute the forward pass through the MLP network using the entire train input data.
            :param x: ndarray
            :return:
        '''

        # Input
        hidden_layer_wmma(self.layers[0].W, x, self.layers[0].b, self.layers[0].z_train, self.layers[0].a_train)

        # hidden layers
        for i in range(1, self.n_layers - 1):
            hidden_layer_wmma(self.layers[i].W, self.layers[i - 1].a_train, self.layers[i].b,
                         self.layers[i].z_train, self.layers[i].a_train)

        # Output layer
        l = self.n_layers - 1
        forward_propagate_wmma(self.layers[l].W, self.layers[l - 1].a_train, self.layers[l].b, self.layers[l].z_train)
        self.layers[l].a_train = self.softmax(self.layers[l].z_train)

        return self.layers[l].a_train

    def __forward_test_dataset_cuda(self, x) -> cp.ndarray:
        '''
        Compute the forward pass through the MLP network using the entire test input data.
        :param x: ndarray
        :return:
        '''

        # Input
        hidden_layer_wmma(self.layers[0].W, x, self.layers[0].b, self.layers[0].z_test,self.layers[0].a_test)

        # hidden layers
        for i in range(1,self.n_layers-1):
            hidden_layer_wmma(self.layers[i].W, self.layers[i-1].a_test, self.layers[i].b,
                         self.layers[i].z_test, self.layers[i].a_test)

        #Output layer
        l = self.n_layers - 1
        forward_propagate_wmma(self.layers[l].W, self.layers[l-1].a_test, self.layers[l].b, self.layers[l].z_test)
        self.layers[l].a_test = self.softmax(self.layers[l].z_test)

        return self.layers[l].a_test


    def __backward(self, x, y):

        output_layer = self.n_layers - 1
        norm_factor = 1./ self.batch_size

        y_output = self.__forward_batch_cuda(x)

        for n in range(output_layer,-1, -1):

            if n == output_layer:
                self.layers[n].gradient = y_output - y
            else:
                hidden_layer_gradient_wmma(self.layers[n+1].W, self.layers[n+1].gradient,
                                      self.layers[n].gradient, self.layers[n].z)

            previous_activation = x if n == 0 else self.layers[n-1].a

            weight_gradient_wmma(self.layers[n].gradient, previous_activation, self.layers[n].dW)
            self.layers[n].db = norm_factor * cp.sum(self.layers[n].gradient, axis=1,
                                                     keepdims=False, dtype=cp.float32)


    def train(self, x, y, epochs, learning_rate = 1e-3, method = 'Adam',beta1 = 0.9, beta2 = 0.999):
        # Setup
        batch_size = self.batch_size
        data_size = x.shape[1]
        iterations = data_size // batch_size

        # Train array allocations
        for layer in self.layers:
            layer.a_train = cp.zeros((layer.a.shape[0],data_size), dtype=cp.float32, order='C')
            layer.z_train = cp.zeros((layer.z.shape[0],data_size), dtype=cp.float32, order='C')

        if method == "SGD":
            optimiser = SGDOptimiser(learning_rate=learning_rate)
        else:
            optimiser = AdamOptimiser(learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=1e-8)

        for epoch in range(epochs):
            input_data, y_data = shuffle_data(x, y)

            for step in range(iterations):
                input_batch = cp.ascontiguousarray(input_data[:, step * batch_size:(step + 1) * batch_size],
                                                   dtype=cp.float32)
                y_batch = cp.ascontiguousarray(y_data[:, step * batch_size:(step + 1) * batch_size],
                                               dtype=cp.float32)

                self.__backward(input_batch, y_batch)

                optimiser.step(self.layers)


            print('epoch = ', epoch + 1, '|',
                  ' Accuracy = ', self.__accuracy(self.__forward_train_dataset_cuda(input_data), y_data), '%',
                  '| Loss = ', self.cross_entropy_loss(self.layers[self.n_layers-1].a_train,y_data))


        print('Training accuracy = ', self.__accuracy(self.__forward_train_dataset_cuda(x), y),'%')


    def test(self, x, y):
        # Test array allocations
        for layer in self.layers:
            layer.a_test = cp.zeros((layer.a.shape[0], y.shape[1]), dtype=cp.float32, order='C')
            layer.z_test = cp.zeros((layer.z.shape[0], y.shape[1]), dtype=cp.float32, order='C')

        print('Test accuracy = ', self.__accuracy(self.__forward_test_dataset_cuda(x), y), '%',' | '
              'Test loss = ', self.cross_entropy_loss(self.layers[self.n_layers-1].a_test, y))


    def plot(self,x):
        batches = 16 if x.shape[1] >= 16 else x.shape[1]

        num_cols = x.shape[1]
        indices = permutation(num_cols)

        if batches == 16:
            random_start = np.random.randint(x.shape[1]-17)
        else:
            random_start = 0

        batch = indices[random_start:random_start + batches]

        predictions = np.asarray(cp.argmax(self.layers[self.n_layers-1].a_test[:,batch],axis=0).get())
        data = 255*np.asarray(x[:,batch].get())

        plt.style.use('dark_background')
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]

        fig = plt.figure(figsize=(8, 8))


        if batches == 16:
            for n in range(batches):
                ax = fig.add_subplot(4,4, n+1)
                ax.imshow(data[:,n].reshape(28,28), cmap='gray', interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Prediction = ' + str(predictions[n]), fontsize=16,fontweight='bold')

            plt.show()
        else:
            for n in range(batches):
                factor = np.gcd(3,batches)

                ax = fig.add_subplot(factor, batches//factor, n + 1)
                ax.imshow(data[:, n].reshape(28, 28), cmap='gray', interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Prediction = ' + str(predictions[n]), fontsize=16,fontweight='bold')

            plt.show()

        return



