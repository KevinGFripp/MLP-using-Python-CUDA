import numpy as np
from src.MLP.shuffle_data import shuffle_data
from src.Optimiser.optimisers import SGDOptimiser, AdamOptimiser
from src.MLP.network_layer import Layer


class MultiLayerPerceptron(object):

    def __init__(self, layer_sizes, batch_size, seed=None):

        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.batch_size = batch_size

        self.layers = []

        for i in range(self.n_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]

            W = self.__he_weights_initialisation((n_out, n_in))
            dW = np.zeros_like(W, dtype=np.float32)

            b = np.zeros(n_out, dtype=np.float32)
            db = np.zeros_like(b, dtype=np.float32)

            a = np.zeros((n_out, batch_size), dtype=np.float32)
            z = np.zeros((n_out, batch_size), dtype=np.float32)

            gradient = np.zeros_like(a, dtype=np.float32)

            a_train = np.zeros_like(a, dtype=np.float32)
            z_train = np.zeros_like(a, dtype=np.float32)

            a_test = np.zeros_like(a, dtype=np.float32)
            z_test = np.zeros_like(a, dtype=np.float32)

            # Adam optimiser
            mW = np.zeros_like(W, dtype=np.float32)
            vW = np.zeros_like(W, dtype=np.float32)

            mb = np.zeros_like(b, dtype=np.float32)
            vb = np.zeros_like(b, dtype=np.float32)

            layer = Layer(W=W, b=b, dW=dW, db=db, gradient=gradient, a=a, z=z,
                          mW=mW, vW=vW, mb=mb, vb=vb, a_train=a_train, z_train=z_train,
                          a_test=a_test, z_test=z_test)

            self.layers.append(layer)

    @staticmethod
    def __xavier_weights_initialisation(shape):
        '''
        Computes the xavier initialization values for a weight matrix of size (shape)
        :param shape: weights dimensions
        :return: weights: ndarray
        '''

        in_dim, out_dim = shape
        xavier_limits = np.sqrt(6.) / np.sqrt(np.float32(in_dim) + np.float32(out_dim))
        weights = np.float32(np.random.uniform(-xavier_limits, xavier_limits, (in_dim, out_dim)))

        return weights

    @staticmethod
    def __he_weights_initialisation(shape):
        '''
               Computes the He weights initialization values for a weight matrix of size (shape)
               :param shape: weights dimensions
               :return: weights: ndarray
        '''

        in_dim, out_dim = shape
        xavier_limits = np.sqrt(6.) / np.sqrt(np.float32(in_dim))
        weights = np.float32(np.random.uniform(-xavier_limits, xavier_limits, (in_dim, out_dim)))

        return weights

    # Activation functions
    @staticmethod
    def __relu(x):
        return np.maximum(0, x)

    @staticmethod
    def __relu_grad(x):
        return (x > 0).astype(x.dtype)

    @staticmethod
    def __softmax(x):
        x = x - np.max(x, axis=0, keepdims=False)
        expx = np.exp(x)

        return expx / np.sum(expx, axis=0, keepdims=False, dtype=np.float32)

    @staticmethod
    def __accuracy(y_prediction, y):
        classification_predictions = np.argmax(y_prediction, axis=0)
        truth = np.argmax(y, axis=0)

        is_equal = np.float32(np.equal(classification_predictions, truth))

        return 100*(np.mean(is_equal, dtype=np.float32))

    @staticmethod
    def cross_entropy_loss(prediction, y):
        '''
              Evaluate the cross-entropy loss function -sum{y*log(p)} over the number of logits,
              and average the loss over the number of samples.
              :param prediction: output from the forward pass of the network
              :param y: truth
              :return: loss
        '''
        eps = 1e-8
        loss = np.sum(y * np.log(prediction + eps))

        return -loss / y.shape[1]


    def __forward(self, x):

        a_layer = x

        for i in range(0,self.n_layers):
            self.layers[i].z = self.layers[i].W @ a_layer + self.layers[i].b[:,None]

            if i == self.n_layers - 1: # Output layer
                self.layers[i].a = self.__softmax(self.layers[i].z)

            else: # Hidden layer
                self.layers[i].a = self.__relu(self.layers[i].z)

            a_layer = self.layers[i].a


        return a_layer

    def __forward_train(self, x):

        a_layer = x

        for i in range(0,self.n_layers):
            self.layers[i].z_train = self.layers[i].W @ a_layer + self.layers[i].b[:,None]

            if i == self.n_layers - 1: # Output layer
                self.layers[i].a_train = self.__softmax(self.layers[i].z_train)

            else: # Hidden layer
                self.layers[i].a_train = self.__relu(self.layers[i].z_train)

            a_layer = self.layers[i].a_train


        return a_layer

    def __forward_test(self, x):

        a_layer = x

        for i in range(0,self.n_layers):
            self.layers[i].z_test = self.layers[i].W @ a_layer + self.layers[i].b[:,None]

            if i == self.n_layers - 1: # Output layer
                self.layers[i].a_test = self.__softmax(self.layers[i].z_test)

            else: # Hidden layer
                self.layers[i].a_test = self.__relu(self.layers[i].z_test)

            a_layer = self.layers[i].a_test


        return a_layer


    def __backward(self, x, y):

        output_layer = self.n_layers - 1
        norm_factor = (1.0 / self.batch_size)

        y_output = self.__forward(x)

        # for softmax output activation
        self.layers[output_layer].gradient = y_output - y

        self.layers[output_layer].dW = norm_factor * self.layers[output_layer].gradient @ self.layers[output_layer-1].a.T
        self.layers[output_layer].db = norm_factor * np.sum(self.layers[output_layer].gradient,
                                                            axis =1, keepdims=False,
                                                            dtype=np.float32)

        for n in range(output_layer-1, -1, -1):
            self.layers[n].gradient = norm_factor * (self.layers[n+1].W.T @ self.layers[n+1].gradient)
            self.layers[n].gradient *= self.__relu_grad(self.layers[n].z)

            previous_activation = x if n == 0 else self.layers[n-1].a

            self.layers[n].dW = norm_factor * self.layers[n].gradient @ previous_activation.T
            self.layers[n].db = norm_factor * np.sum(self.layers[n].gradient,
                                                     axis=1, keepdims=False, dtype=np.float32)


    def train(self, x, y, epochs, learning_rate = 1e-3, method = 'Adam',beta1 = 0.9, beta2 = 0.999):
        # Setup
        batch_size = self.batch_size
        data_size = x.shape[1]
        iterations = data_size // batch_size

        if method == "SGD":
            optimiser = SGDOptimiser(learning_rate)
        else:
            optimiser = AdamOptimiser(learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=1e-8)

        for epoch in range(epochs):
            input_data, y_data = shuffle_data(x, y)

            for step in range(iterations):
                input_batch = input_data[:, step * batch_size:(step + 1) * batch_size]
                y_batch = y_data[:, step * batch_size:(step + 1) * batch_size]

                self.__backward(input_batch, y_batch)
                optimiser.step(self.layers)

            print('epoch = ', epoch + 1, '|',
                  ' Accuracy = ', self.__accuracy(self.__forward_train(input_data), y_data), '%',
                  '| Loss = ', self.cross_entropy_loss(self.layers[self.n_layers-1].a_train,y_data))


        print('Training accuracy = ', self.__accuracy(self.__forward_train(x), y),'%')


    def test(self, x, y):
        # Test array allocations
        for layer in self.layers:
            layer.a_test = np.zeros((layer.a.shape[0], y.shape[1]), dtype=np.float32)
            layer.z_test = np.zeros((layer.z.shape[0], y.shape[1]), dtype=np.float32)

        print('Test accuracy = ', self.__accuracy(self.__forward_test(x), y), '%',' | '
              'Test loss = ', self.cross_entropy_loss(self.layers[self.n_layers-1].a_test, y))



