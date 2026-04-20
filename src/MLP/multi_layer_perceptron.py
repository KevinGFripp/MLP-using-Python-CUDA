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
            n_out = layer_sizes[i+1]

            W = self.__xavier_weights_initialisation((n_out, n_in))
            dW = np.zeros((n_out, n_in), dtype=np.float32)

            b = np.zeros((n_out, 1), dtype=np.float32)
            db = np.zeros((n_out, 1), dtype=np.float32)

            a = np.zeros((n_out, batch_size), dtype=np.float32)
            z = np.zeros((n_out, batch_size), dtype=np.float32)

            # Adam optimiser
            mW = np.zeros_like(W, dtype=np.float32)
            vW = np.zeros_like(W, dtype=np.float32)

            mb = np.zeros_like(b, dtype=np.float32)
            vb = np.zeros_like(b, dtype=np.float32)

            layer = Layer(W=W, b=b, dW=dW, db = db, a = a, z = z,
                          mW = mW, vW = vW, mb = mb, vb = vb)
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

    # Activation functions
    @staticmethod
    def __relu(x):
        return np.maximum(0, x)

    @staticmethod
    def __relu_grad(x):
        return (x > 0).astype(x.dtype)

    @staticmethod
    def __softmax(x):
        x = x - np.max(x, axis=0, keepdims=True)
        expx = np.exp(x)

        return expx / np.sum(expx, axis=0, keepdims=True, dtype=np.float32)

    @staticmethod
    def __accuracy(y_prediction, y):
        classification_predictions = np.argmax(y_prediction, axis=0)
        truth = np.argmax(y, axis=0)

        is_equal = np.float32(np.equal(classification_predictions, truth))

        return 100*(np.mean(is_equal, dtype=np.float32))


    def __forward(self, x):

        a_layer = x

        for i in range(0,self.n_layers):
            self.layers[i].z = self.layers[i].W @ a_layer + self.layers[i].b

            if i == self.n_layers - 1: # Output layer
                self.layers[i].a = self.__softmax(self.layers[i].z)

            else: # Hidden layer
                self.layers[i].a = self.__relu(self.layers[i].z)

            a_layer = self.layers[i].a


        return a_layer


    def __backward(self, x, y):

        norm_factor = (1.0 / self.batch_size)

        y_output = self.__forward(x)

        # for softmax output activation -> dC/da_l+1 * d_al+1/dz_l+1 == a_l+1 - y
        layer_grad = y_output - y

        # for softmax output activation -> dC_out/dW_l+1 = (a_l+1 - y) * (a_l).T
        self.layers[self.n_layers - 1].dW = norm_factor * layer_grad @ self.layers[self.n_layers-2].a.T
        self.layers[self.n_layers - 1].db = norm_factor * np.sum(layer_grad, axis =1, keepdims=True, dtype=np.float32)

        for n in range(self.n_layers-2, -1, -1):
            layer_grad = norm_factor * (self.layers[n+1].W.T @ layer_grad) * self.__relu_grad(self.layers[n].z)

            previous_activation = x if n == 0 else self.layers[n-1].a

            self.layers[n].dW = norm_factor * layer_grad @ previous_activation.T
            self.layers[n].db = norm_factor * np.sum(layer_grad, axis=1, keepdims=True, dtype=np.float32)

    def classify(self, x):

        a_layer = x
        z = []

        for i in range(0,self.n_layers):
            z = self.layers[i].W @ a_layer + self.layers[i].b

            # Output layer
            if i == self.n_layers - 1:
                z = self.__softmax(z)
            # Hidden layer
            else:
                z = self.__relu(z)

            a_layer = z


        return np.argmax(z)

    def train(self, x, y,learning_rate, epochs, method = None):
        # Setup
        batch_size = self.batch_size
        data_size = x.shape[1]
        iterations = data_size // batch_size

        if method == "SGD":
            optimiser = SGDOptimiser(learning_rate)
        else:
            optimiser = AdamOptimiser(learning_rate=learning_rate, beta1=0.9, beta2=0.999, eps=1e-8)

        for epoch in range(epochs):
            input_data, y_data = shuffle_data(x, y)

            for step in range(iterations):
                input_batch = input_data[:, step * batch_size:(step + 1) * batch_size]
                y_batch = y_data[:, step * batch_size:(step + 1) * batch_size]

                self.__backward(input_batch, y_batch)
                optimiser.step(self.layers)

            print('epoch = ', epoch + 1, '|',
                  ' Accuracy = ', self.__accuracy(self.__forward(input_data), y_data), '%',
                  '| Loss = ', self.cross_entropy_loss(y_data))


        print('Training accuracy = ', self.__accuracy(self.__forward(x), y),'%')


    def accuracy(self, x, y):
        print('Test accuracy = ', self.__accuracy(self.__forward(x), y), '%')


    def cross_entropy_loss(self, y):
        eps = 1e-8
        loss = 0.0
        for n in range(y.shape[1]):
            loss += np.sum( y[:,n]* np.log(self.layers[self.n_layers-1].a[:,n] + eps))

        return -loss/y.shape[1]
