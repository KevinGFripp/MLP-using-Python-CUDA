from src.MLP.network_layer import Layer
from numpy import sqrt

class SGDOptimiser:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers: list[Layer]):
        for layer in layers:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db


class AdamOptimiser:

    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, layers: list[Layer]):
        self.t += 1

        for layer in layers:
            # Momentum
            layer.mW *= self.beta1
            layer.mW += (1 - self.beta1) * layer.dW

            layer.mb *= self.beta1
            layer.mb += (1 - self.beta1) * layer.db

            # Variance
            layer.vW *= self.beta2
            layer.vW += (1 - self.beta2) * (layer.dW ** 2)

            layer.vb *= self.beta2
            layer.vb += (1 - self.beta2) * (layer.db ** 2)

            # Corrections
            mW_hat = layer.mW / (1 - self.beta1 ** self.t)
            mb_hat = layer.mb / (1 - self.beta1 ** self.t)

            vW_hat = layer.vW / (1 - self.beta2 ** self.t)
            vb_hat = layer.vb / (1 - self.beta2 ** self.t)

            # Update weights and biases
            layer.W -= self.learning_rate * mW_hat / (sqrt(vW_hat) + self.eps)
            layer.b -= self.learning_rate * mb_hat / (sqrt(vb_hat) + self.eps)