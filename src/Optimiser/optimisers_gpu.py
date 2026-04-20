from src.MLP.network_layer_gpu import Layer
from src.CUDA.Optimisers.adam_optimiser import adam_optimiser_vec_step

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

            adam_optimiser_vec_step(layer.W, layer.b, layer.dW, layer.db,
                                    layer.mW, layer.mb,layer.vW, layer.vb,
                                    self.learning_rate, self.beta1, self.beta2, self.t)
