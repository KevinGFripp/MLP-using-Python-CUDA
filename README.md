# MLP-using-Python-CUDA
A multi-layer perceptron (MLP) network accelerated with CUDA, implemented in Python, for the MNIST digits classification problem.

## Network Features
- Multi-level dense layer perceptron.
- Rectified linear (ReLU) and softmax activation functions for the input/hidden layers and output layer, respectively.
- Network weights initialised with "He"-type random numbers drawn from a uniform distribution.
- Cross-entropy loss function
- Optimisation with either stochastic Adaptive moment-estimation (Adam) or stochastic gradient descent.
- Training with multiple epochs via random shuffling and mini-batches.
- 

## Implementation
- Forward propagation for each layer performed in one CUDA kernel pass (custom CUDA gemm kernel + bias + activation).
- Back propagation for partial derivatives, with respect to the layer and weights, each a single CUDA kernel.
- Custom matrix-matrix multiplication (gemm) kernels with extensive optimisations (double buffered block-tiled shared memory, warp-tiling, register-tiling, and optionally wmma tensor core accumulation).
- Adam optimiser step per layer as a single CUDA kernel. 
 
