# MLP-using-Python-CUDA
A multi-layer perceptron (MLP) network accelerated with CUDA, implemented in Python, for the MNIST digits classification problem.
This repository benchmarks varied CUDA optimisations in the forward and backward passes of the network relative to a PyTorch analog.

<img width="587" height="414" alt="Schematic" src="https://github.com/user-attachments/assets/09def155-d6bd-4be7-b189-475a15e82762" />

## Network Features
- Multi-level dense layer perceptron.
- Rectified linear (ReLU) and softmax activation functions for the input/hidden layers and output layer, respectively.
- Network weights initialised with "He"-type random numbers drawn from a uniform distribution.
- Cross-entropy loss function.
- Optimisation with either stochastic Adaptive moment-estimation (Adam) or stochastic gradient descent.
- Training with multiple epochs via random shuffling and mini-batches.

## Performance : MLP-CPU versus PyTorch-GPU and MLP-CUDA
-Hardware: Ryzen 9 9950x3d and an RTX 4090

<img width="1396" height="693" alt="PyTorchVersusCPUVersusCUDA_revised" src="https://github.com/user-attachments/assets/7803b280-1145-43a9-9ded-fe24208edb9a" />


The GPU (~0.16 seconds) can achieve 40x+ the performance of the CPU NumPy implementation at a batch-size of 4096 due to the problem-size better saturating the GPU.

Versus Pytorch (~0.256 seconds), this implementation achieves a +60% performance increase from aggressive kernel fusion for the forwards and backwards passes, where the minimisation of visits to global memory alongside extensive vectorisation better feed the tensor cores in this GEMM-dominant workload.



## Implementation
- Forward propagation for each layer performed in one CUDA kernel pass.
- Back propagation for partial derivatives, with respect to the layer, weights, and biases, each a single CUDA kernel.
- Adam gradient descent optimiser step per layer as a single CUDA kernel.
- Tensor-core fp16 accelerated fp32 loaded GEMM kernels.

## Training
The train dataset is randomly shuffled per epoch, where the data is sampled contiguously in mini-batch strides. After the data has been traversed, the accuracy over the entire shuffled data is computed along with the cross-entropy loss. The data is then re-shuffled for the next training epoch.

## Optimisations
### Kernel fusion
Instead of launching multiple kernels, incurring kernel overhead and multiple global memory loads, multiple operations are fused into one kernel call. For example, the forward pass through a hidden layer is a single kernel executing ReLU(Weights * Activation + Bias).

### GEMM Optimisation
A large contribution to the overall solve time will be from repeated matrix-matrix multiplications in the forward and backward passes of the network.

#### Tiling layout
--- 128 x 16 block tiling 

----- 64 x 32 warp tiling

-------- 16x16 accumulator tiling

#### Loading optimisations
- Double buffered block-tiled shared memory of size 128x16 to overlap loads with compute.
- Vectorised float4 loads from global memory where possible.
- Costly matrix transposes avoided by vectorised row-major global access -> shared memory indexing transpose
- Shared memory leading dimension padding to suppress bank conflicts.

#### Writing optimisations
- Shared memory staging for the wmma output tile to perform additional operations, e.g. activation or normalisation.
- Conditional branching based on whether the full tile resides within the output matrix -> Fully vectorised writes.
- Each warp handles it's respective output tile and writes back data contiguously based upon the warp lane.

The optimisations implemented here for the GEMM kernels are not exhaustive. More performance could be extracted from tuning of kernel parameters, for example, or using true asynchronous buffering of the data.


### Adaptive Moment Estimation Optimisation
Every iteration of the train loop passes through the weights and biases optimiser based upon propagated gradients in the backward pass. The following optimisations were implemented:

- Vectorised loads and writes to/from global memory.
- Operation fusion into a single kernel.


## Example:
### Download the dataset
```
import kagglehub
# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
```

#### Move data sub-folders to MNIST_dataset folder

### Train and test
    //Load the MNIST dataset
    (train_data,y),(test_data,test) = load_mnist_data()

    //Move data to gpu
    train_data_gpu = asarray(train_data,dtype='float32',order='C')
    y_gpu = asarray(y,dtype='float32',order='C')
    test_data_gpu = asarray(test_data,dtype='float32',order='C')
    test_gpu = asarray(test,dtype='float32',order='C')

    //Setup
    mlp = MultiLayerPerceptronGPU(layer_sizes=[784, 1024, 512, 10], batch_size=512, seed=0)

    mlp.train(train_data_gpu, y_gpu, learning_rate=1e-3, epochs=10)

    mlp.test(test_data_gpu, test_gpu)

  ### Output
  ```
epoch =  1 |  Accuracy =  97.22833 % | Loss =  0.100154914
epoch =  2 |  Accuracy =  98.64333 % | Loss =  0.05184615
epoch =  3 |  Accuracy =  99.48167 % | Loss =  0.023750618
epoch =  4 |  Accuracy =  99.70167 % | Loss =  0.0141675165
epoch =  5 |  Accuracy =  99.86167 % | Loss =  0.008657567
epoch =  6 |  Accuracy =  99.956665 % | Loss =  0.004854481
epoch =  7 |  Accuracy =  99.986664 % | Loss =  0.00238529
epoch =  8 |  Accuracy =  99.97666 % | Loss =  0.0027756281
epoch =  9 |  Accuracy =  100.0 % | Loss =  0.0012471621
epoch =  10 |  Accuracy =  100.0 % | Loss =  0.0007783677
Training accuracy =  100.0 %
Test accuracy =  98.14 %  | Test loss =  0.07018457
  ```

### Plot the predictions
```
mlp.plot(test_data_gpu)
```
<img width="450" height="450" alt="Result" src="https://github.com/user-attachments/assets/2dbfbad3-7fc8-470c-8f8c-879f764a112c" />

