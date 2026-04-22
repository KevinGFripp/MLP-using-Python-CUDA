# MLP-using-Python-CUDA
A multi-layer perceptron (MLP) network accelerated with CUDA, implemented in Python, for the MNIST digits classification problem.

<img width="587" height="414" alt="Schematic" src="https://github.com/user-attachments/assets/09def155-d6bd-4be7-b189-475a15e82762" />



## Network Features
- Multi-level dense layer perceptron.
- Rectified linear (ReLU) and softmax activation functions for the input/hidden layers and output layer, respectively.
- Network weights initialised with "He"-type random numbers drawn from a uniform distribution.
- Cross-entropy loss function
- Optimisation with either stochastic Adaptive moment-estimation (Adam) or stochastic gradient descent.
- Training with multiple epochs via random shuffling and mini-batches.

## Implementation
- Forward propagation for each layer performed in one CUDA kernel pass.
- Back propagation for partial derivatives, with respect to the layer and weights, each a single CUDA kernel.
- Adam gradient descent optimiser step per layer as a single CUDA kernel.

## Training
The train dataset is randomly shuffled per epoch, where the data is sampled contiguously in mini-batch strides. After the data has been traversed, the accuracy over the entire shuffled data is computed along with the cross-entropy loss. The data is then re-shuffled for the next training epoch.

## Optimisations
### Kernel fusion
Instead of launching multiple kernels, incurring kernel overhead and multiple global memory loads, 'fuse' operations into one kernel call. For example, 
the forward pass through a hidden layer is a single kernel executing Activation(Weights * Activation + Bias).

### GEMM Optimisation
The dominant component of the solve time will be from repeatedly matrix-matrix multiplications in the forward and backward passes of the network. The following optimisations were implemented:
- Double buffered block-tiled shared memory of size 128x16 to overlap loads with compute.
- Shared memory leading dimension padding to suppress bank conflicts.
- Warp-level sub-tiling of size 64x32.
- Register-level thread-tiling for matrix-multiply-accumulate of size 8x8.
- In-place loading of the transpose of matrices into shared memory by row -> column major indexing.

The optimisations implemented were not exhaustive. More performance can be extracted from kernel parameter tuning, vectorised loads/writes and/or wmma tensor core accumulation. 

The gemm kernels are in the region of 60-70% the performance of cuBLAS.

### Adaptive Moment Estimation Optimisation
Every iteration of the train loop passes through the weights and biases optimiser based upon propagated gradients in the backward pass. The following optimisations were implemented:

- Vectorised loads and writes to/from global memory with remainder tail handling.
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
    mlp = MultiLayerPerceptronGPU(layer_sizes=[784, 1024, 512, 10], batch_size=256, seed=0)

    mlp.train(train_data_gpu, y_gpu, learning_rate=1e-3, epochs=10)

    mlp.test(test_data_gpu, test_gpu)

  ### Output
  ```
  epoch =  1 |  Accuracy =  97.565 % | Loss =  0.0809
  epoch =  2 |  Accuracy =  98.90667 % | Loss =  0.03887933
  epoch =  3 |  Accuracy =  99.42667 % | Loss =  0.02105977
  epoch =  4 |  Accuracy =  99.70334 % | Loss =  0.012894733
  epoch =  5 |  Accuracy =  99.808334 % | Loss =  0.0088467095
  epoch =  6 |  Accuracy =  99.955 % | Loss =  0.003703843
  epoch =  7 |  Accuracy =  99.985 % | Loss =  0.0019400718
  epoch =  8 |  Accuracy =  99.988335 % | Loss =  0.0013510864
  epoch =  9 |  Accuracy =  99.94334 % | Loss =  0.0025417348
  epoch =  10 |  Accuracy =  100.0 % | Loss =  0.0005004144
  Training accuracy =  100.0 %
  Test accuracy =  98.24 %  | Test loss =  0.06843385
  ```

### Plot the predictions
```
mlp.plot(test_data_gpu)
```
<img width="450" height="450" alt="Result" src="https://github.com/user-attachments/assets/2dbfbad3-7fc8-470c-8f8c-879f764a112c" />


## Performance 
### Ryzen 9950x3D versus RTX 4090 : 10 epochs versus batch size


The GPU can achieve up to 80x the performance of the CPU (~0.4 seconds) at large batch sizes due to the problem-size saturating the GPU. At small batch sizes, the dominant cost becomes kernel launch overhead as the GPU becomes under-utilised.

<img width="981" height="458" alt="CPUvsGPU" src="https://github.com/user-attachments/assets/03969ae9-2963-47db-bf65-b125c584a1f1" />

### Versus PyTorch : 10 epochs versus batch size

Up to batch sizes of 512, the minimised kernel executions leads this implementation to be faster than PyTorch. When the batch size becomes larger, the GPU becomes saturated and the efficiency of the library kernels surpass this implementation.

 <img width="453" height="435" alt="Pytorch_vs_this" src="https://github.com/user-attachments/assets/fb7aee6e-6df2-4223-8d5c-bda6e1a03b02" />
