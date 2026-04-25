# MLP-using-Python-CUDA
A multi-layer perceptron (MLP) network accelerated with CUDA, implemented in Python, for the MNIST digits classification problem.

<img width="587" height="414" alt="Schematic" src="https://github.com/user-attachments/assets/09def155-d6bd-4be7-b189-475a15e82762" />

## Network Features
- Multi-level dense layer perceptron.
- Rectified linear (ReLU) and softmax activation functions for the input/hidden layers and output layer, respectively.
- Network weights initialised with "He"-type random numbers drawn from a uniform distribution.
- Cross-entropy loss function.
- Optimisation with either stochastic Adaptive moment-estimation (Adam) or stochastic gradient descent.
- Training with multiple epochs via random shuffling and mini-batches.

## Performance 
### Ryzen 9950x3D versus RTX 4090 : 10 epochs versus batch size

The GPU (0.27 seconds) can achieve up to ~35x the performance of the CPU NumPy implementation at larger batch sizes due to the problem-size better saturating the GPU. At smaller batch sizes, the dominant cost becomes kernel launch overhead as the GPU becomes under-utilised.

<img width="1036" height="488" alt="CPUvsGPUvsGPUOptimised" src="https://github.com/user-attachments/assets/60478b88-5a7c-4bee-98b2-afde80406508" />


### Versus PyTorch : 10 epochs versus batch size

Up to a batch size of 2048, the minimised kernel executions leads this implementation to be faster than PyTorch. At 2048 batch size, this implementation and PyTorch have converged.

<img width="898" height="426" alt="PyTorchvsGPUvsGPUOptimised" src="https://github.com/user-attachments/assets/65bc12a9-a279-4dd4-98cd-401e76ef1898" />

## Implementation
- Forward propagation for each layer performed in one CUDA kernel pass.
- Back propagation for partial derivatives, with respect to the layer and weights, each a single CUDA kernel.
- Adam gradient descent optimiser step per layer as a single CUDA kernel.
- Tensor-core accelerated or native fp32 GEMM kernels.

## Training
The train dataset is randomly shuffled per epoch, where the data is sampled contiguously in mini-batch strides. After the data has been traversed, the accuracy over the entire shuffled data is computed along with the cross-entropy loss. The data is then re-shuffled for the next training epoch.

## Optimisations
### Kernel fusion
Instead of launching multiple kernels, incurring kernel overhead and multiple global memory loads, multiple operations are fused into one kernel call. For example, the forward pass through a hidden layer is a single kernel executing Activation(Weights * Activation + Bias).

### GEMM Optimisation
A large contribution to the overall solve time will be from repeated matrix-matrix multiplications in the forward and backward passes of the network.

#### GEMM tiling structure
--- 128 x 16 block tiling (double-buffered shared memory)

----- 64 x 32 warp tiling

-------- 8x8 register accumulation (fp32) or 16x16 (half) wmma fragments
   
         
The following optimisations were implemented:
- Double buffered block-tiled shared memory of size 128x16 to overlap loads with compute.
- Vectorised float4 loads from global memory.
- Shared memory leading dimension padding to suppress bank conflicts.
- Per-warp sub-tiling of shared memory tile blocks.
- Register-level thread-tiling for matrix-multiply-accumulate in fp32.
- Tensor core fp32 wmma fragment accumulation for fp16 cast inputs.
- In-place loading of the transpose of matrices into shared memory by row -> column major indexing for the back-propagation steps.
- Vectorised float4 write-backs to global memory.

The optimisations implemented have not been exhaustive. More performance can be extracted from tuning of kernel parameters, for example.
The GEMM kernels implemented here will not be faster than cuBLAS or PyTorch kernels.

### Adaptive Moment Estimation Optimisation
Every iteration of the train loop passes through the weights and biases optimiser based upon propagated gradients in the backward pass. The following optimisations were implemented:

- Vectorised loads and writes to/from global memory.
- Operation fusion into a single kernel to minimise overhead and visits to global memory.


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

