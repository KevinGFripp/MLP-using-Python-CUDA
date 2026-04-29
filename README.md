# MLP-using-Python-CUDA
A high-performance implementation of a multi-layer perceptron (MLP) for MNIST digit classification, using custom CUDA kernels executed through Python.

This project explores low-level GPU optimisation strategies and benchmarks them against a PyTorch equivalent, with a focus on kernel fusion, memory efficiency, and GEMM performance.

<img width="587" height="414" alt="Schematic" src="https://github.com/user-attachments/assets/09def155-d6bd-4be7-b189-475a15e82762" />

## Summary

| Implementation    | Train Time (s)     | Throughput          |
| ----------------- | ------------------ | ------------------- |
| NumPy (CPU)       | ~11 s              | ~10k samples/s      |
| PyTorch (GPU)     | ~0.256 s           | ~224k samples/s     |
| **This CUDA MLP** | **~0.16 s**        | **~360k samples/s** |

(Batch size = 4096, 10 Epochs, 60000 samples)

* ~ **60**x **speedup vs CPU** due to improved saturation of the GPU at larger batch sizes.
* ~**60**% **faster than PyTorch** at this batch size.

### Why faster?
* Aggressive kernel fusion (minimised trips to global memory)
* Strong compute from leveraging **tensor-core GEMM kernels**
* Optimised memory access from careful **indexing** and **vectorisation**

## Features
* Fully-connected feedforward neural network
* ReLU activations (hidden layers) and Softmax (output layer)
* He initialisation (uniform variant)
* Cross-entropy loss
* Optimisers: 
    * Adam
    * Stochastic Gradient Descent (SGD)
* Mini-batch training with per-epoch shuffling
* End-to-end GPU execution via custom CUDA kernel
* Mini-batch training with per-epoch shuffling
* End-to-end GPU execution via custom CUDA kernels

## Performance with batch size
Hardware: Ryzen 9 9950X3D + RTX 4090

<img width="1396" height="693" alt="PyTorchVersusCPUVersusCUDA_revised" src="https://github.com/user-attachments/assets/7803b280-1145-43a9-9ded-fe24208edb9a" />


## Implementation Overview
Each step of the training iteration is implemented with custom CUDA kernels:
* **Forward pass**: One kernel per layer
* **Backward pass**:
    * Weight gradients
    * Bias gradients
    * Layer gradients
* **Optimiser(Adam)**: One kernel per layer

### Mixed Precision
* Leverage tensor cores to accelerate GEMM
* Accumulate and store in fp32

## Optimisations
### 1. Kernel Fusion
Minimise kernel launch overhead (important at small batch sizes) and global memory access (important at large batch sizes).

Example:
```
x1 = ReLU(W @ x0 + b)
```

would incur multiple kernel launches for each operation: GEMM, bias and assignment.

### 2. GEMM Optimisation

Matrix multiplication is a dominant contribution to the runtime, so effort should be concentrated there.

#### Tiling Strategy
```
Block tile:         128 x 16
Warp tile:           64 x 32
Accumulator tile:    16 x 16 (WMMA)
```

#### Memory Access Optimisations

* Double buffered block-tile shared memory (overlap compute and loading)
* Vectorised ```float4``` global memory reads
* Implicit transpose via shared memory indexing (avoids costly strided access from explicit transpose)
* Shared memory padding to avoid bank conflicts and satisfy ```__half``` alignment.

#### Write Optimisations

* Shared memory staging for processing (e.g. activation)
* Vectorised WMMA or otherwise global writes when tiles lay within bounds
* Warp-level assignment of output tiles for coalesced writes


  Further optimisations are possible from parameter tuning or incorporating ```cp.async``` asynchronous pipelining.

### 3. Adaptive Moment Estimation (Adam) Optimisation
* Fused kernel per layer
* Vectorised ```float4``` memory access
* Minimised global memory access

### 4. Bias Gradient Reduction Optimisation
* Vectorised ```float4``` memory access
* Warp-level sum reduction avoiding shared memory storage
* Last warp performs final reduction to write to global memory

## Training
* Dataset shuffled at the start of each epoch
* Mini-batches processed sequentially
* After each epoch:
    * Full-dataset accuracy and cross-entropy loss computed
    * Data reshuffled for next epoch


## Example Usage:
### Download MNIST
```
import kagglehub
# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
```

#### Move data into ```MNIST_dataset/``` directory.

### Train and evaluate
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
epoch =  5 |  Accuracy =  99.86167 % | Loss =  0.008657567
epoch =  10 |  Accuracy =  100.0 % | Loss =  0.0007783677

Training accuracy =  100.0 %
Test accuracy =  98.14 %  | Test loss =  0.07018457
  ```

### Plot the predictions
```
mlp.plot(test_data_gpu)
```
<img width="450" height="450" alt="Result" src="https://github.com/user-attachments/assets/2dbfbad3-7fc8-470c-8f8c-879f764a112c" />

