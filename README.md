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
- 

## Implementation
- Forward propagation for each layer performed in one CUDA kernel pass (custom CUDA gemm kernel + bias + activation).
- Back propagation for partial derivatives, with respect to the layer and weights, each a single CUDA kernel.
- Custom matrix-matrix multiplication (gemm) kernels with extensive optimisations (double buffered block-tiled shared memory, warp-tiling, register-tiling, and optionally wmma tensor core accumulation).
- Adam optimiser step per layer as a single CUDA kernel.

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
### Performance
~1.5 seconds for training and testing on an RTX 4090.
### Plot the predictions
```
mlp.plot(test_data_gpu)
```
<img width="450" height="450" alt="Result" src="https://github.com/user-attachments/assets/2dbfbad3-7fc8-470c-8f8c-879f764a112c" />


 
