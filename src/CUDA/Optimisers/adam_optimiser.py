from cupy import RawKernel
from pathlib import Path
from cupy import ndarray,float32


adam_optimiser_kernel = RawKernel(
    Path('src/CUDA/Optimisers/Kernels/Adam_Optimiser_Kernel.cu').read_text(),'adam_optimiser_kernel')

adam_optimiser_vectorised_kernel = RawKernel(
    Path('src/CUDA/Optimisers/Kernels/Adam_Optimiser_Vectorised_Kernel.cu').read_text(),
    'adam_optimiser_vectorised_kernel')
adam_optimiser_vectorised_kernel.compile()


def adam_optimiser_step(weights: ndarray, biases: ndarray, weight_gradients: ndarray,
                        bias_gradients: ndarray, mW: ndarray, mb: ndarray, vW: ndarray, vb: ndarray,
                        learning_rate, beta1, beta2, t: int):

    M = weights.shape[0]
    N = weights.shape[1]

    beta1_correction = float32(1. /  (1. - beta1**t))
    beta2_correction = float32(1. / (1. - beta2**t))

    adam_optimiser_kernel(*kernel_config(M,N),
                          (weights.data.ptr,
                                biases.data.ptr,
                                weight_gradients.data.ptr,
                                bias_gradients.data.ptr,
                                mW.data.ptr,
                                mb.data.ptr,
                                vW.data.ptr,
                                vb.data.ptr,
                                float32(learning_rate),
                                float32(beta1), float32(beta2),
                                beta1_correction,beta2_correction,
                                int(t), int(M), int(N)))

    return

def adam_optimiser_vec_step(weights: ndarray, biases: ndarray, weight_gradients: ndarray,
                            bias_gradients: ndarray, mW: ndarray, mb: ndarray, vW: ndarray, vb: ndarray,
                            learning_rate, beta1, beta2, t: int):

    M = weights.shape[0]
    N = weights.shape[1]

    beta1_correction = float32(1. /  (1. - beta1**t))
    beta2_correction = float32(1. / (1. - beta2**t))

    adam_optimiser_vectorised_kernel(*kernel_vec_config(M,N),
                          (weights.data.ptr,
                                biases.data.ptr,
                                weight_gradients.data.ptr,
                                bias_gradients.data.ptr,
                                mW.data.ptr,
                                mb.data.ptr,
                                vW.data.ptr,
                                vb.data.ptr,
                                float32(learning_rate),
                                float32(beta1), float32(beta2),
                                beta1_correction,beta2_correction,
                                int(t), int(M), int(N)))

    return


def kernel_config(M: int,N: int):
    thread_x = 16
    thread_y = 16

    block_x = (N + thread_x -1)//thread_x
    block_y = (M + thread_y -1)//thread_y

    return (block_x,block_y,),(thread_x,thread_y)

def kernel_vec_config(M: int,N: int):
    thread_x = 256

    block_x = (N*M + thread_x -1)//thread_x

    return (block_x,),(thread_x,)