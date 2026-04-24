from cupy import RawKernel
from pathlib import Path
from cupy import ndarray, float32


hidden_layer_gradient_kernel = RawKernel(
    Path('src/CUDA/MLP/Backwards/Kernels/hidden_layer_gradient_kernel.cu').read_text(),
                        'hidden_layer_gradient_kernel')
hidden_layer_gradient_kernel.compile()

hidden_layer_gradient_wmma_kernel = RawKernel(
    Path('src/CUDA/MLP/Backwards/Kernels/hidden_layer_gradient_wmma_kernel.cu').read_text(),
                        'hidden_layer_gradient_wmma_kernel')
hidden_layer_gradient_wmma_kernel.compile()

weight_gradient_kernel = RawKernel(
    Path('src/CUDA/MLP/Backwards/Kernels/weight_gradient_kernel.cu').read_text(),
                        'weight_gradient_kernel')
weight_gradient_kernel.compile()


weight_gradient_wmma_kernel = RawKernel(
    Path('src/CUDA/MLP/Backwards/Kernels/weight_gradient_wmma_kernel.cu').read_text(),
                        'weight_gradient_wmma_kernel')
weight_gradient_wmma_kernel.compile()


def hidden_layer_gradient(W: ndarray, gradient: ndarray, this_gradient: ndarray, this_z: ndarray):

    M = W.shape[1]
    N = gradient.shape[1]
    K = W.shape[0]

    norm_factor = float32(1./N)

    hidden_layer_gradient_kernel(*kernel_config(M, N),
                             (W.data.ptr,
                                   gradient.data.ptr,
                                   this_gradient.data.ptr,
                                   this_z.data.ptr,
                                   norm_factor,
                                   M, N, K))
    return

def hidden_layer_gradient_wmma(W: ndarray, gradient: ndarray, this_gradient: ndarray, this_z: ndarray):

    M = W.shape[1]
    N = gradient.shape[1]
    K = W.shape[0]

    norm_factor = float32(1./N)

    hidden_layer_gradient_wmma_kernel(*kernel_config(M, N),
                               (W.data.ptr,
                                   gradient.data.ptr,
                                   this_gradient.data.ptr,
                                   this_z.data.ptr,
                                   norm_factor,
                                   M, N, K))
    return

def weight_gradient(this_gradient: ndarray, previous_activation: ndarray, dW: ndarray):

    M = this_gradient.shape[0]
    N = previous_activation.shape[0]
    K = this_gradient.shape[1]

    norm_factor = float32(1./N)

    weight_gradient_kernel(*kernel_config(M, N),
                           (this_gradient.data.ptr,
                                 previous_activation.data.ptr,
                                 dW.data.ptr,
                                 norm_factor,
                                 M, N, K))
    return

def weight_gradient_wmma(this_gradient: ndarray, previous_activation: ndarray, dW: ndarray):

    M = this_gradient.shape[0]
    N = previous_activation.shape[0]
    K = this_gradient.shape[1]

    norm_factor = float32(1./N)

    weight_gradient_wmma_kernel(*kernel_config(M, N),
                           (this_gradient.data.ptr,
                                 previous_activation.data.ptr,
                                 dW.data.ptr,
                                 norm_factor,
                                 M, N, K))
    return


def kernel_config(M: int,N: int):
    Blocks_N = 128
    Blocks_M = 128
    block_x = (N + Blocks_N - 1)//Blocks_N
    block_y = (M + Blocks_M - 1)//Blocks_M

    return (block_x,block_y,),(256,)