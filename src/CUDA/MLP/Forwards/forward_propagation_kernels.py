from cupy import RawKernel
from pathlib import Path
from cupy import ndarray


forward_propagate_kernel = RawKernel(
    Path('src/CUDA/MLP/Forwards/Kernels/forward_propagate_kernel.cu').read_text(),
                        'forward_propagate_kernel')
forward_propagate_kernel.compile()

forward_propagate_hidden_layer_kernel = RawKernel(
    Path('src/CUDA/MLP/Forwards/Kernels/forward_propagate_hidden_layer_kernel.cu').read_text(),
                        'forward_propagate_hidden_layer_kernel')
forward_propagate_hidden_layer_kernel.compile()

def forward_propagate(W: ndarray, a: ndarray, b: ndarray, z: ndarray):

    M, K = W.shape
    N = a.shape[1]


    forward_propagate_kernel(*kernel_config(M,N),
                             (W.data.ptr,
                                   a.data.ptr,
                                   z.data.ptr,
                                   b.data.ptr,
                                   M,N,K))

    return


def hidden_layer(W: ndarray, aprev: ndarray, b: ndarray, z: ndarray, a: ndarray):

    M, K = W.shape
    N = a.shape[1]

    forward_propagate_hidden_layer_kernel(*kernel_config(M,N),
                                     (W.data.ptr,
                                           aprev.data.ptr,
                                           z.data.ptr,
                                           b.data.ptr,
                                           a.data.ptr,
                                           M,N,K))

    return


def kernel_config(M: int,N: int):
    Blocks_N = 128
    Blocks_M = 128
    block_x = (N + Blocks_N - 1)//Blocks_N
    block_y = (M + Blocks_M - 1)//Blocks_M

    return (block_x,block_y,),(256,)