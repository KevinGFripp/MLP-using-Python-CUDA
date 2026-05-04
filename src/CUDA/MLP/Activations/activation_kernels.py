from cupy import RawKernel
from pathlib import Path
from cupy import ndarray


relu_kernel = RawKernel(
    Path('src/CUDA/MLP/Activations/Kernels/ReLU_kernel.cu').read_text(),
                        'ReLU_kernel')
relu_kernel.compile()

softmax_10_kernel = RawKernel(
    Path('src/CUDA/MLP/Activations/Kernels/softmax_10_kernel.cu').read_text(),
                        'softmax_10_kernel')
softmax_10_kernel.compile()

def relu(z:ndarray, a:ndarray):
    M,N = z.shape

    relu_kernel(*kernel_config(M*N),
                (z.data.ptr,
                    a.data.ptr,
                    M*N))
    return

def softmax_10(z:ndarray, a:ndarray):
    M,N = z.shape

    softmax_10_kernel(*softmax_10_kernel_config(N),
                (z.data.ptr,
                      a.data.ptr,
                      N))
    return


def kernel_config(size):

    threads = 256
    blocks = (size + threads -1)//threads

    return (blocks,),(threads,)

def softmax_10_kernel_config(batch_size):

    threads = 256
    blocks = (batch_size + 16 -1)//16

    return (blocks,),(threads,)