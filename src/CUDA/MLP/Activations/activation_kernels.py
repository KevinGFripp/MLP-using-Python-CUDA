from cupy import RawKernel
from pathlib import Path
from cupy import ndarray


relu_kernel = RawKernel(
    Path('src/CUDA/MLP/Activations/Kernels/ReLU_kernel.cu').read_text(),
                        'ReLU_kernel')
relu_kernel.compile()

def relu(z:ndarray, a:ndarray):
    M,N = z.shape

    relu_kernel(*kernel_config(M*N),
                (z.data.ptr,
                    a.data.ptr,
                    M*N))
    return


def kernel_config(size):

    threads = 256
    blocks = (size + threads -1)//threads

    return (blocks,),(threads,)