from cupy import RawKernel
from pathlib import Path
from cupy import ndarray




matrix_matrix_mul_reg_tiled_kernel = RawKernel(
    Path('src/CUDA/Matrix_Matrix_Multiplication/Kernels/Matrix_Matrix_Mul_Register_Tiled.cu').read_text(),
                        'matrix_matrix_mul_register_tiled')


matrix_matrix_mul_buffered_reg_tiled_kernel = RawKernel(
    Path('src/CUDA/Matrix_Matrix_Multiplication/Kernels/Matrix_Matrix_Mul_Buffered_Register_Tiled.cu').read_text(),
                        'matrix_matrix_mul_buffered_register_tiled')
matrix_matrix_mul_buffered_reg_tiled_kernel.compile()

matrix_matrix_mul_vec_buffered_reg_tiled_kernel = RawKernel(
    Path('src/CUDA/Matrix_Matrix_Multiplication/Kernels/Matrix_Matrix_Mul_Vec_Buffered_Register_Tiled.cu').read_text(),
                        'matrix_matrix_mul_vec_buffered_register_tiled')
matrix_matrix_mul_vec_buffered_reg_tiled_kernel.compile()

matrix_matrix_mul_warp_buffered_reg_tiled_kernel = RawKernel(
    Path('src/CUDA/Matrix_Matrix_Multiplication/Kernels/Matrix_Matrix_Mul_Warp_Buffered_Reg_Tiled.cu').read_text(),
                        'Matrix_Matrix_Mul_Warp_Buffered_Reg_Tiled')
matrix_matrix_mul_warp_buffered_reg_tiled_kernel.compile()

matrix_matrix_mul_wmma_warp_buffered_tiled_kernel = RawKernel(
    Path('src/CUDA/Matrix_Matrix_Multiplication/Kernels/Matrix_Matrix_Mul_WMMA_Warp_Buffered_Tiled.cu').read_text(),
                        'Matrix_Matrix_Mul_WMMA_Warp_Buffered_Tiled')
matrix_matrix_mul_wmma_warp_buffered_tiled_kernel.compile()

def matrix_matrix_mul_wmma_warp_buffered_tiled(A: ndarray, B: ndarray, C: ndarray):

    M, K = A.shape
    N = B.shape[1]

    matrix_matrix_mul_wmma_warp_buffered_tiled_kernel(*matrix_matrix_mul_warp_buffered_reg_tiled_config(M,N),
                                                (A.data.ptr,B.data.ptr,C.data.ptr,M,N,K))

    return C


def matrix_matrix_mul_warp_buffered_reg_tiled(A: ndarray, B: ndarray, C: ndarray):

    M, K = A.shape
    N = B.shape[1]

    matrix_matrix_mul_warp_buffered_reg_tiled_kernel(*matrix_matrix_mul_warp_buffered_reg_tiled_config(M,N),
                                                (A.data.ptr,B.data.ptr,C.data.ptr,M,N,K))

    return C

def matrix_matrix_mul_reg_tiled(A: ndarray, B: ndarray, C: ndarray):

    M, K = A.shape
    N = B.shape[1]

    matrix_matrix_mul_reg_tiled_kernel(*matrix_matrix_kernel_reg_tiled_config(M,N),
                                   (A.data.ptr,B.data.ptr,C.data.ptr,M,N,K),
                                shared_mem=matrix_matrix_kernel_register_shared_mem())

    return C


def matrix_matrix_mul_buffered_reg_tiled(A: ndarray, B: ndarray, C: ndarray):

    M, K = A.shape
    N = B.shape[1]

    matrix_matrix_mul_buffered_reg_tiled_kernel(*matrix_matrix_kernel_buffered_reg_tiled_config(M,N),
                                   (A.data.ptr,B.data.ptr,C.data.ptr,M,N,K),
                                shared_mem=matrix_matrix_kernel_buffered_shared_mem())

    return C

def matrix_matrix_mul_vec_buffered_reg_tiled(A: ndarray, B: ndarray, C: ndarray):

    M, K = A.shape
    N = B.shape[1]

    matrix_matrix_mul_vec_buffered_reg_tiled_kernel(*matrix_matrix_kernel_buffered_reg_tiled_config(M,N),
                                   (A.data.ptr,B.data.ptr,C.data.ptr,M,N,K),
                                shared_mem=matrix_matrix_kernel_vec_buffered_shared_mem())

    return C



def matrix_matrix_mul_warp_buffered_reg_tiled_config(M: int,N: int):
    Blocks_N = 128
    Blocks_M = 128
    block_x = (N + Blocks_N - 1)//Blocks_N
    block_y = (M + Blocks_M - 1)//Blocks_M

    return (block_x,block_y,),(256,)


def matrix_matrix_kernel_buffered_reg_tiled_config(M: int,N: int):
    # M rows by N columns

    tile = 32
    thread_tile = 4
    threads = tile // thread_tile

    blocks_x = (N + tile - 1)//tile
    blocks_y = (M + tile - 1)//tile

    return (blocks_x, blocks_y,), (threads, threads,)


def matrix_matrix_kernel_reg_tiled_config(M: int,N: int):
    # M rows by N columns
    tile = 64
    thread_tile = 8
    threads = tile//thread_tile

    blocks_x = (N + tile - 1)//tile
    blocks_y = (M + tile - 1)//tile

    return (blocks_x, blocks_y,), (threads, threads,)

# def matrix_matrix_kernel_shared_mem():
#     # two 2d arrays for both matrices
#     tile = 64
#     return 2 * tile * tile * 4

def matrix_matrix_kernel_register_shared_mem():
    # two buffered 2d arrays per matrix
    # outermost tile dimension padded by 1 to remove shared memory bank conflicts
    tile = 64
    return 2 * tile * (tile+1) * 4

def matrix_matrix_kernel_buffered_shared_mem():
    # two buffered 2d arrays per matrix
    # outermost tile dimension padded by 1 to remove shared memory bank conflicts
    tile = 32
    return 4 * tile * (tile+1) * 4

def matrix_matrix_kernel_vec_buffered_shared_mem():
    # two buffered 2d arrays per matrix
    # outermost tile dimension padded by 1 to remove shared memory bank conflicts
    tile = 32
    return 4 * tile * (tile + 1) * 4
