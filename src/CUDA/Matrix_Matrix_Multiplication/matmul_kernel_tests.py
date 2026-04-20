from numpy import float32,zeros,asarray,sum
from numpy.random import rand
from src.CUDA.Matrix_Matrix_Multiplication.matrix_matrix_multiplication import matrix_matrix_mul_warp_buffered_reg_tiled
from src.CUDA.Matrix_Matrix_Multiplication.matrix_matrix_multiplication import matrix_matrix_mul_wmma_warp_buffered_tiled
from src.CUDA.Matrix_Matrix_Multiplication.matrix_matrix_multiplication import matrix_matrix_mul_vec_buffered_reg_tiled
from cupy import asarray,sum
from cupy.cuda.runtime import deviceSynchronize
import time as timer

def wmma_matmul_kernel_test():
    M = 1024
    N = 256
    K = 784

    A = asarray(float32(rand(M,K)),dtype=float32)
    B = asarray(float32(rand(K,N)),dtype=float32)

    C = asarray((zeros((M,N),dtype=float32)),dtype=float32)
    C_truth = asarray((zeros((M, N), dtype=float32)))

    tic = timer.time()
    for _ in range(500):
        C = matrix_matrix_mul_wmma_warp_buffered_tiled(A,B,C)

    deviceSynchronize()
    toc = timer.time()
    print('Kernel time = ',toc - tic,' s')

    tic = timer.time()

    for _ in range(500):
        C_truth = A @ B

    deviceSynchronize()
    toc = timer.time()
    print('cuBLAS time = ',toc - tic,' s')

    C_sum = sum(C).get()
    C_truth_sum = sum(C_truth).get()

    print('Kernel result sum = ',C_sum)
    print('cuBLAS result sum = ',C_truth_sum)


def warp_buffered_matmul_kernel_test():
    M = 1024
    N = 1024
    K = 1024

    A = asarray(float32(rand(M,K)),dtype=float32)
    B = asarray(float32(rand(K,N)),dtype=float32)

    C = asarray((zeros((M,N),dtype=float32)),dtype=float32)
    C_truth = asarray((zeros((M, N), dtype=float32)))

    tic = timer.time()
    for _ in range(1000):
        C = matrix_matrix_mul_warp_buffered_reg_tiled(A,B,C)

    deviceSynchronize()
    toc = timer.time()
    print('Kernel time = ',toc - tic,' s')

    tic = timer.time()

    for _ in range(1000):
        C_truth = A @ B

    deviceSynchronize()
    toc = timer.time()
    print('cuBLAS time = ',toc - tic,' s')

    C_sum = sum(C).get()
    C_truth_sum = sum(C_truth).get()

    print('Kernel result sum = ',C_sum)
    print('cuBLAS result sum = ',C_truth_sum)



def vec_buffered_matmul_kernel_test():
    M = 1024
    N = 1024
    K = 1024

    A = asarray(float32(rand(M,K)),dtype=float32)
    B = asarray(float32(rand(K,N)),dtype=float32)

    C = asarray((zeros((M,N),dtype=float32)),dtype=float32)
    C_truth = asarray((zeros((M, N), dtype=float32)))

    tic = timer.time()
    for _ in range(1000):
        C = matrix_matrix_mul_vec_buffered_reg_tiled(A,B,C)

    deviceSynchronize()
    toc = timer.time()
    print('Kernel time = ',toc - tic,' s')

    tic = timer.time()

    for _ in range(1000):
        C_truth = A @ B

    deviceSynchronize()
    toc = timer.time()
    print('cuBLAS time = ',toc - tic,' s')

    C_sum = sum(C).get()
    C_truth_sum = sum(C_truth).get()

    print('Kernel result sum = ',C_sum)
    print('cuBLAS result sum = ',C_truth_sum)
