from cupy.random import permutation
from cupy import asarray,ndarray

def shuffle_data(data:ndarray,truth:ndarray):

    num_cols = data.shape[1]
    indices = permutation(num_cols)

    shuffled_data = asarray(data[:,indices],order='C')
    shuffled_truth = asarray(truth[:,indices],order='C')

    return shuffled_data,shuffled_truth
