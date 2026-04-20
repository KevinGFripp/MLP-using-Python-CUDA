from numpy.random import permutation

def shuffle_data(data,truth):

    num_cols = data.shape[1]
    indices = permutation(num_cols)

    shuffled_data = data[:,indices]
    shuffled_truth = truth[:,indices]

    return shuffled_data,shuffled_truth
