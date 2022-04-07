import numpy as np

def cross_correlate_1d(vec_1: np.ndarray, vec_2: np.ndarray, normalised=False):
    '''Constructs
    '''
    n = len(vec_1)
    if n != len(vec_2):
        raise Exception(f'Vectors {vec_1} ({n}) and'
            f' {vec_2} ({len(vec_2)}) are of unequal length.')

    mean_1 = vec_1.mean()
    st_dev_1 = vec_1.std()
    mean_2 = vec_2.mean()
    st_dev_2 = vec_2.std()

    padding = np.zeros(n - 1)
    # Vector of length 3n-2, containing n-1 zeros either side of vec_2
    vec_2_padded = np.concatenate([padding, vec_2, padding])

    correlated = np.zeros(2 * n - 1)
    for i in range(2 * n - 1):
        if i % 1000 == 0:
            print(i)
        # Take a snapshot
        vec_2_snapshot = vec_2_padded[i : n + i]
        correlated[i] = np.sum((vec_1 - mean_1) * (vec_2_snapshot - mean_2)) / n
        # Divide by standard deviations of each vector if normalising
        if normalised:
            correlated[i] /= (st_dev_1 * st_dev_2)

    return correlated