import numpy as np
from numba import njit

@njit()
def cross_correlate_1d_raw_numba(vec_1, vec_2, normalised=False):
    n = len(vec_1)
    if n != len(vec_2):
        print('Error: Vectors are of unequal length.')
        return None

    def vec_mean(vec):
        return sum(vec) / len(vec)
    def vec_std(vec):
        vec_sq = [x ** 2 for x in vec]
        var = vec_mean(vec_sq) - vec_mean(vec) ** 2
        return var ** 0.5
    mean_1 = vec_mean(vec_1)
    mean_2 = vec_mean(vec_2)
    st_dev_1 = vec_std(vec_1)
    st_dev_2 = vec_std(vec_2)

    padding = [0] * (n - 1)
    # Vector of length 3n-2, containing n-1 zeros either side of vec_2
    vec_2_padded = [*padding, *vec_2, *padding]

    correlated = [0] * (2 * n - 1)
    for i in range(2 * n - 1):
        # Log every 1000 to track progress
        if i % 1000 == 0:
            print(f'{i} / {2 * n - 1}')
        # Take a snapshot
        vec_2_snapshot = vec_2_padded[i : n + i]
        for j, x in enumerate(vec_1):
            correlated[i] += (x - mean_1) * (vec_2_snapshot[j] - mean_2)
        correlated[i] /= n
        # Divide by standard deviations of each vector if normalising
        if normalised:
            correlated[i] /= (st_dev_1 * st_dev_2)

    return correlated

@njit()
def cross_correlate_1d_numba(vec_1, vec_2, normalised=False):
    n = len(vec_1)
    if n != len(vec_2):
        print('Error: Vectors are of unequal length.')
        return None

    mean_1 = vec_1.mean()
    st_dev_1 = vec_1.std()
    mean_2 = vec_2.mean()
    st_dev_2 = vec_2.std()

    padding = np.zeros(n - 1)
    # Vector of length 3n-2, containing n-1 zeros either side of vec_2
    vec_2_padded = np.array([*padding, *vec_2, *padding])

    correlated = np.zeros(2 * n - 1)
    for i in range(2 * n - 1):
        # Log every 1000 to track progress
        if i % 1000 == 0:
            print(i)
        # Take a snapshot
        vec_2_snapshot = vec_2_padded[i : n + i]
        correlated[i] = np.sum((vec_1 - mean_1) * (vec_2_snapshot - mean_2)) / n
        # Divide by standard deviations of each vector if normalising
        if normalised:
            correlated[i] /= (st_dev_1 * st_dev_2)

    return correlated