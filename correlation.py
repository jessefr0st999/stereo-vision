import numpy as np
import math

def cross_correlate_1d_raw(vec_1, vec_2, normalised=False):
    '''Computes the cross-correlation (optionally normalised) of two vectors
    of equal length using raw Python.
    '''
    n = len(vec_1)
    if n != len(vec_2):
        raise Exception(f'Vectors {vec_1} ({n}) and'
            f' {vec_2} ({len(vec_2)}) are of unequal length.')

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

def cross_correlate_1d(vec_1: np.ndarray, vec_2: np.ndarray, normalised=False):
    '''Computes the cross-correlation (optionally normalised) of two vectors
    of equal length. The vectors are input as Numpy arrays, and vectorised
    Numpy operations used for calculations.
    '''
    n = len(vec_1)
    if n != len(vec_2):
        raise Exception(f'Vectors {vec_1} ({n}) and'
            f' {vec_2} ({len(vec_2)}) are of unequal length.')

    mean_1 = vec_1.mean()
    mean_2 = vec_2.mean()
    st_dev_1 = vec_1.std()
    st_dev_2 = vec_2.std()

    padding = np.zeros(n - 1)
    # Vector of length 3n-2, containing n-1 zeros either side of vec_2
    vec_2_padded = np.concatenate([padding, vec_2, padding])

    correlated = np.zeros(2 * n - 1)
    for i in range(2 * n - 1):
        # Log every 1000 to track progress
        if i % 1000 == 0:
            print(f'{i} / {2 * n - 1}')
        # Take a snapshot
        vec_2_snapshot = vec_2_padded[i : n + i]
        correlated[i] = np.sum((vec_1 - mean_1) * (vec_2_snapshot - mean_2)) / n
        # Divide by standard deviations of each vector if normalising
        if normalised:
            correlated[i] /= (st_dev_1 * st_dev_2)

    return correlated

def cross_correlate_2d(template: np.ndarray, region: np.ndarray, step_x=1, step_y=1):
    '''Computes the cross-correlation between a 2D region and template. Numpy is used
    for representation of inputs and required calculations.
    '''
    if template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
        raise Exception('Dimensions of template must not exceed those of region.')

    output_rows = math.ceil(region.shape[0] / step_y)
    output_columns = math.ceil(region.shape[1] / step_x)

    correlated = np.zeros([output_rows, output_columns])

    for i in range(output_rows):
        if i % 10 == 0:
            print(f'{i} / {output_rows}')
        for j in range(output_columns):
            # Take a snapshot of the region based on the overlap between it and the template
            region_snapshot = region[
                i * step_y : template.shape[0] + (i * step_y),
                j * step_x : template.shape[1] + (j * step_x),
            ]
            # Also truncate the template to handle occurrences at bottom and right edges
            template_snapshot = template[
                0 : region_snapshot.shape[0],
                0 : region_snapshot.shape[1],
            ]

            # Now calculate the cross-correlation
            template_snapshot_mean = np.nanmean(template)
            region_snapshot_mean = region_snapshot.mean()
            correlated[i][j] = np.nansum((template_snapshot - template_snapshot_mean) \
                * (region_snapshot - region_snapshot_mean)) / template.size

            # Normalise by dividing by standard deviations of each vector
            template_snapshot_st_dev = np.nanstd(template)
            region_snapshot_st_dev = region_snapshot.std()
            if template_snapshot_st_dev == 0 or region_snapshot_st_dev == 0:
                correlated[i][j] = np.nan
            else:
                correlated[i][j] /= (template_snapshot_st_dev * region_snapshot_st_dev)

    return correlated