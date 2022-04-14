import numpy as np
import math

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

def cross_correlate_2d(template: np.ndarray, region: np.ndarray, step_x=1, step_y=1):
    '''Constructs
    '''
    if template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
        raise Exception('Dimensions of template must not exceed those of region.')

    output_rows = math.ceil(region.shape[0] / step_y)
    output_columns = math.ceil(region.shape[1] / step_x)

    correlated = np.zeros([output_rows, output_columns])

    for i in range(output_rows):
        for j in range(output_columns):
            # Take a snapshot of the based on the overlap between it and the template
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
                    * (region_snapshot - region_snapshot_mean))

            # Normalise by dividing by standard deviations of each vector
            template_snapshot_st_dev = np.nanstd(template)
            region_snapshot_st_dev = region_snapshot.std()
            if template_snapshot_st_dev == 0 or region_snapshot_st_dev == 0:
                correlated[i][j] = np.nan
            else:
                correlated[i][j] /= (template_snapshot_st_dev * region_snapshot_st_dev)

    return correlated