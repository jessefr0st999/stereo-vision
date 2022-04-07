import numpy as np
import math

def cross_correlate_2d(template: np.ndarray, region: np.ndarray, step_x=1, step_y=1,
        normalised=False):
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
            # Also truncate the template to handle occurrences at edge
            template_snapshot = template[
                0 : region_snapshot.shape[0],
                0 : region_snapshot.shape[1],
            ]

            # Now calculate the cross-correlation
            template_snapshot_mean = template_snapshot.mean()
            region_snapshot_mean = region_snapshot.mean()
            correlated[i][j] = np.sum((template_snapshot - template_snapshot_mean) \
                    * (region_snapshot - region_snapshot_mean)) \
                / region_snapshot.size

            # Divide by standard deviations of each vector if normalising
            # How to handle standard deviation = 0?
            if normalised:
                template_snapshot_st_dev = template_snapshot.std()
                region_snapshot_st_dev = region_snapshot.std()
                correlated[i][j] /= (template_snapshot_st_dev * region_snapshot_st_dev)

    return correlated