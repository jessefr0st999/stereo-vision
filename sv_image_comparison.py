'''Script for stereo vision image comparison...
'''
from correlation_spectral import cross_correlate_2d_spectral
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from utils import max_pos
import numpy as np

# TODO: handle zero standard deviation (constant template or region) in correlation_spectral
# TODO: check the arrows are plotted the right way
# TODO: truncate window on boundaries
# TODO: automatic scheme/window size for later stages
# TODO: split up template image for later stages

PLOT_GRIDS = 0
# LOG_WINDOW_SUMMARY = 0
LOG_WINDOW_SUMMARY = 'left'
# LOG_WINDOW_SUMMARY = 1

rect_kwargs = [
    {
        'linewidth': 1,
        'edgecolor': 'blue',
        'facecolor': 'none',
    },
    {
        'linewidth': 1,
        'edgecolor': 'green',
        'facecolor': 'none',
    },
    {
        'linewidth': 1,
        'edgecolor': 'cyan',
        'facecolor': 'none',
    },
    {
        'linewidth': 1,
        'edgecolor': 'orange',
        'facecolor': 'none',
    },
]
arrow_kwargs = {
    'head_width': 3,
    'head_length': 5,
    'edgecolor': 'red',
    'facecolor': 'red',
}

def plot_multi_pass_output(left_image, right_image, stage_results):
    '''Plots the output...
    '''
    # TODO: handle overlap
    figure = plt.figure(figsize=(1, 2))
    left_plot = figure.add_subplot(1, 2, 1)
    left_plot.imshow(left_image)
    right_plot = figure.add_subplot(1, 2, 2)
    right_plot.imshow(right_image)
    # Plot the grid of windows on each image
    for stage, windows in enumerate(stage_results):
        for window_info in windows:
            x, y = window_info['window_centre']
            if stage == 0:
                top_left = (x - int(window_info['size'][0] / 2),
                    y - int(window_info['size'][1] / 2))
                window_rect = lambda: Rectangle(top_left, *window_info['size'],
                    **rect_kwargs[stage])
                left_plot.add_patch(window_rect())
                right_plot.add_patch(window_rect())
            else:
                for region_info in window_info['target_regions']:
                    r_x, r_y = region_info['centre']
                    r_top_left = (r_x - int(region_info['size'][0] / 2),
                        r_y - int(region_info['size'][1] / 2))
                    region_rect = lambda: Rectangle(r_top_left, *region_info['size'],
                        **rect_kwargs[stage])
                    left_plot.add_patch(region_rect())
                    right_plot.add_patch(region_rect())
            # Also plot arrows showing non-zero pixel displacement on the left image
            if stage == len(stage_results) - 1:
                if window_info['dp_x'] or window_info['dp_y']:
                    left_plot.arrow(x, y, window_info['dp_x'], window_info['dp_y'], **arrow_kwargs)
    plt.show()

def plot_single_pass_output(left_image, right_image, windows):
    '''Plots the output...
    '''
    figure = plt.figure(figsize=(1, 2))
    left_plot = figure.add_subplot(1, 2, 1)
    left_plot.imshow(left_image)
    right_plot = figure.add_subplot(1, 2, 2)
    right_plot.imshow(right_image)
    # Plot the grid of windows on each image
    for window_info in windows:
        x, y = window_info['window_centre']
        top_left = (x - int(window_info['size'][0] / 2),
            y - int(window_info['size'][1] / 2))
        window_rect = lambda: Rectangle(top_left, *window_info['size'],
            **rect_kwargs[0])
        left_plot.add_patch(window_rect())
        right_plot.add_patch(window_rect())
        # Also plot arrows showing non-zero pixel displacement on the left image
        if window_info['dp_x'] or window_info['dp_y']:
            left_plot.arrow(x, y, window_info['dp_x'], window_info['dp_y'], **arrow_kwargs)
    plt.show()

def whole_image_search_regions(template_image, region_image, x_window, y_window,
        scheme='3x3', overlap=0):
    '''For a given window size, breaks up each image into windows of that size,
    defined by their top left co-ordinate.
    Then, for each window, determines the search target_regions based on the specified
    scheme, in a dict containing the origins and image slices for each region.
    '''
    image_height, image_width = region_image.shape
    if template_image.shape[0] != image_height or template_image.shape[1] != image_width:
        raise Exception('Dimensions of template and region images must match')
    windows = []
    # x, y are left, top pixels respectively
    y = 0
    window_id = 0
    while y < image_height:
        x = 0
        while x < image_width:
            x_centre = x + x_window / 2
            y_centre = y + y_window / 2
            window_centre = (x_centre, y_centre)
            window_info = {
                'window_id': window_id,
                'window_centre': window_centre,
                'stage_centres': [window_centre],
                'size': (x_window, y_window),
                'image_slice': template_image[y : y + y_window,  x : x + x_window],
                'target_regions': [],
            }
            for _x, _y in region_pairs(x_centre, x_window, y_centre, y_window, scheme):
                # If fully outside image, skip
                if (_x + x_window / 2 <= 0 or \
                    _y + y_window / 2 <= 0 or \
                    _x - x_window / 2 > image_width or \
                    _y - y_window / 2 > image_height):
                    continue
                # If partially outside image, truncate
                x_start = int(max(_x - x_window / 2, 0))
                y_start = int(max(_y - y_window / 2, 0))
                x_end = int(min(_x + x_window / 2, image_width))
                y_end = int(min(_y + y_window / 2, image_height))
                window_info['target_regions'].append({
                    'centre': (_x, _y),
                    'size': (x_end - x_start, y_end - y_start),
                    'image_slice': region_image[y_start : y_end, x_start : x_end],
                })
            windows.append(window_info)
            window_id += 1
            x += int(x_window * (1 - overlap))
        y += int(y_window * (1 - overlap))

    return windows

def centred_search_region(template_window_info, stage, region_image, x_window, y_window,
        scheme='3x3'):
    '''
    '''
    image_height, image_width = region_image.shape
    window_info = {**template_window_info, 'target_regions': []}
    x_centre, y_centre = template_window_info['stage_centres'][stage]
    for _x, _y in region_pairs(x_centre, x_window, y_centre, y_window, scheme):
        # If fully outside image, skip
        if (_x + x_window / 2 <= 0 or \
            _y + y_window / 2 <= 0 or \
            _x - x_window / 2 > image_width or \
            _y - y_window / 2 > image_height):
            continue
        # If partially outside image, truncate
        x_start = int(max(_x - x_window / 2, 0))
        y_start = int(max(_y - y_window / 2, 0))
        x_end = int(min(_x + x_window / 2, image_width))
        y_end = int(min(_y + y_window / 2, image_height))
        # print(x_start, x_end, y_start, y_end)
        window_info['target_regions'].append({
            'centre': (_x, _y),
            'size': (x_end - x_start, y_end - y_start),
            'image_slice': region_image[y_start : y_end, x_start : x_end],
        })
    return window_info

def image_scan(window_info, corr_threshold=0, _print=False):
    '''Scans
    '''
    template = window_info['image_slice']
    window_centre = window_info['window_centre']
    window_size = window_info['size']
    window_corr_max = 0
    window_corr_max_pos = window_centre
    for region_info in window_info['target_regions']:
        target_centre = region_info['centre']
        # Skip if the region falls exactly outside
        if region_info['image_slice'].size == 0:
            continue
        x_corr = cross_correlate_2d_spectral(template, region_info['image_slice'])
        corr_max = np.max(x_corr)

        # TODO
        # print(template.shape, region_info['image_slice'].shape, x_corr.shape)

        # Update the max correlation for the window to the centre of the region if the
        # value found in this region is the highest and exceeds the specified threshold
        if corr_max > window_corr_max and corr_max > corr_threshold:
            window_corr_max = corr_max
            window_corr_max_pos = target_centre
        if PLOT_GRIDS:
            local_corr_max_pos = max_pos(x_corr) # (y, x)
            figure = plt.figure()
            ax = figure.add_subplot(1, 2, 1)
            ax.set_title((f'Centre at {target_centre}, max XC of {round(corr_max, 4)} '
                f'at local pos {reversed(local_corr_max_pos)}'))
            ax.add_patch(Circle(local_corr_max_pos, radius=10, color='red'))
            plt.imshow(region_info['image_slice'])
            ax = figure.add_subplot(1, 2, 2)
            plt.imshow(x_corr)
            plt.show()
    dp_x = window_corr_max_pos[0] - window_centre[0]
    dp_y = window_corr_max_pos[1] - window_centre[1]
    if LOG_WINDOW_SUMMARY:
        # Only log the windows on the left edge
        if LOG_WINDOW_SUMMARY != 'left' or window_centre[0] - window_size[0] / 2 == 0:
            window_summary = ('Window (left, right, top, bottom): '
                f'({window_centre[0] - window_size[0] / 2}, {window_centre[0] + window_size[0] / 2}, '
                f'{window_centre[1] - window_size[1] / 2}, {window_centre[1] + window_size[1] / 2}), '
                f'max correlation of {round(window_corr_max, 5)} at {window_corr_max_pos}, '
                f'shift of ({dp_x}, {dp_y})')
            print(window_summary)
    return dp_x, dp_y

def depth_map(results_grid):
    '''General depth map

    minval = np.percentile(depth, 3)
    maxval = np.percentile(depth, 97)
    depth = np.clip(depth, minval, maxval)
    depth = ((depth - minval) / (maxval - minval)) * 255
    '''
    # TODO: attempt to assign to pixels beneath
    # Reshape the z-values into a grid
    z_grid = np.zeros(results_grid.shape[0 : 2])
    for i, row in enumerate(results_grid):
        for j, values in enumerate(row):
            x_l, y_l, x_r, y_r = values
            if x_l == x_r:
                z_grid[i][j] = 0
            else:
                z_grid[i][j] = 1 / np.abs(x_l - x_r)
    max_depth = np.max(np.abs(z_grid))
    z_grid[z_grid == 0] = max_depth
    return z_grid

def region_pairs(x, x_window, y, y_window, scheme):
    match scheme:
        case '3x3':
            return [
                (x - x_window, y - y_window),
                (x, y - y_window),
                (x + x_window, y - y_window),

                (x - x_window, y),
                (x, y),
                (x + x_window, y),

                (x - x_window, y + y_window),
                (x, y + y_window),
                (x + x_window, y + y_window),
            ]
        case '1x3':
            return [
                (x - x_window, y),
                (x, y),
                (x + x_window, y),
            ]
        case '3x1':
            return [
                (x, y - y_window),
                (x, y),
                (x, y + y_window),
            ]
        case '5x5':
            return [
                (x - 2 * x_window, y - 2 * y_window),
                (x - x_window, y - 2 * y_window),
                (x, y - 2 * y_window),
                (x + x_window, y - 2 * y_window),
                (x + 2 * x_window, y - 2 * y_window),

                (x - 2 * x_window, y - y_window),
                (x - x_window, y - y_window),
                (x, y - y_window),
                (x + x_window, y - y_window),
                (x + 2 * x_window, y - y_window),

                (x - 2 * x_window, y),
                (x - x_window, y),
                (x, y),
                (x + x_window, y),
                (x + 2 * x_window, y),

                (x - 2 * x_window, y + y_window),
                (x - x_window, y + y_window),
                (x, y + y_window),
                (x + x_window, y + y_window),
                (x + 2 * x_window, y + y_window),

                (x - 2 * x_window, y + 2 * y_window),
                (x - x_window, y + 2 * y_window),
                (x, y - 2 * y_window),
                (x + x_window, y + 2 * y_window),
                (x + 2 * x_window, y + 2 * y_window),
            ]
        case _:
            raise Exception(f'Unknown search region scheme: {scheme}')