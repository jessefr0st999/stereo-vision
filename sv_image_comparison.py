'''Script for stereo vision image comparison...
'''
from correlation_spectral import cross_correlate_2d_spectral
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from utils import max_pos

# TODO: handle window sizes as exact divisors of image sizes
# TODO: handle zero standard deviation (constant template or region) in correlation_spectral
# TODO: check the arrows are plotted the right way

PLOT_GRIDS = 0
LOG_WINDOW_SUMMARY = 'left'

rect_kwargs = {
    'linewidth': 2,
    'edgecolor': 'b',
    'facecolor': 'none',
}
arrow_kwargs = {
    'head_width': 10,
    'head_length': 10,
    'edgecolor': 'r',
    'facecolor': 'r',
}

def plot_output(left_image, right_image, windows):
    '''Plots the output...
    '''
    figure = plt.figure(figsize=(1, 2))
    left_plot = figure.add_subplot(1, 2, 1)
    left_plot.imshow(left_image)
    right_plot = figure.add_subplot(1, 2, 2)
    right_plot.imshow(right_image)
    # Plot the grid of windows on each image
    for window_info in windows:
        x, y = window_info['window_centre_top_left']
        window_rect = lambda: Rectangle((x, y), *window_info['size'], **rect_kwargs)
        left_plot.add_patch(window_rect())
        right_plot.add_patch(window_rect())
        # Also plot arrows showing non-zero pixel displacement on the left image
        if window_info['dp_x'] or window_info['dp_y']:
            x += int(window_info['size'][0] / 2)
            y += int(window_info['size'][1] / 2)
            left_plot.arrow(x, y, window_info['dp_x'], window_info['dp_y'], **arrow_kwargs)
    plt.show()

def search_regions_by_window(template_image, region_image, x_window, y_window,
        overlap=0, region_scheme='3x3'):
    '''For a given window size, breaks up each image into windows of that size,
    defined by their top left co-ordinate.
    Then, for each window, determines the search regions based on the specified
    scheme, in a dict containing the origins and image slices for each region.
    '''
    image_height, image_width = region_image.shape
    if template_image.shape[0] != image_height or template_image.shape[1] != image_width:
        raise Exception('Dimensions of template and region images must match')
    windows = []
    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            window_info = {
                'window_centre_top_left': (x, y),
                'size': (x_window, y_window),
                'image_slice': template_image[y : y + y_window, x : x + x_window],
                'regions': [],
            }
            for _x, _y in region_pairs(x, x_window, y, y_window, region_scheme):
                # If fully outside image, skip
                if (_x + x_window <= 0 or _y + y_window <= 0 or \
                        _x > image_width or _y > image_height):
                    continue
                # If partially outside image, truncate
                x_start = max(_x, 0)
                y_start = max(_y, 0)
                x_end = min(_x + x_window, image_width)
                y_end = min(_y + y_window, image_height)
                window_info['regions'].append({
                    'top_left': (x_start, y_start),
                    'size': (x_end - x_start, y_end - y_start),
                    'image_slice': region_image[y_start : y_end, x_start : x_end],
                })
            windows.append(window_info)
            x += int(x_window * (1 - overlap))
        y += int(y_window * (1 - overlap))

    return windows

def image_scan(window_info, corr_threshold=0):
    '''Scans
    '''
    template = window_info['image_slice']
    wctl = window_info['window_centre_top_left']
    window_size = window_info['size']
    if PLOT_GRIDS:
        figure = plt.figure(figsize=(3, 3))
        figure.suptitle(f'WCTL at {wctl}')
    window_corr_max = 0
    window_corr_max_pos = wctl
    for i, region_info in enumerate(window_info['regions']):
        tl = region_info['top_left']
        x_corr = cross_correlate_2d_spectral(template, region_info['image_slice'])
        local_corr_max_pos = max_pos(x_corr) # (y, x)
        corr_max = x_corr[local_corr_max_pos[1], local_corr_max_pos[0]]
        corr_max_pos = (local_corr_max_pos[1] + tl[0], local_corr_max_pos[0] + tl[1]) # (x, y)
        # Update the max correlation for the window if the value found in this region
        # is the highest and exceeds the specified threshold
        if corr_max > window_corr_max and corr_max > corr_threshold:
            window_corr_max = corr_max
            window_corr_max_pos = corr_max_pos
        if PLOT_GRIDS:
            ax = figure.add_subplot(3, 3, i + 1)
            ax.set_title(f'TL at {tl}, max XC of {round(corr_max, 4)} at {corr_max_pos}')
            ax.add_patch(Circle(local_corr_max_pos, radius=10, color='red'))
            plt.imshow(region_info['image_slice'])
    dp_x = window_corr_max_pos[0] - wctl[0]
    dp_y = window_corr_max_pos[1] - wctl[1]
    if LOG_WINDOW_SUMMARY:
        # Only log the windows on the left edge
        if LOG_WINDOW_SUMMARY != 'left' or wctl[0] == 0:
            window_summary = ('Window (left, right, top, bottom): '
                f'({wctl[0]}, {wctl[0] + window_size[0]}, {wctl[1]}, {wctl[1] + window_size[1]}), '
                f'max correlation of {round(window_corr_max, 5)} at {window_corr_max_pos}, '
                f'shift of ({dp_x}, {dp_y}), ')
            print(window_summary)
    if PLOT_GRIDS:
        plt.show()
    return dp_x, dp_y

def image_scan_multi_pass():
    '''Implements the multi-pass algorithm, effectively zooming in on a
    region to find features.
    '''
    pass

def region_pairs(x, x_window, y, y_window, region_scheme):
    match region_scheme:
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
            raise Exception(f'Unknown search region scheme: {region_scheme}')