'''Script for stereo vision image comparison...
'''
# from correlation import cross_correlate_2d as x_corr_2d
from correlation_spectral import cross_correlate_2d_spectral as x_corr_2d
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from utils import max_pos
import numpy as np

# TODO: truncate window on boundaries
# TODO: fix multi pass

PLOT_XCORR = 0
# PLOT_XCORR = 1
# LOG_WINDOW_SUMMARY = 0
# LOG_WINDOW_SUMMARY = 'left'
LOG_WINDOW_SUMMARY = 'nonzero'
# LOG_WINDOW_SUMMARY = 1
CENTRE_TOL = 0.01

def whole_image_search_regions(template_image, region_image, x_window, y_window,
        scheme=(1, 3), scheme_shift_size=(0, 0), window_overlap=0):
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
    row = 0
    window_id = 0
    while y < image_height:
        x = 0
        column = 0
        while x < image_width:
            x_centre = x + x_window / 2
            y_centre = y + y_window / 2
            centre = (x_centre, y_centre)
            window_info = {
                'id': window_id,
                'row': row,
                'column': column,
                'centre': centre,
                'size': (x_window, y_window),
                'image_slice': template_image[y : y + y_window,  x : x + x_window],
                'target_regions': [],
            }
            for _x, _y in region_pairs(x_centre, x_window, y_centre, y_window,
                    scheme, scheme_shift_size):
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
            x += int(x_window * (1 - window_overlap))
            column += 1
        y += int(y_window * (1 - window_overlap))
        row += 1

    return windows

def centred_search_region(template_window_info, centre, size, region_image, factor):
    '''Gets

    Used for later stages of multi-pass
    '''
    image_height, image_width = region_image.shape
    window_info = {**template_window_info, 'target_regions': []}
    x_centre, y_centre = centre
    x_window, y_window = size
    for _x, _y in sub_region_pairs(x_centre, factor * x_window,
            y_centre, factor * y_window, factor):
        # TODO: helper for this shit
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
    return window_info

def image_scan(window_info, corr_threshold=0, _print=False):
    '''For a given
    '''
    template = window_info['image_slice']
    centre = window_info['centre']
    window_size = window_info['size']
    window_corr_max = 0
    window_corr_max_pos = centre
    for region_info in window_info['target_regions']:
        target_centre = region_info['centre']
        # Skip if the region falls exactly outside
        if region_info['image_slice'].size == 0:
            continue
        x_corr = x_corr_2d(template, region_info['image_slice'], window_info["id"])
        corr_max = np.max(x_corr)

        # TODO
        # print(template.shape, region_info['image_slice'].shape, x_corr.shape)

        # Update the max correlation for the window to the centre of the region if the
        # value found in this region is the highest and exceeds the specified threshold
        # Preference the target region with the same centre as the window if there is
        # a tie in max correlations
        update_condition = corr_max > corr_threshold and (
            (target_centre == centre and corr_max >= window_corr_max) or \
            (target_centre != centre and corr_max > window_corr_max + CENTRE_TOL)
        )
        if update_condition:
            window_corr_max = corr_max
            window_corr_max_pos = target_centre
        if PLOT_XCORR and window_info['id'] == 161:
            local_corr_max_pos = max_pos(x_corr) # (y, x)
            figure = plt.figure()
            ax = figure.add_subplot(1, 3, 1)
            ax.set_title(f'Template; centre at {centre}')
            plt.imshow(template)
            ax = figure.add_subplot(1, 3, 2)
            ax.set_title(f'Target region; centre at {target_centre}')
            plt.imshow(region_info['image_slice'])
            ax = figure.add_subplot(1, 3, 3)
            ax.set_title((f'Max XC of {round(corr_max, 4)} '
                f'at local pos {tuple(reversed(local_corr_max_pos))}'))
            ax.add_patch(Circle(local_corr_max_pos, radius=1, color='red'))
            plt.imshow(x_corr)
            plt.show()
    dp_x = window_corr_max_pos[0] - centre[0]
    dp_y = window_corr_max_pos[1] - centre[1]
    if LOG_WINDOW_SUMMARY:
        window_summary = (f'Window {window_info["id"]}: '
            f'centre at {centre}, '
            f'max correlation of {round(window_corr_max, 5)} at {window_corr_max_pos}, '
            f'shift of ({dp_x}, {dp_y})')
        # Only log the windows on the left edge
        if LOG_WINDOW_SUMMARY == 1 or \
            (LOG_WINDOW_SUMMARY == 'left' and centre[0] - window_size[0] / 2 == 0) or \
            (LOG_WINDOW_SUMMARY == 'nonzero' and (dp_x != 0 or dp_y != 0)):
            print(window_summary)
    return window_corr_max_pos, window_corr_max

def sub_region_pairs(x_centre, x_window, y_centre, y_window, factor):
    '''Gets the midpoints of
    '''
    x_vec = np.linspace(x_centre - x_window / 2, x_centre + x_window / 2, 2 * factor + 1)[1 :: 2]
    y_vec = np.linspace(y_centre - y_window / 2, y_centre + y_window / 2, 2 * factor + 1)[1 :: 2]
    pairs = []
    for y in y_vec:
        for x in x_vec:
            pairs.append((x, y))
    return pairs

def region_pairs(x_centre, x_window, y_centre, y_window, scheme, shift_size):
    '''Gets the midpoints of
    '''
    if scheme[0] % 2 != 1 or scheme[1] % 2 != 1:
        raise Exception('Scheme dimensions must be odd.')
    if shift_size[0]:
        x_span_hw = shift_size[0] * (scheme[0] - 1) / 2
    else:
        x_span_hw = x_window * (scheme[0] - 1) / 2
    if shift_size[1]:
        y_span_hw = shift_size[1] * (scheme[1] - 1) / 2
    else:
        y_span_hw = y_window * (scheme[1] - 1) / 2
    x_vec = np.linspace(x_centre - x_span_hw, x_centre + x_span_hw, scheme[0])
    y_vec = np.linspace(y_centre - y_span_hw, y_centre + y_span_hw, scheme[1])
    pairs = [(x_centre, y_centre)]
    for y in y_vec:
        for x in x_vec:
            if x == x_centre and y == y_centre:
                continue
            pairs.append((x, y))
    return pairs