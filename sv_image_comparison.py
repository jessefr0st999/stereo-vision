'''Script for stereo vision image comparison...
'''
from correlation_spectral import cross_correlate_2d_spectral as x_corr_2d
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from utils import max_pos
import numpy as np

# Amount by which normalised XC for a non-central region must exceed that for
# the central region to register a displacement
CENTRE_TOL = 0.01

# Whether to plot the XC; this is disabled by default as it will produce a lot
# of figures
PLOT_XCORR = False

# Only logs non-zero displacements; can be change to True/False; True will produce
# a lot of logs and slow the script down considerably
LOG_WINDOW_SUMMARY = 'nonzero'

def sequence_scan(left_image, right_image, scan_config):
    '''Compares left and right images using window searching and cross-correlation
    for a given sequence of stages.
    '''
    seq_results = []
    image_width = left_image.shape[1]
    image_height = left_image.shape[0]
    print(f'Image width: {image_width}, height: {image_height}')
    for stage, stage_config in enumerate(scan_config):
        # Firstly, discretise the left image and obtain the corresponding search regions
        # in the right image for each window
        windows = {}
        if stage == 0:
            # For the first stage, partition the entire left image as specified in the config
            windows = whole_image_search_regions(
                template_image=left_image,
                region_image=right_image,
                x_window=stage_config['window_width'],
                y_window=stage_config['window_height'],
                scheme=tuple(stage_config['scheme']),
                scheme_shift_size=tuple(stage_config['scheme_shift_size']),
                window_overlap=stage_config['window_overlap'],
            )
            # Use the initial window centre and size for the next stage
            for window_id, window_info in windows.items():
                window_info['stage_centres'] = [window_info['centre']]
                window_info['stage_sizes'] = [window_info['size']]
        else:
            # For later stages, partition both the left and right images based
            # on the factor
            for window_id, window_info in seq_results[stage - 1].items():
                # Skip if the previous stage did not culminate in a resulting displacement
                if (window_info['dp_x'] == 0 and window_info['dp_y'] == 0) or \
                        len(window_info['stage_sizes']) < stage:
                    continue
                updated_window_size = (
                    int(window_info['stage_sizes'][stage - 1][0] / stage_config['factor']),
                    int(window_info['stage_sizes'][stage - 1][1] / stage_config['factor']),
                )
                window_info['stage_sizes'].append(updated_window_size)
                # First, get the target regions
                windows[window_id] = multi_pass_search_regions(
                    template_window_info=window_info,
                    # Centre the region partitions at the "winning" target region in the
                    # previous stage
                    x_centre=window_info['stage_centres'][stage - 1][0] + window_info['dp_x'],
                    y_centre=window_info['stage_centres'][stage - 1][1] + window_info['dp_y'],
                    size=window_info['stage_sizes'][stage],
                    region_image=right_image,
                    factor=stage_config['factor'],
                )
        # Next, perform the actual correlations for each window, and calculate the
        # displacements accordingly
        for window_id, window_info in windows.items():
            if stage == 0:
                max_pos, _ = image_scan(window_info, window_id,
                    stage_config['correlation_threshold'])
                window_info['dp_x'] = max_pos[0] - window_info['centre'][0]
                window_info['dp_y'] = max_pos[1] - window_info['centre'][1]
            else:
                # Now break up the window (left image) based on the factor
                # Centre the window partitions at the previous window centre
                x_centre, y_centre = window_info['stage_centres'][stage - 1]
                x_window, y_window = window_info['stage_sizes'][stage - 1]
                window_partitions = []
                for _x, _y in region_partition_pairs(x_centre, x_window, y_centre, y_window,
                        stage_config['factor']):
                    window_boundaries = get_window_boundaries(_x, _y, x_window,
                        y_window, image_width, image_height)
                    if window_boundaries is None:
                        continue
                    x_start, x_end, y_start, y_end = window_boundaries
                    window_partitions.append({
                        'centre': (_x, _y),
                        'size': (x_end - x_start, y_end - y_start),
                        'image_slice': left_image[y_start : y_end, x_start : x_end],
                    })
                # Find the biggest correlation for each partition of the window in the template image
                # For the overall biggest correlation, update the window and (dp_x, dp_y) appropriately
                # Note that this may not occur for this stage, in which case dp_x, dp_y and centre will
                # not update, and stage_centres will not increase in length
                window_corr_max = 0
                for wp in window_partitions:
                    partitioned_window_info = {**wp, 'target_regions': window_info['target_regions']}
                    max_pos, corr_max = image_scan(partitioned_window_info, window_id,
                        corr_threshold=0)
                    if corr_max > window_corr_max:
                        window_corr_max = corr_max
                        window_info['centre'] = wp['centre']
                        window_info['dp_x'] = max_pos[0] - wp['centre'][0]
                        window_info['dp_y'] = max_pos[1] - wp['centre'][1]
                        if len(window_info['stage_centres']) < stage + 1:
                            window_info['stage_centres'].append(wp['centre'])
                        else:
                            window_info['stage_centres'][stage] = wp['centre']
        seq_results.append(windows)
    return seq_results

def region_pairs(x_centre, x_window, y_centre, y_window, scheme, shift_size):
    '''Gets the midpoints of the regions centred at the desired points up to a
    specified distance away (scheme time either window size or shift pixels).
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
    windows = {}
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
                'row': row,
                'column': column,
                'centre': centre,
                'size': (x_window, y_window),
                'image_slice': template_image[y : y + y_window,  x : x + x_window],
                'target_regions': [],
            }
            for _x, _y in region_pairs(x_centre, x_window, y_centre, y_window,
                    scheme, scheme_shift_size):
                window_boundaries = get_window_boundaries(_x, _y, x_window,
                    y_window, image_width, image_height)
                if window_boundaries is None:
                    continue
                x_start, x_end, y_start, y_end = window_boundaries
                window_info['target_regions'].append({
                    'centre': (_x, _y),
                    'size': (x_end - x_start, y_end - y_start),
                    'image_slice': region_image[y_start : y_end, x_start : x_end],
                })
            windows[window_id] = window_info
            window_id += 1
            x += int(x_window * (1 - window_overlap))
            column += 1
        y += int(y_window * (1 - window_overlap))
        row += 1
    return windows

def region_partition_pairs(x_centre, x_window, y_centre, y_window, factor):
    '''Same as `region_pairs`, but for a given factor rather than scheme.
    '''
    x_vec = np.linspace(x_centre - x_window / 2, x_centre + x_window / 2,
        2 * factor + 1)[1 :: 2]
    y_vec = np.linspace(y_centre - y_window / 2, y_centre + y_window / 2,
        2 * factor + 1)[1 :: 2]
    pairs = []
    for y in y_vec:
        for x in x_vec:
            pairs.append((x, y))
    return pairs

def multi_pass_search_regions(template_window_info, x_centre, y_centre, size,
        region_image, factor):
    '''Performs the same operation as `whole_image_search_regions` but for later
    stages of multi-pass sequences, centred on a given region in the right image and
    partitioned based on a factor.
    '''
    image_height, image_width = region_image.shape
    window_info = {**template_window_info, 'target_regions': []}
    x_window, y_window = size
    for _x, _y in region_partition_pairs(x_centre, factor * x_window,
            y_centre, factor * y_window, factor):
        window_boundaries = get_window_boundaries(_x, _y, x_window,
            y_window, image_width, image_height)
        if window_boundaries is None:
            continue
        x_start, x_end, y_start, y_end = window_boundaries
        window_info['target_regions'].append({
            'centre': (_x, _y),
            'size': (x_end - x_start, y_end - y_start),
            'image_slice': region_image[y_start : y_end, x_start : x_end],
        })
    return window_info

def image_scan(window_info, window_id, corr_threshold=0):
    '''For a given window, cross-correlates with the corresponding search/target
    regions, returning the centre of the target region at which a maximum is attained,
    as well as the maximum cross-correlation value.
    '''
    template = window_info['image_slice']
    centre = window_info['centre']
    window_corr_max = 0
    window_corr_max_pos = centre
    for region_info in window_info['target_regions']:
        target_centre = region_info['centre']
        # Skip if the region falls exactly outside
        if region_info['image_slice'].size == 0:
            continue
        x_corr = x_corr_2d(template, region_info['image_slice'])
        corr_max = np.max(x_corr)
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
        if PLOT_XCORR:
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
    # Log a summary for the window to terminal if configured to do so
    if LOG_WINDOW_SUMMARY == True or \
        (LOG_WINDOW_SUMMARY == 'nonzero' and (dp_x != 0 or dp_y != 0)):
        window_summary = (f'Window {window_id}: '
            f'centre at {centre}, '
            f'max correlation of {round(window_corr_max, 5)} at {window_corr_max_pos}, '
            f'shift of ({dp_x}, {dp_y})')
        print(window_summary)
    return window_corr_max_pos, window_corr_max

def get_window_boundaries(x, y, x_window, y_window, image_width, image_height):
    '''Gets boundaries for a given window, validating them based on the image
    dimensions.
    '''
    # If fully outside image, skip
    if (x + x_window / 2 <= 0 or \
        y + y_window / 2 <= 0 or \
        x - x_window / 2 > image_width or \
        y - y_window / 2 > image_height):
        return None
    # If partially outside image, truncate
    x_start = int(max(x - x_window / 2, 0))
    x_end = int(min(x + x_window / 2, image_width))
    y_start = int(max(y - y_window / 2, 0))
    y_end = int(min(y + y_window / 2, image_height))
    return x_start, x_end, y_start, y_end

grid_rect_kwargs = {
    'linewidth': 1,
    'edgecolor': 'blue',
    'facecolor': 'none',
}
multi_pass_template_kwargs = {
    'linewidth': 1,
    'edgecolor': 'cyan',
    'facecolor': 'none',
}
multi_pass_target_kwargs = {
    'linewidth': 1,
    'edgecolor': 'green',
    'facecolor': 'none',
}
shift_rect_kwargs = {
    'linewidth': 1,
    'edgecolor': 'none',
}
arrow_kwargs = {
    'head_width': 5,
    'head_length': 5,
    'edgecolor': 'red',
    'facecolor': 'red',
}

def plot_sequence_output(left_image, right_image, sequence_results,
        max_shift_magnitude, shift_plot_type):
    '''Plots the left and right images overlaid with results for the given sequence.
    '''
    figure = plt.figure(figsize=(1, 2))
    left_plot = figure.add_subplot(1, 2, 1)
    left_plot.imshow(left_image)
    right_plot = figure.add_subplot(1, 2, 2)
    right_plot.imshow(right_image)
    for stage, windows in enumerate(sequence_results):
        for window_id, window_info in windows.items():
            if len(window_info['stage_centres']) < stage + 1:
                continue
            # Plot the initial (first stage) grid on each image
            if stage == 0:
                x, y = window_info['centre']
                top_left = (x - int(window_info['size'][0] / 2),
                    y - int(window_info['size'][1] / 2))
                window_rect = lambda: Rectangle(top_left, *window_info['size'],
                    **grid_rect_kwargs)
                left_plot.add_patch(window_rect())
                right_plot.add_patch(window_rect())
            else:
                # For subsequent stages, plot the regions over which the templates
                # are searched on both images
                for region_info in window_info['target_regions']:
                    r_x, r_y = region_info['centre']
                    r_top_left = (r_x - int(region_info['size'][0] / 2),
                        r_y - int(region_info['size'][1] / 2))
                    region_rect = lambda: Rectangle(r_top_left, *region_info['size'],
                        **multi_pass_target_kwargs)
                    left_plot.add_patch(region_rect())
                    right_plot.add_patch(region_rect())
                # Also plot the templates on left image
                x, y = window_info['stage_centres'][stage]
                top_left = (x - int(window_info['stage_sizes'][stage][0] / 2),
                    y - int(window_info['stage_sizes'][stage][1] / 2))
                window_rect = lambda: Rectangle(top_left, *window_info['stage_sizes'][stage],
                    **multi_pass_template_kwargs)
                left_plot.add_patch(window_rect())
            # Also plot arrows showing non-zero pixel displacement on the left image
            if stage == len(sequence_results) - 1 and (window_info['dp_x'] or window_info['dp_y']):
                if shift_plot_type == 'arrows':
                    left_plot.arrow(x, y, window_info['dp_x'],
                        window_info['dp_y'], **arrow_kwargs)
                elif shift_plot_type == 'boxes':
                    top_left = (x - int(window_info['stage_sizes'][stage][0] / 2),
                        y - int(window_info['stage_sizes'][stage][1] / 2))
                    shift_magnitude = np.sqrt(window_info['dp_x'] ** 2 + window_info['dp_y'] ** 2)
                    shift_rect_colour = (1, 0, 0) if stage == 0 else (0, 1, 1)
                    shift_rect = lambda: Rectangle(top_left, *window_info['stage_sizes'][stage],
                        facecolor=(*shift_rect_colour, min(shift_magnitude / max_shift_magnitude, 1)),
                        **shift_rect_kwargs)
                    left_plot.add_patch(shift_rect())
    plt.show()