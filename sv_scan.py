'''Script for stereo vision image comparison...
'''
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sv_image_comparison import (whole_image_search_regions, image_scan, sub_region_pairs,
    centred_search_region)

rect_kwargs = [
    {
        'linewidth': 1,
        'edgecolor': 'blue',
        'facecolor': 'none',
    },
    {
        'linewidth': 1,
        'edgecolor': 'cyan',
        'facecolor': 'none',
    },
    {
        'linewidth': 1,
        'edgecolor': 'green',
        'facecolor': 'none',
    },
]
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
shift_plot_type = 'arrows'
# shift_plot_type = 'magnitude'

def sequence_scan(left_image, right_image, scan_config):
    '''
    '''
    seq_results = []
    image_width = left_image.shape[1]
    image_height = left_image.shape[0]
    print(f'Image width: {image_width}, height: {image_height}')
    for stage, stage_config in enumerate(scan_config):
        windows = []
        if stage == 0:
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
            for window_info in windows:
                window_info['stage_centres'] = [window_info['centre']]
                window_info['stage_sizes'] = [window_info['size']]
        else:
            # Partition both the left and right images based on the factor
            for window_info in seq_results[stage - 1]:
                if window_info['dp_x'] == 0 and window_info['dp_y'] == 0:
                    continue
                updated_window_centre = (
                    window_info['stage_centres'][stage - 1][0] + window_info['dp_x'],
                    window_info['stage_centres'][stage - 1][1] + window_info['dp_y'],
                )
                updated_window_size = (
                    int(window_info['stage_sizes'][stage - 1][0] / stage_config['factor']),
                    int(window_info['stage_sizes'][stage - 1][1] / stage_config['factor']),
                )
                window_info['stage_centres'].append(updated_window_centre)
                window_info['stage_sizes'].append(updated_window_size)
                windows.append(centred_search_region(
                    template_window_info=window_info,
                    centre=window_info['stage_centres'][stage],
                    size=window_info['stage_sizes'][stage],
                    region_image=right_image,
                    factor=stage_config['factor'],
                ))
        for window_info in windows:
            if stage > 0:
                x_centre, y_centre = window_info['stage_centres'][stage - 1]
                x_window, y_window = window_info['stage_sizes'][stage - 1]
                # Break up the window (left image) based on the factor
                window_partitions = []
                for _x, _y in sub_region_pairs(x_centre, x_window, y_centre, y_window,
                        stage_config['factor']):
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
                    window_partitions.append({
                        'id': window_info['id'],
                        'centre': (_x, _y),
                        'size': (x_end - x_start, y_end - y_start),
                        'image_slice': left_image[y_start : y_end, x_start : x_end],
                    })
                # Find the biggest correlation for each partition of the window in the template image
                # For the overall biggest correlation, update the window and (dp_x, dp_y) appropriately
                window_corr_max = 0
                for wp in window_partitions:
                    partitioned_window_info = {**wp, 'target_regions': window_info['target_regions']}
                    max_pos, corr_max = image_scan(partitioned_window_info,
                        stage_config['correlation_threshold'])
                    if corr_max > window_corr_max:
                        window_corr_max = corr_max
                        window_info['centre'] = wp['centre']
                        window_info['dp_x'] = max_pos[0] - window_info['centre'][0]
                        window_info['dp_y'] = max_pos[1] - window_info['centre'][1]
                        window_info['stage_centres'][stage] = wp['centre']
            else:
                max_pos, _ = image_scan(window_info,
                    stage_config['correlation_threshold'])
                window_info['dp_x'] = max_pos[0] - window_info['centre'][0]
                window_info['dp_y'] = max_pos[1] - window_info['centre'][1]
        seq_results.append(windows)
    return seq_results

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
        x, y = window_info['centre']
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

def plot_multi_pass_output(left_image, right_image, stage_results):
    '''Plots the output...
    '''
    # TODO: handle overlap
    figure = plt.figure(figsize=(1, 2))
    left_plot = figure.add_subplot(1, 2, 1)
    left_plot.imshow(left_image)
    right_plot = figure.add_subplot(1, 2, 2)
    right_plot.imshow(right_image)
    for stage, windows in enumerate(stage_results):
        for window_info in windows:
            # Plot the initial (first stage) grid on each image
            if stage == 0:
                x, y = window_info['centre']
                top_left = (x - int(window_info['size'][0] / 2),
                    y - int(window_info['size'][1] / 2))
                window_rect = lambda: Rectangle(top_left, *window_info['size'],
                    **rect_kwargs[0])
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
                        **rect_kwargs[2])
                    left_plot.add_patch(region_rect())
                    right_plot.add_patch(region_rect())
                # Also plot the templates on left image
                x, y = window_info['stage_centres'][stage]
                top_left = (x - int(window_info['stage_sizes'][stage][0] / 2),
                    y - int(window_info['stage_sizes'][stage][1] / 2))
                window_rect = lambda: Rectangle(top_left, *window_info['stage_sizes'][stage],
                    **rect_kwargs[1])
                left_plot.add_patch(window_rect())
            # Also plot arrows showing non-zero pixel displacement on the left image
            if stage == 0 and (window_info['dp_x'] or window_info['dp_y']):
                if shift_plot_type == 'arrows':
                    left_plot.arrow(x, y, window_info['dp_x'],
                        window_info['dp_y'], **arrow_kwargs)
                elif shift_plot_type == 'magnitude':
                    top_left = (x - int(window_info['stage_sizes'][stage][0] / 2),
                        y - int(window_info['stage_sizes'][stage][1] / 2))
                    shift_magnitude = np.sqrt(window_info['dp_x'] ** 2 + window_info['dp_y'] ** 2)
                    shift_rect = lambda: Rectangle(top_left, *window_info['stage_sizes'][stage],
                        facecolor=(1, 0, 0, min(shift_magnitude / 15, 1)),
                        **shift_rect_kwargs)
                    left_plot.add_patch(shift_rect())
    plt.show()