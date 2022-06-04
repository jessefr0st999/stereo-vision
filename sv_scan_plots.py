'''Script for stereo vision image comparison...
'''
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plot
import numpy as np

# TODO: handle 3D axes flip

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

def plot_depth_grid(depth_grid, contributions_grid):
    '''Plots the...
    '''
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1, projection='3d')
    x_grid, y_grid = np.meshgrid(range(depth_grid.shape[1]),
        range(depth_grid.shape[0]))
    ax.plot_surface(x_grid, y_grid, depth_grid, cmap=cm.coolwarm)
    plt.show()

    figure = plt.figure()
    ax = figure.add_subplot(1, 2, 1)
    ax.imshow(depth_grid)

    ax = figure.add_subplot(1, 2, 2)
    ax.imshow(contributions_grid)
    plt.show()

def plot_multi_pass_output(left_image, right_image, stage_results,
        max_shift_magnitude, shift_plot_type):
    '''Plots the output...
    '''
    figure = plt.figure(figsize=(1, 2))
    left_plot = figure.add_subplot(1, 2, 1)
    left_plot.imshow(left_image)
    right_plot = figure.add_subplot(1, 2, 2)
    right_plot.imshow(right_image)
    for stage, windows in enumerate(stage_results):
        for window_id, window_info in windows.items():
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
            if stage == len(stage_results) - 1 and (window_info['dp_x'] or window_info['dp_y']):
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