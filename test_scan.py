'''Script for stereo vision image comparison...
'''
import numpy as np
import json
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plot
from sv_scan import sequence_scan, plot_multi_pass_output
from utils import downsample

# TODO: depth map on 3D grid

image_dir = 'images-p2-uncal'
data_dir = 'depth-data'
config_dir = 'configs'

PLOT_COMPARISON = 0

# TODO: refine the depth map as below
'''
minval = np.percentile(depth, 3)
maxval = np.percentile(depth, 97)
depth = np.clip(depth, minval, maxval)
depth = ((depth - minval) / (maxval - minval)) * 255
'''

def main():
    parser = ArgumentParser()
    parser.add_argument('--images', default='box')
    parser.add_argument('--ds_factor', type=int, default=1)
    parser.add_argument('--config', default='scan_config.json')
    parser.add_argument('--depth_output', default=None)
    parser.add_argument('--depth_input', default=None)
    args = parser.parse_args()

    with open(f'{config_dir}/{args.config}') as f:
        config = json.load(f)

    if args.depth_input is None:
        # Read in the left image
        left_image_file = f'left_{args.images}.tiff'
        left_image = Image.open(f'{image_dir}/{left_image_file}').convert('L')
        left_image = np.asarray(left_image)
        left_image = downsample(left_image, args.ds_factor)

        # Repeat for the right image
        right_image_file = f'right_{args.images}.tiff'
        right_image = Image.open(f'{image_dir}/{right_image_file}').convert('L')
        right_image = np.asarray(right_image)
        right_image = downsample(right_image, args.ds_factor)

        image_width = left_image.shape[1]
        image_height = left_image.shape[0]
        # Add all results to a grid, updating results obtained at latter stages
        total_depth_grid = np.zeros(left_image.shape)
        total_contributions_grid = np.zeros(left_image.shape)
        for seq_config in config:
            seq_results = sequence_scan(left_image, right_image, seq_config)
            if PLOT_COMPARISON:
                plot_multi_pass_output(left_image, right_image, seq_results)
            seq_depth_grid = np.zeros(left_image.shape)
            seq_contributions_grid = np.zeros(left_image.shape)
            for stage, windows in enumerate(seq_results):
                for window_info in windows:
                    x, y = window_info['centre']
                    x_window, y_window = window_info['size']
                    x_start = int(max(x - x_window / 2, 0))
                    y_start = int(max(y - y_window / 2, 0))
                    x_end = int(min(x + x_window / 2, image_width))
                    y_end = int(min(y + y_window / 2, image_height))
                    shift_magnitude = np.sqrt(window_info['dp_x'] ** 2 + window_info['dp_y'] ** 2)
                    seq_depth_grid[y_start : y_end, x_start : x_end] += shift_magnitude
                    if shift_magnitude > 0:
                        seq_contributions_grid[y_start : y_end, x_start : x_end] += 1
            total_depth_grid += seq_depth_grid
            total_contributions_grid += seq_contributions_grid
        total_contributions_grid[total_contributions_grid == 0] = np.Infinity
        total_depth_grid = np.divide(total_depth_grid, total_contributions_grid)

        if args.depth_output is not None:
            # Save the raw data
            with open(f'{data_dir}/{args.depth_output}', 'w') as f:
                results = {
                    'depth_grid': total_depth_grid.tolist(),
                    'contributions_grid': total_contributions_grid.tolist(),
                }
                json.dump(results, f)
            print(f'Data saved to file {args.depth_output}')
        else:
            print(f'No output file specified.')

    else:
        with open(f'{data_dir}/{args.depth_input}') as f:
            saved_results = json.load(f)
        total_depth_grid = np.array(saved_results['depth_grid'])
        total_contributions_grid = np.array(saved_results['contributions_grid'])
        print(f'Data read from file {args.depth_input}')

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1, projection='3d')
    X, Y = np.meshgrid(range(total_depth_grid.shape[1]), range(total_depth_grid.shape[0]))
    ax.plot_surface(X, Y, total_depth_grid, cmap=cm.coolwarm)
    plt.show()

    figure = plt.figure()
    ax = figure.add_subplot(1, 2, 1)
    ax.imshow(total_depth_grid)

    ax = figure.add_subplot(1, 2, 2)
    ax.imshow(total_contributions_grid)
    plt.show()

if __name__ == '__main__':
    main()