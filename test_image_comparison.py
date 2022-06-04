'''Script for stereo vision image comparison...
'''
import numpy as np
import json
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime
from sv_image_comparison import sequence_scan
from sv_scan_plots import plot_depth_grid, plot_multi_pass_output
from utils import downsample

image_dir = 'images-p2-uncal'
data_dir = 'depth-data'
config_dir = 'configs'

def main():
    parser = ArgumentParser()
    parser.add_argument('--images', default='test_2_noise')
    parser.add_argument('--ds_factor', type=int, default=1)
    parser.add_argument('--config', default='scan_config.json')
    parser.add_argument('--depth_output', default=None)
    parser.add_argument('--depth_input', default=None)
    parser.add_argument('--shift_plot_type', default='boxes')
    parser.add_argument('--show_comparison', action='store_true', default=False)
    parser.add_argument('--hide_depth', action='store_true', default=False)
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
        total_dp_x_grid = np.zeros(left_image.shape)
        total_dp_y_grid = np.zeros(left_image.shape)
        total_depth_grid = np.zeros(left_image.shape)
        total_contributions_grid = np.zeros(left_image.shape)
        for i, seq_config in enumerate(config):
            start = datetime.now()
            seq_results = sequence_scan(left_image, right_image, seq_config)
            print(f'Time elapsed for sequence {i + 1} scan: {datetime.now() - start}')
            if args.show_comparison:
                plot_multi_pass_output(left_image, right_image, seq_results,
                    max_shift_magnitude=calc_max_shift_magnitude(seq_config[0]),
                    shift_plot_type=args.shift_plot_type)
            seq_depth_grid = np.zeros(left_image.shape)
            seq_dp_x_grid = np.zeros(left_image.shape)
            seq_dp_y_grid = np.zeros(left_image.shape)
            seq_contributions_grid = np.zeros(left_image.shape)
            for stage, windows in enumerate(seq_results):
                for window_id, window_info in windows.items():
                    # Only the results for the last stage of a given window should contribute
                    if len(window_info['stage_centres']) - 1 > stage:
                        continue
                    # Obtain desired quantities
                    x, y = window_info['stage_centres'][stage]
                    shift_magnitude = np.sqrt(window_info['dp_x'] ** 2 + window_info['dp_y'] ** 2)
                    # Obtain the subset of the (x, y) grid on which to set these quantities
                    x_window, y_window = window_info['stage_sizes'][stage]
                    x_start = int(max(x - x_window / 2, 0))
                    y_start = int(max(y - y_window / 2, 0))
                    x_end = int(min(x + x_window / 2, image_width))
                    y_end = int(min(y + y_window / 2, image_height))
                    # Set the quantities
                    seq_dp_x_grid[y_start : y_end, x_start : x_end] += window_info['dp_x']
                    seq_dp_y_grid[y_start : y_end, x_start : x_end] += window_info['dp_y']
                    seq_depth_grid[y_start : y_end, x_start : x_end] += shift_magnitude
                    if shift_magnitude > 0:
                        seq_contributions_grid[y_start : y_end, x_start : x_end] += 1
            total_depth_grid += seq_depth_grid
            total_dp_x_grid += seq_dp_x_grid
            total_dp_y_grid += seq_dp_y_grid
            total_contributions_grid += seq_contributions_grid
        # Average out the estimated values at each (x, y) point by dividing
        # by the number of windows contibuting to each point
        total_contributions_grid[total_contributions_grid == 0] = np.Infinity
        total_depth_grid = np.divide(total_depth_grid, total_contributions_grid)
        total_dp_x_grid = np.divide(total_dp_x_grid, total_contributions_grid)
        total_dp_y_grid = np.divide(total_dp_y_grid, total_contributions_grid)

        if args.depth_output is not None:
            # Save the raw data
            with open(f'{data_dir}/{args.depth_output}', 'w') as f:
                results = {
                    'depth_grid': total_depth_grid.tolist(),
                    'dp_x_grid': total_dp_x_grid.tolist(),
                    'dp_y_grid': total_dp_y_grid.tolist(),
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

    if not args.hide_depth:
        plot_depth_grid(total_depth_grid, total_contributions_grid)

def calc_max_shift_magnitude(config):
    if config['scheme_shift_size'][0]:
        max_dp_x = config['scheme_shift_size'][0] * (config['scheme'][0] - 1) / 2
    else:
        max_dp_x = config['window_width'] * (config['scheme'][0] - 1) / 2
    if config['scheme_shift_size'][1]:
        max_dp_y = config['scheme_shift_size'][1] * (config['scheme'][1] - 1) / 2
    else:
        max_dp_y = config['window_height'] * (config['scheme'][1] - 1) / 2
    return np.sqrt(max_dp_x ** 2 + max_dp_y ** 2)


if __name__ == '__main__':
    main()