'''Script for stereo vision image comparison...
'''
import numpy as np
import json
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sv_image_comparison import (whole_image_search_regions, image_scan,
    plot_multi_pass_output, centred_search_region, depth_map)
from utils import downsample

# TODO: depth map on 3D grid

image_dir = 'images-p2-uncal'

PLOT_COMPARISON = 1
PLOT_DEPTH = 0

def main():
    parser = ArgumentParser()
    parser.add_argument('--images', default='portal')
    parser.add_argument('--ds_factor', type=int, default=1)
    parser.add_argument('--scan_config', default='scan_config_one_seq.json')
    args = parser.parse_args()

    with open(args.scan_config) as f:
        scan_config = json.load(f)

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

    print(f'Image width: {left_image.shape[1]}, height: {left_image.shape[0]}')

    stage_results = []
    for stage, stage_config in enumerate(scan_config):
        search_region_kwargs = dict(
            region_image=right_image,
            x_window=stage_config['window_width'],
            y_window=stage_config['window_height'],
            scheme=stage_config['scheme'],
        )
        if stage == 0:
            windows = whole_image_search_regions(template_image=left_image,
                **search_region_kwargs, overlap=stage_config['overlap'])
        else:
            windows = []
            for window_info in stage_results[stage - 1]:
                if window_info['dp_x'] == 0 and window_info['dp_y'] == 0:
                    continue
                updated_window_centre = (window_info['window_centre'][0] + window_info['dp_x'],
                    window_info['window_centre'][1] + window_info['dp_y'])
                window_info['stage_centres'].append(updated_window_centre)
                windows.append(centred_search_region(window_info, stage,
                    **search_region_kwargs))
        for window_info in windows:
            window_info['dp_x'], window_info['dp_y'] = image_scan(window_info,
                stage_config['correlation_threshold'])
        stage_results.append(windows)

    # Add all results to a list, updating results obtained at latter stages
    total_results = [None] * len(stage_results[0])
    for stage, windows in enumerate(stage_results):
        for window_info in windows:
            x_l, y_l = window_info['window_centre']
            x_r = x_l + window_info['dp_x']
            y_r = y_l + window_info['dp_y']
            total_results[window_info['window_id']] = (x_l, y_l, x_r, y_r)

    # Reshape the values into a grid
    prev_y_l = None
    total_results_grid = []
    for values in total_results:
        y_l = values[1]
        if y_l != prev_y_l:
            total_results_grid.append([])
        total_results_grid[len(total_results_grid) - 1].append(values)
        prev_y_l = y_l
    total_results_grid = np.array(total_results_grid)

    if PLOT_COMPARISON:
        plot_multi_pass_output(left_image, right_image, stage_results)

    if PLOT_DEPTH:
        depth_grid = depth_map(total_results_grid)
        figure = plt.figure()
        plt.imshow(depth_grid)
        plt.show()

if __name__ == '__main__':
    main()