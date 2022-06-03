'''Script for stereo vision image comparison...
'''
import numpy as np
import json
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sv_image_comparison import (whole_image_search_regions, image_scan, sub_region_pairs,
    plot_multi_pass_output, centred_search_region, depth_map)
from utils import downsample

# TODO: depth map on 3D grid

image_dir = 'images-p2-uncal'

PLOT_COMPARISON = 1
PLOT_DEPTH = 0

def main():
    parser = ArgumentParser()
    parser.add_argument('--images', default='box')
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

    image_width = left_image.shape[1]
    image_height = left_image.shape[0]
    print(f'Image width: {image_width}, height: {image_height}')

    stage_results = []
    for stage, stage_config in enumerate(scan_config):
        windows = []
        if stage == 0:
            windows = whole_image_search_regions(
                template_image=left_image,
                region_image=right_image,
                x_window=stage_config['window_width'],
                y_window=stage_config['window_height'],
                scheme=stage_config['scheme'],
                overlap=stage_config['overlap'],
            )
            # Use the initial window centre and size for the next stage
            for window_info in windows:
                window_info['stage_centres'] = [window_info['centre']]
                window_info['stage_sizes'] = [window_info['size']]
        else:
            # Partition both the left and right images based on the factor
            for window_info in stage_results[stage - 1]:
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
        for ii, window_info in enumerate(windows):
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
        stage_results.append(windows)

    # Add all results to a grid, updating results obtained at latter stages
    # total_results = []
    # prev_row = None
    # for stage, windows in enumerate(stage_results):
    #     for window_info in windows:
    #         x_l, y_l = window_info['centre']
    #         x_r = x_l + window_info['dp_x']
    #         y_r = y_l + window_info['dp_y']
    #         if window_info['row'] != prev_row:
    #             total_results.append([])
    #         total_results[len(total_results) - 1].append([x_l, y_l, x_r, y_r])
    #         prev_row = window_info['row']
    # total_results = np.array(total_results)

    if PLOT_COMPARISON:
        plot_multi_pass_output(left_image, right_image, stage_results)

    # if PLOT_DEPTH:
    #     depth_grid = depth_map(total_results)
    #     figure = plt.figure()
    #     plt.imshow(depth_grid)
    #     plt.show()

if __name__ == '__main__':
    main()