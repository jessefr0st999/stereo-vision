'''Script for stereo vision image comparison...
'''
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime
from matplotlib import pyplot as plt
from sv_image_comparison import whole_image_search_regions, image_scan, depth_map, plot_single_pass_output

image_dir = 'images-p2-uncal'

PLOT_COMPARISON = 1
PLOT_DEPTH = 0

def main():
    parser = ArgumentParser()
    parser.add_argument('--images', default='box')
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--x_window', type=int)
    parser.add_argument('--y_window', type=int)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--scheme', default='3x3')
    parser.add_argument('--corr_threshold', type=float, default=0)
    args = parser.parse_args()
    x_window = getattr(args, 'x_window') or args.window
    y_window = getattr(args, 'y_window') or args.window

    # Read in the left image
    left_image_file = f'left_{args.images}.tiff'
    left_image = Image.open(f'{image_dir}/{left_image_file}').convert('L')
    left_image = np.asarray(left_image)

    # Repeat for the right image
    right_image_file = f'right_{args.images}.tiff'
    right_image = Image.open(f'{image_dir}/{right_image_file}').convert('L')
    right_image = np.asarray(right_image)

    print(f'Image width: {left_image.shape[1]}, height: {left_image.shape[0]}')

    windows = whole_image_search_regions(left_image, right_image, x_window, y_window,
        args.scheme, args.overlap)
    start = datetime.now()
    for window_info in windows:
        window_info['dp_x'], window_info['dp_y'] = image_scan(window_info, args.corr_threshold)
    print(f'Time elapsed (image scan): {datetime.now() - start}')

    depth_grid = depth_map(windows)

    if PLOT_COMPARISON:
        plot_single_pass_output(left_image, right_image, windows)

    if PLOT_DEPTH:
        figure = plt.figure()
        plt.imshow(depth_grid)
        plt.show()

if __name__ == '__main__':
    main()