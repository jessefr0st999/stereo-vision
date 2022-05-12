'''Script for stereo vision image comparison...
'''
import numpy as np
import json
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime
import matplotlib.pyplot as plt
from sv_image_comparison import search_regions_by_window, image_scan, plot_output
from sv_calibration import model_func_generator

image_dir = 'images-p2-uncal'
model_input_file = 'model.json'

PLOT_OUTPUT = 0
PLOT_DEPTH_MAP = 1

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

    # Read in the model and define the corresponding function
    with open(model_input_file) as f:
        model = json.load(f)
    def model_func(x_l, y_l, x_r, y_r):
        return (
            model_func_generator(model['x'])(x_l, y_l, x_r, y_r),
            model_func_generator(model['y'])(x_l, y_l, x_r, y_r),
            model_func_generator(model['z'])(x_l, y_l, x_r, y_r),
        )

    # Read in the left image
    left_image_file = f'left_{args.images}.tiff'
    left_image = Image.open(f'{image_dir}/{left_image_file}').convert('L')
    left_image = np.asarray(left_image)

    # Repeat for the right image
    right_image_file = f'right_{args.images}.tiff'
    right_image = Image.open(f'{image_dir}/{right_image_file}').convert('L')
    right_image = np.asarray(right_image)

    print(f'Image width: {left_image.shape[1]}, height: {left_image.shape[0]}')

    windows = search_regions_by_window(left_image, right_image, x_window, y_window,
        args.overlap, args.scheme)
    start = datetime.now()
    z_values = []
    prev_y = None
    for window_info in windows:
        window_info['dp_x'], window_info['dp_y'] = image_scan(window_info, args.corr_threshold)
        x_l, y_l = window_info['window_centre_top_left']
        # Use the window centres as positions
        x_l += window_info['size'][0] / 2
        y_l += window_info['size'][1] / 2
        x_r, y_r = x_l, y_l
        if window_info['dp_x'] or window_info['dp_y']:
            x_r += window_info['dp_x']
            y_r += window_info['dp_y']
        x, y, z = model_func(x_l, y_l, x_r, y_r)
        if y_l != prev_y:
            z_values.append([])
        z_values[len(z_values) - 1].append(z)
        prev_y = y_l
        # print(f'In: {x_l}, {y_l}, {x_r}, {y_r}; out: {x}, {y}, {z}')
    print(f'Time elapsed (image scan): {datetime.now() - start}')

    if PLOT_OUTPUT:
        plot_output(left_image, right_image, windows)
    if PLOT_DEPTH_MAP:
        figure = plt.figure(figsize=(1, 2))
        figure.add_subplot(1, 2, 1)
        plt.imshow(left_image)
        figure.add_subplot(1, 2, 2)
        plt.imshow(z_values)
        plt.show()

if __name__ == '__main__':
    main()