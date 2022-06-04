'''Script for stereo vision image comparison...
'''
import numpy as np
import json
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sv_calibration import model_func_generator
from scipy.interpolate import griddata as scipy_griddata

image_dir = 'images-p2-cal'
data_dir = 'calibration-data'

# TODO: statistical analysis of error when changing test/trial sets
# TODO: different calibration schemes

PLOT_OUTPUT = 1

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_input_file', default='00_40_60_80_model.json')
    parser.add_argument('--data_file', default='data.json')
    parser.add_argument('--calib_type', default='model')
    args = parser.parse_args()

    with open(f'{data_dir}/{args.model_input_file}') as f:
        model = json.load(f)
    def model_func(x_l, y_l, x_r, y_r):
        return (
            model_func_generator(model['x'])(x_l, y_l, x_r, y_r),
            model_func_generator(model['y'])(x_l, y_l, x_r, y_r),
            model_func_generator(model['z'])(x_l, y_l, x_r, y_r),
        )

    with open(f'{data_dir}/{args.data_file}') as f:
        data = json.load(f)
    test_points = data['test_points']
    train_points = data['train_points']
    z_values = data['z_values']

    # Pack the z-values into a grid
    if args.calib_type == 'griddata':
        z_list = scipy_griddata(
            points=np.array(train_points),
            values=np.array(z_values),
            xi=test_points,
            fill_value=0,
        )
    elif args.calib_type == 'model':
        z_list = []
        for i, xyxy in enumerate(test_points):
            x_l, y_l, x_r, y_r = xyxy
            x, y, z = model_func(x_l, y_l, x_r, y_r)
            z_list.append(z)

    print(f'Mean value: {np.mean(z_list)}')
    print(f'Median value: {np.median(z_list)}')

    z_grid = []
    prev_y_l = None
    for i, xyxy in enumerate(test_points):
        x_l, y_l, x_r, y_r = xyxy
        z = z_list[i]
        if prev_y_l is None or np.abs(y_l - prev_y_l) > 10:
            z_grid.append([])
        z_grid[len(z_grid) - 1].append(z)
        prev_y_l = y_l

    left_image_file = 'cal_image_left_1940.tiff'
    left_image = Image.open(f'{image_dir}/{left_image_file}').convert('L')
    left_image = np.asarray(left_image)

    figure = plt.figure(figsize=(1, 2))
    figure.add_subplot(1, 2, 1)
    plt.imshow(left_image)
    figure.add_subplot(1, 2, 2)
    plt.imshow(z_grid)
    plt.show()

if __name__ == '__main__':
    main()