'''Script for stereo vision image comparison...
'''
import numpy as np
import json
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plot
from matplotlib import cm
from sv_calibration import model_func_generator
from scipy.interpolate import griddata as scipy_griddata

# TODO: handle inverse depth

image_dir = 'images-p2-uncal'
depth_data_dir = 'depth-data'
cal_data_dir = 'calibration-data'
PLOT_DP = 0

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_input', default='model.json')
    parser.add_argument('--cal_data_input', default='test_1_depth_data.json')
    parser.add_argument('--depth_input', default='test_1_depth_map.json')
    parser.add_argument('--calib_type', default='model')
    parser.add_argument('--images', default='test_1_dots')
    args = parser.parse_args()

    # Read in the model and define the corresponding function
    with open(f'{cal_data_dir}/{args.model_input}') as f:
        model = json.load(f)
    def model_func(x_l, y_l, x_r, y_r):
        return model_func_generator(model['z'])(x_l, y_l, x_r, y_r)
    vec_model_func = np.vectorize(model_func)

    with open(f'{cal_data_dir}/{args.cal_data_input}') as f:
        cal_data = json.load(f)

    with open(f'{depth_data_dir}/{args.depth_input}') as f:
        depth_data = json.load(f)
    dp_x_grid = np.array(depth_data['dp_x_grid'])
    dp_y_grid = np.array(depth_data['dp_y_grid'])
    grid_shape = dp_x_grid.shape
    flattened_shape = grid_shape[0] * grid_shape[1]


    if args.calib_type == 'griddata':
        x_linspace = range(grid_shape[1])
        y_linspace = range(grid_shape[0])
        x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)

        x_vec = np.reshape(x_grid, flattened_shape)
        y_vec = np.reshape(y_grid, flattened_shape)
        dp_x_vec = np.reshape(dp_x_grid, flattened_shape)
        dp_y_vec = np.reshape(dp_y_grid, flattened_shape)
        xyxy_values = []
        for i, (x, y, dp_x, dp_y )in enumerate(zip(x_vec, y_vec, dp_x_vec, dp_y_vec)):
            xyxy = (x, y, x + dp_x, y + dp_y)
            xyxy_values.append(xyxy)
            # if i > 1683871 and i < 1683881:
            #     xyxy_values.append(xyxy)

        z_vec = scipy_griddata(
            points=np.array(cal_data['train_points']),
            values=np.array(cal_data['z_values']),
            xi=xyxy_values,
            fill_value=0,
            # method='nearest',
        )
        # for xyxy, z in zip(xyxy_values, z_vec):
        #     print(xyxy, z)
        z_grid = np.reshape(z_vec, grid_shape)

    elif args.calib_type == 'model':
        # Rescale the grids to those used for the model calibration:
        # -500 < x < 500
        # -400 < y < 400
        x_linspace = np.linspace(-500, 500, grid_shape[1])
        y_linspace = np.linspace(-400, 400, grid_shape[0])
        x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)
        z_grid = vec_model_func(x_grid, y_grid, x_grid + dp_x_grid,
            y_grid + dp_y_grid)

    left_image_file = f'left_{args.images}.tiff'
    left_image = Image.open(f'{image_dir}/{left_image_file}').convert('L')
    left_image = np.asarray(left_image)

    figure = plt.figure(1)
    figure.suptitle('z')
    ax = figure.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    figure = plt.figure(2, figsize=(1, 2))
    figure.add_subplot(1, 2, 1)
    plt.imshow(left_image)
    figure.add_subplot(1, 2, 2)
    plt.imshow(z_grid)

    if PLOT_DP:
        figure = plt.figure(3)
        figure.suptitle('dp_x')
        ax = figure.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(x_grid, y_grid, dp_x_grid, cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        figure = plt.figure(4)
        figure.suptitle('dp_y')
        ax = figure.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(x_grid, y_grid, dp_y_grid, cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.show()

if __name__ == '__main__':
    main()