'''Script for stereo vision image comparison...
'''
import numpy as np
import json
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sv_calibration import model_func_generator, build_features
from scipy.interpolate import griddata as scipy_griddata

# Dots are arranged in a 17 by 21 grid
grid_height = 17
grid_length = 21
grid_spacing = 50 # millimetres

image_dir = 'images-p2-cal'
data_dir = 'calibration-data'
config_dir = 'configs'

# TODO: this script working
# TODO: statistical analysis of error when changing test/trial sets
# TODO: different calibration schemes
# TODO: make input regions consistent between griddata and polynomial

image_z_values = [
    1900,
    1920,
    1940,
    1960,
    1980,
    2000,
]
image_z_average = 1950

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_file', default='model.json')
    parser.add_argument('--data_file', default='data.json')
    parser.add_argument('--config', default='cal_config_full.json')
    args = parser.parse_args()

    with open(f'{config_dir}/{args.config}') as f:
        config = json.load(f)

    peaks_x_out = np.array([])
    peaks_y_out = np.array([])
    peaks_z_out = np.array([])
    peaks = []
    peaks_poly = []
    peaks_test = []
    # Firstly, build the features for the model
    for i, z in enumerate(image_z_values):
        if z not in config['train_z'] and z not in config['test_z']:
            continue
        _xyxy, _polynomials, _x_out, _y_out, _z_out = build_features(z, image_z_average,
            grid_height, grid_length, grid_spacing)
        include = np.random.choice(a=[True, False], size=len(_xyxy),
            p=[1 - config['exclusion_ratio'], config['exclusion_ratio']])
        train_peaks = _xyxy[include]
        train_polynomials = _polynomials[include]
        train_x_out = _x_out[include]
        train_y_out = _y_out[include]
        train_z_out = _z_out[include]
        if config['test_excluded']:
            test_peaks = _xyxy[~include]
        else:
            test_peaks = _xyxy
        if z in config['train_z']:
            peaks.extend(train_peaks)
            peaks_poly.extend(train_polynomials)
            peaks_x_out = np.append(peaks_x_out, train_x_out)
            peaks_y_out = np.append(peaks_y_out, train_y_out)
            peaks_z_out = np.append(peaks_z_out, train_z_out)
        if z in config['test_z']:
            peaks_test.extend(test_peaks)

    # Save the features as well
    if args.data_file is None:
        print(f'No data file specified; skipping save to file.')
    else:
        with open(f'{data_dir}/{args.data_file}', 'w') as f:
            output_data = {
                'train_points': peaks,
                'test_points': peaks_test,
                'x_values': list(peaks_x_out),
                'y_values': list(peaks_y_out),
                'z_values': list(peaks_z_out),
            }
            json.dump(output_data, f)
            print(f'Data saved to file {args.data_file}')

    # Least squares model for parameters
    peaks_poly = np.array(peaks_poly)
    x_reg = LinearRegression(fit_intercept=False).fit(peaks_poly, peaks_x_out)
    y_reg = LinearRegression(fit_intercept=False).fit(peaks_poly, peaks_y_out)
    z_reg = LinearRegression(fit_intercept=False).fit(peaks_poly, peaks_z_out)
    # Add constant terms back on to model
    model = {
        'x': [0, *list(x_reg.coef_)],
        'y': [grid_spacing * (grid_height - 1) / 2 , *list(y_reg.coef_)],
        'z': [image_z_average, *list(z_reg.coef_)],
    }

    # Save the model
    if args.model_file is None:
        print(f'No model file specified; skipping save to file.')
    else:
        with open(f'{data_dir}/{args.model_file}', 'w') as f:
            json.dump(model, f, indent=2)
            print(f'Model saved to file {args.model_file}')

    with open(f'{data_dir}/{args.model_input_file}') as f:
        model = json.load(f)
    def model_func(x_l, y_l, x_r, y_r):
        return (
            model_func_generator(model['x'])(x_l, y_l, x_r, y_r),
            model_func_generator(model['y'])(x_l, y_l, x_r, y_r),
            model_func_generator(model['z'])(x_l, y_l, x_r, y_r),
        )
    vec_model_func = np.vectorize(model_func)

    with open(f'{data_dir}/{args.data_file}') as f:
        data = json.load(f)
    test_points = data['test_points']
    train_points = data['train_points']
    z_values = data['z_values']

    # Pack the z-values into a grid
    z_list = scipy_griddata(
        points=np.array(train_points),
        values=np.array(z_values),
        xi=test_points,
        fill_value=0,
    )

    z_list = []
    for i, xyxy in enumerate(test_points):
        x_l, y_l, x_r, y_r = xyxy
        x, y, z = model_func(x_l, y_l, x_r, y_r)
        z_list.append(z)
    x_linspace = np.linspace(-500, 500, grid_shape[1])
    y_linspace = np.linspace(-400, 400, grid_shape[0])
    x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)
    z_grid = vec_model_func(x_grid, y_grid, x_grid + dp_x_grid,
        y_grid + dp_y_grid)

    print(f'Mean value: {np.mean(z_list)}')
    print(f'Median value: {np.median(z_list)}')

    z_grid_model = []
    prev_y_l = None
    for i, xyxy in enumerate(test_points):
        x_l, y_l, x_r, y_r = xyxy
        z = z_list[i]
        if prev_y_l is None or np.abs(y_l - prev_y_l) > 10:
            z_grid.append([])
        z_grid[len(z_grid) - 1].append(z)
        prev_y_l = y_l

    figure = plt.figure()
    figure.add_subplot(1, 2, 1)
    plt.imshow(z_grid_interp)
    figure.add_subplot(1, 2, 2)
    plt.imshow(z_grid_model)
    plt.show()

if __name__ == '__main__':
    main()