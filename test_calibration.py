'''Script for stereo vision image comparison...
'''
from datetime import datetime
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

    x_labels = np.array([])
    y_labels = np.array([])
    z_labels = np.array([])
    z_test_labels = np.array([])
    peaks_poly = []
    train_features = []
    test_features = []
    # Firstly, build the features for the model
    start = datetime.now()
    for i, z in enumerate(image_z_values):
        if z not in config['train_z'] and z not in config['test_z']:
            continue
        _xyxy, _polynomials, _x_out, _y_out, _z_out = build_features(z, image_z_average,
            grid_height, grid_length, grid_spacing)
        # Include a random subset of the features for training as specified in config
        include = np.random.choice(a=[True, False], size=len(_xyxy),
            p=[1 - config['exclusion_ratio'], config['exclusion_ratio']])
        if z in config['train_z']:
            train_features.extend(_xyxy[include].tolist())
            peaks_poly.extend(_polynomials[include].tolist())
            x_labels = np.append(x_labels, _x_out[include])
            y_labels = np.append(y_labels, _y_out[include])
            z_labels = np.append(z_labels, _z_out[include])
        if z in config['test_z']:
            if config['test_excluded']:
                test_features.extend(_xyxy[~include].tolist())
                z_test_labels = np.append(z_test_labels, _z_out[~include])
            else:
                test_features.extend(_xyxy.tolist())
                z_test_labels = np.append(z_test_labels, _z_out)
    print(f'Time elapsed (peak detection and feature construction): {datetime.now() - start}')

    # Save the features to file
    if args.data_file is None:
        print(f'No data file specified; skipping save to file.')
    else:
        with open(f'{data_dir}/{args.data_file}', 'w') as f:
            output_data = {
                'train_features': train_features,
                'test_features': test_features,
                'x_labels': list(x_labels),
                'y_labels': list(y_labels),
                'z_labels': list(z_labels),
            }
            json.dump(output_data, f, indent=2)
            print(f'Data saved to file {args.data_file}')

    # Next, build the least squares model for parameters
    peaks_poly = np.array(peaks_poly)
    x_reg = LinearRegression(fit_intercept=False).fit(peaks_poly, x_labels)
    y_reg = LinearRegression(fit_intercept=False).fit(peaks_poly, y_labels)
    z_reg = LinearRegression(fit_intercept=False).fit(peaks_poly, z_labels)

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

    def print_error_metrics(outputs, labels, model_name):
        _max = max(outputs - labels)
        _min = min(outputs - labels)
        _mean = np.mean(np.abs(outputs - labels))
        print(f'\nStatistical summary for {model_name}:')
        print(f'Mean absolute error: {_mean}')
        print(f'Range of errors: [{_min}, {_max}]')

    # Run the testing procedure for griddata interpolation
    griddata_kwargs = {
        'points': np.array(train_features),
        'xi': test_features,
        'fill_value': 0,
    }
    z_list_linear = scipy_griddata(values=np.array(z_labels),
        method='linear', **griddata_kwargs)
    print_error_metrics(z_list_linear, z_test_labels, 'scipy griddata (linear)')

    z_list_nearest = scipy_griddata(values=np.array(z_labels),
        method='nearest', **griddata_kwargs)
    print_error_metrics(z_list_nearest, z_test_labels, 'scipy griddata (nearest)')

    # Run the testing procedure for the polynomial model
    def model_test_func(xyxy):
        [x_l, y_l, x_r, y_r] = xyxy
        return model_func_generator(model['z'])(x_l, y_l, x_r, y_r) - image_z_average
    z_list_model = [model_test_func(xyxy) for xyxy in test_features]
    print_error_metrics(z_list_model, z_test_labels, 'polynomial model')

    if len(config['test_z']) > 1:
        print('More than one test z-value specified in config; skipping plotting.')
        return

    # Pack the outputs into a grid and plot
    grid_shape = (grid_length, grid_height)
    z_grid_linear = np.reshape(z_list_linear, grid_shape)
    z_grid_nearest = np.reshape(z_list_nearest, grid_shape)
    z_grid_model = np.reshape(z_list_model, grid_shape)

    z_label = config['test_z'][0] - image_z_average

    figure, axes = plt.subplots(1, 3)
    axes[0].set_title(f'Scipy griddata; linear interpolation (target: {z_label})')
    axes[0].imshow(z_grid_linear)
    axes[1].set_title(f'Scipy griddata; nearest-neighbour interpolation (target: {z_label})')
    axes[1].imshow(z_grid_nearest)
    axes[2].set_title(f'Polynomial model interpolation (target: {z_label})')
    axes[2].imshow(z_grid_model)
    plt.show()

if __name__ == '__main__':
    main()