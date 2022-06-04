'''Script for stereo vision calibration...
'''
import numpy as np
import json
from correlation_spectral import cross_correlate_2d_spectral
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from argparse import ArgumentParser

# TODO: peak detector in external helper function
# TODO: split test data generation into different function and file

# Dots are arranged in a 17 by 21 grid
grid_height = 17
grid_length = 21
grid_spacing = 50 # millimetres

image_z_values = [
    1900,
    1920,
    1940,
    1960,
    1980,
    2000,
    -1,
]
train_z = set([
    1900,
    # 1920,
    # 1940,
    # 1960,
    # 1980,
    # 2000,
])
test_z = set([
    # 1900,
    # 1920,
    # 1940,
    # 1960,
    1980,
    # 2000,
    # -1,
])
image_z_average = 1950
image_dir = 'images-p2-cal'
data_dir = 'calibration-data'

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_file', default=None)
    parser.add_argument('--data_file', default='data.json')
    args = parser.parse_args()

    peaks_x_out = np.array([])
    peaks_y_out = np.array([])
    peaks_z_out = np.array([])
    peaks_xyxy = []
    peaks_polynomials = []
    peaks_xyxy_test = []
    for i, z in enumerate(image_z_values):
        if z not in train_z and z not in test_z:
            continue
        _xyxy, _polynomials, _x_out, _y_out, _z_out = build_features(z, i == 0)
        if z in train_z:
            peaks_xyxy.extend(_xyxy)
            peaks_polynomials.extend(_polynomials)
            peaks_x_out = np.append(peaks_x_out, _x_out)
            peaks_y_out = np.append(peaks_y_out, _y_out)
            peaks_z_out = np.append(peaks_z_out, _z_out)
        if z in test_z:
            peaks_xyxy_test.extend(_xyxy)

    # Save the raw data
    if args.data_file is None:
        print(f'No data file specified; skipping save to file.')
    else:
        with open(f'{data_dir}/{args.data_file}', 'w') as f:
            output_data = {
                'train_points': peaks_xyxy,
                'test_points': peaks_xyxy_test,
                'x_values': list(peaks_x_out),
                'y_values': list(peaks_y_out),
                'z_values': list(peaks_z_out),
            }
            json.dump(output_data, f, indent=2)
            print(f'Data saved to file {args.data_file}')

    # Least squares model for parameters
    peaks_polynomials = np.array(peaks_polynomials)
    x_reg = LinearRegression(fit_intercept=False).fit(peaks_polynomials, peaks_x_out)
    y_reg = LinearRegression(fit_intercept=False).fit(peaks_polynomials, peaks_y_out)
    z_reg = LinearRegression(fit_intercept=False).fit(peaks_polynomials, peaks_z_out)
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

def build_features(z, plot=False):
    '''
    '''
    peaks_x_out = np.array([])
    peaks_y_out = np.array([])
    peaks_z_out = np.array([])
    peaks_xyxy = []
    peak_polynomials = []
    # Read in the left image and find the locations of its dots
    if z == -1:
        image_dir = 'images-p2-uncal'
        left_image_file = 'left_test_1_dots.tiff'
        right_image_file = 'right_test_1_dots.tiff'
    else:
        image_dir = 'images-p2-cal'
        left_image_file = f'cal_image_left_{str(z)}.tiff'
        right_image_file = f'cal_image_right_{str(z)}.tiff'

    left_image = Image.open(f'{image_dir}/{left_image_file}').convert('L')
    left_image = np.asarray(left_image)
    # neighborhood_size and threshold set via trial and error
    left_x_peaks, left_y_peaks = find_peaks(left_image, neighborhood_size=10, threshold=0.2)
    print(f'Peaks found for {left_image_file}')

    # Repeat for the right image
    right_image = Image.open(f'{image_dir}/{right_image_file}').convert('L')
    right_image = np.asarray(right_image)
    right_x_peaks, right_y_peaks = find_peaks(right_image, neighborhood_size=10, threshold=0.2)
    print(f'Peaks found for {right_image_file}')

    # Lists above ordered row by row, top-to-bottom
    # Left lists are left-to-right, right lists are right-to-left
    # Hence flip the rows in the right lists
    for i in range(grid_height):
        right_x_peaks[grid_length * i : grid_length * (i + 1)] = np.flip(
            right_x_peaks[grid_length * i : grid_length * (i + 1)])
        right_y_peaks[grid_length * i : grid_length * (i + 1)] = np.flip(
            right_y_peaks[grid_length * i : grid_length * (i + 1)])
    # Build the matrix of design variables
    # Add squares and combination terms for linear regression
    for lin_terms in zip(left_x_peaks, left_y_peaks, right_x_peaks, right_y_peaks):
        terms = list(lin_terms)
        combination_terms = [
            lin_terms[0] * lin_terms[1],
            lin_terms[0] * lin_terms[2],
            lin_terms[0] * lin_terms[3],
            lin_terms[1] * lin_terms[2],
            lin_terms[1] * lin_terms[3],
            lin_terms[2] * lin_terms[3],
        ]
        square_terms = [t ** 2 for t in lin_terms]
        terms.extend(combination_terms)
        terms.extend(square_terms)
        peaks_xyxy.append(lin_terms)
        peak_polynomials.append(terms)

    # Build (x, y, z) co-ordinates in actual space of the grid points
    # Centre them for modelling purposes and add the offset on as a known
    # intercept term afterwards
    for i in range(grid_height):
        for j in range(grid_length):
            x = grid_spacing * (j - (grid_length - 1) / 2)
            y = grid_spacing * ((grid_height - 1) / 2 - i)
            peaks_x_out = np.append(peaks_x_out, x)
            peaks_y_out = np.append(peaks_y_out, y)
            peaks_z_out = np.append(peaks_z_out, z - image_z_average)

    if plot:
        figure = plt.figure(figsize=(1, 2))
        figure.add_subplot(1, 2, 1)
        plt.imshow(left_image)
        plt.autoscale(False)
        figure.add_subplot(1, 2, 2)
        plt.imshow(left_image)
        plt.autoscale(False)
        plt.plot(left_x_peaks, left_y_peaks, 'ro')
        # plt.imshow(right_image)
        # plt.autoscale(False)
        # plt.plot(right_x_peaks, right_y_peaks, 'ro')
        plt.show()

    return peaks_xyxy, peak_polynomials, peaks_x_out, peaks_y_out, peaks_z_out

def find_peaks(region, neighborhood_size, threshold):
    '''First, construct a Gaussian for detecting the exact positions of dots in the image.
    Next, use a 2D peak detection algorithm to get the exact locations of the maxima.
    Obtained from here:
    https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
    '''
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 101), np.linspace(-5, 5, 101))
    template = gaussian(x_grid, y_grid)
    data = cross_correlate_2d_spectral(template, region)

    data_max = ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = ndimage.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2
        y.append(y_center)
    x = np.add(x, template.shape[1] / 2)
    y = np.add(y, template.shape[0] / 2)
    return x, y

def model_func_generator(model_params):
    '''Helper for returning a function which uses the given model parameters.
    '''
    def model_func(x_l, y_l, x_r, y_r):
        # Constant term
        output = model_params[0]
        # Linear terms
        output += \
            model_params[1] * x_l + \
            model_params[2] * y_l + \
            model_params[3] * x_r + \
            model_params[4] * y_r
        # Combination terms
        output += \
            model_params[5] * x_l * y_l + \
            model_params[6] * x_l * x_r + \
            model_params[7] * x_l * y_r + \
            model_params[8] * y_l * x_r + \
            model_params[9] * y_l * y_r + \
            model_params[10] * x_r * y_r
        # Square terms
        output += \
            model_params[11] * x_l ** 2 + \
            model_params[12] * y_l ** 2 + \
            model_params[13] * x_r ** 2 + \
            model_params[14] * y_r ** 2
        return output
    return model_func

def gaussian(x, y):
    exponent = -(np.power(x, 2) + np.power(y, 2))
    return np.exp(exponent)

if __name__ == '__main__':
    main()