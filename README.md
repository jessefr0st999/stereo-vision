# stereo-vision
Instruction manual for usage of the `stereo-vision` project.


## Setup Procedure
1. Create a virtual environment in which to install packages:
```
python -m venv .venv
```

2. Enter the virtual environment:
```
.\.venv\Scripts\activate
```

3. Install Python dependencies:
```
pip install -r requirements.txt
```

4. Test/develop your heart out!


## General Structure
This repository follows a flat structure, with all scripts in the root directory.

The code is generally kept in files/modules of two types:
- "Function modules"; contain constant variables and functions to be imported and called elsewhere. Note that object oriented programming has not been leveraged in this project; if it were, classes would also exist in these modules.
- "Driver scripts"; to be run from the command line, possibly with flags. Involve a sequence of commands in a main function. Generally import and call functions from the "function modules".

Refer to the Driver Scripts subsection for tips on how to use this project.


## Function Modules
### correlation.py
Developed as part of sections 1.1 and 1.3 of this project.

For section 1.1, contains functions for calculating the (optionally normalised) cross-correlation between two vectors of equal length *as a vector* i.e. by "sliding" one vector over the other.

One of these functions uses raw Python for looped operations, while the other uses Numpy arrays for this purpose, leading to much better performance.

These processes are slow, so logs are produced regularly (every 1000 correlations) to keep the user informed on progress.

For section 1.3, contains a function for computing the 2D normalised cross-correlation from scratch.

### correlation_spectral.py
Developed for section 1.5. Contains spectral analogues of the above 1D and 2D cross-correlators, using Fourier transforms and convolutions instead of looped operations.

### correlation_numba.py
Developed for section 1.6. Contains versions of the 1D raw and vectorised cross-correlators but decorated to use with the Numba just-in-time compiler (`njit`).

Code in the functions here is very similar to `correlation.py` but differs slightly so as to be compatible with Numba.

### sv_calibration.py
Developed for sections 2.1 and 2.2. Contains a function for reading in a pair of calibration images, detecting the peaks in these images then building Numpy arrays of features to be used for calibration modelling.

Also contains a function of model parameters which itself returns another function configured to use these parameters.

### sv_image_comparison.py
Developed for sections 2.3 and 2.4. Contains a function when runs the image comparison scanning procedure for a particular scanning sequence, based on a specified configuration. Several helper functions used in this sequence procedure are the following:
- Function for retrieving the search regions in a target image based on a window in the template image and region scheme.
- Function for performing the above for later stages in a multi-pass scan sequence.
- Functions for retreiving the midpoints of templates/regions for each of the above.
- Function for actually performing an image scan using a previously-constructed template window with target search regions. Returns the midpoints of the search region at which the maximum correlation was obtained.
- Function for getting the boundaries of a given window, validating that it does not include values from outside the image on which it was placed.

Also has a function for plotting the output of a given sequence in the format displayed in the report.

### utils.py
Contains several utility functions used across the codebase, not specific to any particular module.


## Driver Scripts
### test_1d_xcorr.py
Used for testing the 1-dimensional cross-correlation functions on the signals specified in section 1.1. These functions and the step sizes may be configured using the constants at the top of the script. Displays several plots.

No command line flags supported.

### signal_offset.py
Deploys the 1D cross-correlation on the signal data for section 1.2. Prints relevant results to the terminal. Displays several plots.

Supports the command line flag `--spectral` for optionally using the spectral variety.

### test_2d_xcorr.py
Used for testing the 2D cross-correlation functions for section 1.3. Defaults to run on the Rocket Man images (section 1.4).

Supports the following command line flags:
- `--spectral` for optionally using the spectral variety.
- `--template` and `--region` for specifying the images to be correlated. Must be in the folder 'images-p1'.
- `--step` for the step/stride distance if using non-spectral cross-correlation.

### numba_tests.py
Used for section 1.6; compares the performance of the functions in `correlation.py` with the corresponding Numba-optimised varieties in `correlation_numba.py`.

No command line flags supported.

### music_patterns_africa.py
Used for section 1.7; deploys the spectral 1D correlation on several audio files for Africa by Toto and plots the results.

No command line flags supported.

### test_calibration.py
Used for sections 2.1 and 2.2; uses code from `sv_calibration.py` to build features for model training, before generating a polynomial model (coefficients for which are saved to file) and interpolation models (linear and nearest-neighbour). Each of these are then tested with specified sets of test points, with error metrics printed to terminal and plots of the outputs.

Supports the following command line flags:
- `--model_file` for saving the polynomial model coefficients to file.
- `--data_file`; likewise for the train/test features and labels.
- `--config`; see below:

Configs should be supplied to the script in the following format:
```json
{
  "train_z": [
    1900,
    1920,
    1960,
    1980,
    2000
  ],
  "test_z": [
    1940
  ],
  "exclusion_ratio": 0,
  "test_excluded": false
}
```
A calibration run with this configuration trains the polynomial and griddata models using images with z-values in the `train_z` array, before testing using points from the z = 1940 image. This calibration does not exclude random points from training.

### test_image_comparison.py
Used for sections 2.3 and 2.4; calls the `sequence_scan` function in `sv_image_comparison.py` for each sequence specified in the configuration, builds up a grid of resulting dp_x, dp_y and shift magnitude values then saves these to file. Plots the results at the end of each sequence as well as the total depth grid after all sequences have completed.

Supports the following command line flags:
- `--images` for specifying the appropriately named (prefixed with `left_` and `right_`) image pair to compare (within the `images-p2-uncal` folder)..
- `--ds_factor` for specifying the factor by which to downsmaple the images for quicker testing. Defaults to 1.
- `--config` for... specifying the config fie.
- `--depth_output` for specifying the name of the file to which to output depth and disparity data. If none, no data is output.
- `--depth_input` for specifying a file output by the above flag. Short circuits the sequence scan procedure; useful for quickly testing adjustments to outputs and plots.
- `--shift_plot_type`; either "boxes" or "arrows". Arrows look better for multi-pass plots, generally boxes otherwise.
- `--sequence_plots` for showing plots at the end of each sequence. These plots are hidden by default.
- `--hide_depth` for hiding the depth plots. These plots are shown by default.

Configs should be supplied to the script in the following format:
```json
[
  [
    {
      "window_width": 40,
      "window_height": 40,
      "scheme": [5, 1],
      "scheme_shift_size": [0, 0],
      "window_overlap": 0,
      "correlation_threshold": 0.9
    },
    {
      "factor": 2
    }
  ],
  [
    {
      "window_width": 20,
      "window_height": 20,
      "scheme": [5, 1],
      "scheme_shift_size": [10, 0],
      "window_overlap": 0,
      "correlation_threshold": 0.9
    }
  ],
  [
    {
      "window_width": 10,
      "window_height": 10,
      "scheme": [5, 1],
      "scheme_shift_size": [1, 0],
      "window_overlap": 0,
      "correlation_threshold": 0.9
    }
  ]
]
```
Note the structure of the config; an array of arrays, with items in the first level defining a sequence and items in the second defining a stage.

This particular config defines a search procedure with 3 sequences; the first sequence is multi-pass (with 2 stages) while the second and third are single-pass.

Refer to the report for the effect of other config fields.


### test_scan_calibrated.py
Used for section 2.5; combines results obtained from `test_image_comparison.py` with a specified calibration type, using data obtained from `test_calibration.py`. Produces a 3D grid of resulting z-values alongside the original left image.

Supports the following command line flags:
- `--images` for specifying the appropriately named (prefixed with `left_` and `right_`) image pair to compare (within the `images-p2-uncal` folder).
- `--cal_type` for specifying the calibration type; either "polynomial", "linear" or "nearest".
- `--model_input` for the model data file (within the `calibration-data` folder).
- `--cal_data_input` for the calibration data file (within the `calibration-data` folder).
- `--depth_input` for the depth data file (within the `depth-data` folder).