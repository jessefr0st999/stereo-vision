import numpy as np
from PIL import Image
from skimage.transform import resize
import math

def greyscale_with_nan(path):
    '''Reads an RGBA image and outputs a corresponding greyscale image, with
    all fully transparent pixels of the input image set to NaN in the output.
    Should only be used on an image which...
    '''
    rgba_image = np.asarray(Image.open(path))
    greyscale_image = np.asarray(Image.open(path).convert('L')).astype('float64')
    r, g, b, a = np.rollaxis(rgba_image, axis=-1)
    greyscale_image[a == 0] = np.nan
    return greyscale_image

def max_pos(array: np.ndarray, step_x=1, step_y=1):
    '''
    '''
    max_index = np.nanargmax(array)
    return (
        step_y * (max_index % array.shape[1]),
        step_x * (max_index // array.shape[1]),
    )

def downsample(image, downsample_factor=1):
    '''
    '''
    return resize(image, (
        math.floor(image.shape[0] / downsample_factor),
        math.floor(image.shape[1] / downsample_factor),
    ))