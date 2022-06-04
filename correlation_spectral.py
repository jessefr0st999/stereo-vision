import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2

# Minimum standard deviation above which a region is considered non-homogeneous,
# used to suppress erroneous feature displacements in homogeneous regions
MIN_ST_DEV = 1e-4

# Used to determine whether a homogeneous region is identical enough to the region
# against which it is being compared
MIN_MEAN_DIFF = 1e-9

def cross_correlate_1d_spectral(template: np.ndarray, signal: np.ndarray):
    '''Computes the normalised cross-correlation between a 1D region and signal
    using (discrete) Fourier transforms and the convolution theorem.
    '''
    # Zero-pad to get a waveform with same phase as input
    padded_signal = np.array([*np.zeros(template.size - 1), *signal])

    # Normalise the template and signal
    template = template - np.mean(template)
    signal = signal - np.mean(signal)
    template = template / (np.std(template) * template.size)
    signal = signal / np.std(signal)

    size = signal.size + template.size - 1
    ft_template = fft(template, n=size)
    ft_signal = fft(padded_signal, n=size)

    correlated = np.real(ifft(np.conj(ft_template) * ft_signal))
    return correlated

def cross_correlate_2d_spectral(orig_template: np.ndarray, orig_region: np.ndarray):
    '''Computes the normalised cross-correlation between a 2D target region and template
    using (discrete) Fourier transforms and the convolution theorem.
    '''
    # Normalise the template and target region
    template = orig_template - np.mean(orig_template)
    region = orig_region - np.mean(orig_region)
    shape = region.shape

    # Handle cases where the standard deviations are low
    if np.std(orig_template) < MIN_ST_DEV or np.std(orig_template) < MIN_ST_DEV:
        if np.abs(np.mean(orig_template) - np.mean(orig_region)) < MIN_MEAN_DIFF:
            return np.ones(shape)
        else:
            return np.zeros(shape)
    else:
        region = region / np.std(region)
        template = template / (np.std(template) * template.size)

    ft_template = fft2(template, s=shape)
    ft_region = fft2(region, s=shape)

    correlated = np.real(ifft2(np.conj(ft_template) * ft_region))
    return correlated
