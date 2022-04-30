import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, ifftshift
from matplotlib import pyplot as plt

def cross_correlate_1d_spectral(template: np.ndarray, signal: np.ndarray):
    '''
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

def cross_correlate_2d_spectral(template: np.ndarray, region: np.ndarray):
    '''
    '''
    if template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
        raise Exception('Dimensions of template must not exceed those of region.')

    # Normalise the template and signal
    template = template - np.mean(template)
    region = region - np.mean(region)
    template = template / (np.std(template) * template.size)
    region = region / np.std(region)

    shape = region.shape
    ft_template = fft2(template, s=shape)
    ft_region = fft2(region, s=shape)

    x_corr = np.real(ifft2(np.conj(ft_template) * ft_region))
    return x_corr
