'''Script for cross-correlating two 1D test signals then plotting
these plus the output.
'''
import numpy as np
from correlation import cross_correlate_1d, cross_correlate_1d_raw
from correlation_spectral import cross_correlate_1d_spectral
from matplotlib import pyplot as plt
from datetime import datetime

# STEP_SIZE = np.pi / 5000
STEP_SIZE = np.pi / 2000

def main():
    start = 0
    end = 4 * np.pi
    x_vec = np.arange(start,  end + STEP_SIZE, STEP_SIZE)
    output_x_vec = np.arange(0, 2 * end + STEP_SIZE, STEP_SIZE)

    template = np.cos(x_vec)
    signal = -np.sin(x_vec)

    figure, axis = plt.subplots(5)

    axis[0].plot(x_vec, signal, label='signal')
    axis[0].plot(x_vec, template, label='template')
    axis[0].legend()
    axis[0].set_title('signal and template')

    start = datetime.now()
    custom_correlation = cross_correlate_1d_raw(template, signal)
    print(f'Time elapsed (raw custom correlation) : {datetime.now() - start}')
    axis[1].plot(output_x_vec, custom_correlation)
    axis[1].set_title('raw custom correlation')

    start = datetime.now()
    custom_correlation = cross_correlate_1d(template, signal)
    print(f'Time elapsed (vectorised custom correlation) : {datetime.now() - start}')
    axis[2].plot(output_x_vec, custom_correlation)
    axis[2].set_title('vectorised custom correlation')

    start = datetime.now()
    spectral_correlation = cross_correlate_1d_spectral(template, signal)
    print(f'Time elapsed (spectral correlation) : {datetime.now() - start}')
    axis[3].plot(output_x_vec, spectral_correlation)
    axis[3].set_title('spectral correlation')

    start = datetime.now()
    numpy_correlation = np.correlate(
        (template - np.mean(template)) / (np.std(template) * template.size),
        (signal - np.mean(signal)) / np.std(signal),
        mode='full',
    )
    print(f'Time elapsed (Numpy correlation) : {datetime.now() - start}')
    axis[4].plot(output_x_vec, numpy_correlation)
    axis[4].set_title('numpy "full" correlation')

    plt.show()

if __name__ == '__main__':
    main()