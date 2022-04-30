'''Script for testing the effect of the Numba just-in-time compiler on the
raw custom cross-correlation functions.
'''
import numpy as np
from correlation import cross_correlate_1d_raw
from correlation_numba import cross_correlate_1d_raw_numba
from datetime import datetime

STEP_SIZE = np.pi / 1000

def main():
    template_x_vec = np.arange(0, 4 * np.pi + STEP_SIZE, STEP_SIZE)
    signal_x_vec = np.arange(0, 4 * np.pi + STEP_SIZE, STEP_SIZE)

    template = np.sin(template_x_vec)
    signal = -np.sin(signal_x_vec)

    start = datetime.now()
    _ = cross_correlate_1d_raw(template, signal)
    print(f'Raw python, without Numba JIT; time elapsed: {datetime.now() - start}')

    start = datetime.now()
    _ = cross_correlate_1d_raw_numba(template, signal)
    print(f'Raw python, with Numba JIT; time elapsed: {datetime.now() - start}')

if __name__ == '__main__':
    main()