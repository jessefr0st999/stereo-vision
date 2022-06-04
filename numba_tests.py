'''Script for testing the effect of the Numba just-in-time compiler on the
custom cross-correlation functions.
'''
import numpy as np
from correlation import cross_correlate_1d, cross_correlate_1d_raw
from correlation_numba import cross_correlate_1d_numba, cross_correlate_1d_raw_numba
from datetime import datetime

RAW_STEP_SIZE = np.pi / 1000
VEC_STEP_SIZE = np.pi / 20000

def main():
    # First, run the tests on the raw Python correlation functions
    template_x_vec = np.arange(0, 4 * np.pi + RAW_STEP_SIZE, RAW_STEP_SIZE)
    signal_x_vec = np.arange(0, 4 * np.pi + RAW_STEP_SIZE, RAW_STEP_SIZE)

    template = np.sin(template_x_vec)
    signal = -np.sin(signal_x_vec)

    start = datetime.now()
    _ = cross_correlate_1d_raw(template, signal)
    print(f'Raw python, without Numba JIT; time elapsed: {datetime.now() - start}')

    start = datetime.now()
    _ = cross_correlate_1d_raw_numba(template, signal)
    print(f'Raw python, with Numba JIT; time elapsed: {datetime.now() - start}')


    # Now repeat the tests for the vectorised functions
    template_x_vec = np.arange(0, 4 * np.pi + VEC_STEP_SIZE, VEC_STEP_SIZE)
    signal_x_vec = np.arange(0, 4 * np.pi + VEC_STEP_SIZE, VEC_STEP_SIZE)

    template = np.sin(template_x_vec)
    signal = -np.sin(signal_x_vec)

    start = datetime.now()
    _ = cross_correlate_1d(template, signal)
    print(f'Vectorisation, without Numba JIT; time elapsed: {datetime.now() - start}')

    start = datetime.now()
    _ = cross_correlate_1d_numba(template, signal)
    print(f'Vectorisation, with Numba JIT; time elapsed: {datetime.now() - start}')

if __name__ == '__main__':
    main()