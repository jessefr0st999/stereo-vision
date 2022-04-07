import numpy as np
from correlation_1d import cross_correlate_1d as custom_correlate
from matplotlib import pyplot as plt

STEP_SIZE = np.pi / 10

def main():
    x_vec = np.arange(0, 4 * np.pi + STEP_SIZE, STEP_SIZE)
    correlation_x_vec = np.arange(0, 8 * np.pi + STEP_SIZE, STEP_SIZE)
    f_vec = np.sin(x_vec)
    g_vec = -np.sin(x_vec)

    figure, axis = plt.subplots(3)

    custom_correlation = custom_correlate(f_vec, g_vec, normalised=True)
    axis[0].plot(correlation_x_vec, custom_correlation)
    axis[0].set_title('custom correlation')

    numpy_correlation = np.correlate(f_vec, g_vec, mode='full')
    axis[1].plot(correlation_x_vec, numpy_correlation)
    axis[1].set_title('numpy "full" correlation')

    numpy_correlation = np.correlate(f_vec, g_vec, mode='same')
    axis[2].plot(x_vec, numpy_correlation)
    axis[2].set_title('numpy "same" correlation')

    plt.show()

if __name__ == '__main__':
    main()