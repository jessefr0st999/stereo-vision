import numpy as np
import pandas as pd
from correlation import cross_correlate_1d
from correlation_spectral import cross_correlate_1d_spectral
from matplotlib import pyplot as plt
from datetime import datetime
import argparse

SIGNAL_FREQ = 44_000
SIGNAL_SPEED = 333

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectral', action='store_true', default=False)
    args = parser.parse_args()

    # Read data into Pandas dataframes, leveraging the fact that
    # the data is contained within a single column
    with open('data/sensor1Data.txt') as f:
        df_signal_1 = pd.read_csv(f).squeeze()
    with open('data/sensor2Data.txt') as f:
        df_signal_2 = pd.read_csv(f).squeeze()

    signal_1 = df_signal_1.to_numpy()
    signal_2 = df_signal_2.to_numpy()

    _, axis = plt.subplots(1)

    start = datetime.now()
    if args.spectral:
        correlation = cross_correlate_1d_spectral(signal_1, signal_2)
    else:
        correlation = cross_correlate_1d(signal_1, signal_2)
    print(f'Time elapsed : {datetime.now() - start}')

    max_pos = np.argmax(correlation)
    shift_num = np.abs(max_pos - len(signal_1))
    shift_time = shift_num / SIGNAL_FREQ
    shift_distance = shift_time * SIGNAL_SPEED
    print(f'Max at: {max_pos}')
    print(f'Max value: {correlation[max_pos]}')
    print(f'Number of signals shifted: {shift_num}')
    print(f'Signal shift time: {shift_time}')
    print(f'Signal shift distance: {shift_distance}')

    axis.plot(correlation)
    axis.set_title('signal correlation')
    plt.show()

if __name__ == '__main__':
    main()