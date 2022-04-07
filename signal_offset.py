import numpy as np
import pandas as pd
from correlation_1d import cross_correlate_1d as custom_correlate
from matplotlib import pyplot as plt
from datetime import datetime

def main():
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
    # correlation = custom_correlate(signal_1, signal_2, normalised=True)
    correlation = np.correlate(signal_2, signal_1, mode='full')
    print(f'Took {datetime.now() - start} seconds')
    axis.plot(correlation)
    axis.set_title('signal correlation')
    plt.show()

if __name__ == '__main__':
    main()