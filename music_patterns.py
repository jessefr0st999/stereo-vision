import numpy as np
from correlation_spectral import cross_correlate_1d_spectral
from matplotlib import pyplot as plt
from scipy.io import wavfile
import soundfile

def main():
    _, signal = wavfile.read('audio/creep.wav')
    _, snare_template = wavfile.read('audio/snare.wav')
    word_template, _ = soundfile.read('audio/whatever.flac')
    word2_template, _ = soundfile.read('audio/control.flac')

    # Some audio files are stereo (two channels)
    # Just read the left channel
    signal = signal[:, 0]
    snare_template = snare_template[:, 0]

    figure, axis = plt.subplots(3)

    snare_correlation = cross_correlate_1d_spectral(signal, snare_template)
    axis[0].plot(snare_correlation)
    axis[0].set_title('Correlation with snare drum')

    word_correlation = cross_correlate_1d_spectral(signal, word_template)
    axis[1].plot(word_correlation)
    axis[1].set_title('Correlation with "whatever"')

    word2_correlation = cross_correlate_1d_spectral(signal, word2_template)
    axis[2].plot(word2_correlation)
    axis[2].set_title('Correlation with "control"')

    plt.show()

if __name__ == '__main__':
    main()