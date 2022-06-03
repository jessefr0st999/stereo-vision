'''Script for finding occurrences of certain audio snippets in a
longer audio file using spectral cross-correlation.
'''
from correlation_spectral import cross_correlate_1d_spectral
from matplotlib import pyplot as plt
from scipy.io import wavfile

def main():
    _, signal = wavfile.read('audio/africa.wav')
    _, bass_template = wavfile.read('audio/africa_drum_1.wav')
    _, snare_template = wavfile.read('audio/africa_drum_2.wav')
    _, hurry_template = wavfile.read('audio/africa_hurry.wav')
    _, rains_template = wavfile.read('audio/africa_rains.wav')

    # africa_rains.wav and africa_hurry.wav are stereo (two channels)
    # Just use the left channel
    hurry_template = hurry_template[:, 0]
    rains_template = rains_template[:, 0]

    figure, axes = plt.subplots(2, 2)

    bass_correlation = cross_correlate_1d_spectral(signal, bass_template)
    axes[0, 0].plot(bass_correlation)
    axes[0, 0].set_title('Correlation with bass drum')

    snare_correlation = cross_correlate_1d_spectral(signal, snare_template)
    axes[0, 1].plot(snare_correlation)
    axes[0, 1].set_title('Correlation with snare drum')

    hurry_correlation = cross_correlate_1d_spectral(signal, hurry_template)
    axes[1, 0].plot(hurry_correlation)
    axes[1, 0].set_title('Correlation with "Hurry boy"')

    rains_correlation = cross_correlate_1d_spectral(signal, rains_template)
    axes[1, 1].plot(rains_correlation)
    axes[1, 1].set_title('Correlation with "I bless the rains down in Africa"')

    plt.show()

if __name__ == '__main__':
    main()