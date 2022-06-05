'''Script for finding occurrences of certain audio snippets in a
longer audio file using spectral cross-correlation.
'''
from correlation_spectral import cross_correlate_1d_spectral
from matplotlib import pyplot as plt
from scipy.io import wavfile
from datetime import datetime

def main():
    _, signal = wavfile.read('audio/africa.wav')
    _, verse_template = wavfile.read('audio/africa_first_verse.wav')
    _, snare_template = wavfile.read('audio/africa_drum_snare.wav')
    _, hurry_template = wavfile.read('audio/africa_hurry.wav')
    _, rains_template = wavfile.read('audio/africa_rains.wav')

    # africa_rains.wav and africa_hurry.wav are stereo (two channels)
    # Just use the left channel
    hurry_template = hurry_template[:, 0]
    rains_template = rains_template[:, 0]

    figure, axes = plt.subplots(2, 2)

    start = datetime.now()
    verse_correlation = cross_correlate_1d_spectral(signal, verse_template)
    axes[0, 0].plot(verse_correlation)
    axes[0, 0].set_title('Correlation with first verse')
    print(f'First verse correlation complete; time elapsed: {datetime.now() - start}')

    start = datetime.now()
    snare_correlation = cross_correlate_1d_spectral(signal, snare_template)
    axes[0, 1].plot(snare_correlation)
    axes[0, 1].set_title('Correlation with snare drum')
    print(f'Snare drum correlation complete; time elapsed: {datetime.now() - start}')

    start = datetime.now()
    hurry_correlation = cross_correlate_1d_spectral(signal, hurry_template)
    axes[1, 0].plot(hurry_correlation)
    axes[1, 0].set_title('Correlation with "Hurry boy"')
    print(f'"Hurry boy" correlation complete; time elapsed: {datetime.now() - start}')

    start = datetime.now()
    rains_correlation = cross_correlate_1d_spectral(signal, rains_template)
    axes[1, 1].plot(rains_correlation)
    axes[1, 1].set_title('Correlation with "I bless the rains down in Africa"')
    print(f'"Rains" correlation complete; time elapsed: {datetime.now() - start}')

    plt.show()

if __name__ == '__main__':
    main()