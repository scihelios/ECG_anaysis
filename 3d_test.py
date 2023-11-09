
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import random
def random_color():
    return (random.random(), random.random(), random.random())

def load_signals(filename):
    data = np.fromfile(filename, dtype=np.int16)
    raw_signal = data[::2]
    return raw_signal

def extract_cycles(signal):
    peaks, _ = find_peaks(signal, distance=100, height=70)
    cycles = [signal[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]
    return cycles


def compute_fft(cycle, max_len, sampling_rate=500):
    cycle_padded = np.pad(cycle, (0, max_len - len(cycle)), 'constant')
    N = len(cycle_padded)
    fft_vals = np.fft.fft(cycle_padded)
    magnitude = np.abs(fft_vals) / N
    freqs = np.fft.fftfreq(N, 1 / sampling_rate)
    idx = np.where(freqs >= 100)[0][0]
    return freqs[:idx], magnitude[:idx]

# Extract cycles
raw_signal = load_signals('rec_1.dat')
cycles = extract_cycles(raw_signal)

# Compute the maximum cycle length
max_cycle_len = max([len(cycle) for cycle in cycles])

# Compute FFTs and gather magnitudes for each frequency
magnitudes_for_freqs = []

for cycle in cycles:
    if len(magnitudes_for_freqs) == 0:  # First iteration
        freqs, magnitudes = compute_fft(cycle, max_cycle_len)
        magnitudes_for_freqs = [[] for _ in magnitudes]
    
    _, magnitudes = compute_fft(cycle, max_cycle_len)
    for i, magnitude in enumerate(magnitudes):
        magnitudes_for_freqs[i].append(magnitude)


def bin_frequencies(freqs, magnitudes):
    bins = np.arange(0, 101, 10)  # bins: 0-10Hz, 10-20Hz, ..., 90-100Hz
    binned_magnitudes = []

    for i in range(len(bins) - 1):
        start_freq = bins[i]
        end_freq = bins[i+1]
        indices = np.where((freqs >= start_freq) & (freqs < end_freq))[0]
        summed_magnitude = np.sum([magnitudes_for_freqs[j] for j in indices], axis=0)
        binned_magnitudes.append(summed_magnitude)

    return bins[:-1], binned_magnitudes

# Bin the frequencies
binned_freqs, binned_magnitudes = bin_frequencies(freqs, magnitudes_for_freqs)

# Create a 3D histogram
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define bin edges for amplitudes
magnitude_bins = np.linspace(0, max([max(mags) for mags in binned_magnitudes]), 30)

# Create 3D histogram
for idx, freq_bin in enumerate(binned_freqs):
    hist, _ = np.histogram(binned_magnitudes[idx], bins=magnitude_bins)
    x = [freq_bin] * len(hist)
    y = magnitude_bins[:-1]
    z = np.zeros_like(hist)
    dx = 5  # Width of bars
    dy = magnitude_bins[1] - magnitude_bins[0]
    dz = hist
    ax.bar3d(x, y, z, dx, dy, dz, shade=True)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_zlabel('Count')
plt.show()