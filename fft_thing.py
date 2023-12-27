import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define the sampling rate (in Hz)
sampling_rate = 500  # The sampling rate of the original ECG data is 500 Hz

src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/treated_ECG")
count = 0

for signal_file in os.listdir(src_directory):
    if signal_file.endswith('.json'):
        with open(src_directory / signal_file, "r") as json_file:
            data = json.load(json_file)
        
        your_signal = data['signal']  # The ECG signal

        # Perform FFT
        fft_result = np.fft.fft(your_signal)
        # Frequency values in Hertz
        freqs = np.fft.fftfreq(len(your_signal), 1 / sampling_rate)

        # Perform Inverse FFT to reconstruct the signal
        reconstructed_signal = np.fft.ifft(fft_result)
        reconstructed_signal = reconstructed_signal.real  # Take the real part

        freqs = np.fft.fftfreq(len(your_signal), 1 / sampling_rate)
        positive_freq_indices = np.where(freqs >= 0)  # Indices of positive frequencies
        positive_freqs = freqs[positive_freq_indices]
        positive_fft_result = np.abs(fft_result)[positive_freq_indices]

        # Plotting
        plt.figure(figsize=(18, 6))

        # Plot the original signal
        plt.subplot(1, 3, 1)
        plt.plot(your_signal)
        plt.title("Original Signal")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude")

        # Plot the magnitude of FFT for frequencies between 0 and 400 Hz
        plt.subplot(1, 3, 2)
        # We limit the x-axis from 0 to 400 Hz, although the Nyquist frequency is 250 Hz
        plt.plot(positive_freqs, positive_fft_result)
        plt.title("Magnitude of FFT")
        plt.xlabel("Frequency (Hz)")
        plt.xlim(0, 400)  # Limit x-axis to 400 Hz
        plt.yscale("log")
        plt.ylabel("Magnitude")

        # Plot the reconstructed signal
        plt.subplot(1, 3, 3)
        plt.plot(reconstructed_signal)
        plt.title("Reconstructed Signal")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()