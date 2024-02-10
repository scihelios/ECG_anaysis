import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.optimize as opt
import imageio as io
import copy
from scipy.signal import butter, filtfilt





import numpy as np
from scipy.signal import butter, filtfilt

"""
a priori ici il y a un signal par seconde arbitraire

fs = 1

"""

def bandpass_filter(signal, fs, lowcut, highcut):
    """
    Apply a bandpass filter to a signal.

    Parameters:
    signal (numpy array): The input signal.
    fs (float): Sampling rate of the signal.
    lowcut (float): Lower frequency bound of the filter in Hz.
    highcut (float): Upper frequency bound of the filter in Hz.

    Returns:
    num py array: Filtered signal.
    """
    # Design the Butterworth bandpass filter
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=2, Wn=[low, high], btype='band')

    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def lowpass_filter(signal, fs = 1, cutoff = 0.025):
    """
    Appliquer un filtre passe-bas à un signal.

    Paramètres :
    signal (numpy array) : Le signal d'entrée.
    fs (float) : Fréquence d'échantillonnage du signal.
    cutoff (float) : Fréquence de coupure du filtre passe-bas en Hz.

    Retour :
    numpy array : Signal filtré.
    """
    # Conception du filtre passe-bas Butterworth
    nyq = 0.5 * fs  # Fréquence de Nyquist
    normal_cutoff = cutoff / nyq
    b, a = butter(N=2, Wn=normal_cutoff, btype='low')

    # Appliquer le filtre au signal
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def highpass_filter(signal, fs = 1, cutoff = 0.025):
    """
    Appliquer un filtre passe-haut à un signal.

    Paramètres :
    signal (numpy array) : Le signal d'entrée.
    fs (float) : Fréquence d'échantillonnage du signal.
    cutoff (float) : Fréquence de coupure du filtre passe-haut en Hz.

    Retour :
    numpy array : Signal filtré.
    """
    # Conception du filtre passe-haut Butterworth
    nyq = 0.5 * fs  # Fréquence de Nyquist
    normal_cutoff = cutoff / nyq
    b, a = butter(N=2, Wn=normal_cutoff, btype='high')

    # Appliquer le filtre au signal
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def median_filter(signal, kernel_size=3):
    """
    Appliquer un filtre médian à un signal.

    Paramètres :
    signal (numpy array) : Le signal d'entrée.

    Retour :
    numpy array : Signal filtré.
    """
    return sig.medfilt(signal, kernel_size=kernel_size)
