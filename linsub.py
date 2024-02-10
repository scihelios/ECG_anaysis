import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.optimize as opt
import imageio as io
import copy
from scipy.signal import butter, filtfilt


def substract_linear(signal, pas):
    a = np.mean(signal[:pas])
    b = np.mean(signal[-pas:])
    for i in range(len(signal)):
        signal[i] -= a + ((b-a)/len(signal))*i
    return(signal)