import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.optimize as opt
import imageio as io
import copy
from scipy.signal import butter, filtfilt

def fonction_affine(signal, pas):
    a = np.mean(signal[:pas])
    b = np.mean(signal[-pas:])
    affine = np.zeros(len(signal))
    n = len(signal)
    for i in range(len(signal)):
        affine[i] = a + ((b-a)/n)*i
    return(affine)

def substract_linear(signal, pas):
    a = np.mean(signal[:pas])
    b = np.mean(signal[-pas:])
    n = len(signal)
    for i in range(len(signal)):
        signal[i] -= a + ((b-a)/n)*i
    return(signal)