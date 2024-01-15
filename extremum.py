
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.optimize as opt
import imageio as io
import copy
import filter as flt
import linsub as ls
import time
import parametres as par

learning_rate = {
    'Amplitude' : [3],
    'Centre' : [0.0001],
    'Ecart-type' : [0.001]
}

def ext(arr):
    """
    Fonction qui trouve les extrema d'un signal
    : return: liste des indices des extrema, liste des valeurs des extrema
    """
    extrema_indices = []
    extrema_values = []

    # On détecte tos 
    for i in range(1, len(arr) - 1):
        if arr[i-1] > arr[i] < arr[i+1] or arr[i-1] < arr[i] > arr[i+1]:
            extrema_indices.append(i)
            extrema_values.append(arr[i])

    return extrema_indices, extrema_values


def extremums(filt_beat, beat):
    """
    Fonction qui trouve et filtre les extrema d'un signal
    """
    n = len(filt_beat)
    extrema_indices = []
    extrema_values = []

    for i in range(1, len(filt_beat) - 1):
        if filt_beat[i-1] > filt_beat[i] < filt_beat[i+1] or filt_beat[i-1] < filt_beat[i] > filt_beat[i+1]:
            extrema_indices.append(i)
            extrema_values.append(beat[i])
    x = extrema_indices
    y = extrema_values

    print(x)
    print(y)

    # # Filter for x values between 300 and 500
    # filtered_indices = [i for i, val in enumerate(x) if 0.3 <= val/n <= 0.7]
    # filtered_y = [y[i] for i in filtered_indices]

    # if not filtered_indices:
    #     return None, None

    # # Find index of maximum y value in the filtered range
    # max_y_index = filtered_indices[filtered_y.index(max(filtered_y))]

    if not x or not y:
        return None, None
    
    max_y_index = np.argmax(y)


    # Indices around the max_y_index
    indices = []
    for i in range(-2,3,1):
        if max_y_index + i < len(extrema_indices) and max_y_index + i >= 0:
            indices.append(max_y_index + i)
    new_x = []
    new_y = []

    for index in indices:
        if 0 <= index < len(extrema_indices):  # Check if the index is within the list limits
            new_x.append(x[index] / n * 2 * np.pi - np.pi)
            new_y.append(y[index])

    return new_x, new_y


def init_parameters_extremums(nombre_gaussiennes, amplitudes,centres):
    param = par.parametres()
    param.nombre_gaussiennes = nombre_gaussiennes
    param.amplitudes = amplitudes
    param.centres = centres
    param.ecarts_types = [0.05 for _ in range(nombre_gaussiennes)]
    return param

def loss_function(param, signal):
    """
    Fonction de coût
    """
    signal_gauss = param.signal_gaussiennes(len(signal))
    return np.mean((signal - signal_gauss)**2)

def gradient_descent(param, signal, eps = 0.01, itmax = 1000, learning_rate = learning_rate):
    nombre_gaussiennes = param.nombre_gaussiennes # Assuming all lists in param have the same length

    for iteration in range(itmax):
        for i in range(nombre_gaussiennes):
                
            param.ecarts_types[i] += eps
            loss_plus = loss_function(param, signal)

            param.ecarts_types[i] -= 2*eps
            loss_minus = loss_function(param, signal)

            param.ecarts_types[i] += eps
            grad = (loss_plus - loss_minus)/(2*eps)

            param.ecarts_types[i] -= learning_rate['Ecart-type'][0] * grad 

            param.centres[i] += eps
            loss_plus = loss_function(param, signal)

            param.centres[i] -= 2*eps
            loss_minus = loss_function(param, signal)
            
            param.centres[i] += eps
            grad = (loss_plus - loss_minus)/(2*eps)

            param.centres[i] -= learning_rate['Centre'][0] * grad

            param.amplitudes[i] += eps
            loss_plus = loss_function(param, signal)

            param.amplitudes[i] -= 2*eps
            loss_minus = loss_function(param, signal)

            param.amplitudes[i] += eps
            grad = (loss_plus - loss_minus)/(2*eps)

            param.amplitudes[i] -= learning_rate['Amplitude'][0] * grad
    return param


def gradient_descent_calibre(beat, learning_rate = learning_rate, pas = 10):
    beat = ls.substract_linear(beat, pas)
    filt_beat = beat.copy()
    filt_beat = flt.lowpass_filter(filt_beat)
    c,A = extremums(filt_beat, beat)
    if c is None or A is None:
        nombre_gaussiennes = 0
    else: 
        nombre_gaussiennes = len(c)
    param = init_parameters_extremums(nombre_gaussiennes,A,c)
    print(param)
    return gradient_descent(param, beat, learning_rate=learning_rate), filt_beat

