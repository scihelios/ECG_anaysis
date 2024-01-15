
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
import scipy




def extremums(filt_beat, beat):
    """
    Fonction qui trouve et filtre les extrema d'un signal
    Retourne les indices et les valeurs des extrema

    """
    n = len(filt_beat)
    extrema_indices = [0]
    extrema_values = [filt_beat[0]]

    # On trouve les extrema et leurs indices

    extrema_indices += scipy.signal.find_peaks(abs(filt_beat))[0].tolist()
    extrema_values += list(filt_beat[extrema_indices])

    # # On ajoute les extrema aux extrémités
    extrema_indices.append(len(filt_beat))
    extrema_values.append(filt_beat[-1])

    

    # On calcule les différences entre les valeurs des extrema
    courbature = []

    for i in range(1,len(extrema_values) - 1):
        courbature.append(abs(extrema_values[i - 1] - 2 * extrema_values[i] + extrema_values[i + 1]))


    # On garde les 5 pics avec la plus grande courbature
    indices = []
    for i in range(5):

        indices.append(extrema_indices[courbature.index(max(courbature))-1])
        courbature[courbature.index(max(courbature))] = 0

    indices_pics = []
    valeurs_pics = []

    # Convert the indices to the corresponding x values
    for indice in indices:
        indices_pics.append(indice / n * 2 * np.pi - np.pi)
        valeurs_pics.append(filt_beat[indice])

    return indices_pics, valeurs_pics





def init_parameters_extremums(nombre_gaussiennes, amplitudes,centres):
    """
    Fonction qui initialise les paramètres des gaussiennes
    """
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



def gradient_descent(param, signal, learning_rate, eps = 0.0001, itmax = 2000 ):
    """
    Fonction de descente de gradient
    """
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


def gradient_descent_calibre(beat, learning_rate, pas = 10):
    beat = ls.substract_linear(beat, pas)
    filt_beat = beat.copy()
    filt_beat = flt.lowpass_filter(filt_beat)
    c,A = extremums(filt_beat, beat)
    if c is None or A is None:
        nombre_gaussiennes = 0
    else: 
        nombre_gaussiennes = len(c)
    param = init_parameters_extremums(nombre_gaussiennes,A,c)

    return gradient_descent(param, beat, learning_rate=learning_rate), filt_beat

