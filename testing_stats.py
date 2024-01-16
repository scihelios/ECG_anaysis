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
import extremum as ext
import pandas as pd
import os



numero_enregistrement = input('Numero de l\'enregistrement : ')
input_folder = f'C:/Users/thoma/Documents/Python/ECG_PSC/data/1/beats/'

bit_resolution = 12
max_range = 10 # mV
yscale_factor = max_range / (2**(bit_resolution+1))
iteration_max = 100

learning_rate = {
    'Amplitude' : [0.1],
    'Centre' : [0.00001],
    'Ecart-type' : [0.00001]
}

pas = 10


Amplitude = []
Centre = []
Ecart_type = []



file = os.listdir(f"{input_folder}{numero_enregistrement}/")[0]
beat = np.load(f"{input_folder}{numero_enregistrement}/{file}")
x_unit = np.linspace(-np.pi,np.pi, len(beat))
param, filt_beat = ext.gradient_descent_calibre(beat, learning_rate, pas, iteration_max)


xscale_factor = 180 / np.pi
yscale_factor = 1

plt.plot(xscale_factor * x_unit, yscale_factor * beat,color='b',alpha=0.4, label = 'Signal')
plt.plot(xscale_factor * x_unit, yscale_factor * filt_beat,color='g',alpha=1, label = 'Signal filtré')
plt.plot(xscale_factor * x_unit, yscale_factor * param.signal_gaussiennes(len(beat)) ,color='r',alpha=1, label = 'Signal gaussien')
param.plot_pics(yscale_factor)

print(param)
plt.grid()
plt.legend()
plt.show()



