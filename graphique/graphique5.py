import numpy as np
import matplotlib.pyplot as plt
import parametres as par
import extremum as ext
import scipy.signal as sig
import filter as flt
import linsub as linsub
import kalman as kal

signal = np.load('data/1/beats/1/6.npy')
signal = linsub.substract_linear(signal,10)
x = np.linspace(-180, 180, len(signal))



plt.plot(x, signal, color = 'red')


param, _, _ = kal.kalman_signal(signal, 5)
# Paramétrer le nom des axes 
param_kalman = par.parametres()
param_kalman.amplitudes = param[0:5]
param_kalman.centres = param[5:10]
param_kalman.ecarts_types = param[10:15]
signal_gauss = param_kalman.signal_gaussiennes(len(signal))
plt.plot(x, signal_gauss, color = 'blue')
print(param)
plt.xlabel('Phase (°)')
plt.ylabel('Amplitude (UA)')

plt.show()