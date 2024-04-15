import numpy as np
import matplotlib.pyplot as plt
import parametres as par
import extremum as ext
import scipy.signal as sig
import filter as flt
import linsub as linsub

signal = np.load('data/1/beats/1/1.npy')
x = np.linspace(-np.pi, np.pi, len(signal))

param = par.parametres()

param.amplitudes = [0.2,-0.1,0.5,0.1,0.5]
param.centres = [-2,-0.1,0,0.05,2]
param.ecarts_types = [0.1,0.1,0.1,0.1,0.1]

plt.plot(x, signal, color = 'red')


param = ext.gradient_descent(param, signal, learning_rate={'Amplitude':0.1, 'Centre':0.1, 'Ecart-type':0.1}, itmax=100)
# Paramétrer le nom des axes 

signal_gauss = param.signal_gaussiennes(len(signal))
plt.plot(x, signal_gauss, color = 'blue')
print(param)
plt.xlabel('Phase (°)')
plt.ylabel('Amplitude (UA)')

plt.show()