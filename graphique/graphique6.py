import numpy as np
import matplotlib.pyplot as plt
import parametres as par
import extremum as ext
import scipy.signal as sig
import filter as flt
import linsub as linsub

signal = np.load('data/1/beats/1/1.npy')
x = np.linspace(-np.pi, np.pi, len(signal))

signal = linsub.substract_linear(signal, 10)


plt.plot(x, signal, color = 'red')

learning_rate = {'Amplitude':0.5, 'Centre':0.1, 'Ecart-type':0.2}

param, _ = ext.gradient_descent_calibre(signal, learning_rate)
# Paramétrer le nom des axes 

signal_gauss = param.signal_gaussiennes(len(signal))
plt.plot(x, signal_gauss, color = 'blue')
print(param)
plt.xlabel('Phase (°)')
plt.ylabel('Amplitude (UA)')

plt.show()