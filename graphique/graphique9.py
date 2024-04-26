import numpy as np
import matplotlib.pyplot as plt
import parametres as par
import extremum as ext
import scipy.signal as sig
import filter as flt
import linsub as linsub

signal = np.load('data/1/beats/1/4.npy')
x = np.linspace(-180, 180, len(signal))

signal = linsub.substract_linear(signal,10)

plt.plot(x, signal, color = 'red')
plt.xlabel('Phase (°)')
plt.ylabel('Amplitude (UA)')

param = par.parametres()

param,_ = ext.gradient_descent_calibre(signal)

signal_gauss = param.signal_gaussiennes(len(signal))
plt.plot(x, signal_gauss, color = 'blue')
print(param)


plt.show()