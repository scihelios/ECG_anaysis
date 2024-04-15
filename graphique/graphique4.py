import numpy as np
import matplotlib.pyplot as plt
import parametres as par
import extremum as ext
import scipy.signal as sig
import filter as flt
import linsub as linsub

signal = np.load('data/1/beats/1/1.npy')
x = np.linspace(-np.pi, np.pi, len(signal))
signal_filter = flt.lowpass_filter(signal)

fonction_affine = linsub.fonction_affine(signal, 10)

extrema_indices, extrema_values = ext.extremums(signal_filter, signal)


plt.plot(x, signal, color = 'red')
#plt.plot(x, signal_filter, color = 'blue')
plt.plot(x, fonction_affine, color = 'green')
# Paramétrer le nom des axes 
plt.xlabel('Phase (°)')
plt.ylabel('Amplitude (UA)')

plt.show()