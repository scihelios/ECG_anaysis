import numpy as np
import matplotlib.pyplot as plt
import parametres as par
import extremum as ext
import scipy.signal as sig

signal = np.load('data/1/beats/1/1.npy')
x = np.linspace(-180, 180, len(signal))


plt.plot(x, signal, color = 'red')
# Paramétrer le nom des axes 
plt.xlabel('Phase (°)')
plt.ylabel('Amplitude (UA)')

plt.show()