import numpy as np
import matplotlib.pyplot as plt
import parametres as par
import extremum as ext
import scipy.signal as sig

signal = np.load('data/1/beats/1/1.npy')
x = np.linspace(-np.pi, np.pi, len(signal))
# Enlever les graduations sur les axes
plt.xticks([])
param = par.parametres()
param.amplitudes = [1,-3,1]
param.centres = [1,0,1]
param.ecarts_types = [0.5,0.5,0.5]


gauss = param.signal_gaussiennes(len(signal))

# Faire des flèches double sens

plt.arrow(1.1,-2.7,0,4.33,color = 'red')
plt.arrow(-1.3,-2.7,0,2.5,color = 'red')

plt.plot(x, gauss, color = 'blue')
# Paramétrer le nom des axes 
plt.xlabel('Phase (°)')
plt.ylabel('Amplitude (UA)')

plt.show()