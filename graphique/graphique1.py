import numpy as np
import matplotlib.pyplot as plt
import parametres as par
import extremum as ext
import scipy.signal as sig

signal = np.load('data/1/full_signal/1.npy')[0:10000]
pics, _ = sig.find_peaks(signal, height=75, distance= 600)


plt.plot(signal, color = 'red')
plt.plot(pics, signal[pics], "o", color = 'green')
plt.hlines(75, 0, 10000, color = 'blue')

# Enlever les graduations sur les axes
plt.xticks([])
plt.yticks([])
plt.show()