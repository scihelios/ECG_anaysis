import numpy as np
import matplotlib.pyplot as plt
import parametres as par
import extremum as ext
import scipy.signal as sig
import filter as flt
import linsub as linsub

signal = np.load('data/1/beats/1/8.npy')
x = np.linspace(-180, 180, len(signal))

signal = linsub.substract_linear(signal, 10)
plt.plot(x, signal, color = 'red')
plt.xlabel('Phase (°)')
plt.xlabel('Phase (°)')
plt.ylabel('Amplitude (UA)')

param = par.parametres()
param,_ = ext.gradient_descent_calibre(signal)
print(param)

plt.text(-1.2*180/3.14, 0.1, "Pic P")
plt.text(-0.62*180/3.14, -0.17, "Pic Q")
plt.text(0.18*180/3.14, 0.68, "Pic R")
plt.text(0.4*180/3.14, -0.13, "Pic S")
plt.text(1.7*180/3.14, 0.3, "Pic T")


plt.show()