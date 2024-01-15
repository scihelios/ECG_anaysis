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

input_folder = f'data/1/beats/24/776.npy'
bit_resolution = 12
max_range = 10 # mV
yscale_factor = max_range / (2**(bit_resolution+1))



learning_rate = {
    'Amplitude' : [0.1],
    'Centre' : [0.0000],
    'Ecart-type' : [0.0001]
}

pas = 10


Amplitude = []
Centre = []
Ecart_type = []



index = ['Pic P Amplitude', 'Pic Q Amplitude', 'Pic R Amplitude', 'Pic S Amplitude', 'Pic T Amplitude', 'Pic P Centre', 'Pic Q Centre', 'Pic R Centre', 'Pic S Centre', 'Pic T Centre', 'Pic P Ecart-type', 'Pic Q Ecart-type', 'Pic R Ecart-type', 'Pic S Ecart-type', 'Pic T Ecart-type']

data = []
i = 0
# for folder in os.listdir('data/1/beats'):
#     for file in os.listdir(f'data/1/beats/{folder}'):
#         if file.endswith('.npy'):
#             beat = np.load(f'data/1/beats/{folder}/{file}')
#             x_unit = np.linspace(-np.pi,np.pi, len(beat))
#             param, filt_beat = ext.gradient_descent_calibre(beat)
#         break
#     break
beat = np.load(input_folder)
x_unit = np.linspace(-np.pi,np.pi, len(beat))
param, filt_beat = ext.gradient_descent_calibre(beat, learning_rate, pas)


# fig, ax = plt.subplots(3)


# ax[0].plot(x_unit, yscale_factor * beat,color='g',alpha=0.5)
# ax[1].plot(x_unit, yscale_factor * filt_beat,color='r',alpha=0.5)
# ax[2].plot(x_unit, yscale_factor * param.signal_gaussiennes(len(beat)),color='b',alpha=1)

xscale_factor = 180 / np.pi

plt.plot(xscale_factor * x_unit, yscale_factor * beat,color='b',alpha=0.7, label = 'Signal')
plt.plot(xscale_factor * x_unit, yscale_factor * filt_beat,color='r',alpha=1, label = 'Signal filtré')
plt.plot(xscale_factor * x_unit, yscale_factor * param.signal_gaussiennes(len(beat)) ,color='g',alpha=1, label = 'Signal gaussien')
param.plot_pics(yscale_factor)



# data = np.array(data)
# df = pd.DataFrame(data, columns = index)
# df.hist(bins = 25, figsize = (20,15))
# Make x axis with -pi and pi in legend
# plt.xticks(np.linspace(-np.pi,np.pi,5), ('-π', '-π/2', '0', 'π/2', 'π'))


#df.to_csv('data/1/parametres.csv', index = False)

# plt.xlabel('Angle (°)')
# plt.ylabel('Amplitude (mV)')
print(param)
plt.legend()
plt.grid()
plt.show()



