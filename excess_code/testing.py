import numpy as np
import matplotlib.pyplot as plt

# Load the signals
def load_signals(filename):
    # Since the data is stored as 16-bit integers
    data = np.fromfile(filename, dtype=np.int16)

    
    return data[1::12]

# Assuming the first signal is contained in the data as is
signal = load_signals('C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/excess_code/s0543_re.dat')

from pyts.decomposition import SingularSpectrumAnalysis
X = signal.reshape(1,-1)
ssa = SingularSpectrumAnalysis(window_size=10, groups='auto')
X_ssa = ssa.fit_transform(X)



X_ssa.shape
# Show the results for the first time series and its subseries
plt.figure(figsize=(16, 6))

ax1 = plt.subplot(121)
ax1.plot(X[0], label='Original')
ax1.legend(loc='best', fontsize=14)

ax2 = plt.subplot(122)
for i in range(3):
    ax2.plot(X_ssa[0, i], label='SSA {0}'.format(i + 1))
ax2.legend(loc='best', fontsize=14)

plt.suptitle('Singular Spectrum Analysis', fontsize=20)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# The first subseries consists of the trend of the original time series.
# The second and third subseries consist of noise.