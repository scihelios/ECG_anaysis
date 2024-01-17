import numpy as np
import matplotlib.pyplot as plt

# Load the signals
def load_signals(filename):
    # Since the data is stored as 16-bit integers
    data = np.fromfile(filename, dtype=np.int16)

    # Split the data into the two signals (raw and filtered)
    # Assuming interleave format
    raw_signal = data[::2]
    filtered_signal = data[1::2]

    return raw_signal, filtered_signal

raw_signal, filtered_signal = load_signals('rec_1.dat')




from pyts.decomposition import SingularSpectrumAnalysis
X = raw_signal.reshape(1,-1)
ssa = SingularSpectrumAnalysis(window_size=20, groups='auto')
X_ssa = ssa.fit_transform(X)

X_ssa.shape
# Show the results for the first time series and its subseries
plt.figure(figsize=(16, 6))

from pyts.decomposition import SingularSpectrumAnalysis
Y = filtered_signal.reshape(1,-1)
ssa = SingularSpectrumAnalysis(window_size=20, groups='auto')
Y_ssa = ssa.fit_transform(Y)

Y_ssa.shape
print(Y_ssa[0, 0])

ax1 = plt.subplot(141)
ax1.plot(X[0], label='Original')
ax1.legend(loc='best', fontsize=14)

ax2 = plt.subplot(142)
ax2.plot([i for i in range(len(filtered_signal))],filtered_signal, label='filtered')
ax2.legend(loc='best', fontsize=14)

ax3 = plt.subplot(143)
for i in range(3):
    ax3.plot(X_ssa[0, i], label='SSA {0}'.format(i + 1))
ax3.legend(loc='best', fontsize=14)


ax4 = plt.subplot(144)
for i in range(3):
    ax4.plot(Y_ssa[0, i], label='SSA of fil {0}'.format(i + 1))
ax4.legend(loc='best', fontsize=14)

plt.suptitle('Singular Spectrum Analysis', fontsize=20)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# The first subseries consists of the trend of the original time series.
# The second and third subseries consist of noise.