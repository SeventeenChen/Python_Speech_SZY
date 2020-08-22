# pr_5_4_1
# Remove a linear trend from a vector

from scipy.signal import detrend
from Universal import *

if __name__ == '__main__':
    S = Speech()
    x, fs = S.audioread('bluesky31.wav', 8000)
    t = np.linspace(0, len(x) - 1, num=len(x)) / fs     # time
    y = detrend(x)          # remove linear trend
    y = y / np.max(np.abs(y))   # normalized
    
    # figure
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Signal with linear trend')
    plt.subplot(2, 1, 2)
    plt.plot(t, y)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Signal removed linear trend')
    plt.savefig('images/detrend.png', bbox_inches='tight', dpi=600)
    plt.show()