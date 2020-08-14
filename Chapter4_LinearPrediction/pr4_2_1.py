# pr4_2_1
# lpc

from Universal import *
from audiolazy import lazy_lpc
from scipy.signal import lfilter
import numpy as np

def lpcar2ff(ar, p=None):
    """
    Convert AR coefs to complex spectrum FF=(AR,P)
    :param ar:
    :param p:
    :return ff:
    """
    ar = np.array(ar)
    ar0 = ar.reshape(1, -1)
    nf, p1 = ar0.shape
    
    if p == None:
        p = p1 -1
    ff = (np.fft.rfft(ar0, n = 2 * p + 2)).T **(-1)
    
    return ff


if __name__ == '__main__':
    S = Speech()
    x, fs = S.audioread('aa.wav', 8000)
    L = 240     # frame length
    y = x[8000: 8000 + L]       # one frame
    p = 12              # lpc order
    ar = lazy_lpc.lpc.autocor(y, p)
    ar0 = ar.numerator
    Y = lpcar2ff(ar0, 255)
    ar0.pop(0)
    ar0.insert(0,0)
    b = -1 * np.array(ar0)
    a = 1
    est_x = lfilter(b, 1, y)
    err = y - est_x
    print('LPC:\n {}'.format(ar.numerator))
    
    # figure
    plt.figure(figsize=(48, 27))
    plt.subplot(3, 1, 1)
    plt.plot(x)
    plt.tight_layout()
    plt.title('Waveform of Vowel /a/')
    plt.ylabel('Amplitude')
    plt.subplot(3, 2, 3)
    plt.plot(y)
    plt.xlim([0, L])
    plt.title('One Frame Speech Signal')
    plt.ylabel('Amplitude')
    plt.subplot(3, 2, 4)
    plt.plot(est_x)
    plt.xlim([0, L])
    plt.title('Prediction Signal')
    plt.ylabel('Amplitude')
    plt.subplot(3, 2, 5)
    plt.plot(np.abs(Y))
    plt.xlim([0, L])
    plt.title('LPC Spectrum')
    plt.ylabel('Amplitude')
    plt.xlabel('Sample points')
    plt.subplot(3, 2, 6)
    plt.plot(err)
    plt.xlim([0, L])
    plt.title('Prediction Error')
    plt.ylabel('Amplitude')
    plt.xlabel('Sample points')
    plt.savefig('images/lpc.png', bbox_inches = 'tight', dpi=600)
    plt.show()
    
