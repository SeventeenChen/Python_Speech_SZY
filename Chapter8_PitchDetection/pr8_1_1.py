#
# pr8_1_1

from math import pi

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ellipord, ellip, freqz, group_delay


def freqz_m(b, a):
	"""
	Modified version of freqz subroutine
	:param b: numerator polynomial of H(z)   (for FIR: b=h)
	:param a: denominator polynomial of H(z) (for FIR: a=[1])
	:return db: Relative magnitude in dB computed over 0 to pi radians
	:return mag: absolute magnitude computed over 0 to pi radians
	:return pha: Phase response in radians over 0 to pi radians
	:return grd: Group delay over 0 to pi radians
	:return w: 501 frequency samples between 0 to pi radians
	"""
	w, H = freqz(b, a, 1000, whole=True)
	H = H[0:501]
	w = w[0:501]
	mag = np.abs(H)
	eps = np.finfo(float).eps
	db = 20 * np.log10((mag + eps) / np.max(mag))
	pha = np.angle(H)
	_, grd = group_delay((b, a), w)
	
	return db, mag, pha, grd, w

if __name__ == '__main__':
	fs = 8000                                       # sampling frequency
	fs2 = fs / 2
	Wp = np.array([60, 500]) / fs2                  # filter pass band
	Ws = np.array([20, 2000]) / fs2                 # filter stop band
	Rp = 1                                          # passband ripple
	Rs = 40                                         # stopband attenuation
	n, Wn = ellipord(Wp,Ws,Rp,Rs)                   # filter order
	b, a = ellip(n, Rp, Rs, Wn, 'bandpass')                     # filter coefficients
	print('b = {} \na = {}'.format(b, a))
	db, mag, pha, grd, w = freqz_m(b, a)            # frequency response curve
	
	# figure
	plt.figure(figsize=(16, 9))
	plt.plot(w / pi* fs2, db, linewidth=2)
	plt.grid()
	plt.axis([0, 4000, -90, 10])
	plt.title('Frequency Response of Elliptical 6th-order BPF')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.savefig('images/elliptical_6th-order_BPF.png', bbox_inches='tight', dpi=600)
	plt.show()
	
	