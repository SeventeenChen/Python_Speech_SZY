#
# pr9_3_2

from audiolazy import lazy_lpc
from scipy.signal import lfilter

from LPC import LPC
from Universal import *

if __name__ == '__main__':
	filename = 'snn27.wav'
	speech = Speech()
	x, fs = speech.audioread(filename, None)  # read one frame data
	u = lfilter(b=np.array([1, -0.99]), a=1, x=x)  # pre-emphasis
	wlen = len(u)  # frame length
	p = 12  # LPC order
	ar = lazy_lpc.lpc.autocor(u, p)  # LPC coefficients
	ar0 = ar.numerator
	U = LPC().lpcar2pf(ar0, 255)  # LPC coefficients --> spectrum
	freq = np.arange(257) * fs / 512  # frequency scale in the frequency domain
	df = fs / 512  # frequency resolution
	U_log = 10 * np.log10(U)
	
	n_formnt = 4  # format number
	const = fs / 2 / np.pi  # constant
	rst = np.roots(ar0)  # roots of polynomial
	k = 0  # initialization
	yf = np.zeros(len(ar0) - 1)  # format frequency
	bandw = np.zeros(len(ar0) - 1)  # band width
	
	for i in range(len(ar0) - 1):
		re = np.real(rst[i])  # real of root
		im = np.imag(rst[i])  # image of root
		formn = const * np.arctan2(im, re)  # (9-3-17) format frequency
		bw = -2 * const * np.log(np.abs(rst[i]))  # (9-3-18) band width
		
		if formn > 150 and bw < 700 and formn < fs / 2:
			yf[k] = formn
			bandw[k] = bw
			k = k + 1
	
	yf = yf[0:k]
	bandw = bandw[0:k]
	y = np.sort(yf)  # sort
	ind = np.argsort(yf)  # index
	bw = bandw[ind]
	
	F = np.zeros(4)  # format frequency
	BW = np.zeros(4)  # band width
	F[0: np.min([n_formnt, len(y)])] = y[0:np.min([n_formnt, len(y)])]  # only output 4 format
	BW[0: np.min([n_formnt, len(y)])] = bw[0:np.min([n_formnt, len(y)])]
	
	np.set_printoptions(precision=2)
	print('Format = {}'.format(F))
	print('Band Width = {}'.format(BW))
	
	# figure
	plt.figure(figsize=(16, 9))
	plt.subplot(2, 1, 1)
	plt.plot(u)
	plt.xlabel('Sample points')
	plt.ylabel('Amplitude')
	plt.title('Pre-emphasis Signal Spectrum')
	plt.axis([0, wlen, -0.5, 0.5])
	plt.subplot(2, 1, 2)
	plt.plot(freq, U_log)
	for k in range(4):
		m = int(np.floor(F[k] / df))
		P = U_log[m + 1]
		plt.plot(F[k], P, 'r', marker='o', markersize=8)
		plt.plot(np.array([F[k], F[k]], dtype=object), np.array([-10, P], dtype=object), 'r-.',
		         linewidth=2)
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.title('Vocal Transfer Function Power Spectrum')
	plt.axis([0, 4000, -10, max(U_log) + 2])
	plt.savefig('images/lpc_roots_format_detection.png', bbox_inches='tight', dpi=600)
	plt.show()
