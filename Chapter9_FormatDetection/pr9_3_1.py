#
# pr9_3_1

from audiolazy import lazy_lpc
from scipy.signal import lfilter, find_peaks

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
	Loc, _ = find_peaks(U.reshape(np.size(U), ))  # peak location
	Val = U[Loc]  # peak value
	LenLoc = len(Loc)  # peak number
	F = np.zeros(LenLoc)  # format frequency
	BW = np.zeros(LenLoc)  # band width
	for k in range(LenLoc):
		m = Loc[k]  # set m, m-1, m+1
		m1 = m - 1
		m2 = m + 1
		p = Val[k]  # set P(m), P(m-1), P(m+1)
		p1 = U[m1]
		p2 = U[m2]
		aa = (p1 + p2) / 2 - p  # (9-3-4)
		bb = (p2 - p1) / 2
		cc = p
		dm = -bb / 2 / aa  # (9-3-6)
		pp = - bb * bb / 4 / aa + cc  # (9-3-8)
		m_new = m + dm
		bf = -np.sqrt(bb * bb - 4 * aa * (cc - pp / 2)) / aa  # (9-3-13)
		F[k] = (m_new - 1) * df  # (9-3-7)
		BW[k] = bf * df  # (9-3-14)
	
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
	plt.plot(freq, U)
	for k in range(LenLoc):
		plt.plot(freq[Loc[k]], Val[k], 'r', marker='o', markersize=8)
		plt.plot(np.array([freq[Loc[k]], freq[Loc[k]]], dtype=object), np.array([0, Val[k]], dtype=object), 'r-.',
		         linewidth=2)
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.title('Vocal Transfer Function Power Spectrum')
	plt.axis([0, 4000, 0, max(U) + 2])
	plt.savefig('images/lpc_format_detection.png', bbox_inches='tight', dpi=600)
	plt.show()
