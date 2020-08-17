# pr4_4_1
# LPC spectrum

from Universal import *
from LPC import LPC
from audiolazy import lazy_lpc



def lpcar2zz(ar):
	"""
	Convert ar filter to z-plane poles ZZ=(AR)
	:param ar:
	:return:
	"""
	ar = np.array(ar)
	ar0 = ar.reshape(1, -1)
	nf, p1 = ar0.shape
	zz = np.zeros((nf, p1-1), dtype=complex)
	for k in range(nf):
		p = np.poly1d(ar0[k, :])
		zz[k, :] = p.roots
	
	return zz

if __name__ == '__main__':
	S = Speech()
	x, fs = S.audioread('aa.wav', 8000)
	L = 240
	p = 12
	y = x[8000 : 8000 + L]      # one frame
	ar = lazy_lpc.lpc(y, p)     # LPC
	nfft = 512
	W2 = int(nfft / 2)
	m = np.linspace(0, W2, num=W2+1)
	Y = np.fft.fft(y, n = nfft)
	Y1 = LPC.lpcar2ff(ar.numerator, p = W2 - 1)
	zz = lpcar2zz(ar.numerator).squeeze()
	

	for k in range(p):
		print('{} {}'.format(zz[k].real, zz[k].imag))
	
	# figure
	plt.figure(figsize=(18, 15))
	plt.subplot(2, 1, 1)
	plt.plot(y)
	plt.xlabel('Sample Points')
	plt.ylabel('Amplitude')
	plt.title('One Frame Speech Signal')
	plt.subplot(2, 1, 2)
	plt.plot(m, 20 * np.log10(np.abs(Y[0 : W2+1])), linewidth= 2)
	plt.plot(m, 20 * np.log10(np.abs(Y1)), linewidth= 4)
	plt.axis([0, W2+1, -30, 25])
	plt.legend(['FFT Spectrum', 'LPC Spectrum'])
	plt.xlabel('Sample Points')
	plt.ylabel('Amplitude [dB]')
	plt.title('FFT Spectrum & LPC Spectrum')
	plt.savefig('images/lpc_spectrum.png', dpi=600)
	plt.show()
	
	