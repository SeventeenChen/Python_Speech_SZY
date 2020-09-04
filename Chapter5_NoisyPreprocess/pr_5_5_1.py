# pr5_5_1
# Chebyshev II Filter

import math
from Universal import *
from scipy.signal import cheby2, cheb2ord, freqz, group_delay, lfilter

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
	w=w[0:501]
	mag = np.abs(H)
	eps = np.finfo(float).eps
	db = 20 * np.log10((mag + eps) / np.max(mag))
	pha = np.angle(H)
	_, grd = group_delay((b, a), w)
	
	return db, mag, pha, grd, w

def stftms(x, win, nfft, inc):
	"""
	short time fourier transform
	:param x: siganl
	:param win: window function or frame length(default 'hanning')
	:param nfft: FFT number
	:param inc: frame shift
	:return d: STFT (win x frame number) just positive frequency
	"""
	
	if len([win]) == 1:
		wlen = win
		win = np.hanning(wlen)
	else:
		wlen = len(win)

	x = x.reshape(-1, 1)
	win = win.reshape(-1, 1)
	s = len(x)
	c = 0
	d = np.zeros((1 + int(nfft / 2), int(1 + np.fix((s - wlen) / inc))), dtype=complex)

	for b in np.arange(0, (s - wlen), inc):
		u = win * x[b: (b + wlen)]
		t = np.fft.fft(u, n=nfft, axis=0).squeeze()
		d[:, c] = t[0: int(nfft / 2 + 1)]
		c = c + 1
	
	return d

if __name__ == '__main__':
	fp = 500
	fs = 750    # pass/stop frequency
	Fs = 8000
	Fs2 = int(Fs / 2)    # sample frequency
	Wp = fp / Fs2
	Ws = fs / Fs2   # normalized
	Rp = 3      # passband ripple
	Rs = 50      # stopband attenuation
	
	# Chebyshev II LPF
	n, Wn = cheb2ord(wp = Wp, ws = Ws, gpass = Rp, gstop = Rs)    # filter order
	b, a = cheby2(N = n, rs=Rs, Wn = Wn)        # Chebyshev II LPF coefficient
	[db, mag, pha, grd, w] = freqz_m(b, a)      # frequency response curve
	
	S = Speech()
	s, fs = S.audioread('bluesky3.wav', 8000)
	s = s - np.mean(s)          # DC
	s = s / np.max(np.abs(s))   # normalized
	N = len(s)
	t = np.linspace(0, N-1, num = N) / fs
	
	y = lfilter(b, a, np.squeeze(s))    # LPF
	wlen = 200          # frame length
	inc = 80            # frame shift
	nfft = 512
	
	d = stftms(s, wlen, nfft, inc)          # STFT of original
	fn = d.shape[1]
	frameTime = S.FrameTime(fn, nfft, inc, Fs)
	freq = np.linspace(0, int(nfft/2), num=int(nfft/2 + 1)) * Fs / nfft
	d1 = stftms(y, wlen, nfft, inc)         # STFT of filtered signal
	
	# figure
	plt.figure(1)
	plt.plot(w / math.pi * Fs2, db, 'k', linewidth=2)
	plt.grid()
	plt.axis([0, 4000, -100, 5])
	plt.title('LPF Amplitude Response')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.savefig('images/chebyII.png', bbox_inches='tight', dpi=600)
	plt.show()
	
	plt.figure(2, figsize=(16, 9))
	plt.subplot(2, 1, 1)
	plt.plot(t, s, 'k')
	plt.title('Clean Speech Siganl')
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.subplot(2, 1, 2)
	Xdb = librosa.amplitude_to_db(abs(d))
	librosa.display.specshow(Xdb, sr=Fs, x_coords=frameTime, x_axis='time', y_axis='linear')
	plt.colorbar()
	plt.xlabel('Time/s')
	plt.ylabel('Frequency/Hz')
	plt.title('Spectorgram')
	plt.savefig('images/ChebyCleanSTFT.png', bbox_inches='tight', dpi=600)
	plt.show()
	
	plt.figure(3, figsize=(16, 9))
	plt.subplot(2, 1, 1)
	plt.plot(t, y, 'k')
	plt.title('Filtered Speech Siganl')
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.subplot(2, 1, 2)
	Ydb = librosa.amplitude_to_db(abs(d1))
	librosa.display.specshow(Ydb, sr=Fs, x_coords=frameTime, x_axis='time', y_axis='linear')
	plt.ylim([0, 1000])
	plt.colorbar()
	plt.xlabel('Time [s]')
	plt.ylabel('Frequency [Hz]')
	plt.title('Spectorgram')
	plt.savefig('images/ChebyFilteredSTFT.png', bbox_inches='tight', dpi=600)
	plt.show()
	
	
	
	
	