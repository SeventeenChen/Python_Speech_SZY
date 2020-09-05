# pr5_5_2

from scipy.signal import freqz
from Universal import *
import math

def ideal_lp(wc, M):
	"""
	Ideal Lowpass filter computation
	:param wc:
	:param M:
	:return:
	"""
	alpha = (M - 1) / 2
	n = np.arange(M)
	eps = np.finfo(float).eps
	m = n - alpha + eps
	hd = np.sin( wc * m) / (math.pi * m)
	
	return hd

if __name__ == '__main__':
	As = 50         # stopband min attenuation
	Fs = 8000       # sample rate
	Fs2 = Fs / 2
	fp = 75         # passband frequency
	fs = 60         # stopband frequency
	df = fp - fs    # interim band
	M0 = np.round((As - 7.95) / (14.36 * df / Fs)) + 2     # kaiser window length
	M = M0 + np.mod(M0 + 1, 2)          # M -> odd
	wp = fp / Fs2 * math.pi
	ws = fs / Fs2 * math.pi             # rad/s
	wc = (wp + ws) / 2                  # stop frequency
	beta = 0.5842 * (As - 21) ** 0.4 + 0.07886 * (As - 21)
	print('beta = {:.6f}'.format(beta))
	w_kai = np.kaiser(M, beta)          # kaiser window
	hd = ideal_lp(math.pi, M) - ideal_lp(wc, M)
	b = hd * w_kai
	w, h = freqz(b, a=1, worN=4000)
	db = 20 * np.log10(np.abs(h))
	
	S = Speech()
	s, fs = S.audioread('bluesky3.wav', 8000)
	s = s - np.mean(s)          # DC
	s = s / np.max(np.abs(s))   # normalized
	N = len(s)
	t = np.arange(N) / fs
	ns = 0.5 * np.cos(2 * math.pi * 50 * t)     # 50Hz IF
	x = s + ns
	snr1 = S.SNR_singlech(s, x)
	print('snr1 = {:.4f}'.format(snr1))
	y = np.convolve(b, x)           # FIR output
	
	
	# figure
	plt.figure(1)
	plt.plot(w / math.pi * Fs2, db, 'k', linewidth=2)
	plt.grid()
	plt.axis([0, 150, -100, 10])
	plt.title('Amplitude Response Curve')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.savefig('images/FIR_Filter.png', bbox_inches='tight', dpi=600)
	plt.show()
	
	plt.figure(2, figsize=(9, 16))
	plt.subplot(3, 1, 1)
	plt.plot(t, s, 'k')
	plt.title('Clean Speech Signal')
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude ')
	plt.axis([0, np.max(t), -1.2, 1.2])
	plt.subplot(3, 1, 2)
	plt.plot(t, x, 'k')
	plt.title('Speech with 50 Hz IF')
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude ')
	plt.axis([0, np.max(t), -1.2, 1.2])
	z = y[int(M/2) : len(y) - int(M/2)]
	snr2 =  S.SNR_singlech(s, z)
	print('snr2 = {:.4f}'.format(snr2))
	plt.subplot(3, 1, 3)
	plt.plot(t, z, 'k')
	plt.title('Speech Removed 50 Hz IF')
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude ')
	plt.axis([0, np.max(t), -1.2, 1.2])
	plt.savefig('images/FIR_Preprocess.png', bbox_inches='tight', dpi=600)
	plt.show()
	
	