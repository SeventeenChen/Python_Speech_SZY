# pr5_5_3

from Universal import *
import math
from scipy.signal import firwin, freqz

if __name__ == '__main__':
	As = 50  # stopband min attenuation
	Fs = 8000  # sample rate
	Fs2 = Fs / 2
	fs1 = 49
	fs2 = 51    # stopband frequency
	fp1 = 45
	fp2 = 55    # passband frequency
	df = min(fs1 - fp1, fp2 - fs2)  # interim bandwidth
	M0 = np.round((As - 7.95) / (14.36 * df / Fs)) + 2  # kaiser window length
	M = M0 + np.mod(M0 + 1, 2)  # M -> odd
	wp1 = fp1 / Fs2 * math.pi
	wp2 = fp2 / Fs2 * math.pi
	ws1 = fs1 / Fs2 * math.pi
	ws2 = fs2 / Fs2 * math.pi   # rad/s
	wc1 = (wp1 + ws1) / 2  # stop frequency
	wc2 = (wp2 + ws2) / 2
	beta = 0.5842 * (As - 21) ** 0.4 + 0.07886 * (As - 21)
	print('beta = {:.6f}'.format(beta))
	b = firwin(M, np.array([wc1 / math.pi, wc2 / math.pi]) , window=('kaiser', beta),pass_zero='bandstop')
	w, h = freqz(b, 1, 4000)
	db = 20 * np.log10(np.abs(h))
	
	
	S = Speech()
	s, fs = S.audioread('bluesky3.wav', 8000)
	s = s - np.mean(s)  # DC
	s = s / np.max(np.abs(s))  # normalized
	N = len(s)
	t = np.arange(N) / fs
	ns = 0.5 * np.cos(2 * math.pi * 50 * t)  # 50Hz IF
	x = s + ns
	snr1 = S.SNR_singlech(s, x)
	print('snr1 = {:.4f}'.format(snr1))
	y = np.convolve(b, x)  # FIR output
	z = y[int(np.fix(M/2)) : len(y) - int(np.fix(M/2))]
	snr2 = S.SNR_singlech(s, z)
	print('snr2 = {:.4f}'.format(snr2))
	
	# figure
	plt.figure(1)
	plt.plot(w / math.pi * Fs2, db, 'k', linewidth=2)
	plt.grid()
	plt.axis([0, 100, -60, 5])
	plt.title('Amplitude Frequency Response')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.savefig('images/BSF_Kaiser.png', bbox_inches='tight', dpi=600)
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
	plt.subplot(3, 1, 3)
	plt.plot(t, z, 'k')
	plt.title('Speech Removed 50 Hz IF')
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude ')
	plt.axis([0, np.max(t), -1.2, 1.2])
	plt.savefig('images/BSF_Preprocess.png', bbox_inches='tight', dpi=600)
	plt.show()
	