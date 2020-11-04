# LMS
# pr7_1_1

import scipy.signal as signal
from adaptfilt import lms

from Noisy import *
from Universal import *

if __name__ == '__main__':
	filename = 'bluesky1.wav'
	speech = Speech()
	xx, fs = speech.audioread(filename, 8000)
	xx = xx - np.mean(xx)  # DC
	x = xx / np.max(xx)  # normalized
	N = len(x)
	time = np.arange(N) / fs
	SNR = 5
	r2 = np.random.randn(N)
	b = signal.firwin(32, 0.5)
	r21 = signal.lfilter(b, 1, r2)
	noisy = Noisy()
	r1, r22 = noisy.add_noisedata(x, r21, fs, fs, SNR)
	
	M = 32
	mu = 0.001
	snr1 = noisy.SNR_singlech(x, r1)                # initial SNR
	y, e, w = lms(r1.squeeze(), r2, M + 1, mu)
	output = np.zeros(N)
	output[0 : len(y)] = y
	snr2 = noisy.SNR_singlech(x, x - output)
	print('snr1 = {}\nsnr2 = {}\n'.format(snr1, snr2))
	
	# figure
	plt.figure(figsize=(9, 16))
	plt.subplot(3, 1, 1)
	plt.plot(time, x)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Original Speech Signal')
	plt.subplot(3, 1, 2)
	plt.plot(time, r1)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Signal SNR = {}dB'.format(SNR))
	plt.subplot(3, 1, 3)
	plt.plot(time, x - output)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Denoise Speech Signal SNR = {0:.2f}dB'.format(snr2))
	plt.savefig('images/lms_denoise.png', bbox_inches='tight', dpi=600)
	plt.show()
	
	
	
	
	
