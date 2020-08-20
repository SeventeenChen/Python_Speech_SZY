# pr5_3_1

from Universal import *
import numpy as np

def Gnoisegen(x, snr):
	"""
	Generate Gaussian white noise according to the set SNR, return noisy speech
	:param x: clean speech signal
	:param snr: SNR [dB]
	:return y: noisy speech
	:return noise: gaussian white noise
	"""
	
	noise = np.random.randn(x.size)     # generate noise
	Nx = len(x)
	signal_power = 1 / Nx * np.sum(x * x)
	noise_power = 1 / Nx * np.sum(noise * noise)
	noise_variance = signal_power / (10 ** (snr / 10))
	noise = np.sqrt(noise_variance / noise_power) * noise
	y = x + noise
	
	return y, noise

def SNR_singlech(I,In):
	"""
	calculate SNR of noisy speech signal
	:param I: clean speech siganl
	:param In: noisy speech siganl
	:return snr:
	"""
	
	Ps = np.sum((I - np.mean(I)) ** 2)      # signal power
	Pn = np.sum((I - In) ** 2)              # noise power
	snr = 10 * np.log10(Ps / Pn)
	
	return snr

if __name__ == '__main__':
	S = Speech()
	s, fs = S.audioread('bluesky3.wav', 8000)
	s = s - np.mean(s)      # remove DC
	s = s / np.max(np.abs(s))   # normalized
	N = len(s)
	time = np.linspace(0, N - 1, num=N) / fs
	
	# figure
	plt.figure(figsize=(16, 24))
	plt.subplot(4, 1, 1)
	plt.plot(time, s)
	plt.ylabel('Amplitude')
	plt.title('Clean Speech Siganl')
	
	SNR = np.array([15, 5, 0])
	for i in range(3):
		snr = SNR[i]
		x, noise = Gnoisegen(s, snr)
		plt.subplot(4, 1, i + 2)
		plt.plot(time, x)
		plt.xlabel('Time [s]')
		plt.ylabel('Amplitude')
		snr1 = SNR_singlech(s, x)
		print('i = {}, snr = {}, snr1 = {:.2}'.format(i+1, snr, snr1))
		plt.title('Noisy Speech Siganl \nSNR = {}, calculating SNR = {:.2}'.format(snr, snr1))
	
	plt.savefig('images/pr5_3_1.png',bbox_inches = 'tight' ,dpi=600)
	plt.show()