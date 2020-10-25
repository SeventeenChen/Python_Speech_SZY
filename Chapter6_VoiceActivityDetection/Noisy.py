# class Noisy

import numpy as np


class Noisy:
	def Gnoisegen(self, x, snr):
		"""
		Generate Gaussian white noise according to the set SNR, return noisy speech
		:param x: clean speech signal
		:param snr: SNR [dB]
		:return y: noisy speech
		:return noise: gaussian white noise
		"""
		
		noise = np.random.randn(x.size)  # generate noise
		Nx = len(x)
		signal_power = 1 / Nx * np.sum(x * x)
		noise_power = 1 / Nx * np.sum(noise * noise)
		noise_variance = signal_power / (10 ** (snr / 10))
		noise = np.sqrt(noise_variance / noise_power) * noise
		y = x + noise
		
		return y, noise
	
	def SNR_singlech(self, I, In):
		"""
		calculate SNR of noisy speech signal
		:param I: clean speech siganl
		:param In: noisy speech siganl
		:return snr:
		"""
		I = I.reshape(-1, 1)
		In = In.reshape(-1, 1)
		Ps = np.sum((I - np.mean(I)) ** 2)  # signal power
		Pn = np.sum((I - In) ** 2)  # noise power
		snr = 10 * np.log10(Ps / Pn)
		
		return snr
	
	def add_noisedata(self, s, data, fs, fs1, snr):
		"""
		把任意的噪声数据按设定的信噪比叠加在纯净信号上，构成带噪语音
		:param s: clean speech signal
		:param data: arbitrary noise data
		:param fs: clean signal sample frequency
		:param fs1: data sample frequency
		:param snr: SNR [dB]
		:return noise: noise scaled by the set SNR
		:return signal: noisy (size: n * 1)
		"""
		
		s = s.reshape(-1, 1)
		s = s - np.mean(s)
		sL = len(s)
		
		if fs != fs1:
			x = librosa.resample(data, fs, fs1)  # noise reample
		else:
			x = data
		
		x = x.reshape(-1, 1)
		x = x - np.mean(x)
		xL = len(x)
		
		if xL >= sL:
			x = x[0: sL]
		else:
			print('Warning noise length < signal length, padding with zero')
			x = np.concatenate((x, np.zeros(sL - xL)))
		
		Sr = snr
		Es = np.sum(x * x)
		Ev = np.sum(s * s)
		a = np.sqrt(Ev / Es / (10 ** (Sr / 10)))  # noise ratio
		noise = a * x
		signal = s + noise  # noisy
		
		return signal, noise