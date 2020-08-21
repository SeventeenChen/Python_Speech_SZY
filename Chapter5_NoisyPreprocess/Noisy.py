# class Noisy

import numpy as np


class Noisy:
	def Gnoisegen(x, snr):
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
	
	def SNR_singlech(I, In):
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