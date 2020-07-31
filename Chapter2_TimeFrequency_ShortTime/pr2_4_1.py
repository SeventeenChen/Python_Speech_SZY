# 短时傅里叶变换

from Universal import *
from enframe import enframe
import matplotlib.pyplot as plt
import numpy as np

def STFT(x, win, nfft, inc):
	"""
	计算语音信号的短时傅里叶变换
	:param x:
	:param win:
	:param nfft:
	:param inc:
	:return D : complex matrix
	"""
	D = librosa.stft(x, n_fft=nfft, hop_length=inc, win_length=len(win), window='hann', center=False)

	return D

if __name__ == '__main__':
	Speech =  Speech("bluesky3.wav")
	x, Fs =Speech.audioread(8000)
	x = x / np.max(np.abs(x))
	inc = 64
	wlen = 256
	nfft = 256
	win = np.hanning(wlen)
	N = len(x)
	D = STFT(x, win, nfft, inc)
	Am = np.abs(D)
	Ph = np.angle(D)
	i = input("Which frame do you want to calculate")
	i = int(i)
	time = [i / Fs for i in range(N)]


	fig = plt.figure(figsize=(15, 18))
	plt.subplot(3, 1, 1)
	plt.plot(time, x)
	plt.xlabel('Time/s')
	plt.ylabel('Amplitude')
	plt.title('Speech Waveform')
	plt.subplot(3, 1, 2)
	plt.plot(Am[:, i + 1])
	plt.xlabel('Frequency/Hz')
	plt.xticks([0, 32, 64, 96, 128]\
			   ,[0, 32 * 8000 / 256, 64 * 8000 / 256, 96 * 8000 / 256, 128 * 8000 / 256])
	plt.ylabel('Amplitude')
	plt.title('STFT Amplitude in the {} Frame'.format(i))
	plt.subplot(3, 1, 3)
	plt.plot(Ph[:, i + 1])
	plt.xlabel('Frequency/Hz')
	plt.xticks([0, 32, 64, 96, 128]\
			   ,[0, 32 * 8000 / 256, 64 * 8000 / 256, 96 * 8000 / 256, 128 * 8000 / 256])
	plt.ylabel('Phase/rad')
	plt.title('STFT Phase in the {} Frame'.format(i))
	plt.savefig('images/stft.png')
	plt.show()
