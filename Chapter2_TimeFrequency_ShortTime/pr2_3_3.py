# 短时自相关

from Universal import *
from enframe import enframe
import matplotlib.pyplot as plt
import numpy as np

def STACOR(x):
    """
    计算短时相关函数
    :param x:
    :return:
    """
    acor = np.zeros(x.shape)
    fn = x.shape[0]
    for i in range(fn):
        R = np.correlate(x[i, :], x[i, :], 'full')
        acor[i, :] = R[(x.shape[1] - 1) :]
    return acor

if __name__ == '__main__':
	Speech =  Speech("bluesky3.wav")
	x, Fs =Speech.audioread(8000)
	# x = x / np.max(np.abs(x))
	inc = 80
	wlen = 200
	win = np.hanning(wlen)
	N = len(x)
	X = enframe(x, win, inc)
	acor = STACOR(X)
	fn = X.shape[0]
	i = input("Which frame do you want to calculate")
	i = int(i)
	time = [i / Fs for i in range(N)]

	frameTime = Speech.FrameTime(fn, wlen, inc, Fs)

	fig = plt.figure(figsize=(15, 13))
	plt.subplot(3, 1, 1)
	plt.plot(time, x)
	plt.xlabel('Time/s')
	plt.ylabel('Amplitude')
	plt.title('Speech Waveform')
	plt.subplot(3, 1, 2)
	plt.plot(X[i, :])
	plt.xlabel('Sampling Points')
	plt.ylabel('Amplitude')
	plt.title('Onr Frame Speech Waveform')
	plt.subplot(3, 1, 3)
	plt.plot(acor[i, :])
	plt.xlabel('Sampling Points')
	plt.ylabel('Amplitude')
	plt.title('Short Time Autocorrelation in Frame')
	plt.savefig('images/acor.png')
	plt.show()