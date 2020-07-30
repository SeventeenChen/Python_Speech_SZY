# 短时平均幅度差

from Universal import *
from enframe import enframe
import matplotlib.pyplot as plt
import numpy as np

def STAMD(x):
	"""
	计算短时平均幅度差函数
	:param x
	:return:
	"""
	fn = x.shape[0]
	wlen = x.shape[1]
	amd = np.zeros((fn, wlen))
	for i in range(fn):
		u = x[i, :]
		for k in range(wlen):
			amd[i, k] = np.sum(np.abs(u[k:]-u[:len(u)-k]))
	return amd

if __name__ == '__main__':
	Speech =  Speech("bluesky3.wav")
	x, Fs =Speech.audioread(8000)
	# x = x / np.max(np.abs(x))
	inc = 80
	wlen = 200
	win = np.hanning(wlen)
	N = len(x)
	X = enframe(x, win, inc)
	amdf = STAMD(X)
	fn = X.shape[0]
	i = input("Which frame do you want to calculate")
	i = int(i)
	time = [i / Fs for i in range(N)]

	frameTime = Speech.FrameTime(fn, wlen, inc, Fs)

	fig = plt.figure(figsize=(15, 18))
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
	plt.plot(amdf[i, :])
	plt.xlabel('Sampling Points')
	plt.ylabel('Amplitude')
	plt.title('Short Time Average Magnitude Difference in Frame')
	plt.savefig('images/amd.png')
	plt.show()