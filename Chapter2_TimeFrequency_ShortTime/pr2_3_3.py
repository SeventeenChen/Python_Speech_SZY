# 短时自相关

import matplotlib.pyplot as plt
import numpy as np
from spectrum import xcorr

from Universal import *

if __name__ == '__main__':
	Speech =  Speech("bluesky3.wav")
	x, Fs =Speech.audioread(8000)
	# x = x / np.max(np.abs(x))
	inc = 80                        # frame shift
	wlen = 200                      # frame length
	win = np.hanning(wlen)          # window function
	N = len(x)                      # data length
	X = Speech.enframe(x, list(win), inc).T
	fn = X.shape[1]                 # frame number
	time = [i / Fs for i in range(N)]
	R = np.zeros(X.shape)
	for i in range(fn):
		u = X[:, i]
		R0, _ = xcorr(u)
		R[:, i] = R0[wlen - 1 : 2 * wlen - 1]


	i = input("Which frame do you want to plot")
	i = int(i)


	frameTime = Speech.FrameTime(fn, wlen, inc, Fs)

	fig = plt.figure(figsize=(9, 16))
	plt.subplot(3, 1, 1)
	plt.plot(time, x)
	plt.xlabel('Time/s')
	plt.ylabel('Amplitude')
	plt.title('Speech Waveform')
	plt.subplot(3, 1, 2)
	plt.plot(X[:, i])
	plt.xlabel('Sampling Points')
	plt.ylabel('Amplitude')
	plt.title('Onr Frame Speech Waveform')
	plt.subplot(3, 1, 3)
	plt.plot(R[:, i])
	plt.xlabel('Sampling Points')
	plt.ylabel('Amplitude')
	plt.title('Short Time Autocorrelation in Frame')
	plt.savefig('images/acor.png', bbox_inches='tight', dpi=600)
	plt.show()