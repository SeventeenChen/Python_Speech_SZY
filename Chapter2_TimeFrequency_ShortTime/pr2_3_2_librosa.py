# 短时平均过零率

from Universal import *
from enframe import enframe
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
	Speech =  Speech("bluesky3.wav")
	x, Fs =Speech.audioread(8000)
	# x = x / np.max(np.abs(x))
	inc = 80
	wlen = 200
	win = np.hanning(wlen)
	N = len(x)
	zcr = librosa.feature.zero_crossing_rate(x, frame_length = wlen, hop_length = inc, center = False)
	X = enframe(x, win, inc)
	fn = X.shape[0]

	time = [i / Fs for i in range(N)]

	frameTime = Speech.FrameTime(fn, wlen, inc, Fs)

	fig = plt.figure(figsize=(10, 13))
	plt.subplot(2, 1, 1)
	plt.plot(time, x)
	plt.xlabel('Time/s')
	plt.ylabel('Amplitude')
	plt.title('Speech Waveform')
	plt.subplot(2, 1, 2)
	plt.plot(frameTime, zcr.T)
	plt.xlabel('Time/s')
	plt.ylabel('Times')
	plt.title('Short Time Zero Cross Times')
	plt.savefig('images/zcr.png')
	plt.show()

