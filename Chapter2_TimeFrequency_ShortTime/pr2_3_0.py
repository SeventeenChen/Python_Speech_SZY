# 短时平均幅度

from Universal import *
from enframe import enframe
import matplotlib.pyplot as plt
import numpy as np

def STM(x, win, inc):
    """
    短时平均幅度函数
    :param x:
    :param win:
    :param inc:
    :return:
    """
    X = enframe(x, win, inc)
    s = np.abs(X)
    return np.mean(s, axis=1)

def FrameTime(frameNum, frameLen, inc, fs):
    """
    分帧后计算每帧对应的时间
    """
    l = np.array([i for i in range(frameNum)])
    return ((l - 1) * inc + frameLen / 2) / fs

if __name__ == '__main__':
	x, fs = Speech("bluesky3.wav").audioread(8000)
	x = x / np.max(np.abs(x))
	inc = 80
	wlen = 200
	win = np.hanning(wlen)
	N = len(x)
	Am = STM(x, win, inc)
	Am = Am / np.max(np.abs(Am))
	X = enframe(x, win, inc)
	fn = X.shape[0]

	time = [i / fs for i in range(N)]
	frameTime = FrameTime(fn, wlen, inc, fs)

	fig = plt.figure(figsize=(10, 13))
	plt.subplot(2, 1, 1)
	plt.plot(time, x)
	plt.xlabel('Time/s')
	plt.ylabel('Normalized Amplitude')
	plt.title('Speech Waveform')
	plt.subplot(2, 1, 2)
	plt.plot(frameTime, Am)
	plt.xlabel('Time/s')
	plt.ylabel('Amplitude')
	plt.title('Normalized Short Time Amplitude')
	plt.savefig('images/amplitude.png')
	plt.show()
