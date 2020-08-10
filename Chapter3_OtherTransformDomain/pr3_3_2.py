# pr3_3_2
# MFCC 参数比较


from scipy.signal import *
import matplotlib.pylab as plt
from Universal import *
from MFCC import *
import numpy as np
import librosa
import math

def mel_dist(x1, x2, fs, num, wlen, inc):
	"""
	计算两信号x1，x2的MFCC参数和距离
	:param x1: signal 1
	:param x2: signal 2
	:param fs: sample frequency
	:param num: the number we select in MFCC
	:param wlen: frame length
	:param inc: frame shift
	:return Dcep: distance
	:return Ccep1, Ccep2: num MFCC
	"""
	M = MFCC()
	ccc1 = M.mfcc(x1, Fs, num, wlen, inc)		# MFCC
	ccc2 = M.mfcc(x2, Fs, num, wlen, inc)
	fn1 = np.shape(ccc1)[0]		# frame number
	Ccep1 = ccc1[:, 0 : num]
	Ccep2 = ccc2[:, 0 : num]

	Dcep = np.zeros(fn1)	# distance
	for i in range(fn1):
		Cn1 = Ccep1[i, :]
		Cn2 = Ccep2[i, :]
		Dstu = 0
		for k in range(num):
			Dstu = Dstu + (Cn1[k] - Cn2[k]) ** 2

		Dcep[i] = np.sqrt(Dstu)


	return Dcep, Ccep1, Ccep2

if __name__ == '__main__':
	x1, Fs = librosa.load('s1.wav', sr = 8000)
	x2, _ = librosa.load('s2.wav', sr = 8000)
	x3, _ = librosa.load('a1.wav', sr=8000)


	wlen = 200		# frame length
	inc = 80		# frame shift
	num = 16

	x1 = x1 / np.max(x1)	# normalized
	x2 = x2 / np.max(x2)
	x3 = x3 / np.max(x3)

	Dcep, Ccep1, Ccep2 = mel_dist(x1, x2, Fs, 16, wlen, inc)

	plt.figure(1, dpi = 600)
	plt.plot(Ccep1[2, :], Ccep2[2, :], 'k+')
	plt.plot(Ccep1[6, :], Ccep2[6, :], 'rx')
	plt.plot(Ccep1[11, :], Ccep2[11, :], 'b^')
	plt.plot(Ccep1[15, :], Ccep2[15, :], 'gh')
	plt.legend(['3rd frame', '7th frame', '12th frame', '16th frame'], loc='best')
	plt.xlabel('Signal x1')
	plt.ylabel('Signal x2')
	plt.axis([-12, 12, -12, 12])
	plt.plot([-12, 12], [-12, 12], 'k--')
	plt.title('The Distance of x1 & x2')
	plt.savefig('images/mel_dist_12.png')
	plt.show()

	Dcep, Ccep1, Ccep2 = mel_dist(x1, x3, Fs, 16, wlen, inc)

	plt.figure(2, dpi = 600)
	plt.plot(Ccep1[2, :], Ccep2[2, :], 'k+')
	plt.plot(Ccep1[6, :], Ccep2[6, :], 'rx')
	plt.plot(Ccep1[11, :], Ccep2[11, :], 'b^')
	plt.plot(Ccep1[15, :], Ccep2[15, :], 'gh')
	plt.legend(['3rd frame', '7th frame', '12th frame', '16th frame'], loc='best')
	plt.xlabel('Signal x1')
	plt.ylabel('Signal x3')
	plt.axis([-12, 12, -12, 12])
	plt.plot([-12, 12], [-12, 12], 'k--')
	plt.title('The Distance of x1 & x3')
	plt.savefig('images/mel_dist_13.png')
	plt.show()







