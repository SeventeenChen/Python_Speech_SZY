# mfcc function
# 提取MFCC参数

from Universal import Speech
from scipy.signal import *
import numpy as np
import math



class MFCC:
	def melbankm(self, p, n, fs, fl = 0, fh = 0.5, w = 't'):
		"""
		再Mel频率上设计平均分布的滤波器
		:param p: fl和fh之间设计的Mel滤波器的个数
		:param n: FFT长度
		:param fs: 采样频率
		:param fl: 设计滤波器的最低频率（用fs归一化，一般取0）
		:param fh: 设计滤波器的最高频率（用fs归一化，一般取0.5）
		:param w: 窗函数，'t'=triangle，'n'=hanning， 'm'=hanmming
		:return bank: 滤波器频率响应，size = p x (n/2 + 1), 只取正频率部分
		"""
		bl = 1125 * np.log(1 + fl * fs / 700)
		bh = 1125 * np.log(1 + fh * fs / 700) # Hz -> Mel
		B = bh - bl  						  # Mel Bandwidth
		y = np.linspace(0, B, p + 2)  		  # uniformed Mel
		Fb = 700 * (np.exp(y / 1125) - 1)     # Mel -> Hz
		W = int(n / 2 + 1)
		df = fs / n
		# freq = [i * df for i in range(W)]  # sample frequency
		bank = np.zeros((p, W))

		for m in range(1, p + 1):
			f0, f1, f2 = Fb[m], Fb[m - 1], Fb[m + 1]	# m, (m-1), (m+1) centeral frequency
			n0 = f0 / df						# frequency -> sampling point
			n1 = f1 / df
			n2 = f2 / df
			for k in range(W):
				if (n1 < k <= n0) & (w == 't') :
					bank[m - 1, k] = (k - n1) / (n0 - n1)
				elif (n1 < k <= n0) & (w == 'n'):
					bank[m - 1, k] = 0.5 - 0.5 * np.cos((k - n1) / (n0 - n1) * math.pi)
				elif (n1 < k <= n0) & (w == 'm'):
					bank[m - 1, k] = 25 / 46 -  21 / 46 * np.cos((k - n1) / (n0 - n1) * math.pi)
				elif (n0 < k <= n2) & (w == 't') :
					bank[m - 1, k]= (n2 - k) / (n2 - n0)
				elif (n0 < k <= n2) & (w == 'n') :
					bank[m - 1, k] = 0.5 - 0.5 * np.cos((n2 - k) / (n2 - n0) * math.pi)
				elif (n0 < k <= n2) & (w == 'm') :
					bank[m - 1, k] = 25 / 46 -  21 / 46 * np.cos((n2 - k) / (n2 - n0) * math.pi)

		return bank
	
	def mfcc(self, x, fs, p, frameSize, inc):
		"""
		提取MFCC参数
		:param x: discrete speech signal
		:param fs: sampling frequency
		:param p: filter number
		:param frameSize: frame length = FFT length
		:param inc: frame shift
		:return ccc: mfcc
		"""
		bank = self.melbankm(p, frameSize, fs, 0, 0.5, 't')

		# DCT coefficient
		dctcoef = np.zeros((12, p))
		for k in range(12):
			for n in range(p):
				dctcoef[k, n] = np.cos((2 * n + 1) * (k + 1) * math.pi / (2 * p))

		# ceps improvement window
		w = np.zeros((1, 12))
		for k in range(12):
			w[:, k] = 1 + 6 * np.sin(math.pi * (k + 1) / 12)
		w = w / np.max(w)

		# pre-emphasis, enframe
		xx = lfilter([1, -0.9375], [1], x)
		S = Speech()
		xx = S.enframe(x=xx, win=frameSize, inc = inc)
		n2 = int(np.fix(frameSize / 2))

		# calculate MFCC
		m = np.zeros((np.shape(xx)[0], 12))
		for i in range(np.shape(xx)[0]):
			y = xx[i, :]
			s = y.T * np.hamming(frameSize)
			t = np.abs(np.fft.fft(s)) ** 2
			c1 = np.dot(dctcoef, np.log(np.dot(bank, t[0: n2 + 1])))
			c2 = c1 * w
			m[i, :] = c2

		dtm = np.zeros((np.shape(m)))
		for i in range(2, (np.shape(m)[0] - 2)):
			dtm[i, :] = -2 * m[i - 2, :] - m[i - 1, :] + m[i + 1, :] + 2 * m[i + 2, :]

		dtm = dtm / 3

		ccc = np.concatenate((m, dtm), axis=1)

		ccc = ccc[2 : np.shape(m)[0]-2, :]

		return ccc

# if __name__ == '__main__':
# 	p = 24
# 	frameSize = 256
# 	inc = 80
# 	S =  Speech()
# 	x, Fs = S.audioread("s1.wav",8000)
# 	x = x / np.max(np.abs(x))
#
# 	MFCC = MFCC()
# 	c = MFCC.mfcc(x, Fs, p, frameSize, inc)
# 	# plt.figure(figsize=(18, 10))
# 	# plt.imshow(c, cmap = 'jet')
# 	# plt.xticks(np.arange(0, 24, step=1))
# 	# plt.yticks(np.arange(0, 18, step=1))
# 	# plt.colorbar()
# 	ax = sns.heatmap(c, linewidth=0.5)
# 	plt.title('MFCC')
# 	# plt.savefig('images/mfcc.png')
# 	plt.show()



