# Mel频率倒谱系数
# Mel滤波器组

import matplotlib.pyplot as plt
import numpy as np
import math

def melbankm(p, n, fs, fl = 0, fh = 0.5, w = 't'):
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
	bl = 1125 * np.log(1 + fl * fs / 700)	# Hz -> Mel
	bh = 1125 * np.log(1 + fh * fs / 700)
	B = bh - bl		# Mel Bandwidth
	y = np.linspace(0, B, p + 2)		# Equally spaced Mel freq
	Fb = 700 * (np.exp(y / 1125) - 1)		# Mel -> Hz
	W = int(n/2 + 1)					# positive frequency
	df = fs / n							# frequency resolution
	bank = np.zeros((p, W))
	for k in range(1, p + 1):
		f0, f1, f2 = Fb[k], Fb[k - 1], Fb[k + 1]	# m, (m-1), (m+1) centeral frequency
		n0 = np.floor(f0 / df)						# frequency -> sampling point
		n1 = np.floor(f1 / df)
		n2 = np.floor(f2 / df)
		for i in range(1, W):
			if (n1 <= i <= n0) & (w == 't') :
				bank[k - 1, i] = (i - n1) / (n0 - n1)
			elif (n1 <= i <= n0) & (w == 'n'):
				bank[k - 1, i] = 0.5 - 0.5 * np.cos((i - n1) / (n0 - n1) * math.pi)
			elif (n1 <= i <= n0) & (w == 'm'):
				bank[k - 1, i] = 0.54 - 0.46 * np.cos((i - n1) / (n0 - n1) * math.pi)
			elif (n0 <= i <= n2) & (w == 't') :
				bank[k - 1, i]= (n2 - i) / (n2 - n0)
			elif (n0 <= i <= n2) & (w == 'n') :
				bank[k - 1, i] = 0.5 - 0.5 * np.cos((n2 - i) / (n2 - n0) * math.pi)
			elif (n0 <= i <= n2) & (w == 'm') :
				bank[k - 1, i] = 0.54 - 0.46 * np.cos((n2 - i) / (n2 - n0) * math.pi)
			elif i > n2:
				break

	return bank

if __name__ == '__main__':
	p = 24
	n = 256
	fs = 8000
	Ps = int(n / 2 + 1)
	w = 't'
	h1 = melbankm(p, n, fs, 0, 0.5, w)
	freq = [int(i * fs / n) for i in range(Ps)]
	plt.figure()
	for k in range(1, p + 1):
		plt.plot(freq, h1[k - 1, :],  'r', linewidth = 2)
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Relative Amplitude')
	plt.title('Triangle Window Frequency Reponse')
	plt.savefig('images/mel_tri.png')
	plt.show()

	w = 'n'
	h2 = melbankm(p, n, fs, 0, 0.5, w)
	freq = [int(i * fs / n) for i in range(Ps)]
	plt.figure()
	for k in range(1, p + 1):
		plt.plot(freq, h2[k - 1, :],  'k', linewidth = 2)
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Relative Amplitude')
	plt.title('Hanning Window Frequency Reponse')
	plt.savefig('images/mel_hanning.png')
	plt.show()

	w = 'm'
	h3 = melbankm(p, n, fs, 0, 0.5, w)
	freq = [int(i * fs / n) for i in range(Ps)]
	plt.figure()
	for k in range(1, p + 1):
		plt.plot(freq, h3[k - 1, :],  'b', linewidth = 2)
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Relative Amplitude')
	plt.title('Hamming Window Frequency Reponse')
	plt.savefig('images/mel_hamming.png')
	plt.show()



