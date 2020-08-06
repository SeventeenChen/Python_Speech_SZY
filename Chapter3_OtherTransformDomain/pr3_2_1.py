# DCT 离散余弦变换

from scipy.fft import dct, idct
import matplotlib.pyplot as plt
import numpy as np
import math

def DCT(x):
	"""
	计算离散余弦变换，MATLAB dct()
	:param x:
	:return:
	"""
	return dct(x, norm='ortho')

def IDCT(x):
	"""
	离散余弦逆变换
	:param x:
	:return:
	"""
	return idct(x, norm='ortho')

if __name__ == '__main__':
	f = 50
	fs = 1000
	N = 1000
	xn = np.zeros(N)
	for i in range(N):
		xn[i] = np.cos(2 * math.pi * f * i / fs)
	y = DCT(xn)
	y = np.where(abs(y) < 5, 0, y)
	zn = IDCT(y)
	rp = 100 - np.linalg.norm((xn - zn)) / np.linalg.norm(xn) * 100  # reconstruction percent
	print('rp = {}'.format(rp))

	plt.figure(figsize=(15, 18))
	plt.subplot(2, 1, 1)
	plt.plot(xn)
	plt.xlabel('n')
	plt.ylabel('Amplitude')
	plt.title('Original Sequence x(n)')
	plt.subplot(2, 1, 2)
	plt.plot(zn)
	plt.xlabel('n')
	plt.ylabel('Amplitude')
	plt.title('DCT Reconstruction Sequence z(n)')
	plt.savefig('images/dct.png')
	plt.show()