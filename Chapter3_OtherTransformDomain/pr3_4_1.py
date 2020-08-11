# pr3_4_1
# 小波变换

from scipy.signal import find_peaks
from Universal import *
import numpy as np
import pywt

def mwavdec(data, wavelet, level):
	"""
	参考MATLAB wavdec 对时间序列进行一维多分辨率分解
	:param data: 待分析时间序列
	:param wavlet: 选区的小波函数
	:param level: 分解层数
	:return C: 时间序列的一维多分辨率分解系数
	:return L: 各系数长度，末尾元素为总长度
	"""

	coeffs = pywt.wavedec(data = data, wavelet = wavelet, level = level)
	C = np.concatenate(coeffs, axis=0)
	L = np.zeros(len(coeffs) + 1)
	for i in range(len(coeffs)):
		L[i] = (np.array(coeffs[i])).size
	L[-1] = C.size

	return C, L


if __name__ == '__main__':
	S = Speech()
	x, fs = S.audioread('awav.wav', 8000)
	N = len(x)
	x = x - np.mean(x)		# DC
	J = 2 			# order = 2

	C, L = mwavdec(x, 'db1', J)
	CaLen = int(N / 2 ** J)		# approximate length
	Ca = C[0 : CaLen]
	Ca = (Ca - np.min(Ca)) / (np.max(Ca) - np.min(Ca))	# normalized
	for i in range(CaLen):
		if(Ca[i] < 0.8):
			Ca[i] = 0

	K, _ = find_peaks(Ca, distance=6)
	V = Ca[K]
	lk = len(K)
	dis = np.zeros(lk-1)
	if lk:
		for i in range(1, lk):
			dis[i-1] = K[i] - K[i - 1] + 1
		distance = np.mean(dis)
		pit = fs/2**J/distance
	else:
		pit = 0


	plt.figure(figsize=(10,12), dpi = 600)
	plt.subplot(2, 1, 1)
	plt.plot(x)
	plt.title('One Frame Speech Signal')
	plt.subplot(2, 1, 2)
	plt.plot(Ca)
	plt.title('The Peak of The Center Factor'\
			  '\nClipping after Wavelet Decomposition', ha='center')
	plt.savefig('images/wavelet.png')
	plt.show()



