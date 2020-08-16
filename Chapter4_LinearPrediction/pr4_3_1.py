# pr4_3_1
# lpc latticem

from audiolazy import lazy_lpc
from Universal import *
import numpy as np

def latticem(x, L, p):
	"""
	用几何平均格型法求出线性预测的系数
	:param x: 一帧语音数据 （长度大于等于L+p）
	:param L: 该帧数据中做格型法处理的长度
	:param p: 线性预测的系数
	:return E: 最小均方误差
	:return G: 增益系数
	:return k: 反射系数
	:return aphaql: 预测系数（size：p x p）
	"""
	N = len(x)
	e = np.zeros((N, p + 1))
	b = np.zeros((N, p + 1))
	alphal = np.zeros((p, p))
	e[:, 0] = x
	b[:, 0] = x
	k = np.zeros(p)
	k[0] = np.sum(e[p: p + L, 0] * b[p - 1: p + L - 1, 0]) / \
		   np.sqrt((np.sum(e[p: p + L, 0] ** 2) * np.sum(b[p - 1: p + L - 1, 0] ** 2)))
	
	alphal[0, 0] = k[0]
	btemp = np.concatenate((np.zeros(1), b[:, 0].T))
	e[0: L + p, 1] = e[0: L + p, 0] - k[0] * btemp[0: L + p]
	b[0: L + p, 1] = btemp[0: L + p] - k[0] * e[0: L + p, 0]
	
	for i in range(1, p):
		k[i] = np.sum(e[p: p + L, i] * b[p - 1: p + L - 1, i]) \
			   / np.sqrt((np.sum(e[p: p + L, i] ** 2) * np.sum(b[p - 1:p + L - 1, i] ** 2)))
		alphal[i, i] = k[i]
		for j in range(i):
			alphal[j, i] = alphal[j, i - 1] - k[i] * alphal[i - j - 1, i - 1]
			btemp = np.concatenate((np.zeros(1), b[:, i].T))
		e[0: L + p, i + 1] = e[0: L + p, i] - k[i] * btemp[0: L + p]
		b[0: L + p, i + 1] = btemp[0: L + p] - k[i] * e[0: L + p, i]
	
	E = np.sum((x[p: p + L]) ** 2)
	
	for i in range(p):
		E = E * (1 - (k[i] ** 2))
	
	G = np.sqrt(E)
	
	
	return E, alphal, G, k
	
def lpcar2pf(ar, p=None):
	"""
	Convert AR coefs to power spectrum PF=(AR,NP)
	:param ar:
	:param p:
	:return pf:
	"""
	ar = np.array(ar)
	ar0 = ar.reshape(1, -1)
	nf, p1 = ar0.shape
	
	if p == None:
		p = p1 - 1
	pf = (np.abs((np.fft.rfft(ar0, n=2 * p + 2)).T ))** (-2)
	
	return pf

if __name__ == '__main__':
	S = Speech()
	x, fs = S.audioread('aa.wav', 8000)
	L = 240
	p = 12
	y = x[8000 : 8240 + p]
	
	EL, alpha1, GL, K = latticem(y, L, p)
	ar = alpha1[:, p - 1]
	a1 = lazy_lpc.lpc.autocor(y, p)
	Y = lpcar2pf(a1.numerator, 255)
	Y1 = lpcar2pf(np.concatenate((np.ones(1), -1 * ar)), 255)

	print('AR1 coefficients: \n {}'.format(-1 * ar))
	print('AR2 coefficients: \n {}'.format(a1.numerator[1 : p]))
	
	# figure
	plt.figure()
	m = np.linspace(0, 256, num=257)
	freq = m * fs / 512
	plt.plot(freq, 10 * np.log10(Y))
	plt.grid()
	plt.plot(freq, 10 * np.log10(Y1), linewidth = 3)
	plt.legend(['Normal Prediction', 'Lattice Prediction'])
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.title('Power Spectrum Response of \n Normal Prediction Lattice Prediction')
	plt.savefig('images/lpc_lattice.png', bbox_inches = 'tight',dpi = 600)
	plt.show()
	
	


	