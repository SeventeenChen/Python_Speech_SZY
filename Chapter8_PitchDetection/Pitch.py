# class Pitch Detection

from VAD import *

class Pitch:
	def linsmoothm(self, x, n=3):
		"""
		linear smoothing
		:param x:
		:param n: default = 3
		:return:
		"""
		win = np.hanning(n)  # hanning window
		win = win / np.sum(win)  # normalized
		x = x.reshape(-1, 1)  # --> column vector
		
		length = len(x)
		y = np.zeros(length)  # initialization
		# complement before and after the output to ensure the length is the same as x
		x = x.squeeze()
		if n % 2 == 0:
			L = int(n / 2)
			x = np.concatenate([np.ones(1) * x[0], x, np.ones(L) * x[L]]).T
		else:
			L = int((n - 1) / 2)
			x = np.concatenate([np.ones(1) * x[0], x, np.ones(L + 1) * x[L]]).T
		
		for k in range(length):
			y[k] = np.dot(win.T, x[k: k + n])
		
		return y
	
	def pitfilterm1(self, x, vseg, vsl):
		"""
		smoothing fundamental frequency T0
		:param x:
		:param vseg: voice segment
		:param vsl: voice sgement length
		:return y: linear smoothing T0
		"""
		y = np.zeros(x.shape)  # initialization
		for i in range(vsl):  # data segment
			ixb = vseg['begin'][i]
			ixe = vseg['end'][i]
			u0 = x[ixb: ixe]  # one segment data
			y0 = VAD().multimidfilter(u0, 5)  # 5 point medfilter
			v0 = self.linsmoothm(y0, 5)  # linear smoothing
			y[ixb: ixe] = v0
		
		return y