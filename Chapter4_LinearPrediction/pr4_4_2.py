# pr4_4_2
# LPCC

from Universal import *
from audiolazy.lazy_lpc import lpc

def lpc2lpccm(ar, n_lpc, n_lpcc):
	"""
	已知预测系数ar求LPCC
	:param ar: prediction coefficient
	:param n_lpc: length of ar
	:param n_lpcc: length of lpcc
	:return lpcc:
	"""
	lpcc = np.zeros(n_lpcc)
	lpcc[0] = ar[0]
	for n in range(1, n_lpc):
		lpcc[n] = ar[n]
		for k in range(n):
			lpcc[n] = lpcc[n] + ar[k] * lpcc[n - k - 1] * (n - k) / (n + 1)
		
	
	for n in range(n_lpc, n_lpcc):
		lpcc[n] = 0
		for k in range(n_lpc):
			lpcc[n] = lpcc[n] + ar[k] * lpcc[n - k - 1] * (n - k) / (n + 1)
			
	lpcc = -1 * lpcc
	
	return lpcc

def lpcc_dist(s1, s2, wlen, inc, p):
	"""
	已知两个信号，计算LPCC系数和距离
	:param s1: signal 1
	:param s2: signal 2
	:param wlen: frame length
	:param inc: frame shift
	:param p: LPC order
	:return DIST: LPCC distance
	:return s1lpcc: LPCC of s1 (size: p)
	:return s2lpcc: LPCC of s2 (size: p)
	"""
	S = Speech()
	y1 = S.enframe(x = s1, win = wlen, inc = inc)
	y2 = S.enframe(x = s2, win = wlen, inc = inc)
	# print(y1.shape)
	y1 = y1.T
	y2 = y2.T
	fn = y1.shape[1]
	s1lpcc = np.zeros((fn, p))
	s2lpcc = np.zeros((fn, p))
	
	for i in range(fn):
		u1 = y1[:, i]
		ar1 = lpc.autocor(u1, p)
		lpcc1 = lpc2lpccm(ar1.numerator, p, p)
		s1lpcc[i, :] = lpcc1
		u2 = y2[:, i]
		ar2 = lpc.autocor(u2, p)
		lpcc2 = lpc2lpccm(ar2.numerator, p, p)
		s2lpcc[i, :] = lpcc2
	
	DIST = np.zeros(fn)
	for i in range(fn):
		Cn1 = s1lpcc[i, :]
		Cn2 = s2lpcc[i, :]
		Dstu = 0
		for k in range(p):
			Dstu = Dstu + (Cn1[k] - Cn2[k]) ** 2
		
		DIST[i] = Dstu
		
	return DIST, s1lpcc, s2lpcc

if __name__ == '__main__':
	S = Speech()
	x1, fs = S.audioread('s1.wav', 8000)
	x2, _ = S.audioread('s2.wav', 8000)
	x3, _ = S.audioread('a1.wav', 8000)
	wlen = 200      # frame length
	inc = 80        # frame shift
	x1 = x1 / np.max(np.abs(x1))    # normalized
	x2 = x2 / np.max(np.abs(x2))
	x3 = x3 / np.max(np.abs(x3))
	p = 12                          # LPC order

	DIST12, y1lpcc, y2lpcc = lpcc_dist(x1, x2, wlen, inc, p)
	DIST13, y1lpcc, y3lpcc = lpcc_dist(x1, x3, wlen, inc, p)
	
	plt.figure(figsize=(32, 18))
	plt.plot(y1lpcc[2, :], y2lpcc[2, :], '+')
	plt.plot(y1lpcc[6, :], y2lpcc[6, :], 'x')
	plt.plot(y1lpcc[11, :], y2lpcc[11, :], '^')
	plt.plot(y1lpcc[15, :], y2lpcc[15, :], 'h')
	plt.legend(['3rd frame','7th frame','12th frame','16th frame'])
	plt.plot([-6, 6], [-6, 6], 'k--', linewidth = 2)
	plt.title('The Comparision of LPCC of /i1/ & /a1/')
	plt.xlabel('Signal x1')
	plt.ylabel('Signal x3')
	plt.axis([-6, 6, -6, 6])
	plt.savefig('images/lpcc.png', bbox_inches = 'tight', dpi=600)
	plt.show()
	