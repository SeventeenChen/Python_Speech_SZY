# Class Pitch Detection

from pywt import dwt, upcoef
from spectrum import xcorr

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
	
	def ACF_corr(self, y, fn, vseg, vsl, lmax, lmin):
		"""
		Auto correlation function Pitch detection
		:param y: enframe matrix (size: window length * frame number)
		:param fn: frame number
		:param vseg: vad
		:param vsl: vad
		:param lmax: min pitch period
		:param lmin: max pitch period
		:return period: pitch period
		"""
		pn = y.shape[1]
		if pn != fn:
			y = y.T
		wlen = y.shape[0]  # frame length
		period = np.zeros(fn)  # pitch period
		
		for i in range(vsl):  # only for voice segment
			ixb = vseg['begin'][i]  # segment begin index
			ixe = vseg['end'][i]  # segment end index
			ixd = ixe - ixb + 1  # segment duration
			for k in range(ixd):
				u = y[:, k + ixb - 1]  # one frame data
				ru, _ = xcorr(u, norm='coeff')
				ru = ru[wlen: 2 * wlen - 1]
				tloc = np.argmax(ru[lmin: lmax])  # find max in [lmin : lmax]
				period[k + ixb - 1] = lmin + tloc - 1
		
		return period
	
	def ACF_clip(self, y, fn, vseg, vsl, lmax, lmin):
		"""
		Auto correlation & clip function Pitch detection
		:param y: enframe matrix (size: window length * frame number)
		:param fn: frame number
		:param vseg: vad
		:param vsl: vad
		:param lmax: min pitch period
		:param lmin: max pitch period
		:return period: pitch period
		"""
		pn = y.shape[1]
		if pn != fn:
			y = y.T
		wlen = y.shape[0]  # frame length
		period = np.zeros(fn)  # pitch period
		
		for i in range(vsl):  # only for voice segment
			ixb = vseg['begin'][i]  # segment begin index
			ixe = vseg['end'][i]  # segment end index
			ixd = ixe - ixb + 1  # segment duration
			for k in range(ixd):
				u = y[:, k + ixb - 1]  # one frame data
				rate = 0.7                      # clip parameter
				CL = np.max(u) * rate           # threshold
				for j in range(wlen):
					if u[j] > CL:
						u[j] = u[j] - CL
					elif u[j] <= (-CL):
						u[j] = u[j] + CL
					else:
						u[j] =0
				u = u / np.max(u)  # normalization
				ru, _ = xcorr(u, norm='coeff')
				ru = ru[wlen: 2 * wlen - 1]
				tloc = np.argmax(ru[lmin: lmax])  # find max in [lmin : lmax]
				period[k + ixb - 1] = lmin + tloc - 1
		
		return period
	
	def ACF_threelevel(self, y, fn, vseg, vsl, lmax, lmin):
		"""
		Auto correlation & three level clip function Pitch detection
		:param y: enframe matrix (size: window length * frame number)
		:param fn: frame number
		:param vseg: vad
		:param vsl: vad
		:param lmax: min pitch period
		:param lmin: max pitch period
		:return period: pitch period
		"""
		pn = y.shape[1]
		if pn != fn:
			y = y.T
		wlen = y.shape[0]  # frame length
		period = np.zeros(fn)  # pitch period
		
		for i in range(vsl):  # only for voice segment
			ixb = vseg['begin'][i]  # segment begin index
			ixe = vseg['end'][i]  # segment end index
			ixd = ixe - ixb + 1  # segment duration
			for k in range(ixd):
				u = y[:, k + ixb - 1]  # one frame data
				px1 = u[0 : 100]        # first 100 points
				px2 = u[wlen - 100 : wlen] # last 100 points
				CLm = min(max(px1), max(px2))
				CL = CLm * 0.68  # threshold
				ThreeLevel = np.zeros(wlen)     # initialization
				for j in range(wlen):
					if u[j] > CL:
						u[j] = u[j] - CL
						ThreeLevel[j] = 1
					elif u[j] <= (-CL):
						u[j] = u[j] + CL
						ThreeLevel[j] = -1
					else:
						u[j] = 0
						ThreeLevel[j] = 0
				u = u / np.max(u)  # normalization
				ru, _ = xcorr(ThreeLevel, u, norm='coeff')
				ru = ru[wlen: 2 * wlen - 1]
				tloc = np.argmax(ru[lmin: lmax])  # find max in [lmin : lmax]
				period[k + ixb - 1] = lmin + tloc - 1
		
		return period
	
	def AMDF_1(self, y, fn, vseg, vsl, lmax, lmin):
		"""
		average magnitude difference function --> pitch detection
		:param y: enframe matrix (size: window length * frame number)
		:param fn: frame number
		:param vseg: vad
		:param vsl: vad
		:param lmax: min pitch period
		:param lmin: max pitch period
		:return period: pitch period
		"""
		
		pn = y.shape[1]
		if pn != fn:
			y = y.T
		wlen = y.shape[0]  # frame length
		period = np.zeros(fn)  # pitch period
		
		for i in range(vsl):  # only for voice segment
			ixb = vseg['begin'][i]  # segment begin index
			ixe = vseg['end'][i]  # segment end index
			ixd = ixe - ixb + 1  # segment duration
			R0 = np.zeros(wlen)  # average magnitude difference
			for k in range(ixd):
				u = y[:, k + ixb - 1]  # one frame data
				for m in range(wlen):
					R0[m] = np.sum(np.abs(u[m: wlen - 1] - u[0: wlen - m - 1]))     # AMDF
				Rmax = max(R0)
				Rth = 0.6 * Rmax   # threshold
				Rm = np.where(R0[lmin: lmax] <= Rth) # find range < threshold in [Pmin, Pmax]
				Rm = Rm[0]
				if Rm.size == 0:
					T0 = 0
				else:
					m11 = Rm[0]
					m22 = lmax
					T = np.argmin(R0[m11:m22])
					if not T:
						T0 = 0
					else:
						T0 = T + m11 - 1    # valley point in min
					period[k + ixb - 1] = T0
		return period
	
	def AMDF_mod(self, y, fn, vseg, vsl, lmax, lmin):
		"""
		modified average magnitude difference function --> pitch detection
		:param y: enframe matrix (size: window length * frame number)
		:param fn: frame number
		:param vseg: vad
		:param vsl: vad
		:param lmax: min pitch period
		:param lmin: max pitch period
		:return period: pitch period
		"""
	
		pn = y.shape[1]
		if pn != fn:
			y = y.T
		wlen = y.shape[0]  # frame length
		period = np.zeros(fn)  # pitch period
		
		for i in range(vsl):  # only for voice segment
			ixb = vseg['begin'][i]  # segment begin index
			ixe = vseg['end'][i]  # segment end index
			ixd = ixe - ixb + 1  # segment duration
			R0 = np.zeros(wlen)     # average magnitude difference
			R = np.zeros(wlen)  # linear transformation
			for k in range(ixd):
				u = y[:, k + ixb - 1]  # one frame data
				for m in range(wlen):
					R0[m] = np.sum(np.abs(u[m : wlen - 1] - u[0 : wlen - m - 1]))
				Rmax = max(R0)
				Nmax = np.argmax(R0)
				for j in range(wlen):
					R[j] = Rmax * (wlen - j) / (wlen - Nmax) - R0[j]
				T = np.argmax(R[lmin: lmax])  # find max in [lmin : lmax]
				T0 = T + lmin - 1
				period[k + ixb - 1] = T0
		
		return period
	
	def CAMDF(self, y, fn, vseg, vsl, lmax, lmin):
		"""
		circular average magnitude difference function --> pitch detection
		:param y: enframe matrix (size: window length * frame number)
		:param fn: frame number
		:param vseg: vad
		:param vsl: vad
		:param lmax: min pitch period
		:param lmin: max pitch period
		:return period: pitch period
		"""
		
		pn = y.shape[1]
		if pn != fn:
			y = y.T
		wlen = y.shape[0]  # frame length
		period = np.zeros(fn)  # pitch period
		
		for i in range(vsl):  # only for voice segment
			ixb = vseg['begin'][i]  # segment begin index
			ixe = vseg['end'][i]  # segment end index
			ixd = ixe - ixb + 1  # segment duration
			R = np.zeros(wlen)  # circular average magnitude difference
			for k in range(ixd):
				u = y[:, k + ixb - 1]  # one frame data
				for m in range(wlen - 1):
					R[m + 1] = 0
					for n in range(wlen - 1):  # circular average magnitude difference
						R[m + 1] = R[m + 1] + np.abs(u[(m + n) % wlen] - u[n + 1])
				Rmax = max(R[0 : lmin])
				Rth = 0.6 * Rmax  # threshold
				Rm = np.where(R[lmin: lmax] <= Rth)  # find range < threshold in [Pmin, Pmax]
				Rm = Rm[0]
				if Rm.size == 0:
					T0 = 0
				else:
					m11 = Rm[0]
					m22 = lmax
					T = np.argmin(R[m11:m22])
					if not T:
						T0 = 0
					else:
						T0 = T + m11 - 1  # valley point in min
					period[k + ixb - 1] = T0
		
		return period
	
	def CAMDF_mod(self, y, fn, vseg, vsl, lmax, lmin):
		"""
		modified circular average magnitude difference function --> pitch detection
		:param y: enframe matrix (size: window length * frame number)
		:param fn: frame number
		:param vseg: vad
		:param vsl: vad
		:param lmax: min pitch period
		:param lmin: max pitch period
		:return period: pitch period
		"""
		
		pn = y.shape[1]
		if pn != fn:
			y = y.T
		wlen = y.shape[0]  # frame length
		period = np.zeros(fn)  # pitch period
		
		for i in range(vsl):  # only for voice segment
			ixb = vseg['begin'][i]  # segment begin index
			ixe = vseg['end'][i]  # segment end index
			ixd = ixe - ixb + 1  # segment duration
			R0 = np.zeros(wlen)  # circular average magnitude difference
			R = np.zeros(wlen)
			for k in range(ixd):
				u = y[:, k + ixb - 1]  # one frame data
				for m in range(wlen - 1):
					R0[m + 1] = 0
					for n in range(wlen - 1):  # circular average magnitude difference
						R0[m + 1] = R0[m + 1] + np.abs(u[(m + n) % wlen] - u[n + 1])
				Rmax = max(R0)
				Nmax = np.argmax(R0)
				for j in range(wlen):
					R[j] = Rmax * (wlen - j) / (wlen - Nmax) - R0[j]
				T = np.argmax(R[lmin: lmax])  # find max in [lmin : lmax]
				T0 = T + lmin - 1
				period[k + ixb - 1] = T0
		
		return period
	
	def ACFAMDF_corr(self, y, fn, vseg, vsl, lmax, lmin):
		"""
			  auto correlation function (ACF)
		-------------------------------------------- >>> pitch detection
		average magnitude difference function (AMDF)
		:param y: enframe matrix (size: window length * frame number)
		:param fn: frame number
		:param vseg: vad
		:param vsl: vad
		:param lmax: min pitch period
		:param lmin: max pitch period
		:return period: pitch period
		"""
		pn = y.shape[1]
		if pn != fn:
			y = y.T
		wlen = y.shape[0]  # frame length
		period = np.zeros(fn)  # pitch period
		
		Acm = np.zeros(wlen)    # ACF/AMDF
		for i in range(vsl):  # only for voice segment
			ixb = vseg['begin'][i]  # segment begin index
			ixe = vseg['end'][i]  # segment end index
			ixd = ixe - ixb + 1  # segment duration
			
			for k in range(ixd):
				u = y[:, k + ixb - 1]  # one frame data
				ru, _ = xcorr(u, norm='coeff')  # auto correlation
				ru = ru[wlen: 2 * wlen - 1]
				R = np.zeros(wlen)  # average magnitude difference
				for m in range(wlen):
					R[m] = np.sum(np.abs(u[m : wlen - 1] - u[0 : wlen - m - 1]))    # AMDF
				R = R[0 : ru.size]  # the same length as ru
				Rindex = np.where(R)
				Acm[Rindex] = ru[Rindex] / R[Rindex]        # ACF/AMDF
				tloc = np.argmax(Acm[lmin: lmax])  # find max in [lmin : lmax]
				period[k + ixb - 1] = lmin + tloc - 1
		
		return period
	
	def Wavelet_corrm1(self, y, fn, vseg, vsl, lmax, lmin):
		"""
		Pitch detection based on autocorrelation of
		low frequency components of discrete wavelet transform
		:param y: enframe matrix (size: window length * frame number)
		:param fn: frame number
		:param vseg: vad
		:param vsl: vad
		:param lmax: min pitch period
		:param lmin: max pitch period
		:return period: pitch period
		"""
		pn = y.shape[1]
		
		if pn != fn:
			y = y.T                 # frame number * frame length
		period = np.zeros(fn)       # pitch period
		
		for i in range(vsl):
			ixb = vseg['begin'][i]  # segment begin index
			ixe = vseg['end'][i]  # segment end index
			ixd = ixe - ixb + 1  # segment duration
			
			for k in range(ixd):
				u = y[:, k + ixb - 1]       # one frame
				ca1, cd1 = dwt(u, 'db4')      # wavelet
				a1 = upcoef('a', ca1, 'db4', 1) # reconstruction with low frequency parameter
				ru, _ = xcorr(a1, norm = 'coeff')  # normalizaed correlation coefficient
				aL = len(a1)
				ru = ru[aL:]
				tloc = np.argmax(ru[lmin : lmax])
				period[k + ixb] = lmin + tloc - 1
		
		return period