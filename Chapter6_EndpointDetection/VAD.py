# class Vad

import numpy as np

class VAD:
	def zc2(self, y, fn):
		"""
		short-term average zero-cross
		:param y: signal
		:param fn: frame number
		:return zcr: zero-cross
		"""
		if y.shape[1] != fn:
			y = y.T
		
		wlen = y.shape[0]  # frame length
		zcr = np.zeros(fn)  # initialization
		delta = 0.01  # small threshold
		
		for i in range(fn):
			yn = y[:, i]
			ym = np.zeros(len(yn))
			for k in range(wlen):
				if yn[k] >= delta:
					ym[k] = yn[k] - delta
				elif yn[k] < -delta:
					ym[k] = yn[k] + delta
				else:
					ym[k] = 0
			
			zcr[i] = np.sum(ym[0: -1] * ym[1: len(ym)] < 0)
		
		return zcr
	
	def findSegemnt(self, express):
		"""
		find voice start, end and length
		:param express: speech index
		:return soundSegment:
		"""
		express = express[0]
		if express[0] == 0:  # find where express = 1
			voiceIndex = np.where(express)
		else:
			voiceIndex = express
		
		soundSegment = {}
		soundSegment.setdefault('begin', []).append(voiceIndex[0])
		k = 1
		
		for i in range(len(voiceIndex) - 1):
			if voiceIndex[i + 1] - voiceIndex[i] > 1:
				soundSegment.setdefault('end', []).append(voiceIndex[i])
				soundSegment.setdefault('begin', []).append(voiceIndex[i + 1])
				k = k + 1
		
		soundSegment.setdefault('end', []).append(voiceIndex[-1])
		
		for i in range(k):
			duration = soundSegment['end'][i] - soundSegment['begin'][i] + 1
			soundSegment.setdefault('duration', []).append(duration)
		
		return soundSegment
	