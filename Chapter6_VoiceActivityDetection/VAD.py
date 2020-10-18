# class Vad

import numpy as np
from scipy.signal import medfilt

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
	
	def multimidfilter(self, x, m):
		"""
		Multiple calls medfilt
		:param x: signal
		:param m: call times
		:return y:
		"""
		a = x
		for k in range(m):
			b = medfilt(a, 5)
			a = b
		
		y = a
		
		return y
	
	def vad_param1D(self, dst1, T1, T2):
		"""
		VAD with one parameter
		:param dst1:
		:param T1: threshold
		:param T2:
		:return voiceseg:
		:return vsl:
		:return SF:
		:return NF:
		"""
		dst1 = dst1.reshape(-1, 1)
		fn = dst1.shape[0]  # frame number
		maxsilence = 8  # initialization
		minlen = 5
		status = 0
		count = [0]
		silence = [0]
		
		# Endpoint Detection
		xn = 0
		x1 = [0]
		x2 = [0]
		for n in range(1, fn):
			if status == 0 or status == 1:  # 0:silence, 1: maybe start
				if dst1[n] > T2:
					x1[xn] = np.max([n - count[xn], 1])
					status = 2
					silence[xn] = 0
					count[xn] = count[xn] + 1
				elif dst1[n] > T1:  # maybe voice segment
					status = 1
					count[xn] = count[xn] + 1
				else:
					status = 0  # silence
					count[xn] = 0
					x1[xn] = 0
					x2[xn] = 0
			elif status == 2:  # voice
				if dst1[n] > T1:
					count[xn] = count[xn] + 1
				else:
					silence[xn] = silence[xn] + 1
					if silence[xn] < maxsilence:  # silence exists, voice hasn't ended
						count[xn] = count[xn] + 1
					elif count[xn] < minlen:  # voice length too short, considered silent or noisy
						status = 0
						silence[xn] = 0
						count[xn] = 0
					else:  # voice end
						status = 3
						x2[xn] = x1[xn] + count[xn]
			elif status == 3:  # voice end, prepare for next
				status = 0
				xn = xn + 1
				count.append(0)
				silence.append(0)
				x1.append(0)
				x2.append(0)
		
		el = len(x1)
		if x1[el - 1] == 0:
			el = el - 1  # x1 real length
		if x2[el - 1] == 0:
			print('Error: Not find endding point! \n')
			x2[el - 1] = fn
		
		SF = np.zeros(fn)  # define SF & NF, according to x1, x2
		NF = np.ones(fn)
		
		for i in range(el):
			SF[(x1[i] - 1): x2[i]] = 1
			NF[(x1[i] - 1): x2[i]] = 0
		
		SpeechIndex = np.where(SF == 1)  # voice segemnt
		vad = VAD()
		voiceseg = vad.findSegemnt(SpeechIndex)
		vsl = len(voiceseg['begin'])
		
		return voiceseg, vsl, SF, NF
	
	def vad_param1D_revr(self, dst1, T1, T2):
		"""
		VAD with one parameter reverse threshold
		:param dst1:
		:param T1: threshold
		:param T2:
		:return voiceseg:
		:return vsl:
		:return SF:
		:return NF:
		"""
		dst1 = dst1.reshape(-1, 1)
		fn = dst1.shape[0]  # frame number
		maxsilence = 8  # initialization
		minlen = 5
		status = 0
		count = [0]
		silence = [0]
		
		# Endpoint Detection
		xn = 0
		x1 = [0]
		x2 = [0]
		for n in range(1, fn):
			if status == 0 or status == 1:  # 0:silence, 1: maybe start
				if dst1[n] < T2:
					x1[xn] = np.max([n - count[xn], 1])
					status = 2
					silence[xn] = 0
					count[xn] = count[xn] + 1
				elif dst1[n] < T1:  # maybe voice segment
					status = 1
					count[xn] = count[xn] + 1
				else:
					status = 0  # silence
					count[xn] = 0
					x1[xn] = 0
					x2[xn] = 0
			elif status == 2:  # voice
				if dst1[n] < T1:
					count[xn] = count[xn] + 1
				else:
					silence[xn] = silence[xn] + 1
					if silence[xn] < maxsilence:  # silence exists, voice hasn't ended
						count[xn] = count[xn] + 1
					elif count[xn] < minlen:  # voice length too short, considered silent or noisy
						status = 0
						silence[xn] = 0
						count[xn] = 0
					else:  # voice end
						status = 3
						x2[xn] = x1[xn] + count[xn]
			elif status == 3:  # voice end, prepare for next
				status = 0
				xn = xn + 1
				count.append(0)
				silence.append(0)
				x1.append(0)
				x2.append(0)
		
		el = len(x1)
		if x1[el - 1] == 0:
			el = el - 1  # x1 real length
		if x2[el - 1] == 0:
			print('Error: Not find endding point! \n')
			x2[el - 1] = fn
		
		SF = np.zeros(fn)  # define SF & NF, according to x1, x2
		NF = np.ones(fn)
		
		for i in range(el):
			SF[(x1[i] - 1): x2[i]] = 1
			NF[(x1[i] - 1): x2[i]] = 0
		
		SpeechIndex = np.where(SF == 1)  # voice segemnt
		vad = VAD()
		voiceseg = vad.findSegemnt(SpeechIndex)
		vsl = len(voiceseg['begin'])
		
		return voiceseg, vsl, SF, NF
	
	def vad_param2D_revr(self, dst1, dst2, T1, T2, T3, T4=None):
		"""
		VAD with two parameters, dst1 & dst2, dst2: reverse comparison
		:param dst1: parameter1
		:param dst2: parameter2
		:param T1, T2, T3, T4: threshold
		:return voiceseg:
		:return vsl:
		:return SF:
		:return NF:
		"""
		fn = len(dst1)  # frame number
		maxsilence = 8  # initialization
		minlen = 8
		status = 0
		count = [0]
		silence = [0]
		
		# Endpoint Detection
		xn = 0
		x1 = [0]
		x2 = [0]
		for n in range(fn):
			if status == 0 or status == 1:  # 0:silence, 1: maybe start
				if dst1[n] > T2 or (T4 != None and dst2[n] < T4):
					x1[xn] = np.max([n - count[xn], 1])
					status = 2
					silence[xn] = 0
					count[xn] = count[xn] + 1
				elif dst1[n] > T1 or dst2[n] < T3:  # maybe voice segment
					status = 1
					count[xn] = count[xn] + 1
				else:
					status = 0  # silence
					count[xn] = 0
					x1[xn] = 0
					x2[xn] = 0
			elif status == 2:  # voice
				if dst1[n] > T1 or dst2[n] < T3:
					count[xn] = count[xn] + 1
				else:
					silence[xn] = silence[xn] + 1
					if silence[xn] < maxsilence:  # silence exists, voice hasn't ended
						count[xn] = count[xn] + 1
					elif count[xn] < minlen:  # voice length too short, considered silent or noisy
						status = 0
						silence[xn] = 0
						count[xn] = 0
					else:  # voice end
						status = 3
						x2[xn] = x1[xn] + count[xn]
			elif status == 3:  # voice end, prepare for next
				status = 0
				xn = xn + 1
				count.append(0)
				silence.append(0)
				x1.append(0)
				x2.append(0)
		
		el = len(x1)
		if x1[el - 1] == 0:
			el = el - 1  # x1 real length
		if x2[el - 1] == 0:
			print('Error: Not find endding point! \n')
			x2[el - 1] = fn
		
		SF = np.zeros(fn)  # define SF & NF, according to x1, x2
		NF = np.ones(fn)
		
		for i in range(el):
			SF[(x1[i] - 1): x2[i]] = 1
			NF[(x1[i] - 1): x2[i]] = 0
		
		SpeechIndex = np.where(SF == 1)  # voice segemnt
		vad = VAD()
		voiceseg = vad.findSegemnt(SpeechIndex)
		vsl = len(voiceseg['begin'])
		
		return voiceseg, vsl, SF, NF
	
	def vad(self, signal, noise, NoiseCounter=0, NoiseMargin=3.0, Hangover=8):
		"""
		Spectral Distance Voice Activity Detector
		:param signal: the current frames magnitude spectrum which is to labeld as noise or speech
		:param noise: noise magnitude spectrum template (estimation)
		:param NoiseCounter: the number of imediate previous noise frames
		:param NoiseMargin: (default 3) the spectral distance threshold
		:param Hangover: (default 8) the number of noise segments after which the SPEECHFLAG is reset (goes to zero)
		:return NoiseCounter: the number of previous noise segments
		:return Dist: spectral distance
		"""
		FreqResol = len(signal)
		
		SpectralDist = 20 * (np.log10(signal) - np.log10(noise))
		SpectralDist[np.where(SpectralDist < 0)] = 0
		
		Dist = np.mean(SpectralDist)
		
		if (Dist < NoiseMargin):
			NoiseFlag = 1
			NoiseCounter = NoiseCounter + 1
		else:
			NoiseFlag = 0
			NoiseCounter = 0
		
		# Detect noise only periods and attenuate the signal
		if (NoiseCounter > Hangover):
			SpeechFlag = 0
		else:
			SpeechFlag = 1
		
		return NoiseFlag, SpeechFlag, NoiseCounter, Dist