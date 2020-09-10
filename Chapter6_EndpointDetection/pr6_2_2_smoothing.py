#
# pr6_2_2_smoothing

from Universal import *
from Noisy import *
from VAD import *
from scipy.signal import medfilt

def vad_ezrm(x, wlen, inc, NIS):
	"""
	 Smoothing Processing
	 Speech Endpoint Detection using Short-term average energy & Short-term average zero-crossing
	:param x: signal
	:param wlen: frame length
	:param inc: frame shift
	:param NIS: number of leading non-sentence frames
	:return voiceseg:
	:return vsl:
	:return SF:
	:return NF:
	"""
	x = x.reshape(-1, 1).squeeze()
	maxsilence = 15  # initilization
	minlen = 5
	status = 0
	count = [0]
	silence = [0]
	
	speech = Speech()
	y = speech.enframe(x, wlen, inc).T  # enframe
	fn = y.shape[1]  # frame number
	amp = np.sum(y ** 2, axis=0)  # short-term average energy
	vad = VAD()
	zcr = vad.zc2(y, fn)  # short-term average zero-cross
	ampm = multimidfilter(amp, 5)   # medfilter smoothing
	zcrm = multimidfilter(zcr, 5)
	ampth = np.mean(ampm[0: NIS])  # non-speech segment energy
	zcrth = np.mean(zcrm[0: NIS])  # non-speech segment zero-cross
	amp2 = 1.2 * ampth
	amp1 = 1.5 * ampth  # energy threshold
	zcr2 = 0.8 * zcrth  # zreo-cross threshold
	
	# Endpoint Detection
	xn = 0
	x1 = [0]
	x2 = [0]
	for n in range(fn):
		if status == 0 or status == 1:  # 0:silence, 1: maybe start
			if amp[n] > amp1:
				x1[xn] = np.max([n - count[xn], 1])
				status = 2
				silence[xn] = 0
				count[xn] = count[xn] + 1
			elif amp[n] > amp2 or zcr[n] < zcr2:  # maybe voice segment
				status = 1
				count[xn] = count[xn] + 1
			else:
				status = 0  # silence
				count[xn] = 0
				x1[xn] = 0
				x2[xn] = 0
		elif status == 2:  # voice
			if amp[n] > amp2 or zcr[n] < zcr2:
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
	voiceseg = vad.findSegemnt(SpeechIndex)
	vsl = len(voiceseg['begin'])
	
	return voiceseg, vsl, SF, NF

def multimidfilter(x, m):
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
	

if __name__ == '__main__':
	S = Speech()
	xx, fs = S.audioread('bluesky1.wav', 8000)
	xx = xx / np.max(np.abs(xx))  # normalized
	N = len(xx)
	time = np.arange(N) / fs
	noisy = Noisy()
	SNR = 10
	x, _ = noisy.Gnoisegen(xx, SNR)
	
	wlen = 200
	inc = 80  # enframe
	IS = 0.25  # IS parameter
	overlap = wlen - inc
	NIS = int((IS * fs - wlen) / inc + 1)
	fn = int((N - wlen) / inc) + 1
	frameTime = S.FrameTime(fn, wlen, inc, fs)
	
	voiceseg, vsl, SF, NF = vad_ezrm(x, wlen, inc, NIS)
	
	# figure
	plt.figure(figsize=(16, 9))
	plt.subplot(2, 1, 1)
	plt.plot(time, xx)
	plt.ylabel('Amplitude')
	plt.xlabel('Time [s]')
	plt.axis([0, np.max(time), -1, 1])
	plt.title('Endpoint Detection')

	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		nx3 = voiceseg['duration'][k]
		print('k = {}, begin = {}, end = {}, duration = {}'.format(k + 1, nx1, nx2, nx3))
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1.5, 1.5]), 'k', linewidth=2)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1.5, 1.5]), 'k--', linewidth=2)

	plt.subplot(2, 1, 2)
	plt.plot(time, x)
	plt.axis([0, max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech SNR = 10dB')
	plt.savefig('images/vad_ezrm.png', bbox_inches='tight', dpi=600)
	plt.show()