# VAD with 2 parameters
# pr6_2_2

from Noisy import *
from Universal import *
from VAD import *


def vad_param2D_revr(dst1, dst2, T1, T2, T3, T4=None):
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
	fn = len(dst1)              # frame number
	maxsilence = 8             # initialization
	minlen = 5
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
	voiceseg = vad.findSegemnt(SpeechIndex)
	vsl = len(voiceseg['begin'])
	
	return voiceseg, vsl, SF, NF

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
	inc = 80            # enframe
	IS = 0.25           # IS parameter
	overlap = wlen - inc
	NIS = int((IS * fs - wlen) / inc + 1)
	y = S.enframe(x, wlen, inc).T       # enframe
	fn = y.shape[1]                     # frame number
	amp = np.sum(y ** 2, axis=0)  # short-term average energy
	vad = VAD()
	zcr = vad.zc2(y, fn)  # short-term average zero-cross
	ampm = vad.multimidfilter(amp, 5)  # medfilter smoothing
	zcrm = vad.multimidfilter(zcr, 5)
	ampth = np.mean(ampm[0: NIS])  # non-speech segment energy
	zcrth = np.mean(zcrm[0: NIS])  # non-speech segment zero-cross
	amp2 = 1.1 * ampth
	amp1 = 1.3 * ampth  # energy threshold
	zcr2 = 0.9 * zcrth  # zreo-cross threshold
	
	frameTime = S.FrameTime(fn, wlen, inc, fs)
	
	voiceseg, vsl, SF, NF = vad_param2D_revr(amp, zcr, amp2, amp1, zcr2)
	
	# figure
	plt.figure(figsize=(16, 9))
	plt.subplot(2, 1, 1)
	plt.plot(time, xx)
	plt.ylabel('Amplitude')
	plt.xlabel('Time [s]')
	plt.axis([0, np.max(time), -1, 1])
	plt.title('VAD 2D parameters Reverse Comparison')

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
	plt.title('Noisy Speech SNR = {}dB'.format(SNR))
	plt.savefig('images/vad_param2D.png', bbox_inches='tight', dpi=600)
	plt.show()