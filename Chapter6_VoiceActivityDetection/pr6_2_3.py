# VAD normalized short-term average energy
# pr6_2_3

from Universal import *
from VAD import *


def vad_param1D(dst1,T1,T2):
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
	fn = dst1.shape[0]              # frame number
	maxsilence = 8                  # initialization
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
			elif dst1[n] > T1:         # maybe voice segment
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
	

if __name__ == '__main__':
	S = Speech()
	xx, fs = S.audioread('bluesky1.wav', 8000)
	x = xx / np.max(np.abs(xx))                # normalized
	N = len(xx)
	time = np.arange(N) / fs
	
	wlen = 200
	inc = 80  # enframe
	IS = 0.25  # IS parameter
	overlap = wlen - inc
	NIS = int((IS * fs - wlen) / inc + 1)
	y = S.enframe(x, wlen, inc).T  # enframe
	fn = y.shape[1]  # frame number
	etemp = np.sum(y ** 2, axis=0)  # short-term average energy
	etemp = etemp / np.max(etemp)       # energy normalized
	T1 = 0.002                          # threshold
	T2 = 0.01
	
	frameTime = S.FrameTime(fn, wlen, inc, fs)
	
	voiceseg, vsl, SF, NF = vad_param1D(etemp,T1,T2)
	
	# figure
	plt.figure(figsize=(16, 9))
	plt.subplot(2, 1, 1)
	plt.plot(time, x)
	plt.ylabel('Amplitude')
	plt.xlabel('Time [s]')
	plt.axis([0, np.max(time), -1, 1])
	plt.title('Clean Speech Signal')
	
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		nx3 = voiceseg['duration'][k]
		print('{}, begin = {}, end = {}, duration = {}'.format(k + 1, nx1, nx2, nx3))
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1, 1]), 'k', linewidth=2)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1, 1]), 'k--', linewidth=2)
	
	plt.subplot(2, 1, 2)
	plt.plot(frameTime, etemp, 'k')
	plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=1)
	plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=1)
	plt.axis([0, max(time), -0.1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('short-term Energy')
	plt.savefig('images/vad_param1D.png', bbox_inches='tight', dpi=600)
	plt.show()