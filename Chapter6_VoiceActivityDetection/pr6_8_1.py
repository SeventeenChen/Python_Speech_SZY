# Mean amplitude product of wavelet decomposition short time coefficients
# pr6_8_1

import pywt

from Noisy import *
from Universal import *
from VAD import *


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
	# Set_I
	IS = 0.25  # unvoice segemnt length
	wlen = 200  # frame length 25ms
	inc = 80  # frame shift
	filename = 'bluesky1.wav'
	SNR = 10
	
	# PART_I
	speech = Speech()
	xx, fs = speech.audioread(filename, 8000)
	xx = xx - np.mean(xx)  # DC
	x = xx / np.max(xx)  # normalized
	N = len(x)
	time = np.arange(N) / fs
	noisy = Noisy()
	signal, _ = noisy.Gnoisegen(x, SNR)  # add noise
	wnd = np.hamming(wlen)  # window function
	overlap = wlen - inc
	NIS = int((IS * fs - wlen) / inc + 1)  # unvoice segment frame number
	y = speech.enframe(signal, list(wnd), inc).T
	fn = y.shape[1]  # frame number
	frameTime = speech.FrameTime(fn, wlen, inc, fs)  # frame to time
	
	# wavelet parameter
	start = np.array([1, 8, 15, 22, 29, 37, 47, 60, 79, 110, 165])
	send = np.array([7, 14, 21, 28, 36, 46, 59, 78, 109, 164, 267])
	duration = send - start + 1
	
	E = np.zeros(10)            # average amplitude
	MD = np.zeros(fn)
	for i in range(fn):
		u = y[:, i]
		c, l = mwavdec(u, 'db4', 10)
		for k in range(10):
			E[9 - k] = np.mean(np.abs(c[start[k + 1] : send[k + 1]]))
		M1 = np.max(E[0 : 5])
		M2 = np.max(E[5 : 10])
		MD[i] = M1 * M2
		
	
	Vad = VAD()
	MDm = Vad.multimidfilter(MD, 10)  # smoothing
	MDmth = np.mean(MDm[0 : NIS])
	T1 = 2 * MDmth
	T2 = 3 * MDmth
	[voiceseg, vsl, SF, NF] = Vad.vad_param1D(MDm, T1, T2)  # vad in ecr with 2 thresholds
	
	# figure
	plt.figure(figsize=(9, 16))
	plt.subplot(3, 1, 1)
	plt.plot(time, x)
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		print('{}, begin = {}, end = {}'.format(k + 1, nx1, nx2))
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1, 1]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1, 1]), 'k--', linewidth=1)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Clean Speech Signal')
	plt.subplot(3, 1, 2)
	plt.plot(time, signal)
	plt.axis([0, np.max(time), np.min(signal), np.max(signal)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Signal SNR = {}dB'.format(SNR))
	plt.subplot(3, 1, 3)
	plt.plot(frameTime, MDm)
	plt.axis([0, np.max(time), 0, 1.2 * np.max(MDm)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Mean amplitude product of wavelet decomposition short time coefficients')
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2 * np.max(MDm)]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2 * np.max(MDm)]), 'k--', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=1)
	plt.savefig('images/vad_wavelet.png', bbox_inches='tight', dpi=600)
	plt.show()