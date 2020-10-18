# Short-term Cepstrum Distance
# pr6_5_2

from Noisy import *
from Universal import *
from VAD import *


def rcceps(x):
    """
    计算实倒谱
    """
    y = np.fft.fft(x)

    return np.fft.ifft(np.log(np.abs(y))).real



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
	
	U = np.zeros((wlen, fn))
	for i in range(fn):
		u = y[:, i]                             # one frame
		U[:, i] = rcceps(u)                     # real cepstrum
	
	C0 = np.mean(U[:, 0 : 4], axis=1)           # first 5th frame cepstrum coefficient average as background noise cepstrum coefficient
	
	Dcep = np.zeros(fn)
	for i in range(5, fn):
		Cn = U[:, i]
		Dst0 = (Cn[0] - C0[0]) ** 2
		Dstm= 0
		for k in range(1, 12):
			Dstm += (Cn[k] - C0[k]) ** 2
		Dcep[i] = 4.3429 * np.sqrt(Dst0 + Dstm)     # cepstrum distance
		
	Dcep[0:4] = Dcep[5]
	
	
	Vad = VAD()
	Dstm = Vad.multimidfilter(Dcep, 10)             # smoothing
	dth = np.max(Dstm[0 : NIS])
	T1 = dth
	T2 = 1.5 * dth
	[voiceseg, vsl, SF, NF] = Vad.vad_param1D(Dstm, T1, T2)
	
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
	plt.plot(frameTime, Dstm)
	plt.axis([0, np.max(time), 0, 1.2 * np.max(Dstm)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Short-term Cepstrum Distance')
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2 * np.max(Dstm)]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2 * np.max(Dstm)]), 'k--', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=1)
	plt.savefig('images/vad_cepstrum_distance.png', bbox_inches='tight', dpi=600)
	plt.show()
