# Short Term Energy Entropy Ratio
# pr6_7_2

from Noisy import *
from Universal import *
from VAD import *

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
	
	aparam = 2                                                  # parameter of energy
	Esum = np.zeros(fn)
	H = np.zeros(fn)
	Ef = np.zeros(fn)
	for i in range(fn):
		S = np.abs(np.fft.fft(y[:, i]))                         # FFT
		Sp = S[0 : int(wlen / 2 + 1)]                           # positive frequency
		Esum[i] = np.log10(1 + np.sum(Sp ** 2) / aparam)        # log energy
		prob = Sp / np.sum(Sp)                                  # probability
		EPS = np.finfo(float).eps
		H[i] = -np.sum(prob * np.log(prob + EPS))               # spectral entropy
		Ef[i] = np.sqrt(1 + np.abs(Esum[i] / H[i]))             # energy entropy ratio
	
	Vad = VAD()
	Enm = Vad.multimidfilter(Ef, 10)                            # smoothing
	Me = np.max(Enm)
	eth = np.mean(Enm[0 : NIS])
	Det = Me - eth                                              # threshold
	T1 = 0.05 * Det + eth
	T2 = 0.1 * Det + eth
	[voiceseg, vsl, SF, NF] = Vad.vad_param1D(Enm, T1, T2)  # vad in ecr with 2 thresholds
	
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
	plt.plot(frameTime, Enm)
	top = Det * 1.1 + eth
	bottom = eth - 0.1 * Det
	plt.axis([0, np.max(time), bottom, top])
	plt.xlabel('Time [s]')
	plt.ylabel('Energy Entropy Ratio')
	plt.title('Short Term Energy Entropy Ratio')
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([bottom, top]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([bottom, top]), 'k--', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=1)
	plt.savefig('images/vad_energy_entropy_ratio.png', bbox_inches='tight', dpi=600)
	plt.show()