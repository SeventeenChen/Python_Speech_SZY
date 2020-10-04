#
# pr6_6_1

from Universal import *
from Noisy import *
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
	frameTime = speech.FrameTime(fn, wlen, inc, fs) # frame to time
	
	H = np.zeros(fn)                                    # spectral entropy function
	for i in range(fn):
		Sp = np.abs(np.fft.fft(y[:, i]))                # FFT
		Sp = Sp[0 : int(wlen/2) + 1]                    # positive frequency
		Ep = Sp ** 2                                    # energy
		prob = Ep / np.sum(Ep)                          # probability density
		EPS = np.finfo(float).eps
		H[i] = - np.sum(prob * np.log(prob + EPS))    # spectral entropy
	
	
	Vad = VAD()
	Enm = Vad.multimidfilter(H, 10)  # smoothing
	Me = np.min(Enm)
	eth = np.mean(Enm[0: NIS])
	Det = eth - Me
	T1 = 0.98 * Det + Me
	T2 = 0.93 * Det + Me
	[voiceseg, vsl, SF, NF] = Vad.vad_param1D_revr(Enm, T1, T2)
	
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
	top = 1.1 * Det + Me
	botton = Me - 0.1 * Det
	plt.axis([0, np.max(time), botton, top])
	plt.xlabel('Time [s]')
	plt.ylabel('Spectral Entropy')
	plt.title('Short-term Spectral Entropy')
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([botton, top]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([botton, top]), 'k--', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=1)
	plt.savefig('images/vad_spectral_entropy.png', bbox_inches='tight', dpi=600)
	plt.show()
