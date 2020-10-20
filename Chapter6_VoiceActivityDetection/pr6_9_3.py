# Multitaper Spectrum & Energy Entropy Ratio
# pr6_9_3

from spectrum import *

from Noisy import *
from Universal import *
from VAD import *


def Mtmpsd_ssb(signal, wlen, inc, NIS, alpha, beta, c):
	"""
	Spectral Subtraction
	Multitaper Spectrum Estimation
	Short-term Energy Entropy Ratio
	:param signal: noisy speech
	:param wlen: frame length
	:param inc: frame shift
	:param NIS: leding unvoiced (noise) frame number
	:param alpha: over subtraction factor in spectral subtraction
	:param beta: gain compensation factor
	:param c: gain factor (0: power spectrum, 1: amplitude spectrum)
	:return output: denoise speech
	"""
	w2 = int(wlen / 2) + 1
	wind = np.hamming(wlen)                             # hamming window
	y = Speech().enframe(signal, list(wind), inc).T     # enframe
	fn = y.shape[1]                                     # frame number
	N = len(signal)                                     # signal length
	fft_frame = np.fft.fft(y, axis = 0)                 # FFT
	abs_frame = np.abs(fft_frame[0 : w2, :])            # positive frequency amplitude
	ang_frame = np.angle(fft_frame[0 : w2, :])          # positive frequency phase
	
	# smoothing in 3 neighbour frame
	abs_frame_backup = abs_frame
	for i in range(1 , fn - 1, 2):
		abs_frame_backup[:, i] = 0.25 * abs_frame[:, i - 1] + 0.5 * abs_frame[:, i] + 0.25 * abs_frame[:, i + 1]
	abs_frame = abs_frame_backup
	
	# multitaper spectrum estimation power spectrum
	PSDFrame = np.zeros((w2, fn))                           # PSD in each frame
	for i in range(fn):
		# PSDFrame[:, i] = pmtm(y[:, i], NW = 3, NFFT=wlen)
		Sk_complex, weights, eigenvalues = pmtm(y[:, i], NW = 3, NFFT=wlen)
		Sk = (np.abs(Sk_complex) ** 2).transpose()
		PSDTwoSide = np.mean(Sk * weights, axis=1)
		PSDFrame[:, i] = PSDTwoSide[0 : w2]
		
	PSDFrameBackup = PSDFrame
	for i in range(1 , fn - 1, 2):
		PSDFrameBackup[:, i] = 0.25 * PSDFrame[:, i - 1] + 0.5 * PSDFrame[:, i] + 0.25 * PSDFrame[:, i + 1]
	PSDFrame = PSDFrameBackup
	
	# average PSD of leading unvoiced segment
	NoisePSD = np.mean(PSDFrame[:, 0 : NIS], axis = 1)
	
	# spectral subtraction -> gain factor
	g = np.zeros((w2, fn))          # gain factor
	g_n = np.zeros((w2, fn))
	for k in range(fn):
		g[:, k] = (PSDFrame[:, k] - alpha * NoisePSD) / PSDFrame[:, k]
		g_n[:, k] = beta * NoisePSD / PSDFrame[:, k]
		gix = np.where(g[:, k] < 0)
		g[gix, k] = g_n[gix, k]
	
	gf = g
	if c == 0:
		g = gf
	else:
		g = np.sqrt(gf)
	
	SubFrame = g * abs_frame                # spectral subtraction amplitude
	output = Speech().OverlapAdd2(SubFrame, ang_frame, wlen, inc)   # synthesis
	output = output / np.max(np.abs(output))                        # normalized
	ol = len(output)
	if ol < N:
		output = np.concatenate((output, np.zeros(N - ol)))
	
	return output

if __name__ == '__main__':
	# Set_I
	IS = 0.25  # unvoice segemnt length
	wlen = 200  # frame length 25ms
	inc = 80  # frame shift
	filename = 'bluesky1.wav'
	SNR = 0
	
	# PART_I
	speech = Speech()
	xx, fs = speech.audioread(filename, 8000)
	xx = xx - np.mean(xx)  # DC
	x = xx / np.max(xx)  # normalized
	N = len(x)
	time = np.arange(N) / fs
	noisy = Noisy()
	signal, _ = noisy.Gnoisegen(x, SNR)  # add noise
	snr1 = noisy.SNR_singlech(x, signal)
	wnd = np.hamming(wlen)  # window function
	overlap = wlen - inc
	NIS = int((IS * fs - wlen) / inc + 1)  # unvoice segment frame number
	y = speech.enframe(signal, list(wnd), inc).T
	fn = y.shape[1]  # frame number
	frameTime = speech.FrameTime(fn, wlen, inc, fs)  # frame to time
	
	alpha = 2.8
	beta = 0.001
	c = 1
	# multitaper spectral estimation, spectral subtraction
	output = Mtmpsd_ssb(signal, wlen, inc, NIS, alpha, beta, c)
	snr2 = noisy.SNR_singlech(x, output)
	print('snr2 = {0:.2f}'.format(snr2))
	y = speech.enframe(output, list(wnd), inc).T        # enframe
	aparam = 2
	
	Esum = np.zeros(fn)
	H = np.zeros(fn)
	Ef = np.zeros(fn)
	for i in range(fn):
		S = np.abs(np.fft.fft(y[:, i]))  # FFT
		Sp = S[0: int(wlen / 2 + 1)]  # positive frequency
		Esum[i] = np.log10(1 + np.sum(Sp ** 2) / aparam)  # log energy
		prob = Sp / np.sum(Sp)  # probability
		EPS = np.finfo(float).eps
		H[i] = -np.sum(prob * np.log(prob + EPS))  # spectral entropy
		Ef[i] = np.sqrt(1 + np.abs(Esum[i] / H[i]))  # energy entropy ratio
	
	Vad = VAD()
	Enm = Vad.multimidfilter(Ef, 10)  # smoothing
	Me = np.max(Enm)
	eth = np.mean(Enm[0: NIS])
	Det = Me - eth  # threshold
	T1 = 0.05 * Det + eth
	T2 = 0.1 * Det + eth
	[voiceseg, vsl, SF, NF] = Vad.vad_param1D(Enm, T1, T2)  # vad in ecr with 2 thresholds
	
	# figure
	plt.figure(figsize=(9, 21))
	plt.subplot(4, 1, 1)
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
	plt.subplot(4, 1, 2)
	plt.plot(time, signal)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Signal SNR = {}dB'.format(SNR))
	plt.subplot(4, 1, 3)
	plt.plot(time, output)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Denoise Speech Signal SNR = {0:.2f}dB'.format(snr2))
	plt.subplot(4, 1, 4)
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
	plt.savefig('images/vad_multitaper_energy_entropy_ratio.png', bbox_inches='tight', dpi=600)
	plt.show()