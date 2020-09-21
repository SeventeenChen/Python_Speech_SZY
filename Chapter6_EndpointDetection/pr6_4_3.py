#
# pr6_4_3

from Universal import *
from Noisy import *
from VAD import *
from scipy.io import loadmat
from scipy.signal import resample

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
	
	# Bark sub-band
	Fk = loadmat('Fk.mat')['Fk']
	
	# interpolation
	fs2 = int(fs/2)
	y = y.T
	sourfft = np.zeros((fn, wlen), dtype=complex)
	sourfft1 = np.zeros((fn, int(wlen / 2)))
	sourre = np.zeros((fn, int(fs / 2)))
	for i in range(fn):
		sourfft[i, :] = np.fft.fft(y[i, :], wlen)                   # FFT
		sourfft1[i, :] = np.abs(sourfft[i, 0 : int(wlen / 2)])      # positive frequency
		sourre[i, :] = resample(sourfft1[i, :], fs2)  # spectral line interpolation
		
	# Bask filter number
	for k in range(25):
		if Fk[k, 2] > fs2:
			break
	
	num = k
	Dst = np.zeros(num)
	Dvar = np.zeros(fn)
	for i in range(fn):
		Sr = sourre[i, :]                   # one frame
		for k in range(num):
			m1 = Fk[k, 1]
			m2 = Fk[k, 2]                   # Bask filter cutoff frequency
			Srt = Sr[m1: m2]                # frequency line
			Dst[k] = np.var(Srt)
		Dvar[i] = np.mean(Dst)
	
	vad = VAD()
	Dvarm = vad.multimidfilter(Dvar, 10)    # smoothing
	dth = np.mean(Dvarm[0 : NIS])           # threshold
	T1 = 1.5 * dth
	T2 = 3 * dth
	voiceseg, vsl, SF, NF = vad.vad_param1D(Dvar, T1, T2)
	
	# figure
	plt.figure(figsize=(9, 16))
	plt.subplot(3, 1, 1)
	plt.plot(time, x)
	plt.axis([0, np.max(time), -1, 1])
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		nx3 = voiceseg['duration'][k]
		print('{}, begin = {}, end = {}, duration = {}'.format(k + 1, nx1, nx2, nx3))
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1, 1]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1, 1]), 'k--', linewidth=1)
	
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
	plt.plot(frameTime, Dvar)
	plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=0.5)
	plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=0.5)
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2 * np.max(Dvar)]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2 * np.max(Dvar)]), 'k--', linewidth=1)
	plt.grid()
	plt.axis([0, np.max(time), -0.1, 1.2 * np.max(Dvar)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Frequency Band Variance of Short-time BASK Subband')
	plt.savefig('images/vad_fre_bask_sub_band_var.png', bbox_inches='tight', dpi=600)
	plt.show()
	
