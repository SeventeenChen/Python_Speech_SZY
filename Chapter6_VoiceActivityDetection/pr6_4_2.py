# Frequency Band Variance of Short-time Uniform Subband
# pr6_4_1

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
	
	Y = np.fft.fft(y, axis=0)           # FFT
	N2 = int(wlen / 2 + 1)              # positive frequency
	n2 = np.arange(N2)
	Y_abs = np.abs(Y[n2, :])            # amplitude
	M = int(np.fix(N2 / 4))                  # subband number
	SY = np.zeros((M, fn))
	Dvar = np.zeros(fn)
	for k in range(fn):
		for i in range(M):
			j = i *4 + 1                    # 4 spectral line
			SY[i, k] = Y_abs[j, k] + Y_abs[j + 1, k] + Y_abs[j + 2, k] + Y_abs[j + 3, k]
		Dvar[k] = np.var(SY[:, k])          # frequency sub-band variance per frame
	
	vad = VAD()
	Dvarm = vad.multimidfilter(Dvar, 10)        # smoothing
	dth = np.mean(Dvar[0 : NIS])                # threshold
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
	
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2 * np.max(Dvar)]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2 * np.max(Dvar)]), 'k--', linewidth=1)
	plt.grid()
	plt.axis([0, np.max(time), 0, 1.2 * np.max(Dvar)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Frequency Band Variance of Short-time Uniform Subband')
	plt.savefig('images/vad_fre_sub_band_var.png', bbox_inches='tight', dpi=600)
	plt.show()
	
			
	