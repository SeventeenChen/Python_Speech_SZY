#
# pr6_9_1

from scipy.signal import resample

from Universal import *
from VAD import *


def add_noisefile(s, filepath_name, SNR, fs):
	"""
	add arbitrary noise wav to clean speech according to SNR
	:param s: clean speech signal
	:param filepath_name: arbitrary noise filename (.wav)
	:param SNR: dB
	:param fs: sample frequency
	:return y: noisy speech
	:return noise: noise scaled by SNR
	"""
	
	s = s.squeeze()                                     # transform signal to n x 1
	wavin, fs1 = Speech().audioread(filepath_name, None)   # noise data
	if fs1 != fs:
		wavin1 = resample(wavin, int(len(wavin) / fs1 * fs))
	else:
		wavin1 = wavin
	wavin1 = wavin1 - np.mean(wavin1)                   # DC
	ns = len(s)
	noise = wavin1[0 : ns]                              # noise length = ns
	noise = noise - np.mean(noise)                      # DC
	siganl_power = 1 / ns * np.sum(s ** 2)
	noise_power = 1 / ns * np.sum(noise ** 2)
	noise_variance = siganl_power / pow(10 , SNR / 10)
	noise = np.sqrt(noise_variance/noise_power) * noise
	y = s + noise
	
	return y, noise
	
if __name__ == '__main__':
	speech = Speech()
	filename = 'bluesky1.wav'
	xx, fs = speech.audioread(filename, 8000)
	xx = xx - np.mean(xx)  # DC
	x = xx / np.max(np.abs(xx))  # normalized
	N = len(x)
	time = np.arange(N) / fs
	IS = 0.3                        # unvoice segment length
	wlen = 200                      # frame length 25ms
	inc = 80                        # frame shift
	SNR = 20
	wnd = np.hamming(wlen)          # window function
	overlap = wlen - inc            # frame overlap
	NIS = int((IS * fs - wlen)/inc + 1)     # unvoiced segment frame number
	
	noisefilename = 'destroyerops.wav'
	signal, noise = add_noisefile(x, noisefilename, SNR, fs)
	y = speech.enframe(signal, list(wnd), inc).T
	fn = y.shape[1]                                     # frame number
	frameTime = speech.FrameTime(fn, wlen, inc, fs)
	
	Y = np.fft.fft(y, axis=0)                                   # FFT
	Y = np.abs(Y[0 : int(wlen / 2 + 1), :])             # positive frequency
	N = np.mean(Y[:, 0 : NIS], axis = 1)                # average spectrum of unvoiced segment
	NoiseCounter = 0
	NoiseLength = 9
	
	SF = np.zeros(fn)                                   # speech flag
	NF = np.zeros(fn)                                   # noise flag
	D = np.zeros(fn)                                    # spectral distance
	TNoise = np.zeros(Y.shape)                          # noise spectrum
	SN = np.zeros(fn)                                   # noise spectral amplitude
	Vad = VAD()
	for i in range(fn):
		if i <= NIS:                                    # unvoiced: NF=1, SF=0
			SpeechFlag = 0
			NoiseCounter = 100
			SF[i] = 0
			NF[i] = 1
			TNoise[:, i] = N
		else:
			NoiseFlag, SpeechFlag, NoiseCounter, Dist = Vad.vad(Y[:, i], N, NoiseCounter, 2.5, 8)
			SF[i] = SpeechFlag
			NF[i] = NoiseFlag
			D[i] = Dist
			if SpeechFlag == 0:                          # noise segment
				N = (NoiseLength * N + Y[:, i]) / (NoiseLength + 1)
			TNoise[:, i] = N
		SN[i] = np.sum(TNoise[:, i])
		
	# figure
	plt.figure(figsize=(9, 16))
	plt.subplot(4, 1, 1)
	plt.plot(time, x, 'k')
	plt.axis([0, np.max(time), -1.2, 1.2])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Clean Speech Signal')
	plt.plot(frameTime, SF, 'k--')
	plt.subplot(4, 1, 2)
	plt.plot(time, signal, 'k')
	plt.axis([0, np.max(time), -1.5, 1.5])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Signal SNR = {}dB'.format(SNR))
	plt.subplot(4, 1, 3)
	plt.plot(frameTime, D, 'k')
	plt.grid()
	plt.axis([0, np.max(time), 0, 1.2 * np.max(D)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Log Spectral Distance')
	plt.subplot(4, 1, 4)
	plt.plot(frameTime, SN, 'k', linewidth=2)
	plt.grid()
	plt.axis([0, np.max(time), np.min(SN), 1.2 * np.max(SN)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Sum of Noise Amplitude ')
	plt.savefig('images/vad_noise_estimation_lsd.png', bbox_inches='tight', dpi=600)
	plt.show()
	
			
		
	
	