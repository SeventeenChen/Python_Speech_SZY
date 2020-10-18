# Log Spectral Distance
# pr6_5_1

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
	
	Y = np.fft.fft(y, axis=0)  # FFT
	Y = np.abs(Y[0 : int(wlen / 2) + 1, :])             # positive frequency
	N = np.mean(Y[:, 0 : NIS], axis = 1).reshape(-1,1)     # average spectrum of the leading unvoiced noise segement
	NoiseCounter = 0
	
	SF = np.zeros(fn)                      # noise flag
	NF = np.zeros(fn)                      # speech flag
	D = np.zeros(fn)
	Vad = VAD()
	for i in range(fn):
		if i <= NIS:                       # NF=1,SF=0 in the leading unvoiced noise segement
			SpeechFlag = 0
			NoiseCounter = 100
			SF[i] = 0
			NF[i] = 1
		else:
			NoiseFlag, SpeechFlag, NoiseCounter, Dist = Vad.vad(Y[:, i], N, NoiseCounter, 2.5, 8)
			SF[i] = SpeechFlag
			NF[i] = NoiseFlag
			D[i] = Dist
		
	Vad = VAD()
	sindex = np.where(SF ==1)             # VAD in SF
	voiceseg = Vad.findSegemnt(sindex)
	vosl = len(voiceseg['begin'])
	
	# figure
	plt.figure(figsize=(9, 16))
	plt.subplot(3, 1, 1)
	plt.plot(time, x)
	for k in range(vosl):
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
	plt.plot(frameTime, D)
	plt.ylim([0, 8])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Log Spectral Distance')
	for k in range(vosl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 8]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 8]), 'k--', linewidth=1)
	plt.savefig('images/vad_log_spectral_distance.png', bbox_inches='tight', dpi=600)
	plt.show()
