#
# pr6_3_3

from Universal import *
from Noisy import *
from VAD import *
from scipy.signal import butter, buttord, filtfilt

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
	
	n, Wn = buttord(300 / (fs / 2), 600 / (fs / 2), 3, 20)      # filter order, bandwidth
	b, a = butter(n, Wn)                                        # digital filter coefficient
	Ru = np.zeros(fn)
	for k in range(1, fn):
		u = y[:, k]                                 # one frame
		ru = np.correlate(u, u, 'full')             # self-correlate
		rnu = ru / np.max(ru)                       # normalized
		rpu = filtfilt(b, a, rnu)                   # digital filter
		Ru[k] = np.max(rpu)
	
	vad = VAD()
	Rum = vad.multimidfilter(Ru, 10)  # smoothing
	Rum = Rum / np.max(Rum)  # normalized
	thredth = np.max(Rum[0: NIS])  # threshold
	T1 = 1.1 * thredth
	T2 = 1.3 * thredth
	voiceseg, vsl, SF, NF = vad.vad_param1D(Rum, T1, T2)
	
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
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Signal SNR = {}dB'.format(SNR))
	plt.subplot(3, 1, 3)
	plt.plot(frameTime, Rum)
	plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=0.8)
	plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=0.8)
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2]), 'k--', linewidth=1)
	plt.grid()
	plt.axis([0, np.max(time), -0.05, 1.2])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Short-Term Normalized Self-Correlation')
	plt.savefig('images/vad_normalized_self_corr.png', bbox_inches='tight', dpi=600)
	plt.show()
	
