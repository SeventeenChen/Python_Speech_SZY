# Short-term Auto-correlation Primary and Secondary Peak Ratio
# pr6_3_4


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
	
	R1 = np.zeros(fn)
	for k in range(1, fn):
		u = y[:, k]  # one frame
		ru = np.correlate(u, u, 'full')  # self-correlate
		ru0 = ru[wlen - 1]
		ru1 = np.max(ru[wlen + 16 : wlen + 133])        # first secondary peak
		R1[k] = ru0 / ru1
	
	vad = VAD()
	Rum = vad.multimidfilter(R1, 20)  # smoothing
	Rum = Rum / np.max(Rum)  # normalized
	thredth = np.max(Rum[0: NIS])  # threshold
	T1 = 0.95 * thredth
	T2 = 0.75 * thredth
	voiceseg, vsl, SF, NF = vad.vad_param1D_revr(Rum, T1, T2)
	
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
	plt.axis([0, np.max(time), 0, 1.2])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Short-term Auto-correlation Primary and Secondary Peak Ratio')
	plt.savefig('images/vad_ratio_self_corr.png', bbox_inches='tight', dpi=600)
	plt.show()

