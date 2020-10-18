# Short Term Energy Zero-crossing Ratio
# pr6_7_1

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
	
	aparam = 2                      # parameter of energy
	bparam = 1                      # parameter of zero-crossing
	etemp = np.sum(y ** 2, axis=0)       # energy
	etemp1 = np.log10(1 + etemp / aparam)
	Vad = VAD()
	zcr = Vad.zc2(y, fn)
	Ecr = etemp1 / (zcr + bparam)   # energy zero-crossing ratio
	Ecrm = Vad.multimidfilter(Ecr, 2)           # smoothing
	dth = np.mean(Ecrm[0 : NIS])                # threshold
	T1 = 1.2 * dth
	T2 = 2 * dth
	
	[voiceseg, vsl, SF, NF] = Vad.vad_param1D(Ecrm, T1, T2)     # vad in ecr with 2 thresholds
	
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
	plt.plot(frameTime, Ecrm)
	plt.axis([0, np.max(time), 0, 1.2 * np.max(Ecrm)])
	plt.xlabel('Time [s]')
	plt.ylabel('Energy Zero-crossing Ratio')
	plt.title('Short Term Energy Zero-crossing Ratio')
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2 * np.max(Ecrm)]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2 * np.max(Ecrm)]), 'k--', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=1)
	plt.savefig('images/vad_energy_zerocrossing_ratio.png', bbox_inches='tight', dpi=600)
	plt.show()
