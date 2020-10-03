#
# pr6_5_3

from Universal import *
from Noisy import *
from VAD import *
from MFCC import *


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
	
	Mfcc = MFCC()
	ccc = Mfcc.mfcc(signal, fs, 16, wlen, inc)          # MFCC
	fn1 = ccc.shape[0]                                  # frame number
	frameTime1 = frameTime[2 : fn - 2]
	Ccep = ccc[:, 0 : 16]                               # MFCC coefficient
	C0 = np.mean(Ccep[0 : 5, :], axis = 0)              # calculate approximate average noise MFCC coefficient
	Dcep = np.zeros(fn)
	for i in range(5, fn1):
		Cn = Ccep[i, :]                                 # one frame MFCC cepstrum coefficient
		Dstu = 0
		for k in range(16):                             # calculate the MFCC cepstrum distance
			Dstu += (Cn[k] - C0[k]) ** 2                # between each frame and noise
		Dcep[i] = np.sqrt(Dstu)
	Dcep[0 : 5] = Dcep[5]
	
	Vad = VAD()
	Dstm = Vad.multimidfilter(Dcep, 10)  # smoothing
	dth = np.max(Dstm[0: NIS])
	T1 = dth
	T2 = 1.5 * dth
	[voiceseg, vsl, SF, NF] = Vad.vad_param1D(Dstm, T1, T2)
	
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
	plt.plot(frameTime, Dstm)
	plt.axis([0, np.max(time), 0, 1.2 * np.max(Dstm)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Short-term MFCC Cepstrum Distance')
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2 * np.max(Dstm)]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2 * np.max(Dstm)]), 'k--', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=1)
	plt.savefig('images/vad_mfcc_cepstrum_distance.png', bbox_inches='tight', dpi=600)
	plt.show()
