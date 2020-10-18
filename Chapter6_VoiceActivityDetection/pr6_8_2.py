# EMD decomposition Average Teager Energy
# pr6_8_2

from PyEMD import EMD

from Noisy import *
from Universal import *
from VAD import *


def steager(z):
	"""
	Teager energy of one dimensional signal
	:param z: one dimensional signal
	:return tz: teager energy
	"""
	
	N = len(z)
	tz = np.zeros(N)
	for k in range(1, N - 1):
		tz[k] = z[k] ** 2 - z[k - 1] * z[k + 1]
	tz[0] = 2 * tz[1] - tz[2]               # extrapolation
	tz[N - 1] = 2 * tz[N - 2] - tz[N - 3]
	
	return tz

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
	
	imf = EMD().emd(signal)                   # EMD decomposition
	M = imf.shape[0]                        # EMD order
	u = np.zeros(N)
	for k in range(2, M):                   # reconstruct without 1st, 2nd IMF
		u += imf[k, :]
	z = speech.enframe(u, list(wnd), inc).T       # reconstruction signal enframe
	
	Tg = np.zeros(z.shape).T
	Tgf = np.zeros(fn)
	for k in range(fn):
		v = z[:, k]                                 # one frame
		imf = EMD().emd(v)                          # EMD decomposition
		L = imf.shape[0]                            # EMD order
		Etg = np.zeros(wlen)
		for i in range(L):                          # average Teager energy in each frame
			Etg += steager(imf[i, :])
		Tg[k, :] = Etg
		Tgf[k] = np.mean(Etg)                       # average Teager energy in recent frame
		
	Vad = VAD()
	Zcr = Vad.zc2(y, fn)                            # zero-crossing
	Tgfm = Vad.multimidfilter(Tgf, 10)                # smoothing
	Zcrm = Vad.multimidfilter(Zcr, 10)
	Mtg = np.max(Tgfm)
	Tmth = np.mean(Tgfm[0 : NIS])
	Zcrth = np.mean(Zcrm[0: NIS])
	T1 = 1.5 * Tmth
	T2 = 3 * Tmth
	T3 = 0.9 * Zcrth
	T4 = 0.8 * Zcrth
	[voiceseg,vsl,SF,NF]=Vad.vad_param2D_revr(Tgfm,Zcrm,T1,T2,T3,T4)
	
	# figure
	plt.figure(figsize=(9, 24))
	plt.subplot(5, 1, 1)
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
	plt.subplot(5, 1, 2)
	plt.plot(time, signal)
	plt.axis([0, np.max(time), np.min(signal), np.max(signal)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Signal SNR = {}dB'.format(SNR))
	plt.subplot(5, 1, 3)
	plt.plot(time, u)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('EMD reconstruction speech')
	plt.subplot(5, 1, 4)
	plt.plot(frameTime, Tgfm)
	plt.axis([0, np.max(time), np.min(Tgfm), np.max(Tgfm)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('EMD decomposition Average Teager Energy')
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2 * np.max(Mtg)]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2 * np.max(Mtg)]), 'k--', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=1)
	plt.subplot(5, 1, 5)
	plt.plot(frameTime, Zcrm)
	plt.axis([0, np.max(time), np.min(Zcrm), np.max(Zcrm)])
	plt.xlabel('Time [s]')
	plt.ylabel('Zero-crossing')
	plt.title('Short-term Zero-crossing')
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2 * np.max(Zcrm)]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2 * np.max(Zcrm)]), 'k--', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T3, T3]), 'b', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T4, T4]), 'r--', linewidth=1)
	plt.savefig('images/vad_emd.png', bbox_inches='tight', dpi=600)
	plt.show()