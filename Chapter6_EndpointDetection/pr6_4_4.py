#
# pr6_4_4

from Universal import *
from Noisy import *
from VAD import *
from pywt import WaveletPacket

def wavlet_barkms(x,wname,fs):
	"""
	decompose speech into 5 layers wavelet packet according to wnmae
	:param x:
	:param wname: wavelet generating function
	:param fs: 8000
	:return y: 17 BASK sub-band filter
	"""
	if fs != 8000:
		print('fs must be 8000Hz, change fs!!!')
		return
	y = np.zeros((17, len(x)))
	n = 5  # decomposition level
	T = WaveletPacket(data=x, wavelet='db2', maxlevel=n, mode='zero')  # 一维小波包分解
	new_wp = WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	
	# 计算各节点对应系数
	map = []
	NodeName = []
	for row in range(n):
		map.append([])
		NodeName.append([])
		for i in [node.path for node in T.get_level(level=row + 1, order='natural')]:
			map[row].append(T[i].data)
			NodeName[row].append(i)
	
	# 按指定的节点，对时间序列分解的一位小波包系数重构
	for i in range(8):
		new_wp[NodeName[4][i]] = map[4][i]
		y[i, :] = new_wp.reconstruct(update=False)
		new_wp = WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	
	new_wp[NodeName[3][4]] = map[3][4]
	y[8, :] = new_wp.reconstruct(update=False)
	new_wp = WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	new_wp[NodeName[3][5]] = map[3][5]
	y[9, :] = new_wp.reconstruct(update=False)
	new_wp = WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	new_wp[NodeName[4][11]] = map[4][11]
	y[10, :] = new_wp.reconstruct(update=False)
	new_wp = WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	new_wp[NodeName[4][12]] = map[4][12]
	y[11, :] = new_wp.reconstruct(update=False)
	new_wp = WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	new_wp[NodeName[3][7]] = map[3][7]
	y[12, :] = new_wp.reconstruct(update=False)
	new_wp = WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	
	for i in range(4, 8):
		new_wp[NodeName[2][i]] = map[2][i]
		y[9 + i, :] = new_wp.reconstruct(update=False)
		new_wp = WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	
	return y
	
	

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
	
	Dst = np.zeros(17)
	Dvar = np.zeros(fn)
	for i in range(fn):
		u = y[:, i]                 # ith frame
		v = wavlet_barkms(u, 'db2', fs)         # 17 BASK sub-band with wavelet packet
		num = v.shape[0]
		for k in range(num):
			Srt = v[k, :]                       # kth BASK sub-band
			Dst[k] = np.var(Srt)                # variance in BASK sub-band
		Dvar[i] = np.mean(Dst)                  # mean variance between 17 band
	
	vad = VAD()
	Dvarm = vad.multimidfilter(Dvar, 10)  # smoothing
	dth = np.mean(Dvarm[0: NIS])  # threshold
	T1 = 1.5 * dth
	T2 = 2.5 * dth
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
	plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=1)
	plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=1)
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2 * np.max(Dvar)]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2 * np.max(Dvar)]), 'k--', linewidth=1)
	plt.grid()
	plt.axis([0, np.max(time), 0, 1.2 * np.max(Dvar)])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Wavelet Packet Short-term BARK Subband Variance')
	plt.savefig('images/vad_wavelet_packet_BASK_var.png', bbox_inches='tight', dpi=600)
	plt.show()
		
		
	