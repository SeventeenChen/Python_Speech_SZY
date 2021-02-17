#
# pr9_3_2

from audiolazy import lazy_lpc
from scipy.signal import lfilter

from Universal import *
from VAD import VAD


def Ext_frmnt(y, p, thr1, fs):
	"""
	LPC Formant Detection
	:param y: enframe signal ( N x frame number(fn))
	:param p: LPC order
	:param thr1: threshold
	:param fs: sample frequency
	:return formant: 3 x fn
	"""
	fn = y.shape[1]  # frame number
	formant = np.zeros((fn, 3))  # formant
	for i in range(fn):
		u = y[:, i]  # one frame data
		ar = lazy_lpc.lpc.autocor(u, p)  # LPC coefficients
		a = ar.numerator
		root = np.roots(a)  # roots of polynomial
		
		mag_root = np.abs(root)  # magnitude
		arg_root = np.angle(root)  # angle
		f_root = arg_root / np.pi * fs / 2  # angle --> frequency
		k = 0
		fmn = []
		for j in range(p):
			if mag_root[j] > thr1:
				if arg_root[j] > 0 and arg_root[j] < np.pi and f_root[j] > 200:
					fmn.append(f_root[j])
					k = k + 1
		if fmn:
			f1 = len(fmn)
			fmn = np.sort(fmn)
			formant[i, 0: min(f1, 3)] = fmn[0: min(f1, 3)]  # just need three frequency
	
	return formant


if __name__ == '__main__':
	filename = 'vowels8.wav'
	speech = Speech()
	xx, fs = speech.audioread(filename, None)  # read one frame data
	x = xx - np.mean(xx)  # DC
	y = lfilter(b=np.array([1, -0.99]), a=1, x=x)  # pre-emphasis
	wlen = 200  # frame length
	inc = 80  # frame shift
	xy = speech.enframe(y, wlen, inc).T  # enframe
	fn = xy.shape[1]  # frame number
	Nx = len(y)  # data length
	time = np.arange(0, Nx) / fs  # time scale
	frameTime = speech.FrameTime(fn, wlen, inc, fs)  # frame to time
	T1 = 0.1
	miniL = 20  # voice segment minimal frame number
	p = 9
	thr1 = 0.75
	voiceseg, vosl, SF, Ef = VAD().pitch_vad1(xy, fn, T1, miniL)  # VAD
	Msf = np.tile(SF.reshape((len(SF), 1)), (1, 3))  # SF ---> fn x 3
	formant = Ext_frmnt(xy, p, thr1, fs)
	
	Fmap = np.multiply(Msf, formant)  # just voice segment
	findex = np.where(Fmap == 0)
	Fmap[findex] = np.nan
	
	nfft = 512
	d = speech.stftms(y, wlen, nfft, inc)  # spectrogram
	W2 = int(1 + nfft / 2)
	n2 = np.arange(0, W2)
	freq = n2 * fs / nfft
	
	# figure
	plt.figure(1, figsize=(16, 9))
	plt.subplot(2, 1, 1)
	plt.plot(time, x)
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('/a-i-u/ Three Vowels Signal')
	plt.axis([0, max(time), -1, 1])
	plt.subplot(2, 1, 2)
	plt.plot(frameTime, Ef)
	plt.plot(np.array([0, max(time)], dtype=object), np.array([T1, T1], dtype=object), 'k--', linewidth=2)
	for k in range(vosl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1, 1]), 'k-.', linewidth=2)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1, 1]), 'k-.', linewidth=2)
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Normalized Power Entropy Ratio')
	plt.axis([0, max(time), 0, 1])
	plt.savefig('images/simple_lpc_format_detection_1.png', bbox_inches='tight', dpi=600)
	plt.figure(2, figsize=(16, 9))
	Xdb = librosa.amplitude_to_db(abs(d))
	librosa.display.specshow(Xdb, sr=fs, x_coords=frameTime, x_axis='time', y_axis='linear')
	plt.plot(frameTime, Fmap, 'k', linewidth=2)
	plt.colorbar()
	plt.xlabel('Time [s]')
	plt.ylabel('Frequency [Hz]')
	plt.title('Spectorgram')
	
	plt.savefig('images/simple_lpc_format_detection_2.png', bbox_inches='tight', dpi=600)
	plt.show()
