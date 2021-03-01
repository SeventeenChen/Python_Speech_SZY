#
# pr9_3_2

from audiolazy import lazy_lpc
from scipy.interpolate import interp1d
from scipy.signal import lfilter, butter, filtfilt

from Universal import *
from VAD import VAD


def seekfmts(sig, Nt, fs, Nlpc=None):
	"""
	Boston University Formant Tracker in MATLAB
	:param sig: signal
	:param Nt: frame number
	:param fs: sample
	:param Nlpc: LPC order
	:return fmt: normalized formant, size = 3 x Nt
	"""
	if Nlpc is None:
		Nlpc = round(fs / 1000) + 2
	ls = len(sig)  # data length
	Nwin = int(np.floor(ls / Nt))  # frame length
	fmt = np.zeros((3, Nt))  # formant
	for m in range(Nt):
		lpcsig = sig[Nwin * m: min(Nwin * (m + 1), ls)]
		
		if lpcsig.size != 0:
			ar = lazy_lpc.lpc.autocor(lpcsig, Nlpc)  # LPC coefficients
			a = ar.numerator
			const = fs / 2 / np.pi  # constant
			rst = np.roots(a)  # roots of polynomial
			k = 0  # initialization
			yf = np.zeros(len(a) - 1)  # format frequency
			bandw = np.zeros(len(a) - 1)  # band width
			
			for i in range(len(a) - 1):
				re = np.real(rst[i])  # real of root
				im = np.imag(rst[i])  # image of root
				formn = const * np.arctan2(im, re)  # (9-3-17) format frequency
				bw = -2 * const * np.log(np.abs(rst[i]))  # (9-3-18) band width
				
				if formn > 150 and bw < 700 and formn < fs / 2:
					yf[k] = formn
					bandw[k] = bw
					k = k + 1
			
			yf = yf[0:k]
			# bandw = bandw[0:k]
			y = np.sort(yf)  # sort
			# ind = np.argsort(yf)  # index
			# bw = bandw[ind]
			
			F = np.ones(3) * np.nan  # format frequency
			# BW = np.zeros(3)  # band width
			F[0: np.min([3, len(y)])] = y[0:np.min([3, len(y)])]  # only output 4 format
			# BW[0: np.min([3, len(y)])] = bw[0:np.min([3, len(y)])]
			fmt[:, m] = F / fs * 2  # normalized frequency
	
	return fmt


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
	voiceseg, vosl, SF, Ef = VAD().pitch_vad1(xy, fn, T1, miniL)  # VAD
	Msf = np.tile(SF.reshape((len(SF), 1)), (1, 3))  # SF ---> fn x 3
	Fsamps = 256  # frequency range length
	Tsamps = fn  # time range length
	ct = 0
	numiter = 10  # loop times
	iv = 2 ** (10 - 10 * np.exp(-np.linspace(2, 10, num=numiter) / 1.9))  # 10 number in 0 ~ 1024
	ft2 = np.zeros((3, fn, numiter))
	for j in range(numiter):
		i = iv[j]
		iy = int(np.fix(len(y) / np.round(i)))  # frame number
		ft1 = seekfmts(y, iy, fs, 10)  # frame number --> formant
		ct = ct + 1
		InterpFunction = interp1d(np.linspace(1, len(y), num=iy).T, Fsamps * ft1.T, axis=0)
		ft2[:, :, j] = InterpFunction(np.linspace(1, len(y), num=Tsamps).T).T
	
	ft3 = np.squeeze(np.nanmean(ft2.transpose((2, 1, 0)), axis=0))
	tmap = np.tile(np.arange(0, Tsamps).reshape(Tsamps, 1), (1, 3))
	Fmap = np.ones(tmap.shape) * np.nan
	idx = np.where(np.invert(np.isnan(np.sum(ft3, axis=1))))
	fmap = np.squeeze(ft3[idx, :])
	
	b, a = butter(9, 0.1)
	fmap1 = np.round(filtfilt(b, a, fmap, axis=0, padlen=0))
	fmap2 = (fs / 2) * (fmap1 / 256)
	Ftmp_map = np.zeros((fn, 3))
	Ftmp_map[idx, :] = fmap2
	
	Fmap1 = np.multiply(Msf, Ftmp_map)  # just voice segment
	findex = np.where(Fmap1 == 0)
	Fmap = Fmap1
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
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, 1.2]), 'k-.', linewidth=2)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, 1.2]), 'k-.', linewidth=2)
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Normalized Power Entropy Ratio')
	plt.axis([0, max(time), 0, 1])
	plt.savefig('images/advanced_lpc_format_detection_1.png', bbox_inches='tight', dpi=600)
	plt.figure(2, figsize=(16, 9))
	Xdb = librosa.amplitude_to_db(abs(d))
	librosa.display.specshow(Xdb, sr=fs, x_coords=frameTime, x_axis='time', y_axis='linear')
	plt.plot(frameTime, Fmap, 'k', linewidth=2)
	plt.colorbar()
	plt.xlabel('Time [s]')
	plt.ylabel('Frequency [Hz]')
	plt.title('Spectorgram')
	
	plt.savefig('images/advanced_lpc_format_detection_2.png', bbox_inches='tight', dpi=600)
	plt.show()
