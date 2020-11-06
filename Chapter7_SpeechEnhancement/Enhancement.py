#
# Class Enhancement

from Universal import *
from VAD import *

class Enhancement:
	def simplesubspec(self, signal, wlen, inc, NIS, a, b):
		"""
		simple spectrum subtraction
		:param signal: noisy speech
		:param wlen: window length
		:param inc: frame shift
		:param NIS: leading noise segment length
		:param a: over subtraction factor
		:param b: gain factor
		:return output: denoise speech
		"""
		wnd = np.hamming(wlen)  # window function
		N = len(signal)  # signal length
		
		speech = Speech()
		y = speech.enframe(signal, list(wnd), inc).T  # enframe
		fn = y.shape[1]  # frame number
		
		y_fft = np.fft.fft(y, axis=0)  # FFT
		y_a = np.abs(y_fft)  # amplitude
		y_phase = np.angle(y_fft)  # phase
		y_a2 = y_a ** 2  # energy
		Nt = np.mean(y_a2[:, 0: NIS], 1)  # average energy in noise segment
		nl2 = int(wlen / 2) + 1  # positvie frequency
		
		temp = np.zeros(nl2)  # energy
		U = np.zeros(nl2)  # one frame amplitude
		X = np.zeros((nl2, fn))  # amplitude
		for i in range(fn):  # spectrum subtraction
			for k in range(nl2):
				if (y_a2[k, i] > a * Nt[k]):
					temp[k] = y_a2[k, i] - a * Nt[k]
				else:
					temp[k] = b * y_a2[k, i]
				U[k] = np.sqrt(temp[k])
			X[:, i] = U
		
		output = speech.OverlapAdd2(X, y_phase[0:nl2, :], wlen, inc)  # synthesis
		Nout = len(output)  # spectrum subtraction length = original length?
		
		if Nout > N:
			output = output[0: N]
		else:
			output = np.concatenate([output, np.zeros(N - Nout)])
		
		output = output / np.max(np.abs(output))  # normalization
		
		return output
	
	def segment(self, siganl, W=256, SP=0.4, Window=np.hamming(256)):
		"""
		chops a signal to overlapping windowed segments
		:param siganl: one dimentional signal
		:param W: sample number per window (default = 25)
		:param SP: shift percent (default =  0.4)
		:param Window: window function (default: hamming)
		:return Seg: segment matrix
		"""
		if (W != 256):
			Window = np.hamming(W)
		Window = Window.reshape(-1, 1)  # make it a column vector
		
		L = len(siganl)
		SP = int(W * SP)
		N = int((L - W) / SP + 1)  # number of segments
		
		Index = np.tile(np.arange(0, W), (N, 1)) + np.tile(SP * np.arange(0, N).reshape(-1, 1), (1, W))
		Index = Index.T
		hw = np.tile(Window, (1, N))
		Seg = siganl[Index] * hw
		
		return Seg
	
	def SSBoll79(self, signal, fs, IS=None):
		"""
		Spectral Subtraction based on Boll 79. Amplitude spectral subtraction
		Includes Magnitude Averaging and Residual noise Reduction
		:param signal: noisy signal
		:param fs: sampling frequency
		:param IS: initial silence (noise only) length in seconds (default value is .25 sec)
		:return output: denoise signal
		"""
		if not IS:
			IS = 0.25  # seconds
		elif isinstance(IS, float):
			W = int(0.025 * fs)  # window length 25ms
			nfft = W
			# overlap-add method works good with this shift value
			SP = 0.4  # frame shift 40% (10ms)
			wnd = np.hamming(W)
		
		# IGNORE THIS SECTION FOR COMPATIBILITY WITH ANOTHER PROGRAM FROM HERE.....
		if isinstance(IS, dict):
			W = IS['windowsize']
			SP = IS['shiftsize'] / W
			nfft = IS['nfft']
			wnd = IS['window']
			if hasattr(IS, 'IS'):
				IS = IS['IS']
			else:
				IS = 0.25
		
		# .......IGNORE THIS SECTION FOR COMPATIBILITY WITH ANOTHER PROGRAM T0 HERE
		
		NIS = int((IS * fs - W) / (SP * W) + 1)  # number of initial silence segments
		Gamma = 1  # 1: magnitude, 2: power spectrum
		
		y = self.segment(signal, W, SP, wnd)
		Y = np.fft.fft(y, axis=0)
		FreqResol, NumberofFrames = Y.shape
		YPhase = np.angle(Y[0: int(NumberofFrames / 2) + 1, :])  # noisy speech phase
		Y = np.abs(Y[0: int(NumberofFrames / 2) + 1, :]) ** Gamma  # Spectrogram
		
		N = np.mean(Y[:, 0:NIS].T, axis=0).T  # initial noise power spectrum mean
		NRM = np.zeros(N.shape)  # Noise Residual Maximum (Initialization)
		NoiseCounter = 0
		NoiseLength = 9  # smoothing factor for noise updating
		
		Beta = 0.03
		YS = Y  # Y magnitude average
		for i in np.arange(1, NumberofFrames - 1):
			YS[:, i] = (YS[:, i - 1] + YS[:, i] + YS[:, i + 1]) / 3
		
		X = np.zeros(Y.shape)
		D = np.zeros(FreqResol)
		for i in range(NumberofFrames):
			# Magnitude Spectrum Distance VAD
			NoiseFlag, SpeechFlag, NoiseCounter, Dist = VAD().vad(Y[:, i] ** (1 / Gamma), N ** (1 / Gamma),
			                                                      NoiseCounter)
			if SpeechFlag == 0:
				N = (NoiseLength * N + Y[:, i]) / (NoiseLength + 1)  # update and smooth noise
				NRM = np.maximum(NRM, YS[:, i] - N)  # update maximum noise  residue
				X[:, i] = Beta * Y[:, i]
			else:
				D = YS[:, i] - N  # spectral subtraction
				if i > 0 and i < NumberofFrames - 1:  # residual noise reduction
					for j in range(len(D)):
						if D[j] < NRM[j]:
							D[j] = np.min(np.array([D[j], YS[j, i - 1] - N[j], YS[j, i + 1] - N[j]]))
				D[D < 0] = 0
				X[:, i] = D
		
		output = Speech().OverlapAdd2(X ** (1 / Gamma), YPhase, int(W), int(SP * W))
		
		return output
	
