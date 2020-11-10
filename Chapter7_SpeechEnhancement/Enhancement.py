#
# Class Enhancement

from scipy.signal import lfilter
from spectrum import pmtm

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
	
	def SSBoll79_2(self, signal, fs, T1, IS=None):
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
			W = int(0.025 * fs)  # window length 25ms
			nfft = W
			# overlap-add method works good with this shift value
			SP = 0.4  # frame shift 40% (10ms)
			wnd = np.hamming(W)
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
		fn = NumberofFrames
		miniL = 5
		voiceseg, vosl, SF, Ef = VAD().pitch_vad1(y, fn, T1, miniL)
		YS = Y  # Y magnitude average
		for i in np.arange(1, NumberofFrames - 1):
			YS[:, i] = (YS[:, i - 1] + YS[:, i] + YS[:, i + 1]) / 3
		
		X = np.zeros(Y.shape)
		D = np.zeros(FreqResol)
		for i in range(NumberofFrames):
			# Magnitude Spectrum Distance VAD
			NoiseFlag, SpeechFlag, NoiseCounter, Dist = VAD().vad(Y[:, i] ** (1 / Gamma), N ** (1 / Gamma),
			                                                      NoiseCounter)
			SpeechFlag = SF[i]
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
		output = output / np.max(np.abs(output))            # normalized
		
		return output
	
	def Mtmpsd_ssb(self, signal, wlen, inc, NIS, alpha, beta, c):
		"""
		Spectral Subtraction
		Multitaper Spectrum Estimation
		Short-term Energy Entropy Ratio
		:param signal: noisy speech
		:param wlen: frame length
		:param inc: frame shift
		:param NIS: leding unvoiced (noise) frame number
		:param alpha: over subtraction factor in spectral subtraction
		:param beta: gain compensation factor
		:param c: gain factor (0: power spectrum, 1: amplitude spectrum)
		:return output: denoise speech
		"""
		w2 = int(wlen / 2) + 1
		wind = np.hamming(wlen)  # hamming window
		y = Speech().enframe(signal, list(wind), inc).T  # enframe
		fn = y.shape[1]  # frame number
		N = len(signal)  # signal length
		fft_frame = np.fft.fft(y, axis=0)  # FFT
		abs_frame = np.abs(fft_frame[0: w2, :])  # positive frequency amplitude
		ang_frame = np.angle(fft_frame[0: w2, :])  # positive frequency phase
		
		# smoothing in 3 neighbour frame
		abs_frame_backup = abs_frame
		for i in range(1, fn - 1, 2):
			abs_frame_backup[:, i] = 0.25 * abs_frame[:, i - 1] + 0.5 * abs_frame[:, i] + 0.25 * abs_frame[:, i + 1]
		abs_frame = abs_frame_backup
		
		# multitaper spectrum estimation power spectrum
		PSDFrame = np.zeros((w2, fn))  # PSD in each frame
		for i in range(fn):
			# PSDFrame[:, i] = pmtm(y[:, i], NW = 3, NFFT=wlen)
			Sk_complex, weights, eigenvalues = pmtm(y[:, i], NW=3, NFFT=wlen)
			Sk = (np.abs(Sk_complex) ** 2).transpose()
			PSDTwoSide = np.mean(Sk * weights, axis=1)
			PSDFrame[:, i] = PSDTwoSide[0: w2]
		
		PSDFrameBackup = PSDFrame
		for i in range(1, fn - 1, 2):
			PSDFrameBackup[:, i] = 0.25 * PSDFrame[:, i - 1] + 0.5 * PSDFrame[:, i] + 0.25 * PSDFrame[:, i + 1]
		PSDFrame = PSDFrameBackup
		
		# average PSD of leading unvoiced segment
		NoisePSD = np.mean(PSDFrame[:, 0: NIS], axis=1)
		
		# spectral subtraction -> gain factor
		g = np.zeros((w2, fn))  # gain factor
		g_n = np.zeros((w2, fn))
		for k in range(fn):
			g[:, k] = (PSDFrame[:, k] - alpha * NoisePSD) / PSDFrame[:, k]
			g_n[:, k] = beta * NoisePSD / PSDFrame[:, k]
			gix = np.where(g[:, k] < 0)
			g[gix, k] = g_n[gix, k]
		
		gf = g
		if c == 0:
			g = gf
		else:
			g = np.sqrt(gf)
		
		SubFrame = g * abs_frame  # spectral subtraction amplitude
		output = Speech().OverlapAdd2(SubFrame, ang_frame, wlen, inc)  # synthesis
		output = output / np.max(np.abs(output))  # normalized
		ol = len(output)
		if ol < N:
			output = np.concatenate((output, np.zeros(N - ol)))
		
		return output
	
	
	def WienerScalart96m_2(self, signal, fs, T1, IS):
		"""
		Wiener filter based on tracking a priori SNR usingDecision-Directed
		method, proposed by Scalart et al 96. In this method it is assumed that
		SNRpost=SNRprior +1. based on this the Wiener Filter can be adapted to a
		model like Ephraims model in which we have a gain function which is a
		function of a priori SNR and a priori SNR is being tracked using Decision
		Directed method.
		:param signal: noisy signal
		:param fs: sampling frequency
		:param IS: initial silence (noise only) length in seconds (default value is .25 sec)
		:param T1: threshold
		:return output: denoise signal
		"""
		if not IS:
			IS = 0.25  # seconds
			W = int(0.025 * fs)  # window length 25ms
			nfft = W
			# overlap-add method works good with this shift value
			SP = 0.4  # frame shift 40% (10ms)
			wnd = np.hamming(W)
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
		
		pre_emph = 0                                                # pre_emphasis parameter
		signal = lfilter(np.array([1, -1 * pre_emph]), 1, signal)   # pre-emphasis
		NIS = int((IS * fs - W) / (SP * W) + 1)                     # number of initial silence segments
		y = self.segment(signal, W, SP, wnd)                        # enframe
		Y = np.fft.fft(y, axis=0)                                   # FFT
		FreqResol, NumberofFrames = Y.shape
		YPhase = np.angle(Y[0: int(NumberofFrames / 2) + 1, :])     # noisy speech phase
		Y = np.abs(Y[0: int(NumberofFrames / 2) + 1, :])            # Spectrogram
		LambdaD = np.mean(Y[:, 0 : NIS] ** 2).T                     # initial noise power spectrum variance
		N = np.mean(Y[:, 0:NIS].T, axis=0).T                        # initial average noise power spectrum
	
		alpha = 0.99
		fn = NumberofFrames
		miniL = 5
		voiceseg, vosl, SF, Ef = VAD().pitch_vad1(y, fn, T1, miniL) # vad
		
		NoiseCounter = 0
		NoiseLength = 9                                             # smoothing factor for noise updating
		G = np.ones(N.shape)                                        # power estimation initialization
		Gamma = G
		X = np.zeros(Y.shape)                                       # Y magnitude average
		for i in np.arange(1, NumberofFrames - 1):
			SpeechFlag = SF[i]
			if i <= NIS:                                            # leading unvoiced segment
				SpeechFlag = 0
				NoiseCounter = 100
			if SpeechFlag == 0:                                     # update noise spectrum in unvoiced segment
				N = (NoiseLength * N + Y[:, i])/(NoiseLength + 1)
				LambdaD = (NoiseLength * LambdaD + Y[:, i] ** 2)/(NoiseLength + 1) # update and smoothing noise variance
		
		
			gammaNew = (Y[:, i] ** 2)/LambdaD                           # post SNR
			xi = alpha * (G ** 2) * Gamma + (1 - alpha) * np.max(gammaNew - 1, 0)   # senior SNR
			Gamma = gammaNew
			G = (xi/(xi + 1))                                           # wiener spectrum estimation
			X[:, i] = G * Y[:, i]                                       # wiener filter spectrum
			
		
		output = Speech().OverlapAdd2(X, YPhase, int(W), int(SP * W))
		output = lfilter([1], np.array([1, -1 * pre_emph]), output)
		output = output / np.max(np.abs(output))  # normalized
		
		return output
		