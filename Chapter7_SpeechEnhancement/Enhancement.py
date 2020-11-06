#
# Class Enhancement

from Universal import *

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