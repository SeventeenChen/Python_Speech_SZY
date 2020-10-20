# spectral subtraction & variance VAD
# pr6_9_2

import cmath

from Noisy import *
from Universal import *
from VAD import *


def simplesubspec(signal, wlen, inc, NIS, a, b):
	"""
	simple spectral subtraction denoise
	:param signal: noisy speech signal
	:param wlen: frame length
	:param inc: frame shift
	:param NIS: leading unvoiced segment frame number
	:param a: over subtraction factor
	:param b: gain compensation factor
	:return output: denoise speech
	"""
	wnd = np.hamming(wlen)              # window function
	N = len(signal)
	
	y = Speech().enframe(signal, list(wnd), inc).T          # enframe
	fn = y.shape[1]                                         # frame number
	
	y_fft = np.fft.fft(y, axis=0)                           # FFT
	y_a = np.abs(y_fft)                                     # amplitude
	y_phase = np.angle(y_fft)                               # phase
	y_power = y_a ** 2                                      # power
	Nt = np.mean(y_power[:, 0 : NIS], axis= 1)              # noise average power
	Np = int(wlen / 2) + 1                                  # positive frequency
	
	temp = np.zeros(Np)                                     # denoise power
	U = np.zeros(Np)                                        # denoise amplitude
	X = np.zeros((Np, fn))
	for i in range(fn):                                     # spectral subtraction
		for k in range(Np):
			if(y_power[k, i] > a * Nt[k]):
				temp[k] = y_power[k, i] - a * Nt[k]
			else:
				temp[k] = b * y_power[k, i]
			U[k] = np.sqrt(temp[k])                         # amplitude = sqrt(power)
		X[:, i] = U
	
	output = OverlapAdd2(X, y_phase[0 : Np, :], wlen, inc)  # reconstruction
	Nout = len(output)                                      # add zero
	
	if Nout > N:
		output = output[0 : N]
	else:
		output = np.concatenate((output, np.zeros(N - Nout)))
	
	output = output / np.max(np.abs(output))
		
	return output

def OverlapAdd2(X,A = None,W = None,S = None):
	"""
	reconstruction signal form spectrogram
	:param X: FFT spectrum matrix (each column: frame fft)
	:param A: phase angle (dimension = X), default = 0
	:param W: window length (default: 2 * fft length)
	:param S: shift length (default: W/2)
	:return Y: reconstructed signal from its spectrogram
	"""
	if A is None:
		A = np.angle(X)
	if W is None:
		W = X.shape[0] * 2
	if S is None:
		S = int(W / 2)
	
	if int(S) != S:                 # frame shift not an integer
		S = int(S)
		print('The shift length have to be an integer as it is the number of samples.\n')
		print('shift length is fixed to {}'.format(S))
	
	FreqRes, FrameNum = X.shape             # frame number, fft number
	Spec = X * np.exp(A * cmath.sqrt(-1))        # complex spectrum
	if np.mod(W, 2):
		Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1::-1, :]))))         # negative frequency
	else:
		Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1:(len(Spec) - 1), :]))))     # negative frequency
	
	sig = np.zeros((FrameNum - 1) * S + W)      # initialization
	weight = sig
	for i in range(FrameNum):                   # overlap
		start = i * S                           # start sample point of ith frame
		spec = Spec[:, i]                       # ith frame spectrum
		sig[start : (start + W)] = sig[start : (start + W)] + np.real(np.fft.ifft(spec, W, axis = 0))
	Y = sig
	
	return Y
	
if __name__ == '__main__':
	# Set_I
	IS = 0.25  # unvoice segemnt length
	wlen = 200  # frame length 25ms
	inc = 80  # frame shift
	filename = 'bluesky1.wav'
	SNR = 0
	
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
	
	snr1 = noisy.SNR_singlech(x, signal)                # initial SNR
	print('snr1 = {0:.2f}'.format(snr1))
	a = 3
	b = 0.01
	output = simplesubspec(signal, wlen, inc, NIS, a, b)
	snr2 = noisy.SNR_singlech(x, output)
	print('snr2 = {0:.2f}'.format(snr2))
	y = speech.enframe(output, list(wnd), inc).T        # denoise enframe
	Np = int(wlen / 2) + 1                              # positive frequency
	Y = np.fft.fft(y, axis = 0)
	YAbs = np.abs(Y[0 : Np, :])                         # positive frequency amplitude
	M = int(np.floor(Np / 4))                                # sub-band number
	SY = np.zeros((M, fn))                              # sub-band amplitude
	Dvar = np.zeros(fn)                                 # sub-band variance
	for k in range(fn):
		for i in range(M):
			j = i * 4
			SY[i, k] = YAbs[j, k] + YAbs[j + 1, k] + YAbs[j + 2, k] + YAbs[j + 3, k]
		Dvar[k] = np.var(SY[:, k])
	
	Vad = VAD()
	Dvarm = Vad.multimidfilter(Dvar, 10)                # smoothing
	dth = np.max(Dvarm[0 : NIS])                        # threshold
	T1 = 1.5 * dth
	T2 = 3 * dth
	voiceseg, vsl, SF, NF = Vad.vad_param1D(Dvarm, T1, T2)
	
	# figure
	plt.figure(figsize=(9, 24))
	plt.subplot(4, 1, 1)
	plt.plot(time, x)
	plt.axis([0, np.max(time), -1, 1])
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		nx3 = voiceseg['duration'][k]
		print('{}, begin = {}, end = {}'.format(k + 1, nx1, nx2))
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1, 1]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1, 1]), 'k--', linewidth=1)
	plt.xlabel('(a) Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Clean Speech Signal')
	plt.subplot(4, 1, 2)
	plt.plot(time, signal)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('(b) Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Signal SNR = {}dB'.format(SNR))
	plt.subplot(4, 1, 3)
	plt.plot(time, output)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('(c) Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Denoise Speech Signal SNR = {0:.2f}dB'.format(snr2))
	plt.subplot(4, 1, 4)
	plt.plot(frameTime, Dvarm)
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([0, np.max(Dvar)]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([0, np.max(Dvar)]), 'k--', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T1, T1]), 'b', linewidth=1)
		plt.plot(np.array([0, np.max(time)]), np.array([T2, T2]), 'r--', linewidth=1)
	plt.grid()
	plt.axis([0, np.max(time), -0.1, np.max(Dvarm)])
	plt.xlabel('(d) Time [s]')
	plt.ylabel('Variance')
	plt.title('Spectral subtraction short-time uniform sub-band frequency band variance')
	plt.savefig('images/vad_spectral_subtraction_var.png', bbox_inches='tight', dpi=600)
	plt.show()
		
	
	