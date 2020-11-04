#
# pr7_2_1

from Noisy import *
from Universal import *


def simplesubspec(signal, wlen, inc, NIS, a, b):
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
	wnd = np.hamming(wlen)              # window function
	N = len(signal)                     # signal length
	
	y = speech.enframe(signal, list(wnd), inc).T        # enframe
	fn = y.shape[1]                                     # frame number
	
	y_fft = np.fft.fft(y, axis=0)                       # FFT
	y_a = np.abs(y_fft)                                 # amplitude
	y_phase = np.angle(y_fft)                           # phase
	y_a2 = y_a ** 2                                     # energy
	Nt = np.mean(y_a2[:, 0 : NIS], 1)                   # average energy in noise segment
	nl2 = int(wlen / 2) + 1                             # positvie frequency
	
	temp = np.zeros(nl2)                                # energy
	U = np.zeros(nl2)                                   # one frame amplitude
	X = np.zeros((nl2, fn))                             # amplitude
	for i in range(fn):                                 # spectrum subtraction
		for k in range(nl2):
			if(y_a2[k, i] > a * Nt[k]):
				temp[k] = y_a2[k, i] - a * Nt[k]
			else:
				temp[k] = b * y_a2[k,i]
			U[k] = np.sqrt(temp[k])
		X[:, i] = U
	
	output = speech.OverlapAdd2(X, y_phase[0:nl2, :], wlen, inc)        # synthesis
	Nout = len(output)                                                  # spectrum subtraction length = original length?
	
	if Nout > N:
		output = output[0 : N]
	else:
		output = np.concatenate([output, np.zeros(N - Nout)])
		
	output = output / np.max(np.abs(output))                            # normalization
	
	return output
	
if __name__ == '__main__':
	# Set_I
	IS = 0.25  # unvoice segemnt length
	wlen = 200  # frame length 25ms
	inc = 80  # frame shift
	filename = 'bluesky1.wav'
	SNR = 5
	
	# PART_I
	speech = Speech()
	xx, fs = speech.audioread(filename, 8000)
	xx = xx - np.mean(xx)  # DC
	x = xx / np.max(xx)  # normalized
	N = len(x)
	time = np.arange(N) / fs
	noisy = Noisy()
	signal, _ = noisy.Gnoisegen(x, SNR)  # add noise
	snr1 = noisy.SNR_singlech(x, signal)
	wnd = np.hamming(wlen)  # window function
	overlap = wlen - inc
	NIS = int((IS * fs - wlen) / inc + 1)  # unvoice segment frame number
	y = speech.enframe(signal, list(wnd), inc).T
	fn = y.shape[1]  # frame number
	frameTime = speech.FrameTime(fn, wlen, inc, fs)  # frame to time
	
	a = 4
	b = 0.001                   # parameter
	output = simplesubspec(signal, wlen, inc, NIS, a, b)
	snr2 = noisy.SNR_singlech(x, output)
	print('snr1 = {:.2f} dB \nsnr2 = {:.2f} dB'.format(snr1, snr2))
	
	# figure
	plt.figure(figsize=(9, 16))
	plt.subplot(3, 1, 1)
	plt.plot(time, x)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Clean Speech Signal')
	plt.subplot(3, 1, 2)
	plt.plot(time, signal)
	plt.axis([0, np.max(time), np.min(signal), np.max(signal)])
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Signal')
	plt.subplot(3, 1, 3)
	plt.plot(time, output)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Spectrum Subtraction Denoise Waveform')
	plt.savefig('images/spectrum_subtraction_denoise.png', bbox_inches='tight', dpi=600)
	plt.show()
	