#
# pr7_2_1

from Enhancement import *
from Noisy import *

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
	enh = Enhancement()
	output = enh.simplesubspec(signal, wlen, inc, NIS, a, b)
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
	