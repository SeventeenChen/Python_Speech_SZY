# Multitaper Spectrum & Energy Entropy Ratio
# pr7_3_1

from Enhancement import *
from Noisy import *

if __name__ == '__main__':
	# Set_I
	IS = 0.15  # unvoice segemnt length
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
	snr1 = noisy.SNR_singlech(x, signal)
	
	alpha = 2.8  # over subtraction factor
	beta = 0.001  # gain factor
	# c = 0, power spectrum -> gain matrix without sqrt; c = 1, sqrt needed
	c = 1
	wlen = 200  # frame length
	inc = 80  # frame shift
	NIS = int((IS * fs - wlen) / inc + 1)  # leading silence segment frame number
	enh = Enhancement()
	# spectral subtraction using multitaper spectrum estimation
	output = enh.Mtmpsd_ssb(signal, wlen, inc, NIS, alpha, beta, c)
	snr2 = noisy.SNR_singlech(x, output)
	print('snr1 = {:.2f} \nsnr2 = {:.2f}'.format(snr1, snr2))
	
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
	plt.savefig('images/multitaper_spectral_estimation_spectrum_subtraction.png', bbox_inches='tight', dpi=600)
	plt.show()



