# Multitaper Spectrum & Energy Entropy Ratio
# pr7_3_1

from Enhancement import *
from Noisy import *

	

if __name__ == '__main__':
	# Set_I
	IS = 0.15  # unvoice segemnt length
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

	enh = Enhancement()
	output = enh.WienerScalart96m_2(signal, fs, 0.12, IS)
	LenOut = len(output)
	if LenOut < N:
		output = np.concatenate([output, np.zeros(N - LenOut)], axis=0)
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
	plt.title('Wiener Denoise Waveform')
	plt.savefig('images/wiener_denoise.png', bbox_inches='tight', dpi=600)
	plt.show()



