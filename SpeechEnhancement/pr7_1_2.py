# LMS Adaptive Notch Filter
# pr7_1_2

from Noisy import *
from Universal import *

if __name__ == '__main__':
	filename = 'bluesky1.wav'                   # file path
	speech = Speech()                           # Speech class instantiation
	ss, fs = speech.audioread(filename, 8000)   # read data
	ss = ss - np.mean(ss)                       # DC
	s = ss / np.max(ss)                         # normalized
	N = len(s)                                  # signal length
	time = np.arange(N) / fs                    # time
	ns = 0.5 * np.cos(2 * np.pi * 50 * time)    # 50Hz power frequency
	x = s + ns                                  # noisy
	noisy = Noisy()                             # Noisy class instantiation
	snr1 = noisy.SNR_singlech(s, x)
	
	x1 = np.cos(2 * np.pi * 50 * time)
	x2 = np.sin(2 * np.pi * 50 * time)
	w1 = 0.1                                    # weight 1
	w2 = 0.1                                    # weight 2
	e = np.zeros(N)                             # initialization
	y = np.zeros(N)
	mu = 0.05
	for i in range(N):                          # LMS adaptive notch filter
		y[i] = w1 * x1[i] + w2 * x2[i]
		e[i] = x[i] - y[i]
		w1 = w1 + mu * e[i] * x1[i]             # iteration
		w2 = w2 + mu * e[i] * x2[i]
	
	output = e
	snr2 = noisy.SNR_singlech(s, output)        # SNR after notch filter
	snr = snr2 - snr1
	print('snr1 = {} \nsnr2 = {} \n'.format(snr1, snr2))
	
	# figure
	plt.figure(figsize=(9, 16))
	plt.subplot(3, 1, 1)
	plt.plot(time, s)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Original Speech Signal')
	plt.subplot(3, 1, 2)
	plt.plot(time, x)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Signal SNR = {:.2f}dB'.format(snr1))
	plt.subplot(3, 1, 3)
	plt.plot(time, output)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('LMS Notch Filter Output Speech Signal \nSNR = {:.2f}dB'.format(snr2))
	plt.savefig('images/lms_notch_denoise.png', bbox_inches='tight', dpi=600)
	plt.show()