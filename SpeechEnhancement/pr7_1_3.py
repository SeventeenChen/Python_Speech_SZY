# LMS Adaptive Notch Filter for ECG
# pr7_1_3

from scipy.io import loadmat

from Universal import *

if __name__ == '__main__':
	mat = loadmat('ecg_m.mat')              # load mat data
	x = mat['x'].squeeze()
	s = x
	N = len(x)                              # data length
	fs = 1000                               # sample frequency
	tt = np.arange(N) / fs                  # time
	ff = np.arange(int(N/2)) * fs / N       # frequency
	X = np.fft.fft(x, axis=0)               # spectrum analysis
	
	for k in range(5):                      # Adaptive Notch Filter
		j = k * 2  + 1                      # 50 Hz, odd order harmoinic frequency
		f0 = 50 * j
		x1 = np.cos(2 * np.pi * tt * f0)
		x2 = np.sin(2 * np.pi * tt * f0)
		w1 = 0                              # weight
		w2 = 1
		e = np.zeros(N)  # initialization
		y = np.zeros(N)
		mu = 0.1
		for i in range(N):  # LMS adaptive notch filter
			y[i] = w1 * x1[i] + w2 * x2[i]
			e[i] = x[i] - y[i]
			w1 = w1 + mu * e[i] * x1[i]  # iteration
			w2 = w2 + mu * e[i] * x2[i]
		x = e
		
	output = e
		
	# figure
	plt.figure(figsize=(9, 16))
	plt.subplot(3, 1, 1)
	plt.plot(tt, s)
	plt.axis([0, np.max(tt), -3000, 6500])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Original ECG')
	plt.subplot(3, 1, 2)
	X = X / np.max(np.abs(X))
	n2 = int(N / 2)
	XSpectrum = np.abs(X[0 : n2])
	plt.plot(ff, XSpectrum)
	plt.axis([0, np.max(ff), np.min(XSpectrum), np.max(XSpectrum)])
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude')
	plt.title('ECG spectrum')
	plt.subplot(3, 1, 3)
	plt.plot(tt, output)
	plt.axis([0, 10, -2000, 6500])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('LMS Notch Filter Output ECG')
	plt.savefig('images/lms_notch_denoise_ecg.png', bbox_inches='tight', dpi=600)
	plt.show()