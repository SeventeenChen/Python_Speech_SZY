# spectrum subtraction & auto-correlation pitch detection
# pr8_7_2

from Enhancement import *
from Noisy import *
from Pitch import *

if __name__ == '__main__':
	
	filename = 'tone4.wav'
	wlen = 320  # frame length
	inc = 80  # frame shift
	overlap = wlen - inc  # frame overlap
	speech = Speech()
	x, fs = speech.audioread(filename, None)
	x = x - np.mean(x)  # DC
	x = x / np.max(np.abs(x))  # normalized
	N = len(x)
	SNR = 0
	noisy = Noisy()
	signal, _ = noisy.Gnoisegen(x, SNR)  # add noise
	snr1 = noisy.SNR_singlech(x, signal)  # initial SNR
	time = np.arange(N) / fs
	wnd = np.hamming(wlen)  # window function
	IS = 0.15       # leading unvoiced segment length
	NIS = int((IS * fs - wlen)/ inc + 1)    # leading unvoiced segment frame number
	
	a = 3
	b = 0.001
	output = Enhancement().simplesubspec(signal, wlen, inc, NIS, a, b)  # spectrum subtraction
	snr2 = noisy.SNR_singlech(x, output)
	y = speech.enframe(output, list(wnd), inc).T
	fn = y.shape[1]  # frame number
	frameTime = speech.FrameTime(fn, wlen, inc, fs)  # frame to time
	T1 = 0.12  # pitch detection parameter
	
	voiceseg, vosl, SF, Ef = VAD().pitch_vad1(y, fn, T1)  # pitch detection
	# filter coefficients
	b = np.array([0.012280, -0.039508, 0.042177, 0.000000, -0.042177, 0.039508, -0.012280])
	a = np.array([1.000000, -5.527146, 12.854342, -16.110307, 11.479789, -4.410179, 0.713507])
	z = lfilter(b, a, output)  # BPF
	yy = speech.enframe(z, list(wnd), inc).T  # filtered signal enframe
	
	lmin = int(fs / 500)  # min pitch period
	lmax = int(fs / 60)  # max pitch period
	period = Pitch().ACF_corr(yy, fn, voiceseg, vosl, lmax, lmin)  # pitch detection with correlation
	tindex = np.where(period)
	F0 = np.zeros(fn)  # fundamental frequency
	F0[tindex] = fs / period[tindex]
	TT = Pitch().pitfilterm1(period, voiceseg, vosl)  # smoothing
	FF = Pitch().pitfilterm1(F0, voiceseg, vosl)
	
	# figure
	plt.figure(figsize=(9, 24))
	plt.subplot(6, 1, 1)
	plt.plot(time, x)
	for k in range(vosl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		nx3 = voiceseg['duration'][k]
		# print('nx1 = {}, nx2 = {}, nx3 = {} \n'.format(nx1, nx2, nx3))
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1, 1]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1, 1]), 'k--', linewidth=1)
	
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Original Speech Siganl')
	plt.axis([0, np.max(time), -1, 1])
	
	plt.subplot(6, 1, 2)
	plt.plot(time, signal)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Siganl')
	
	plt.subplot(6, 1, 3)
	plt.plot(time, output)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Denoise Speech Siganl')
	plt.axis([0, np.max(time), -1, 1])
	
	plt.subplot(6, 1, 4)
	plt.plot(frameTime, Ef)
	plt.plot(np.array([0, max(frameTime)]), np.array([T1, T1]), 'k-.', linewidth=1)
	plt.text(3.25, T1 + 0.05, 'T1')
	for k in range(vosl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		nx3 = voiceseg['duration'][k]
		# print('nx1 = {}, nx2 = {}, nx3 = {} \n'.format(nx1, nx2, nx3))
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1, 1]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1, 1]), 'k--', linewidth=1)
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Energy Entropy Ratio')
	plt.axis([0, np.max(time), 0, max(Ef)])

	plt.subplot(6, 1, 5)
	plt.plot(frameTime, TT, 'k', linewidth=2)
	plt.axis([0, np.max(time), 0, 80])
	plt.xlabel('Time [s]')
	plt.ylabel('Frequency [Hz]')
	plt.title('Pitch Frequency')
	
	plt.subplot(6, 1, 6)
	plt.plot(frameTime, FF, 'k', linewidth=2)
	plt.axis([0, np.max(time), 0, 450])
	plt.xlabel('Time [s]')
	plt.ylabel('Frequency [Hz]')
	plt.title('Pitch Frequency')
	plt.savefig('images/spectrum_subtraction_auto-correlation_pitch_detection.png', bbox_inches='tight', dpi=600)
	plt.show()