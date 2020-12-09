# wavelet auto-correlation pitch detection
# pr8_7_1

from scipy.signal import lfilter

from Noisy import *
from Pitch import *
from Universal import *

if __name__ == '__main__':
	# Set_II
	filename = 'tone4.wav'
	wlen = 320  # frame length
	inc = 80  # frame shift
	overlap = wlen - inc  # frame overlap
	T1 = 0.23  # pitch endpoint detection parameter
	# PART_II
	speech = Speech()
	xx, fs = speech.audioread(filename, None)
	xx = xx - np.mean(xx)  # DC
	xx = xx / np.max(np.abs(xx))  # normalized
	N = len(xx)
	SNR = 5
	x, _ = Noisy().Gnoisegen(xx, SNR)
	time = np.arange(N) / fs
	wnd = np.hamming(wlen)  # window function
	y = speech.enframe(x, list(wnd), inc).T
	fn = y.shape[1]  # frame number
	frameTime = speech.FrameTime(fn, wlen, inc, fs)  # frame to time
	voiceseg, vsl, SF, Ef = VAD().pitch_vad1(y, fn, T1)  # pitch detection
	
	# filter coefficients
	b = np.array([0.012280, -0.039508, 0.042177, 0.000000, -0.042177, 0.039508, -0.012280])
	a = np.array([1.000000, -5.527146, 12.854342, -16.110307, 11.479789, -4.410179, 0.713507])
	z = lfilter(b, a, x)  # BPF
	yy = speech.enframe(z, list(wnd), inc).T  # filtered signal enframe
	lmin = int(fs / 500)  # min pitch period
	lmax = int(fs / 60)  # max pitch period
	
	period = Pitch().Wavelet_corrm1(yy, fn, voiceseg, vsl, lmax, lmin)
	tindex = np.where(period)
	F0 = np.zeros(fn)       # fundamental frequency
	F0[tindex] = fs / period[tindex]
	TT = Pitch().pitfilterm1(period, voiceseg, vsl)     # smoothing
	FF = Pitch().pitfilterm1(F0, voiceseg, vsl)
	
	# figure
	plt.figure(figsize=(9, 24))
	plt.subplot(5, 1, 1)
	plt.plot(time, xx)
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		nx3 = voiceseg['duration'][k]
		print('nx1 = {}, nx2 = {}, nx3 = {} \n'.format(nx1, nx2, nx3))
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1, 1]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1, 1]), 'k--', linewidth=1)
		
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Original Speech Siganl')
	plt.axis([0, np.max(time), -1, 1])
	
	plt.subplot(5, 1, 2)
	plt.plot(frameTime, Ef)
	plt.plot(np.array([0, max(frameTime)]), np.array([T1, T1]), 'k-.', linewidth=1)
	plt.text(3.25, T1 + 0.05, 'T1')
	for k in range(vsl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		nx3 = voiceseg['duration'][k]
		# print('nx1 = {}, nx2 = {}, nx3 = {} \n'.format(nx1, nx2, nx3))
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1, 1]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1, 1]), 'k--', linewidth=1)
		
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Energy Entropy Ratio')
	plt.axis([0, np.max(time), 0, 1])
	plt.subplot(5, 1, 3)
	plt.plot(time, x)
	plt.axis([0, np.max(time), -1, 1])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Noisy Speech Siganl')
	plt.axis([0, np.max(time), -1, 1])
	plt.subplot(5, 1, 4)
	plt.plot(frameTime, TT, 'k', linewidth = 2)
	plt.axis([0, np.max(time), 0, 120])
	plt.xlabel('Time [s]')
	plt.ylabel('Sample Points')
	plt.title('Pitch Period')
	plt.subplot(5, 1, 5)
	plt.plot(frameTime, FF, 'k', linewidth = 2)
	plt.axis([0, np.max(time), 0, 500])
	plt.xlabel('Time [s]')
	plt.ylabel('Frequency [Hz]')
	plt.title('Pitch Frequency')
	plt.savefig('images/wavelet_auto-correlation_pitch_detection.png', bbox_inches='tight', dpi=600)
	plt.show()
