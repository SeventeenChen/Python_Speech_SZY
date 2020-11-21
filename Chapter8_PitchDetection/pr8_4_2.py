#
# pr8_4_2

from scipy.signal import lfilter

from Pitch import *
from Universal import *

if __name__ == '__main__':
	# Set_II
	filename = 'tone4.wav'
	wlen = 320  # frame length
	inc = 80  # frame shift
	overlap = wlen - inc  # frame overlap
	T1 = 0.05  # pitch endpoint detection parameter
	# PART_II
	speech = Speech()
	x, fs = speech.audioread(filename, 8000)
	x = x - np.mean(x)  # DC
	x = x / np.max(np.abs(x))  # normalized
	N = len(x)
	time = np.arange(N) / fs
	wnd = np.hamming(wlen)  # window function
	y = speech.enframe(x, list(wnd), inc).T
	fn = y.shape[1]  # frame number
	frameTime = speech.FrameTime(fn, wlen, inc, fs)  # frame to time
	voiceseg, vosl, SF, Ef = VAD().pitch_vad1(y, fn, T1)  # pitch detection
	
	# filter coefficients
	b = np.array([0.012280, -0.039508, 0.042177, 0.000000, -0.042177, 0.039508, -0.012280])
	a = np.array([1.000000, -5.527146, 12.854342, -16.110307, 11.479789, -4.410179, 0.713507])
	xx = lfilter(b, a, x)  # BPF
	yy = speech.enframe(xx, list(wnd), inc).T  # filtered signal enframe
	lmin = int(fs / 500)  # min pitch period
	lmax = int(fs / 60)  # max pitch period
	# period = np.zeros(fn)                       # pitch period initialization
	# auto correlation pitch detection
	pitch = Pitch()
	period = pitch.ACFAMDF_corr(yy, fn, voiceseg, vosl, lmax, lmin)
	T0 = pitch.pitfilterm1(period, voiceseg, vosl)
	
	# figure
	plt.figure(figsize=(16, 9))
	plt.subplot(2, 1, 1)
	plt.plot(time, x)
	plt.grid()
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Speech Siganl')
	plt.axis([0, np.max(time), -1, 1])
	
	plt.subplot(2, 1, 2)
	plt.plot(frameTime, period, 'g', linewidth=3)
	plt.plot(frameTime, T0, 'k')
	plt.xlabel('Time [s]')
	plt.ylabel('Sampling Points')
	plt.legend(['Initial estimation', 'Smoothing estimation'])
	plt.title('Pitch Period')
	plt.axis([0, np.max(time), 0, 120])
	plt.savefig('images/autocorrelation_average_magnitude_difference_pitch_detection.png', bbox_inches='tight',
	             dpi=600)
	plt.show()