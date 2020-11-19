#
# pr8_3_3


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
	period = pitch.ACF_threelevel(yy, fn, voiceseg, vosl, lmax, lmin)
	T0 = pitch.pitfilterm1(period, voiceseg, vosl)
	
	# figure
	plt.figure(figsize=(16, 9))
	plt.subplot(2, 1, 1)
	plt.plot(time, x)
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Speech Siganl')
	plt.axis([0, np.max(time), -1, 1])
	for k in range(vosl):
		nx1 = voiceseg['begin'][k]
		nx2 = voiceseg['end'][k]
		nx3 = voiceseg['duration'][k]
		print('nx1 = {}, nx2 = {}, nx3 = {} \n'.format(nx1, nx2, nx3))
		plt.plot(np.array([frameTime[nx1], frameTime[nx1]]), np.array([-1, 1]), 'k', linewidth=1)
		plt.plot(np.array([frameTime[nx2], frameTime[nx2]]), np.array([-1, 1]), 'k--', linewidth=1)
	
	plt.subplot(2, 1, 2)
	plt.plot(frameTime, T0)
	plt.xlabel('Time [s]')
	plt.ylabel('Sampling Points')
	plt.title('Smoothing Pitch Period')
	plt.axis([0, np.max(time), np.min(period), np.max(period)])
	plt.savefig('images/3level_clip_autocrrelation_pitch_detection.png', bbox_inches='tight', dpi=600)
	plt.show()
