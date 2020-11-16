#
# pr8_2_1

from Universal import *
from VAD import *

if __name__ == '__main__':
	# Set_II
	filename = 'tone4.wav'
	wlen = 320                  # frame length
	inc = 80                    # frame shift
	overlap = wlen - inc        # frame overlap
	T1 = 0.05                   # pitch endpoint detection parameter
	# PART_II
	speech = Speech()
	x, fs = speech.audioread(filename, 8000)
	x = x - np.mean(x)  # DC
	x = x / np.max(x)  # normalized
	N = len(x)
	time = np.arange(N) / fs
	wnd = np.hamming(wlen)  # window function
	y = speech.enframe(x, list(wnd), inc).T
	fn = y.shape[1]  # frame number
	frameTime = speech.FrameTime(fn, wlen, inc, fs)  # frame to time
	
	voiceseg, vosl, SF, Ef = VAD().pitch_vad1(y, fn, T1)        # pitch detection
	lmin = int(fs / 500)                                # min pitch period
	lmax = int(fs / 60)                                 # max pitch period
	period = np.zeros(fn)                               # pitch period initialization
	R = np.zeros(fn)                                    # max in frame
	Lc = np.zeros(fn)                                   # location
	EPS = np.finfo(float).eps                           # machine epsilon
	for k in range(fn):
		if SF[k] == 1:                                  # voice frame
			y1 = y[:, k] * wnd                          # add window to one frame
			xx = np.fft.fft(y1, axis=0)                 # FFT
			a = 2 * np.log(np.abs(xx) + EPS)            # abs --> log
			b = np.fft.ifft(a, axis=0)                  # cesprtum
			Lc[k] = np.argmax(b[lmin : lmax])           # find max in [lmin : lmax]
			period[k] = Lc[k] + lmin - 1                # pitch period
	
	# figure
	plt.figure(figsize=(9, 16))
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
	plt.plot(frameTime, period)
	plt.xlabel('Time [s]')
	plt.ylabel('Sampling Points')
	plt.title('Pitch Period')
	plt.axis([0, np.max(time), np.min(period), np.max(period)])
	plt.savefig('images/cepstrum_pitch_detection.png', bbox_inches='tight', dpi=600)
	plt.show()
	