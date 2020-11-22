#
# pr8_5_1

from audiolazy import lazy_lpc
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


	lmin = int(fs / 500)  # min pitch period
	lmax = int(fs / 60)  # max pitch period
	# period = np.zeros(fn)                       # pitch period initialization
	p = 12
	EPS = np.finfo(float).eps   # machine epsilon
	Lc = np.zeros(fn)       # pitch location in one frame
	period = np.zeros(fn)   # pitch period
	for k in range(fn):
		if SF[k] == 1:  # voice segment
			u = y[:, k] * wnd   # one frame * window
			ar = lazy_lpc.lpc.autocor(u, p)  # lpc
			ar = np.array(ar.numerator)
			z = lfilter(np.concatenate([np.array([0]), -ar[2 : len(ar)]]), 1, u)  # LPC inverse filter output
			E = u - z       # predict error
			xx = np.fft.fft(E, axis=0)  # FFT
			a = 2 * np.log(np.abs(xx) + EPS)    # abs --> ln
			b = np.fft.ifft(a, axis=0)  # cepstrum
			Lc[k] = np.argmax(b[lmin : lmax])   # find maximum in [Pmin, Pmax]
			period[k] = Lc[k] + lmin - 1
	
	pitch = Pitch()
	T1 = pitch.pitfilterm1(period, voiceseg, vosl)
	
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
	plt.plot(frameTime, T1, 'k')
	plt.xlabel('Time [s]')
	plt.ylabel('Sampling Points')
	plt.legend(['Initial estimation', 'Smoothing estimation'])
	plt.title('Pitch Period')
	plt.axis([0, np.max(time), 0, 150])
	plt.savefig('images/lpc_error_cepstrum_pitch_detection.png', bbox_inches='tight', dpi=600)
	plt.show()