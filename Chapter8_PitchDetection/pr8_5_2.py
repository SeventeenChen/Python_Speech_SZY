#
# pr8_5_2

from audiolazy import lazy_lpc
from scipy.signal import lfilter, ellipord, ellip, resample

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
	x, fs = speech.audioread(filename, None)
	x = x - np.mean(x)  # DC
	x = x / np.max(np.abs(x))  # normalized
	N = len(x)
	time = np.arange(N) / fs
	wnd = np.hamming(wlen)  # window function
	y = speech.enframe(x, list(wnd), inc).T
	fn = y.shape[1]  # frame number
	frameTime = speech.FrameTime(fn, wlen, inc, fs)  # frame to time
	voiceseg, vosl, SF, Ef = VAD().pitch_vad1(y, fn, T1)  # pitch detection
	
	# BPF
	Rp = 1      # passband dripple
	Rs = 50     # stopband attenuation
	fs2 = fs / 2
	Wp = np.array([60, 500]) / fs2  # bandpass: 60-500Hz
	Ws = np.array([20, 1000]) / fs2 # stopband: 20-1000Hz
	n, Wn = ellipord(Wp, Ws, Rp, Rs)  # filter order
	b, a = ellip(n, Rp, Rs, Wn, 'bandpass')  # filter coefficients
	x1 = lfilter(b, a, x)   # BPF
	x1 = x1 / np.max(np.abs(x)) # normalization
	x2 = resample(x1, int(x1.size/4))   # down sample
	
	lmin = int(fs / 500)  # min pitch period
	lmax = int(fs / 60)  # max pitch period
	# period = np.zeros(fn)                       # pitch period initialization
	wind = np.hanning(int(wlen / 4))    # window function
	y2 = speech.enframe(x2, list(wind), int(inc/4)).T
	p = 4
	period = np.zeros(fn)  # pitch period
	for i in range(vosl):
		ixb = voiceseg['begin'][i]      # voice segment
		ixe = voiceseg['end'][i]
		ixd = ixe - ixb + 1             # duration
		for k in range(ixd):
			u = y2[:, k + ixb - 1]   # one frame * window
			ar = lazy_lpc.lpc.autocor(u, p)  # lpc
			ar = np.array(ar.numerator)
			z = lfilter(np.concatenate([np.array([0]), -ar[2: len(ar)]]), 1, u)  # LPC inverse filter output
			E = u - z  # predict error
			ru1 = xcorr(E, norm='coeff')    # normalized correlation coefficient
			ru1 = ru1[0]
			ru1 = ru1[int(wlen / 4) : ]
			ru = resample(ru1, ru1.size * 4)    # upsample
			tloc = np.argmax(ru[lmin: lmax])  # find maximum in [Pmin, Pmax]
			period[k + ixb - 1] = tloc + lmin - 1
	
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
	plt.savefig('images/simplified_inverse_filter_pitch_detection.png', bbox_inches='tight', dpi=600)
	plt.show()