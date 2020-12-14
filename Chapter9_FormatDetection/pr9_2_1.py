#
# pr9_2_1

from scipy.signal import lfilter, find_peaks

from Universal import *

if __name__ == '__main__':
	filename = 'snn27.wav'
	speech = Speech()
	x, fs = speech.audioread(filename, None)   # read one frame data
	u = lfilter(b = np.array([1, -0.99]), a = 1, x = x)     # pre-emphasis
	wlen = len(u)       # frame length
	cepstL = 6          # cepstrum window length
	wlen2 = int(wlen/2)
	freq = np.arange(wlen2 - 1) * fs / wlen     # frequency scale in the frequency domain
	u2 = u * np.hamming(wlen)       # window function
	U = np.fft.fft(u2)
	UAbs = np.log(np.abs(U[0 : wlen2 - 1]))
	Cepst = np.fft.ifft(UAbs)
	cepst = np.zeros(wlen2 - 1, dtype=complex)
	cepst[0 : cepstL] = Cepst[0 : cepstL]
	cepst[wlen2 - cepstL + 1 : wlen2 - 1] = Cepst[wlen2 - cepstL + 1 : wlen2 - 1]
	spect = np.real(np.fft.fft(cepst))
	Loc, _ = find_peaks(spect)              # format frequency
	Val = spect[Loc]
	FRMNT = freq[Loc]
	print('Format = {}'.format(FRMNT))
	
	# figure
	plt.figure(figsize=(16, 9))
	plt.plot(freq, UAbs)
	plt.plot(freq, spect, 'k', linewidth = 2)
	plt.grid()
	
	for k in range(4):
		plt.plot(freq[Loc[k]], Val[k], 'k',marker = 'o', markersize=16)
		plt.plot(np.array([freq[Loc[k]], freq[Loc[k]]]), np.array([-6, Val[k]]), 'k-.', linewidth=2)
	
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.title('Siganl Spectrum, Envelope, Format')
	plt.axis([0, 4000, -6, 2])
	plt.savefig('images/cepstrum_format_detection.png', bbox_inches='tight', dpi=600)
	plt.show()
	
