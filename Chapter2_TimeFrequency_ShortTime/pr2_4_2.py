# Short Time Power Spectrum Density
# 短时功率谱密度

from Universal import *
from enframe import enframe
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np

def STPSD(x, nwind, noverlap, w_nwind, w_noverlap, nfft, Fs):
	"""
	计算语音信号的短时功率谱密度函数
	:param x: discrete signal
	:param nwindwin: frame length
	:param noverlap: frame overlap
	:param w_nwind: divide frame into segments, the length each segment
	:param w_noverlap: overlap of the segment
	:param nfft:
	:return f: frequency
	:return Pxx : Short Time Power Spectrum Density
	"""
	inc = nwind - noverlap
	X = Speech.enframe(x, nwind, inc)
	fn = X.shape[0]
	Pxx = np.zeros((int(w_nwind/2 + 1), fn))
	f = np.zeros((int(w_nwind/2 + 1), fn))
	for i in range(fn):
		f[:, i], Pxx[:, i] = welch(X[i,:], fs=Fs, nperseg = w_nwind, noverlap = w_noverlap, nfft = nfft, detrend=False)

	return f, Pxx

if __name__ == '__main__':
	Speech =  Speech("bluesky3.wav")
	x, Fs =Speech.audioread(8000)
	x = x / np.max(np.abs(x))
	nwind = 240
	noverlap = 160
	w_nwind = 200
	w_noverlap = 195
	nfft = 200
	inc = nwind - noverlap
	N = len(x)
	f, Pxx = STPSD(x, nwind, noverlap, w_nwind, w_noverlap, nfft, Fs)
	Pdb = librosa.amplitude_to_db(Pxx)
	fn = Pxx.shape[1]
	frameTime = Speech.FrameTime(fn, nwind, inc, Fs)
	j = input('Which frame plot:')
	j = int(j)

	time = [i / Fs for i in range(N)]
	fig = plt.figure(figsize=(15, 18))
	plt.subplot(3, 1, 1)
	plt.plot(time, x)
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Speech Waveform')
	plt.subplot(3, 1, 2)
	plt.plot(f[:, j - 1], Pxx[:, j - 1])
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('PSD [V**2/Hz]')
	plt.title('Short Time Power Spectrum Density of {} Frame'.format(j))
	plt.subplot(3, 1, 3)
	librosa.display.specshow(Pdb, sr=Fs, x_coords= frameTime, x_axis='time', y_axis='linear')
	plt.colorbar()
	plt.xlabel('Time [s]')
	plt.ylabel('Frequency [Hz]')
	plt.title('Short Time Power Spectrum Density')
	plt.savefig('images/psd.png')
	plt.show()

