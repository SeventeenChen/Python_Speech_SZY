# Spectrogram 语谱图

from Universal import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	Speech =  Speech("bluesky3.wav")
	x, Fs =Speech.audioread(8000)
	x = x / np.max(np.abs(x))

	inc = 64
	wlen = 256
	nfft = 256
	win = np.hanning(wlen)
	N = len(x)
	X = librosa.stft(x, nfft, inc, wlen, window='hann', center=False)
	fn = X.shape[1]
	Xdb = librosa.amplitude_to_db(abs(X))
	time = [i / Fs for i in range(N)]
	frameTime = Speech.FrameTime(fn, wlen, inc, Fs)

	fig = plt.figure(figsize=(15, 18))
	plt.subplot(2, 1, 1)
	plt.plot(time, x)
	plt.xlabel('Time/s')
	plt.ylabel('Amplitude')
	plt.title('Speech Waveform')
	plt.subplot(2, 1, 2)
	librosa.display.specshow(Xdb, sr=Fs, x_coords= frameTime, x_axis='time', y_axis='linear')
	plt.colorbar()
	plt.xlabel('Time/s')
	plt.ylabel('Frequency/Hz')
	plt.title('Spectorgram')
	plt.savefig('images/spectrogram.png')
	plt.show()
