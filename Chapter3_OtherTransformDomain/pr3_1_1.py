# 同态滤波解卷


import matplotlib.pyplot as plt
import numpy as np

def cceps(x):
    """
    计算复倒谱
    """
    y = np.fft.fft(x)
    return np.fft.ifft(np.log(y))


def icceps(y):
    """
    计算复倒谱的逆变换
    """
    x = np.fft.fft(y)
    return np.fft.ifft(np.exp(x))


def rcceps(x):
    """
    计算实倒谱
    """
    y = np.fft.fft(x)

    return np.fft.ifft(np.log(np.abs(y))).real

def lceps(x):
	"""
	信号倒频谱的对数幅度谱
	:param x:
	:return:
	"""
	y = np.fft.fft(x)

	return np.log(np.abs(y))

if __name__ == '__main__':
	x = np.loadtxt('su1.txt')
	Fs = 16000
	mcep = 29
	N = len(x)
	lceps = lceps(x)
	rcceps = rcceps(x)
	SoundTrack = rcceps[0 : mcep + 1]
	SoundTrack = np.concatenate([SoundTrack.T, np.zeros((1000-2*mcep-1)), SoundTrack[len(SoundTrack) : 0 : -1].T])
	Glottal = np.concatenate([np.zeros((mcep+1)), rcceps[mcep+1 : - mcep], np.zeros((mcep))])
	FTGlottal = np.fft.fft(Glottal).real
	ST = np.fft.fft(SoundTrack).real
	freq = np.fft.fftfreq(N) * Fs
	time = [i / Fs for i in range(N)]

	fig = plt.figure(figsize=(15, 18))
	plt.subplot(2, 1, 1)
	plt.plot(time, x)
	plt.axis([0, time[N - 1], -0.7, 0.7])
	plt.grid()
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Speech Waveform')
	plt.subplot(2, 1, 2)
	plt.plot(time[0 : int(N/2)], rcceps[0 : int(N/2)])
	plt.axis([0, time[int(N/2)], -0.2, 0.2])
	plt.grid()
	plt.xlabel('1/Frequency [s]')
	plt.ylabel('Amplitude')
	plt.title('Speech Cepstrum')
	plt.savefig('images/cepstrum.png')
	plt.show()
	fig = plt.figure(figsize=(15, 18))
	plt.subplot(2, 1, 1)
	plt.plot(freq[0 : int(N/2)], lceps[0 : int(N/2)], label = 'Log Spectrum')
	plt.plot(freq[0 : int(N/2)], ST[0 : int(N/2)], linewidth = 4 ,label = 'Sound Track Impluse Response Spectrum')
	plt.legend(loc = 'upper right')
	plt.ylim([-4, 5])
	plt.grid()
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude')
	plt.title('Speech Log Spectrum & Sound Track Impluse Response Spectrum')
	plt.subplot(2, 1, 2)
	plt.plot(freq[0 : int(N/2)], FTGlottal[0 : int(N/2)])
	plt.grid()
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude')
	plt.title('Glottal Impluse Spectrum')
	plt.savefig('images/sound_track_glottal.png')
	plt.show()

