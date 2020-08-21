# pr5_3_2
# 把任意的噪声数据按设定的信噪比叠加在纯净信号上，构成带噪语音

from Universal import *
from Noisy import *
import librosa
import math

def add_noisedata(s,data,fs,fs1,snr):
    """
    把任意的噪声数据按设定的信噪比叠加在纯净信号上，构成带噪语音
    :param s: clean speech signal
    :param data: arbitrary noise data
    :param fs: clean signal sample frequency
    :param fs1: data sample frequency
    :param snr: SNR [dB]
    :return noise: noise scaled by the set SNR
    :return signal: noisy (size: n * 1)
    """
    
    s = s.reshape(-1, 1)
    s = s - np.mean(s)
    sL = len(s)
    
    if fs != fs1:
        x = librosa.resample(data, fs, fs1)    # noise reample
    else:
        x = data
    
    x = x.reshape(-1, 1)
    x = x - np.mean(x)
    xL = len(x)
    
    if xL >= sL:
        x = x[0 : sL]
    else:
        print('Warning noise length < signal length, padding with zero')
        x = np.concatenate((x, np.zeros(sL - xL)))
        
    Sr = snr
    Es = np.sum(x * x)
    Ev = np.sum(s * s)
    a = np.sqrt(Ev / Es / (10 ** (Sr / 10)))    # noise ratio
    noise = a * x
    signal = s + noise      # noisy
    
    return signal, noise

if __name__ == '__main__':
    S = Speech()
    s, fs = S.audioread('bluesky3.wav', 8000)
    # s = s - np.mean(s)
    s = s / np.max(np.abs(s))   # normalized
    N = len(s)
    time = np.linspace(0, N - 1, num = N) / fs
    
    # figure
    plt.figure(figsize=(16, 28))
    plt.subplot(4, 1, 1)
    plt.plot(time, s)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Clean Speech Siganl')
    
    SNR = np.array([5, 0, -5])
    for k in range(3):
        snr = SNR[k]
        data = np.sin(2 * math.pi * 100 * time)
        x, noise = add_noisedata(s, data, fs, fs, snr)
        plt.subplot(4, 1, k + 2)
        plt.plot(time, x)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.ylim([-2, 2])
        snr1 = Noisy.SNR_singlech(s, x)
        print('k = {}, snr = {}, snr1 = {:.2f}\n'.format(k+1, snr, snr1))
        plt.title('Noisy Speech Siganl \nSNR = {} dB, calculating SNR = {:.2} dB'.format(snr, np.round(snr1 * 1000/ 1000)))
    
    plt.savefig('images/add_noisedata.png', bbox_inches='tight', dpi=600)
    plt.show()
    
    
    
    
    
    