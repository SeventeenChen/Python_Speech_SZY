# 通用类

import cmath
import wave

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pyaudio


class Speech:

    def audiorecorder(self, path, len=2, formater=pyaudio.paInt16, rate=16000, frames_per_buffer=1024, channels=2):
        p = pyaudio.PyAudio()
        stream = p.open(format=formater, channels=channels, rate=rate, input=True, frames_per_buffer=frames_per_buffer)
        print("start recording......")
        frames = []
        for i in range(0, int(rate / frames_per_buffer * len)):
            data = stream.read(frames_per_buffer)
            frames.append(data)
        print("stop recording......")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(formater))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def audioplayer(self, path, frames_per_buffer=1024):
        wf = wave.open(self.path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(frames_per_buffer)
        while data != b'':
            stream.write(data)
            data = wf.readframes(frames_per_buffer)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def audiowrite(self):
        pass

    def audioread(self, path, sr = None):
        data, sample_rate = librosa.load(path, sr = sr)
        return data, sample_rate

    def soundplot(self, data=[], sr=22050, size=(14, 5)):
        if len(data) == 0:
            data, _ = self.audioread()
        plt.figure(figsize=size)
        librosa.display.waveplot(data, sr=sr)
        plt.show()

    def enframe(self, x, win, inc=None):
        # print(x)
        nx = len(x)
        if isinstance(win, list):
            nwin = len(win)
            nlen = nwin  # 帧长=窗长
        elif isinstance(win, int):
            nwin = 1
            nlen = win  # 设置为帧长
        if inc is None:
            inc = nlen
        nf = (nx - nlen + inc) // inc  # 计算帧数
        frameout = np.zeros((nf, nlen))  # 初始化
        indf = np.multiply(inc, np.array([i for i in range(nf)]))  # 设置每帧在x中的位移量位置
        for i in range(nf):
            frameout[i, :] = x[indf[i]:indf[i] + nlen]  # 分帧
        if isinstance(win, list):
            frameout = np.multiply(frameout, np.array(win))  # 每帧乘以窗函数的值
        return frameout

    def FrameTime(self, frameNum, frameLen, inc, fs):
        """
        分帧后计算每帧对应的时间
        """
        l = np.array([i for i in range(frameNum)])
        return (l * inc + frameLen / 2) / fs

    def SNR_singlech(self, I, In):
        """
		calculate SNR of noisy speech signal
		:param I: clean speech siganl
		:param In: noisy speech siganl
		:return snr:
		"""
    
        Ps = np.sum((I - np.mean(I)) ** 2)  # signal power
        Pn = np.sum((I - In) ** 2)  # noise power
        snr = 10 * np.log10(Ps / Pn)
    
        return snr

    def OverlapAdd2(self, X, A=None, W=None, S=None):
        """
		reconstruction signal form spectrogram
		:param X: FFT spectrum matrix (each column: frame fft)
		:param A: phase angle (dimension = X), default = 0
		:param W: window length (default: 2 * fft length)
		:param S: shift length (default: W/2)
		:return Y: reconstructed signal from its spectrogram
		"""
        if A is None:
            A = np.angle(X)
        if W is None:
            W = X.shape[0] * 2
        if S is None:
            S = int(W / 2)
    
        if int(S) != S:  # frame shift not an integer
            S = int(S)
            print('The shift length have to be an integer as it is the number of samples.\n')
            print('shift length is fixed to {}'.format(S))
    
        FreqRes, FrameNum = X.shape  # frame number, fft number
        Spec = X * np.exp(A * cmath.sqrt(-1))  # complex spectrum
        if np.mod(W, 2):
            Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1::-1, :]))))  # negative frequency
        else:
            Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1:(len(Spec) - 1), :]))))  # negative frequency
    
        sig = np.zeros(int((FrameNum - 1) * S + W))  # initialization
        weight = sig
        for i in range(FrameNum):  # overlap
            start = i * S  # start sample point of ith frame
            spec = Spec[:, i]  # ith frame spectrum
            sig[start: (start + W)] = sig[start: (start + W)] + np.real(np.fft.ifft(spec, W, axis=0))
        Y = sig
    
        return Y




