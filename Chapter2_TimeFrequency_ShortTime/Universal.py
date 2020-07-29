# 通用类
import pyaudio
import wave
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class Speech:
    def __init__(self, path):
        self.path = path

    def audiorecorder(self, len=2, formater=pyaudio.paInt16, rate=16000, frames_per_buffer=1024, channels=2):
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
        wf = wave.open(self.path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(formater))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def audioplayer(self, frames_per_buffer=1024):
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

    def audioread(self, sr):
        data, sample_rate = librosa.load(self.path, sr = sr)
        return data, sample_rate

    def soundplot(self, data=[], sr=22050, size=(14, 5)):
        if len(data) == 0:
            data, _ = self.audioread()
        plt.figure(figsize=size)
        librosa.display.waveplot(data, sr=sr)
        plt.show()

    def enframe(self, x, win, inc=None):
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
        return ((l - 1) * inc + frameLen / 2) / fs




