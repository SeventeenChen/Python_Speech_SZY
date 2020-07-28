# 短时能量

from Universal import *
from enframe import enframe
import matplotlib.pyplot as plt
import numpy as np


def STE(x, win, inc):
    """
    计算短时能量函数
    :param x:
    :param win:
    :param inc:
    :return: short time energy
    """
    X = enframe(x, win, inc)
    s = np.multiply(X, X)
    return np.sum(s, axis=1)

def FrameTime(frameNum, frameLen, inc, fs):
    """
    分帧后计算每帧对应的时间
    """
    l = np.array([i for i in range(frameNum)])
    return ((l - 1) * inc + frameLen / 2) / fs

if __name__ == '__main__':
    x, fs = Speech("bluesky3.wav").audioread(8000)
    inc = 80
    wlen = 200
    win = np.hanning(wlen)
    N = len(x)
    En = STE(x, win, inc)
    X = enframe(x, win, inc)
    fn = X.shape[0]
    time = [i / fs for i in range(N)]
    frameTime = FrameTime(fn, wlen, inc, fs)

    fig = plt.figure(figsize=(10, 13))
    plt.subplot(2, 1, 1)
    plt.plot(time, x)
    plt.xlabel('Time/s')
    plt.ylabel('Amplitude')
    plt.title('Speech Waveform')
    plt.subplot(2, 1, 2)
    plt.plot(frameTime, En)
    plt.xlabel('Time/s')
    plt.ylabel('Amplitude')
    plt.title('Short Time Energy')
    plt.savefig('images/energy.png')
    plt.show()




