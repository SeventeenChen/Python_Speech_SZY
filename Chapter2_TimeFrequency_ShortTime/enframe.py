# 分帧
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc	# 计算帧数
    frameout = np.zeros((nf, nlen))		# 初始化
    indf = np.multiply(inc, np.array([i for i in range(nf)]))	# 设置每帧在x中的位移量位置
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]		# 分帧
    if isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))		# 每帧乘以窗函数的值
    return frameout

if __name__ == '__main__':
    fs, data = wavfile.read('bluesky3.wav')

    inc = 100
    wlen = 200
    en = enframe(data, wlen, inc)
    i = input('Start frame(i):')
    i = int(i)
    tlabel = i

    plt.subplot(4, 1, 1)
    x = [i for i in range((tlabel - 1) * inc, (tlabel - 1) * inc + wlen)]
    plt.plot(x, en[tlabel, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(a)The {} Frame Waveform'.format(tlabel))

    plt.subplot(4, 1, 2)
    x = [i for i in range((tlabel + 1 - 1) * inc, (tlabel + 1 - 1) * inc + wlen)]
    plt.plot(x, en[i + 1, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(b)The {} Frame Waveform'.format(tlabel + 1))

    plt.subplot(4, 1, 3)
    x = [i for i in range((tlabel + 2 - 1) * inc, (tlabel + 2 - 1) * inc + wlen)]
    plt.plot(x, en[i + 2, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(c)The {} Frame Waveform'.format(tlabel + 2))

    plt.subplot(4, 1, 4)
    x = [i for i in range((tlabel + 3 - 1) * inc, (tlabel + 3 - 1) * inc + wlen)]
    plt.plot(x, en[i + 3, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(d)The {} Frame Waveform'.format(tlabel + 3))


    plt.savefig('images/enframe.png')
    plt.show()
    plt.close()

