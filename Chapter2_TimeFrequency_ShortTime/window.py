# 窗函数（矩形、汉明、汉宁）
import matplotlib.pyplot as plt
import numpy as np


N = 32
nn = [i for i in range(N)]
plt.figure(figsize=(16, 12))
plt.subplot(3, 1, 1)
plt.stem(np.ones(N))
plt.title('Rectangle window')

# w = 0.54 - 0.46 * np.cos(np.multiply(nn, 2 * np.pi) / (N - 1))
w = np.hamming(N)
plt.subplot(3, 1, 2)
plt.stem(w)
plt.title('Hamming window')

# w = 0.5 * (1 - np.cos(np.multiply(nn, 2 * np.pi) / (N - 1)))
w = np.hanning(N)
plt.subplot(3, 1, 3)
plt.stem(w)
plt.title('Hanning window')
plt.savefig('images/window.png')
plt.show()
plt.close()
