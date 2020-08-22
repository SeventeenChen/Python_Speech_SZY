# pr5_4_2
# remove poly trend

from Universal import *
import numpy as np

def polydetrend(x, fs, m):
	"""
	remove poly trend
	:param x:
	:param fs:
	:param m:
	:return:
	"""
	
	x = x.reshape(-1, 1)
	N = len(x)
	t = np.linspace(0, N - 1, num=N) / fs
	a = np.polyfit(t, x, m)
	xtrend = np.polyval(a, t)
	xtrend = xtrend.reshape(-1, 1)
	y = x - xtrend
	
	return y, xtrend

if __name__ == '__main__':
	S = Speech()
	x, fs = S.audioread('bluesky32.wav', 8000)
	t = np.linspace(0, len(x) - 1, num=len(x)) / fs     # time
	y, xtrend = polydetrend(x, fs, 2)
	
	# figure
	plt.figure(figsize=(16, 12))
	plt.subplot(2, 1, 1)
	plt.plot(t, x)
	plt.plot(t, xtrend, linewidth = 3)
	plt.ylim([-1.5, 1])
	plt.legend(['Signal with Trend', 'Trend'])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Signal with trend')
	plt.subplot(2, 1, 2)
	plt.plot(t, y)
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.title('Signal Removed Trend')
	plt.savefig('images/ploydetrend.png', bbox_inches='tight', dpi=600)
	plt.show()