#
# pr9_5_1

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

if __name__ == '__main__':
	fs = 400  # sample frequency
	N = 400  # data length
	n = np.arange(0, N)
	dt = 1 / fs
	t = n * dt  # time sequence
	A = 0.5  # pluse modulation amplitude
	# signal
	x = (1 + 0.5 * np.cos(2 * np.pi * 5 * t)) * np.cos(2 * np.pi * 50 * t + A * np.sin(2 * np.pi * 10 * t))
	z = hilbert(x)
	a = np.abs(z)  # envelope
	fnor = (np.diff(np.unwrap(np.angle(z)))) / (2 * np.pi)  # instantaneous frequency
	FNOR = np.zeros(len(z))
	FNOR[0] = fnor[0]
	FNOR[1:] = fnor
	
	# figure
	plt.figure(1, figsize=(16, 9))
	plt.subplot(3, 1, 1)
	plt.plot(t, x)
	plt.axis([0, max(t), min(x), max(x)])
	plt.xlabel('Times [s]')
	plt.ylabel('Amplitude')
	plt.title('Singal')
	plt.subplot(3, 1, 2)
	plt.plot(t, a)
	plt.axis([0, max(t), 0.4, 1.6])
	plt.xlabel('Times [s]')
	plt.ylabel('Amplitude')
	plt.title('Envelope')
	plt.subplot(3, 1, 3)
	plt.plot(t, FNOR * fs)
	plt.axis([0, max(t), 43, 57])
	plt.xlabel('Times [s]')
	plt.ylabel('Amplitude')
	plt.title('Instantaneous frequency')
	plt.savefig('images/Hilbert_Transform', bbox_inches='tight', dpi=600)
	plt.show()
