# pr_3_5_1
# EMD变换分解

from PyEMD import EMD
import matplotlib.pylab as plt
import numpy as np
import math

if __name__ == '__main__':
	fs = 5000		# sample frequency
	N = 500			# sample point number
	n = np.linspace(0, N-1, num=N)
	t1 = n/fs		# set time
	x1 = np.sin(2 * math.pi * 50 * t1)
	x2 = np.sin(2 * math.pi * 150 * t1) / 3
	z = x1 + x2
	IMF = EMD().emd(z, t1, max_imf=2)
	m, n = IMF.shape

	# figure
	plt.figure(num = 1, figsize=(12, 20), dpi=600)
	plt.subplot(m+1, 1, 1)
	plt.plot(t1, z)
	plt.title('Original Signal')
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	for i, imf in enumerate(IMF):
		plt.subplot(m+1, 1, i+2)
		plt.plot(t1, imf)
		plt.xlabel('Time [s]')
		plt.ylabel('Amplitude')
		plt.title('IMF{}'.format(i+1))
	plt.tight_layout()
	plt.savefig('images/emd.png')
	plt.show()

