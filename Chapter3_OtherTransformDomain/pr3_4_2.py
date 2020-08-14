# pr3_4_2
# 利用小波包分解构成Bark滤波器

from Universal import *
import pywt

if __name__ == '__main__':
	S = Speech()
	xx, fs = S.audioread('aa.wav', 8000)
	x = xx - np.mean(xx)		# DC
	x = x/np.max(np.abs(x))		# normalized
	N = len(x)
	n = 5	# decomposition level
	T = pywt.WaveletPacket(data = x, wavelet = 'db2', maxlevel = n, mode='zero')		# 一维小波包分解
	new_wp = pywt.WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')


	# 计算各节点对应系数
	map =[]
	NodeName = []
	for row in range(n):
		map.append([])
		NodeName.append([])
		for i in [node.path for node in T.get_level(level = row + 1, order = 'natural')]:
			map[row].append(T[i].data)
			NodeName[row].append(i)

	# 按指定的节点，对时间序列分解的一位小波包系数重构
	y = np.zeros((17, N))
	for i in range(8):
		new_wp[NodeName[4][i]] = map[4][i]
		y[i, :] = new_wp.reconstruct(update=False)
		new_wp = pywt.WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')

	new_wp[NodeName[3][4]] = map[3][4]
	y[8, :] = new_wp.reconstruct(update=False)
	new_wp = pywt.WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	new_wp[NodeName[3][5]] = map[3][5]
	y[9, :] = new_wp.reconstruct(update=False)
	new_wp = pywt.WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	new_wp[NodeName[4][11]] = map[4][11]
	y[10, :] = new_wp.reconstruct(update=False)
	new_wp = pywt.WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	new_wp[NodeName[4][12]] = map[4][12]
	y[11, :] = new_wp.reconstruct(update=False)
	new_wp = pywt.WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')
	new_wp[NodeName[3][7]] = map[3][7]
	y[12, :] = new_wp.reconstruct(update=False)
	new_wp = pywt.WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')

	for i in range(4, 8):
		new_wp[NodeName[2][i]] = map[2][i]
		y[9 + i, :] = new_wp.reconstruct(update=False)
		new_wp = pywt.WaveletPacket(data=np.zeros(len(x)), wavelet='db2', mode='zero')


	# figure
	plt.figure(1, figsize=(18, 15), dpi=600)
	plt.subplot(5, 1, 1)
	plt.plot(x)
	plt.autoscale(enable=True, tight=True)
	for i in range(4):
		plt.subplot(5, 2, 2 * i + 3)
		plt.plot(y[i * 2 + 1, :])
		plt.autoscale(enable=True, tight=True)
		plt.ylabel('y' + str(i*2+1))
		plt.subplot(5, 2, 2 * i + 4)
		plt.plot(y[i * 2 -2, :])
		plt.autoscale(enable=True, tight=True)
		plt.ylabel('y' + str(i*2 +2))
	plt.savefig('images/bark1.png')
	plt.show()

	plt.figure(2, figsize=(18, 15), dpi=600)
	for i in range(4):
		plt.subplot(5, 2, 2 * i + 1)
		plt.plot(y[i * 2 + 8, :])
		plt.autoscale(enable=True, tight=True)
		plt.ylabel('y' + str(i * 2 + 9))
		plt.subplot(5, 2, 2 * i + 2)
		plt.plot(y[i * 2 + 9, :])
		plt.autoscale(enable=True, tight=True)
		plt.ylabel('y' + str(i * 2 + 10))
	plt.subplot(5, 2, 9)
	plt.plot(y[16, :])
	plt.autoscale(enable=True, tight=True)
	plt.ylabel('y17')
	plt.savefig('images/bark2.png')
	plt.show()



