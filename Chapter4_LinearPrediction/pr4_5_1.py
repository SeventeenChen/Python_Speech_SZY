# pr4_5_1
# LPC <-> LSP

from Universal import *
from audiolazy.lazy_lpc import lpc
from LPC import LPC
from scipy.signal import deconvolve, convolve
import cmath
import math

def ar2lsp(a):
	"""
	Convert LPC ai to LSP
	:param a: LPC(order = p, lpc size = p + 1)
	:return lsf: LSP angle frequency
	"""
	a = np.array(a)
	a = a.reshape(-1, )
	if (np.real(a) != a).all():
		print('Line spectral frequencies are not defined for complex polynomials.')
		return
	if a[0] != 1.0:
		a = a / a[0]
	# print(a)
	a0 = np.poly1d(a)
	if np.max(np.abs(a0.roots)) >= 1.0:
		print('The polynomial must have all roots inside of the unit circle. ')
		return
	
	p = len(a) - 1      # polynomial order
	a1 = np.concatenate((a, np.zeros(1)))
	a2 = a1[::-1]
	P1 = a1 + a2
	Q1 = a1 - a2
	
	if (p % 2) :
		Q = deconvolve(Q1, np.array([1, 0, -1]))
		P = P1
	else:
		Q = deconvolve(Q1, np.array([1, -1]))
		P = deconvolve(P1, np.array([1, 1]))
	
	P0 = np.poly1d(P[0])
	Q0 = np.poly1d(Q[0])
	rP = P0.roots
	rQ = Q0.roots
	aP = np.angle(rP[0:len(rP):2])
	aQ = np.angle(rQ[0:len(rQ):2])
	aLSF = np.concatenate((aP, aQ))
	lsf = np.sort(aLSF)
	
	return lsf


def lsp2ar(lsf):
	"""
	Convert LSP to LPC ai
	:param lsf: LSP angle frequency
	:return a: LPC(order = p, lpc size = p + 1)
	"""
	lsf = np.array(lsf)
	lsf = lsf.reshape(-1, )
	if (np.real(lsf) != lsf).all():
		print('Line spectral frequencies are not defined for complex polynomials.')
		return
	if np.max(lsf) > math.pi or np.min(lsf) < 0:
		print( 'Line spectral frequencies must be between 0 and pi. ')
		return
	
	p = len(lsf)
	z = np.exp(cmath.sqrt(-1) * lsf)
	rP = z[0 : len(z) : 2]
	rQ = z[1 : len(z) : 2]
	rP = np.concatenate((rP, rP.conjugate()))
	rQ = np.concatenate((rQ, rQ.conjugate()))
	P = np.poly(rP)
	Q = np.poly(rQ)
	
	if (p % 2) :
		Q1 = convolve(Q, np.array([1, 0, -1]))
		P1 = P
	else:
		Q1 = convolve(Q, np.array([1, -1]))
		P1 = convolve(P, np.array([1, 1]))
		
	a = (P1 + Q1) / 2
	a = np.delete(a, -1)
	
	return a
	
	
	
	
if __name__ == '__main__':
    S = Speech()
    x, fs = S.audioread('aa.wav', 8000)
    x = x / np.max(np.abs(x))
    time = np.linspace(0, len(x), num=len(x)) / fs
    N = 200     # frame length
    M = 80      # frame shift
    xn = S.enframe(x, N, M)
    s = xn[99, :]    # 100th frame
    
    p = 12          # lpc ORDER
    num = 257       # nfft
    a2 = lpc(s, p)  # lpc
    n0 = num - 2
    Hw = LPC.lpcar2ff(a2.numerator, n0)
    Hw_abs = np.abs(Hw)
    lsf = ar2lsp(a2.numerator)
    P_w = lsf[0:p:2]    # frequency of P,Q
    Q_w = lsf[1:p:2]
    P_f = P_w * fs / 2 / math.pi    # rad -> Hz
    Q_f = Q_w * fs / 2 / math.pi
    
    # figure
    plt.figure()
    plt.plot(time, x)
    plt.title('Waveform of aa.wav')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim([0, np.max(time)])
    plt.savefig('images/lsf_1.png', dpi=600)
    plt.show()
    
    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    plt.plot(s)
    plt.title('One Frame of aa.wav')
    plt.xlabel('Sample points')
    plt.ylabel('Amplitude')

    
    freq = np.linspace(0, num - 1, num=num) * fs / 512
    K = len(Q_w)
    ar = lsp2ar(lsf)
    Hw1 = LPC.lpcar2ff(ar, n0)
    Hw1_abs = np.abs(Hw1)
    
    plt.subplot(2, 1, 2)
    plt.plot(freq, 20 * np.log10(Hw_abs[0:num]/np.max(Hw_abs[0:num])), 'k', linewidth=2)
    plt.plot(freq + 1, 20 * np.log10(Hw1_abs[0:num]/np.max(Hw1_abs[0:num])),'--', linewidth=4)
    for i in range(K):
	    plt.plot(np.array([Q_f[i], Q_f[i]]),np.array([-35, 5]), 'k--')
	    plt.plot(np.array([P_f[i], P_f[i]]),np.array([-35, 5]), 'k-')
	
    plt.axis([0, fs/2, -35, 5])
    plt.title('LPC spectrum & LPC spectrum reconstruct from LSP')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.savefig('images/lsf_2.png', dpi=600)
    plt.show()
    
    for i in range(p+1):
	    print('{}, {}, {}'.format(i+1, a2.numerator[i], ar[i]))
	    
    