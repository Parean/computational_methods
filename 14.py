#!/usr/bin/env python3
from operator import itemgetter
import numpy as np
import math as ma

################################################################################
solution = lambda x: 1 / (x**2 + 1)
p = lambda x: 1
q = lambda x: -(x**2 + alpha)
r = lambda x: -2*x
f = lambda x: 2 * (3 * x**2 - alpha) / (x**2 + alpha)**3
alpha = 1
al1 = 1
al2 = -2
al3 = 1 / alpha
bet1 = 1
bet2 = 0
bet3 = 1 / (1 + alpha)
################################################################################

def kth_diag_indices(a, k):
	rows, cols = np.diag_indices_from(a)
	if k < 0:
		return rows[-k:], cols[:k]
	elif k > 0:
		return rows[:-k], cols[k:]
	else:
		return rows, cols

def solve(a,b,c,n):
	ss = []
	tt = []
	s = -c[0] / b[0]
	ss.append(s)
	t = g[0] / b[0]
	tt.append(t)

	print("\nПрогоночные коэффициенты:")
	for i in range(1,n):
		predS = s
		print(i-1, s, t)
		s = -c[i] / (b[i] + a[i]*predS)
		t = (-a[i] * t + g[i]) / (b[i] + a[i]*predS)

		ss.append(s)
		tt.append(t)
	print(n,s,t)
	y = t
	yy = [y]
	for i in range(n-2,-1,-1):
		y=ss[i] * y + tt[i]
		yy.insert(0,y)

	print("\nВектор неизвестных: \n%s" % yy)
	return np.array(yy)

def getLinearBoundings(h,a,b,c,g):
	b0 = al1 * h - al2
	c0 = al2
	g0 = h * al3

	an = -bet2
	bn = bet1 * h + bet2
	gn = bet3 * h

	return b0,c0,an,bn,g0,gn

def getSquareBoundings(h,a,b,c,g):
	b0 = al1 + al2 * (a[0] / c[0] - 3) / (2*h)
	c0 = al2 * (b[0] / c[0] + 4) / (2*h)
	g0 = al3 + al2 * (g[0] / c[0]) / (2*h)

	an = -bet2 * (4 + b[-1] / a[-1]) / (2*h)
	bn = bet1 + bet2 * (3 - c[-1] / a[-1]) / (2*h)
	gn = bet3 - bet2 * (g[-1] / a[-1]) / (2*h)

	return b0,c0,an,bn,g0,gn

def createDiags(n,boundF):
	h = 1 / n
	a = []
	b = []
	c = []
	g = []
	for i in range(1,n):
		xi = i * h
		a.append(p(xi) - h * q(xi) / 2)
		b.append(-2*p(xi) + r(xi) * h**2)
		c.append(p(xi) + h * q(xi) / 2)
		g.append(f(xi) * h**2)

	b0,c0,an,bn,g0,gn = boundF(h,a,b,c,g)

	b.insert(0,b0)
	c.insert(0,c0)
	g.insert(0,g0)

	a.append(an)
	b.append(bn)
	g.append(gn)

	return np.array(a),np.array(b),np.array(c),np.array(g)

def getNormV(v):
	return sum(abs(v))

n = 25
h = 1 / n
np.set_printoptions(precision=2)
print("-----------------------------------------------------------------------")
print("O(n)\n")
a,b,c,g = createDiags(n,getLinearBoundings)

systemMatrix = np.zeros((n+1,n+1))
upperDiag = kth_diag_indices(systemMatrix,1)
mainDiag = kth_diag_indices(systemMatrix,0)
lowDiag = kth_diag_indices(systemMatrix,-1)

systemMatrix[mainDiag] = b
systemMatrix[upperDiag] = c
systemMatrix[lowDiag] = a
a = np.insert(a,0,0)
c = np.append(c,0)

extendedMatrix = systemMatrix.copy()
extendedMatrix=np.append(extendedMatrix, g[np.newaxis].T, axis=1)

print("Расширенная матрица системы:")
print(extendedMatrix)
yy = solve(a,b,c,n+1)

print("\nНорма вектора невязки % 1.20f" % getNormV(np.matmul(systemMatrix,yy)-g))
solutions = np.array([solution(i*h) for i in range(n+1)])
print("\n|u - u точное|\n%s" % abs(yy - solutions))

print("-----------------------------------------------------------------------")
print("O(n^2)\n")
a,b,c,g = createDiags(n,getSquareBoundings)

systemMatrix = np.zeros((n+1,n+1))
upperDiag = kth_diag_indices(systemMatrix,1)
mainDiag = kth_diag_indices(systemMatrix,0)
lowDiag = kth_diag_indices(systemMatrix,-1)

systemMatrix[mainDiag] = b
systemMatrix[upperDiag] = c
systemMatrix[lowDiag] = a
a = np.insert(a,0,0)
c = np.append(c,0)

extendedMatrix = systemMatrix.copy()
extendedMatrix=np.append(extendedMatrix, g[np.newaxis].T, axis=1)

print("Расширенная матрица системы:")
print(extendedMatrix)
yy = solve(a,b,c,n+1)

print("\nНорма вектора невязки % 1.20f" % getNormV(np.matmul(systemMatrix,yy)-g))
solutions = np.array([solution(i*h) for i in range(n+1)])
print("\n|(u - u точное)|\n%s" % abs(yy - solutions))
