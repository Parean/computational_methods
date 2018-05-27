#!/usr/bin/env python3
from operator import itemgetter
import numpy as np
import math

def getMaxIndex(a,k,n):
	index = k
	max = a[k][k]
	for i in range(k+1,n):
		if a[i][k] > max:
			index = i
	return index

def smartGaus(input,n):
	a = input.copy()
	x = [0] * n
	for k in range(n):
		maxIndex = getMaxIndex(a,k,n)
		# print(a,end="\n\n")
		if maxIndex != k:
			a[[maxIndex,k],k:] = a[[k,maxIndex],k:]
		# print(a,end="\n\n")

		temp = a[k][k]
		if(temp):
			for j in range(k, n + 1):
				a[k][j] /= temp

		for i in range(k + 1, n):
			temp = a[i][k]
			for j in range(k, n + 1):
				a[i][j] = a[i][j] - a[k][j] * temp

	for i in range(n-1,-1,-1):
		s = 0
		for j in range(i,n):
			s += a[i][j] * x[j]
		x[i] = a[i][n] - s

	return x

def getNormM(a):
	max = -1
	for i in range(len(a)):
		s = 0
		for j in range(len(a[0])):
			s += abs(a[i][j])
		if(s > max):
			max = s

	return max

def getNormV(a):
	return sum(abs(a))

def getEigenvaluesEstimation(a,n):
	min = 10e10
	max = -10e10
	for i in range(n):
		s = 0
		for j in range(n):
			if i != j:
				s += abs(a[i][j])
		if a[i][i] - s < min:
			min = a[i][i] - s
		if a[i][i] + s > max:
			max = a[i][i] + s

	return min, max

def getApriory(H,g):
	e = 10e-8
	normH = getNormM(H)
	degNormH = 1
	normG = getNormV(g)
	error = e
	k = 0
	while(error >= e):
		degNormH *= normH
		error = degNormH * normG / (1 - normH)
		k += 1
		if k == 14:
			break
	return k, error

def getAposteriory(H,g,n):
	e = 10e-8
	normH = getNormM(H)
	xk = [0.] * n
	k = 0
	error = e
	while(error >= e):
		predXk = xk
		xk = np.matmul(H,xk) + g
		error = getNormV(xk - predXk) * normH / (1 - normH)
		k += 1
		# print("k = %d, xk = %s, погрешность = %f" % (k,xk,error))

	maxEigH = abs(np.linalg.eig(H)[0]).max()
	lusternik = predXk + 1 / (1-maxEigH) * (xk-predXk)
	return lusternik, xk, k, error

def getSeidel(H,g,n):
	low = np.tril_indices(n,-1)
	up = np.triu_indices(n)
	HL = H.copy()
	HR = H.copy()
	HL[up] = 0
	HR[low] = 0
	temp = np.linalg.inv(np.identity(n) - HL)
	HSeid = np.matmul(temp,HR)
	gSeid = np.matmul(temp,g)
	return HSeid, gSeid

def getDiadHAndG(a,b,n):
	H = a.copy()
	g = [0.] * n
	for i in range(n):
		for j in range(n):
			if i == j:
				H[i][j] = 0
			else:
				H[i][j] = -a[i][j]/a[i][i]
		g[i] = b[i]/a[i][i]
	return H,g

def getR(a):
	return abs(np.linalg.eig(a)[0]).max()

def getRelax(H,g,n):
	e = 10e-8
	normH = getNormM(H)
	xk = np.array([0.] * n)
	k = 0
	q = 2 / (1 + math.sqrt(1 - getR(H)**2))
	error = e
	while(error >= e):
		predXk = xk.copy()

		for i in range(n):
			temp = 0
			for j in range(i):
				temp += H[i][j]*xk[j]

			for j in range(i+1,n):
				temp += H[i][j]*predXk[j]

			temp += g[i] - predXk[i]
			xk[i] = temp * q + predXk[i]

		error = getNormV(xk - predXk) * normH / (1 - normH)
		k += 1
		# print("k = %d, xk = %s, погрешность = %f" % (k,xk,error))

	return xk, k, error

b = np.array([0., 1., 0])
a = np.array([[6.687233, 0.80267, -2.06459, 0], [0.80267, 5.07816, 0.48037, 1], [-2.06459, 0.48037, 4.02934, 0]])
aa = np.array([[6.687233, 0.80267, -2.06459], [0.80267, 5.07816, 0.48037], [-2.06459, 0.48037, 4.02934]])
n = len(aa)

x = smartGaus(a,n)
m, M = getEigenvaluesEstimation(aa,len(aa))
parameter = 2 / (m + M)
H = np.identity(n) - parameter * aa
g = parameter * b

print("Матрица А: \n%s\n" % aa)
print("Матрица системы Ax=b: \n%s\n" % a)
print("Решение системы методом Гаусса: \n%s\n" % x)
print("M = %f, m = %f, alpha = %f\n" % (m, M, parameter))
print("Матрица H: \n%s\n" % H)
print("Вектор g: \n%s\n" % g)
print("Норма матрицы H: %f\n" % getNormM(H))
print("------------------------------------------\n")

aprioryK, aprioryError = getApriory(H,g)
lusternik, xk, aposterioryK, aposterioryError = getAposteriory(H,g,n)
print("Метод простых итераций")
print("Априорное k = %d, априорная оценка = %f\n" % (aprioryK,aprioryError))
print("Апостериорное k = %d, апостериорная оценка = %f, фактическая погрешность = %f\n" % (aposterioryK,aposterioryError,getNormV(x-xk)))
print("x = %s, \nxk = %s\n" % (x,xk))
print("Вектор невязки системы: \n%s\n" % abs(b - np.matmul(aa, xk)))
print("------------------------------------------\n")

HSeid, gSeid = getSeidel(H,g,n)
lusternik, xk, aposterioryK, aposterioryError = getAposteriory(HSeid,gSeid,n)
print("Метод Зейделя")
print("Апостериорное k = %d, апостериорная оценка = %f, фактическая погрешность = %f\n" % (aposterioryK,aposterioryError,getNormV(x-xk)))
print("Спектральный радиус матрицы перехода Hseid = %f\n" % abs(np.linalg.eig(HSeid)[0]).max())
print("x = %s, \nxk = %s, \nУточнение по Люстернику = %s, \nФактическая погрешность уточнения = %1.10f\n" % (x,xk,lusternik,getNormV(lusternik-x)))
print("Вектор невязки системы: \n%s\n" % abs(b - np.matmul(aa, xk)))
print("------------------------------------------\n")

H,g = getDiadHAndG(aa,b,n)
xk, aposterioryK, aposterioryError = getRelax(H,g,n)
print("Метод релаксаций")
print("Апостериорное k = %d, апостериорная оценка = %f, фактическая погрешность = %f\n" % (aposterioryK,aposterioryError,getNormV(x-xk)))
print("x = %s, \nxk = %s, \nУточнение по Люстернику = %s, \nФактическая погрешность уточнения = %1.10f\n" % (x,xk,lusternik,getNormV(lusternik-x)))
print("Вектор невязки системы: \n%s\n" % abs(b - np.matmul(aa, xk)))
