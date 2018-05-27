#!/usr/bin/env python3
from operator import itemgetter

import numpy as np
def gaus(input,n,warning):
	a = input.copy()
	x = [0] * n
	for k in range(n):
		# print(a, end='\n\n')
		temp = a[k][k]
		if warning and abs(temp) < 10e-3:
			print("----------------------------------------------------------------------\n"\
				  "!!Внимание!! Происходит деление на очень маленький элемент = %f\n"\
				  "----------------------------------------------------------------------\n" % temp)
		if(temp):
			for j in range(k, n + 1):
				a[k][j] /= temp

		for i in range(k + 1, n):
			temp = a[i][k]
			for j in range(k, n + 1):
				a[i][j] = a[i][j] - a[k][j] * temp
		# print(a, end='\n\n')

	for i in range(n-1,-1,-1):
		s = 0
		for j in range(i,n):
			s += a[i][j] * x[j]
		x[i] = a[i][n] - s

	return x

def getMaxIndex(a,k,n):
	index = k
	max = a[k][k]
	for i in range(k+1,n):
		if a[i][k] > max:
			index = i
	return index

def optimalGaus(input,n):
	a = input.copy()
	x = [0] * n
	for k in range(n):
		maxIndex = getMaxIndex(a,k,n)
		if maxIndex != k:
			a[[maxIndex,k],k:] = a[[k,maxIndex],k:]

		temp = a[k][k]
		if abs(temp) < 10e-3:
			print("----------------------------------------------------------------------\n"\
				  "!!Внимание!! Происходит деление на очень маленький элемент = %f\n"\
				  "----------------------------------------------------------------------\n" % temp)
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

def getColumn(a,n,i):
	v = [0] * n
	v[i] = 1
	for j in range(n):
		a[j][n] = v[j]
	return optimalGaus(a,n)

def inverseMatrix(a,n):
	columns = []
	for i in range(n):
		columns.append(getColumn(a.copy(),n,i))

	return np.transpose(columns)

def getNormM(a):
	max = -1
	for j in range(len(a[0])):
		s = 0
		for i in range(len(a)):
			s += abs(a[i][j])
		if(s > max):
			max = s

	return max

def getCond(a):
	aInv = np.linalg.inv(a)
	return getNormM(a) * getNormM(aInv)

b = [0., 1., 0]
a = np.array([[6.687233, 0.80267, -2.06459, 0], [0.80267, 5.07816, 0.48037, 1], [-2.06459, 0.48037, 4.02934, 0]])
aa = np.array([[6.687233, 0.80267, -2.06459], [0.80267, 5.07816, 0.48037], [-2.06459, 0.48037, 4.02934]])

c = np.array([[6.687233*10e-8, 0.80267, -2.06459, 0], [0.80267, 5.07816, 0.48037, 1], [-2.06459, 0.48037, 4.02934, 0]])
cc = np.array([[6.687233*10e-8, 0.80267, -2.06459], [0.80267, 5.07816, 0.48037], [-2.06459, 0.48037, 4.02934]])

print("Матрица А: \n%s\n" % aa)
print("Обратная матрица: \n%s\n" % inverseMatrix(a,len(a)))
print("Число обусловленности: %f\n" % getCond(aa))
print("------------------------------------------\n")

print("Матрица системы Ax=b: \n%s\n" % a)
print("Решение системы Ax=b по схеме единственного деления: \n%s\n" % gaus(a,len(a),1))
print("Вектор невязки системы Ax=b по схеме единственного деления: \n%s\n" % abs(b - np.matmul(aa, gaus(a,len(a),0))))
print("Решение системы Ax=b по схеме с выбором главного элемента: \n%s\n" % optimalGaus(a,len(a)))
print("Вектор невязки системы Ax=b по схеме с выбором главного элемента: \n%s\n" % abs(b - np.matmul(aa, optimalGaus(a,len(a)))))
print("------------------------------------------\n")

print("Матрица системы Cx=b: \n%s\n" % c)
print("Решение системы Cx=b по схеме единственного деления: \n%s\n" % gaus(c,len(c),1))
print("Вектор невязки системы Cx=b по схеме единственного деления: \n%s\n" % abs(b - np.matmul(cc, gaus(c,len(c),0))))
print("Решение системы Cx=b по схеме с выбором главного элемента: \n%s\n" % optimalGaus(c,len(c)))
print("Вектор невязки системы Cx=b по схеме с выбором главного элемента: \n%s\n" % abs(b - np.matmul(cc, optimalGaus(c,len(c)))))
