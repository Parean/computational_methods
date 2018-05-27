#!/usr/bin/env python3
from operator import itemgetter
import numpy as np
import math as ma

def getEigDeg(a,n):
	e = 10e-12
	yk = np.array([1]*n)
	k = 0
	predYk = yk.copy()
	stop = 0
	while stop != n:
		k+=1
		predYk = yk.copy()
		yk = np.matmul(a,yk)
		l = yk[0] / predYk[0]
		stop = 0
		for i in range(n):
			stop += 1
			if(abs(yk[i] / predYk[i] - l) >= e):
				break
	print("Число шагов = %d" % k)
	return l,yk

def getEigScalar(a,n):
	e = 10e-12
	yk = np.array([1]*n)
	k = 0
	predYk = yk.copy()
	l = e
	predL = 0
	while abs(predL - l) >= e:
		predL = l
		k+=1
		predYk = yk.copy()
		yk = np.matmul(a,yk)
		l = np.dot(yk,predYk) / np.dot(predYk,predYk)

	print("Число шагов = %d" % k)
	return l,yk

def Jakobi(a,n):
	e = 10e-12
	x = np.identity(n)
	E = np.identity(n)
	phi = 0
	k = 0
	while True:
		k += 1
		up = np.vstack(np.triu_indices(n,1)).T
		maxIndex = np.argmax(abs(a[up[:,0],up[:,1]]))
		i,j = up[maxIndex]
		if(abs(a[i][j]) < e):
			break
		# print(i,j)

		phi = ma.atan(2 * a[i][j] / (a[i][i] - a[j][j])) / 2
		Vij = E.copy()
		Vij[i][i] = Vij[j][j] = ma.cos(phi)
		Vij[i][j] = -ma.sin(phi)
		Vij[j][i] = ma.sin(phi)

		x = np.matmul(x,Vij)
		a = np.matmul(a,Vij)
		a = np.matmul(Vij.T,a)

	print("Число шагов = %d" % k)
	return a,x

a = np.array([[-0.881923,-0.046444,0.334218], [-0.046444,0.560226,0.010752], [0.334218,0.010752,-0.883417]])
n = len(a)
print("Эпсилон = 10e-12")
print("------------------------------------------------------")
print("Степенной метод")
print("------------------------------------------------------")
l,yk = getEigDeg(a,n)
print("Максимальное по модулю собственное значение = %f" % l)
print("Ему соответствует собственный вектор: %s\n" % yk)
print("Вектор невязки \nA * yk - yk * lambda: \n%s" % abs(np.matmul(a,yk) - l*yk))
ll,yk = getEigDeg(a-l*np.identity(n),n)
print("Противоположная граница спектра = %f" % (ll+l))
lll = sum(a[np.diag_indices_from(a)]) - l - l - ll
print("Недостающее собственное значение = %f\n" % lll)

print("------------------------------------------------------")
print("Метод скалярных произведений")
print("------------------------------------------------------")
l,yk = getEigScalar(a,n)
print("Максимальное по модулю собственное значение = %f" % l)
print("Ему соответствует собственный вектор: %s\n" % yk)
print("Вектор невязки \nA * yk - yk * lambda: \n%s" % abs(np.matmul(a,yk) - l*yk))

print("------------------------------------------------------")
print("Метод Якоби")
print("------------------------------------------------------")
ls,xs = Jakobi(a,n)
print("Собственные значения: %f, %f, %f\n" % (ls[0][0],ls[1][1],ls[2][2]))
print("Приближение диагональной матрицы: \n%s\n" % ls)
print("Им соответствуют собственные вектора: \n%s\n" % xs)
print("1й вектор невязки \nA * y0 - y0 * lambda0: \n%s\n" % abs(np.matmul(a,xs[:,0]) - ls[0][0]*xs[:,0]))
print("2й вектор невязки \nA * y2 - y2 * lambda2: \n%s\n" % abs(np.matmul(a,xs[:,1]) - ls[1][1]*xs[:,1]))
print("3й вектор невязки \nA * y3 - y3 * lambda3: \n%s\n" % abs(np.matmul(a,xs[:,2]) - ls[2][2]*xs[:,2]))
