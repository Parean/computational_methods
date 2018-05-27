#!/usr/bin/env python3

import numpy as np

def getNormM(a):
	max = -1
	for j in range(len(a[0])):
		s = 0
		for i in range(len(a)):
			s += abs(a[i][j])
		if(s > max):
			max = s

	return max

def getNormV(v):
	return sum(abs(v))

def getCond(a):
	aInv = np.linalg.inv(a)
	return getNormM(a) * getNormM(aInv)

b = np.array([200, -600])
db = np.array([199,-601])
a = np.array([[-401,200],[1200,-601]])
x = np.linalg.solve(a, b)
dx = np.linalg.solve(a, db)
cond = getCond(a)

print("Решение системы Аx=b: %s" % x)
print("Решение системы Аx=b': %s" % dx)
print("Число обусловленности: %f" % cond)
print("Фактическая относительная погрешность: %f" % (getNormV(dx-x) / getNormV(x)))
print("относительная погрешность возмущения: %f" % ((getNormV(db-b) / getNormV(b))))
print("Оценка погрешости: %f" % (cond * (getNormV(db-b) / getNormV(b))))
