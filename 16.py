#!/usr/bin/env python3
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def multiply(x,f,g):
	return f(x) * g(x)

def multiply2(y,u,H,x):
	return u(y) * H(x,y)

def createKernel(n):
	alpha = []
	betha = []
	alpha.append(lambda x: 1)
	betha.append(lambda x: 1 / 5)
	p = 1

	for i in range(1, n):
		p *= i
		alpha.append(lambda x, i=i, p=p: x**i / p)
		betha.append(lambda x, i=i: x**i / 5)

	return alpha, betha

def getC(alpha,betha,n):
	A = np.zeros((n,n))
	b = np.zeros(n)

	for i in range(n):
		for j in range(n):
			d = 0
			if (i == j):
				d = 1
			A[i][j] = d - integrate.quad(multiply, 0, 1, args=(alpha[j],betha[i]))[0]

		b[i] = integrate.quad(multiply, 0, 1, args=(func,betha[i]))[0]

	c = np.linalg.solve(A,b)
	return c

def getValues(U,N):
	h = 1 / N
	ys = []
	for i in range(N+1):
		ys.append(U(i*h))
	return np.array(ys)

def getG(nodes):
	g = []
	for no in nodes:
		g.append(func(no))

	return np.array(g)

def getZ(n,nodes,coeffs,g):
	D = np.zeros((n,n))
	for j in range(n):
		for k in range(n):
			d = 0
			if (j == k):
				d = 1
			D[j][k] = d - coeffs[k] * H(nodes[j],nodes[k])

	return np.linalg.solve(D,g)

H = lambda x,y: np.exp(x * y) / 5
func = lambda x: 1 - x**2

print("a = %d, b = %d, N = %d" % (0,1,10))

print("------------------------------------------------------")
print("Вырожденное ядро")
print("------------------------------------------------------")
print("\nРанг 3")
n = 3
print("Ядро: 1*1/5 + x*y/5 + x^2*y^2/5")
alpha3, betha3 = createKernel(n+1)
c3 = getC(alpha3,betha3,n+1)
U3 = lambda x: func(x) + sum(c3 * [i(x) for i in alpha3])
ys3 = getValues(U3,10)
print("Значения на сетке:")
print(ys3)

# h = 1/10
# for i in range(11):
# 	asd= U3(h*i) - integrate.quad(multiply2, 0, 1, args=(U3, H, h*i))[0] - func(h*i)
# 	print(asd)

print("\nРанг 4")
print("Ядро: 1*1/5 + x*y/5 + x^2*y^2/5 + x^3*y^3/5")
n = 4
alpha4, betha4 = createKernel(n+1)
c4 = getC(alpha4,betha4,n+1)
U4 = lambda x: func(x) + sum(c4 * [i(x) for i in alpha4])
ys4 = getValues(U4,10)
print("Значения на сетке:")
print(ys4)

print("\nДельта max = %f\n" % max(abs(ys3 - ys4)))
# xs = [i for i in range(11)]
# plt.plot(xs,abs(ys3 - ys4))
# plt.show()

print("------------------------------------------------------")
print("ММК")
print("------------------------------------------------------")
print("\n3 Узла")
n = 3
c = 1 / 2.
nodes3 = [-0.7745966692, 0, 0.7745966692]
coeffs3 = [0.5555555556,0.8888888889,0.5555555556]

for i in range (0, n):
	coeffs3[i] = c * coeffs3[i]
	nodes3[i] = c * nodes3[i] + 1 / 2.

g3 = getG(nodes3)
z3 = getZ(n,nodes3,coeffs3,g3)
U3 = lambda x: func(x) + sum(coeffs3 * z3 * [H(x,nodes3[i]) for i in range(n)])
ys3 = getValues(U3,10)
print("Значения на сетке:")
print(ys3)

# for j in range(n):
# 	s = 0
# 	for k in range(n):
# 		s += z3[k]*coeffs3[k]*H(nodes3[j],nodes3[k])
# 	print(z3[j] - s - func(nodes3[j]))

# h = 1/10
# for i in range(11):
# 	asd= U3(h*i) - integrate.quad(multiply2, 0, 1, args=(U3, H, h*i))[0] - func(h*i)
# 	print(asd)

print("\n4 Узла")
n = 4
nodes4 = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]
coeffs4 = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]
с = 1 / 2
for i in range (0, n):
	coeffs4[i] = c * coeffs4[i]
	nodes4[i] = c * nodes4[i] + 1 / 2

g4 = getG(nodes4)
z4 = getZ(n,nodes4,coeffs4,g4)
U4 = lambda x: func(x) + sum(coeffs4 * np.array([H(x,nodes4[i]) for i in range(n)]) * z4)
ys4 = getValues(U4,10)
print("Значения на сетке:")
print(ys4)


# for j in range(n):
# 	s = 0
# 	for k in range(n):
# 		s += z4[k]*coeffs4[k]*H(nodes4[j],nodes4[k])
# 	print(z4[j] - s - func(nodes4[j]))
#
# print()
# for i in range(11):
# 	asd= U4(h*i) - integrate.quad(multiply2, 0, 1, args=(U4, H, h*i))[0] - func(h*i)
# 	print(asd)

# print()
# h = 1/10
# for j in range(11):
# 	s=func(h*j)
# 	for k in range(n):
# 		s += z4[k]*coeffs4[k]*H(h*j,nodes4[k])
# 	print(s)

# nodes6 = [0.9324695142, -0.9324695142, 0.6612093865, -0.6612093865, 0.2386191861, -0.2386191861]
# coeffs6 = [0.1713244924, 0.1713244924, 0.3607615730, 0.3607615730, 0.4679139346, 0.4679139346]
