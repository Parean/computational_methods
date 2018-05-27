#!/usr/bin/env python3
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def multiply(x,f,g):
	return f(x) * g(x)

def multiply2(y,u,H,x):
	return u(y) * H(x,y)

def kth_diag_indices(a, k):
	rows, cols = np.diag_indices_from(a)
	if k < 0:
		return rows[-k:], cols[:k]
	elif k > 0:
		return rows[:-k], cols[k:]
	else:
		return rows, cols

U = lambda x,t: np.exp(-4*t) * (np.sin(2*x) + 1 - x)
func = lambda x,t: -4 * np.exp(-4 * t) * (1 - x)

def getSolution(m,n,h,tau):
	UU = np.zeros((n+1,m+1))
	for k in range(0,m+1):
		for i in range (0,n+1):
			UU[i][k] = U(i*h,k*tau)

	return np.array(UU)


def explicitSheme(m,n,h,tau):
	UU = np.zeros((n+1,m+1))
	for i in range(n+1):
		UU[i][0] = U(h*i,0)

	for k in range(1,m+1):
		for i in range (1,n):
			UU[i][k] = (tau / h**2) * (UU[i+1][k-1] + UU[i-1][k-1]) + (1 - 2 * tau / h**2)*UU[i][k-1] + tau * func(h*i,tau*(k-1))
		UU[0][k] = U(0,tau*k)
		UU[n][k] = U(1,tau*k)

	return np.array(UU)

def nonExplicitScheme(m,n,h,tau):
	UU = np.zeros((n+1,m+1))
	for i in range(n+1):
		UU[i][0] = U(h*i,0)

	for k in range(1,m+1):
		a = [0]
		b = [1]
		c = [0]
		g = [U(0,k*tau)]

		for i in range(1,n):
			a.append(-tau / h**2)
			b.append(1 + 2 * tau / h**2)
			c.append(-tau / h**2)
			g.append(UU[i][k-1] + tau * func(h*i,k*tau))

		a.append(0)
		b.append(1)
		c.append(0)
		g.append(U(1,k*tau))
		UU[:,k] = solve(a,b,c,g,n+1)
	return UU


def solve(a,b,c,g,n):
	ss = []
	tt = []
	s = -c[0] / b[0]
	ss.append(s)
	t = g[0] / b[0]
	tt.append(t)

	for i in range(1,n):
		predS = s
		s = -c[i] / (b[i] + a[i]*predS)
		t = (-a[i] * t + g[i]) / (b[i] + a[i]*predS)

		ss.append(s)
		tt.append(t)
	y = t
	yy = [y]
	for i in range(n-2,-1,-1):
		y=ss[i] * y + tt[i]
		yy.insert(0,y)

	return yy

n = 3
h = 1 / n
tau = h**2 / 2
m = int(1 / tau)

print("tau + h^2: %f" % (tau + h**2))
print("\nЯвная схема:")
explicit = explicitSheme(m,n,h,tau)
print(explicit)

print("\nНеявная схема:")
nonExplicit = nonExplicitScheme(m,n,h,tau)
print(nonExplicit)

print("\nТочное решение:")
solution = getSolution(m,n,h,tau)
print(solution)

print("\n|u*(xi,tk) - uявн(xi,tk)|:")
print(abs(explicit-solution))

print("\nmax по i |u*(xi,tk) - uявн(xi,tk)|:")
maxs=[]
for k in range(m+1):
	maxs.append(max(abs(explicit[:,k] - solution[:,k])))
print(maxs)

print("\n|u*(xi,tk) - uнеявн(xi,tk)|:")
print(abs(nonExplicit-solution))

print("\nmax по i |u*(xi,tk) - uнеявн(xi,tk)|:")
maxs=[]
for k in range(m+1):
	maxs.append(max(abs(nonExplicit[:,k] - solution[:,k])))
print(maxs)
