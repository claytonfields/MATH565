#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:40:47 2020

@author: claytonfields
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


##Problem 1 

#Find square root of real number a>0
def sqroot_finder(a=5, x_0=5, kmax=20, tol=1e-12):
    xold = x_0
    xnew=0
    k=0
    while (abs(xnew-xold))>tol or k<=kmax:
        xnew = .5*(xold+a/xold)
        xold = xnew
        k += 1
    return xnew
   
#find squre root of a   
a = 5
print("The approximated solution of sqrt(%f) is " %a, sqroot_finder(a))        
#numpy solustion is:
print("The solution of sqrt(%f) given by np.sqrt() is " %a, np.sqrt(5))
#the error vs np.sqrt() is:
np.sqrt(5) - sqroot_finder()
print("the error of approx vs np.sqrt() is ", np.sqrt(5) - sqroot_finder())
print("How about that!")

#Trapezoidal rule
def trapz(f,a,b,n=10):
    h = (b-a)/n
    x = np.array([a + i*h for i in range(1,n)])
    sumz = np.sum(f(x))
    return (1/(2*n))*(f(a)+2*sumz+f(b))

def f(x):
    return 1/(1+x**2)


I_trap = trapz(f, 0, 1 ,n=10)
I = quad(f,0,1)[0]
errorvquad = I_trap - I
print(errorvquad)

I_trap = trapz(f, 0, 1, n=20)
errorvquad = I_trap - I
print(errorvquad)

#Polynomial
# def polyval(coef,x):
#     n = len(list(coef))
#     ##coef = coef[::-1]
#     P_x = coef[1]+x
#     x_n = np.copy(x)
#     for i in range(2,n):
#         P_x *= x
#         P_x += coef[i]
#     P_x += coef[0]
#     return P_x

def polyval(coef,x):
    n = len(list(coef))
    P_x = coef[0]*x+coef[1]
    for i in range(2,n):
        P_x *= x
        P_x += coef[i]
    return P_x




coef = [3.1, np.pi, -1, 0, 4.7, 4]
polyval(coef,1)

domain = np.linspace(-2,2,100)
plt.plot(domain, polyval(coef, domain))
    