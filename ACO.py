#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 05:33:56 2018
@author: aditya
"""

import numpy as np
import random

def prob(e,t):
    e=np.transpose(e)
    t=np.transpose(t)
    a=2
    b=1
    num=np.multiply(np.power(t[1],a),np.power(e[1],b))
    denom=np.add(np.multiply(np.power(t[1],a),np.power(e[1],b)),np.multiply(np.power(t[0],a),np.power(e[0],b)))
    p=np.divide(num,denom)
    return np.transpose(p)
    
def weight(g,g1):
    sumarr=np.sum(matMul(g,g1), axis=1)
    return np.min(sumarr)

def matMul(X,Y):
    result = np.zeros((X.shape[0],Y.shape[1]),dtype=np.int64)
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                result[i][j] ^= (X[i][k] * Y[k][j])
                
    return result
            

k=57
n=127
t=np.random.normal(0.5,0.2,(k,2))
e=np.ones((k,2),dtype=np.float64)
tmax=100
m=50
q=10
roh=0.9
g_ini=np.loadtxt("/home/aditya/Desktop/Generator-Matrix-input/bch127_57_23.txt",dtype=np.int32,delimiter='\t')
start_mat=np.identity(k,dtype=np.int64)
G = [start_mat] * m
best=128
bestg=np.zeros((k,n))

for step in range(0,tmax):
    delta_t = np.zeros((k,2))
    s=np.zeros(k)
    for ants in range(0,m):
        g=G[ants]
        dt=np.zeros((k,2))
        alpha=np.zeros(k)
        p=prob(e,t)
        for i in range(0,k):
            curr_weight=weight(g,g_ini)
            if(p[i]>random.uniform(0,1)):
                swap=random.randint(0,k-1)
                while(swap == i and swap>=50):
                    swap=random.randint(0,k-1)
                g[:,i]=np.bitwise_xor(g[:,i],g[:,swap])
                dt[i][1]=q/curr_weight
                alpha[i]=1
            else:
                dt[i][0]=q/curr_weight
                #maybe alpha needs to changed
            if(curr_weight<best):
                best=curr_weight
                bestg=g
        print(best)
                
        s=np.add(s,alpha)
        delta_t=np.add(delta_t, dt)
    t=np.add(np.multiply(roh,t),delta_t)
    e[:,0]=np.subtract(np.add(np.full(k,m),np.ones(k)),s)
    e[:,1]=np.add(s,np.ones(k))    
print(best)
print(bestg)
    
          
                   