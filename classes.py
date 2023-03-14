'''
Ce code définit les classes qui vont être utilisées par la suite dans le projet 
'''



import numpy as np
import pylops
from pylops.optimization.sparsity import ISTA
from pylops.optimization.sparsity import FISTA
import pandas as pd
import time
'''
Ce code définit les classes qui vont être utilisées par la suite dans le projet 
'''


#la Classe random contient les fonctions qui simulent les variables aléatoires qui seront utilisée pour étudier les données simulés.

print("test")
class random:  
    def matrix_normal(n,p,mu=0,sigma=1):  # n est le nombre de lignes et p le nombre des colonnes, mu est la moyenne et sigma est l'écart type
        return (np.random.randn(n,p)*(sigma**2))+mu
    def vect_normal(n,mu=0,sigma=1):
        return (np.random.randn(n)*(sigma**2))+mu
    def beta(a,s,n): # s et a sont à préciser tel que s= 0,1*p et n> 2*s*log(p/2) pour commencer on peut utilisr a=1
        return a*(np.random.binomial(1,s/n , size=(n,)))
    def outcome(n,p,a,s,mu=0,sigma=1):
        X=random.matrix_normal(n,p,mu,sigma)
        beta=random.beta(a,s,p)
        epsilon=random.vect_normal(n,mu,sigma)
        Y=X @ beta+epsilon
        return Y,X,beta,epsilon


#la classe algo contient les algorithmes qui seront utilisés nottament ISTA et IHT
class algo:


    def HardThreshold(x,lamda):
        return x*(np.abs(x)>=lamda)
    def SoftThreshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


    def IHT(X, Y,beta=np.zeros(1) ,C=0.9,step=0.0001,max_iterations=3000,lamda=0.1, tol=1e-6,sparse='False'):
        n,m=X.shape
        Z,Beta=np.zeros(m),np.ones(m)
        loss=[]
        cost=[]
        check_vect=np.zeros(m)
        test=np.zeros(1)
        if np.array_equal(beta, test, equal_nan=False):
          beta=Beta.copy()
          print('We are in the unknow beta case,the cost function is not significant')
        start_time = time.time()        
        for i in range(max_iterations):
            Z=Beta+(step*(X.T)@(Y-X@Beta))
            Beta=algo.HardThreshold(Z, lamda)
            if sparse=='True':
                Beta= np.where(np.isclose(Beta, 1, atol=0.8), 1, 0)
            cost.append(np.linalg.norm(-beta+Beta))
            lamda*=C
            loss.append(Beta[-1]-Beta[-2])
            if np.linalg.norm(Beta -check_vect ) < tol:
                break
        end_time = time.time()
        time_taken = end_time - start_time
        print("IHT execution time :", time_taken, "seconds")

        return Beta,cost,loss

    def ISTA(X, Y,beta=np.zeros(1) ,step=0.0001,max_iterations=3000,lamda=0.01, tol=1e-6,sparse='False'):
        n,m=X.shape
        Z,Beta=np.zeros(m),np.ones(m)
        check_vect=np.zeros(m)
        test=np.zeros(1)
        cost=[]
        loss=[]
        if np.array_equal(beta, test, equal_nan=False):
          beta=Beta.copy()
          print('We are in the unknow beta case,the cost function is not significant')
        start_time = time.time()
        for i in range(max_iterations):
            Z=Beta+(step*(X.T)@(Y-X@Beta))
            Beta=algo.SoftThreshold(Z, lamda)
            if sparse=='True':
              Beta= np.where(np.isclose(Beta, 1, atol=0.8), 1, 0)
            cost.append(np.linalg.norm(-beta+Beta))
            loss.append(Beta[-1]-Beta[-2])
            if np.linalg.norm(Beta -check_vect ) < tol:
                break
        end_time = time.time()
        time_taken = end_time - start_time
        print("ISTA execution time :", time_taken, "seconds")
        return Beta,cost,loss