'''
Ce code définie les classes qui vont être utilisées par la suite dans le projet 
'''


#la Classe random contient les fonctions qui simulent les variables aléatoires qui seront utilisée pour étudier les données simulés.
import numpy as np
import pylops
from pylops.optimization.sparsity import ISTA
print("test")
class random:  
    def matrix_normal(n,p,mu=0,sigma=1):  # n est le nombre de lignes et p le nombre des colonnes, mu est la moyenne et sigma est l'écart type
        return (np.random.randn(n,p)*(sigma**2))+mu
    def vect_normal(n,mu=0,sigma=1):
        return (np.random.randn(n)*(sigma**2))+mu
    def beta(a,s,n): # s et a sont à préciser tel que s= 0,1*p et n> 2*s*log(p/2) pour commencer on peut utilisr a=1
        return a*(np.random.binomial(1,s/n , size=(n,)))
    def outcome(n,p,a,s,mu=0,sigma=1):
        return matrix_normal(n,p,mu,sigma) @ beta(a,s,p)+vect_normal(n,mu,sigma)


#la classe algo contient les algorithmes qui seront utilisés nottament ISTA et IHT
class algo:
    def ista(X,Y,n,alpha): #pour executer cet algorithme il faut installer la version 1.5 de pylops. Pour ceci on utilise la commande "pip install  pylops==1.5 "
        Op=pylops.MatrixMult(X)
        beta, niter, cost = pylops.optimization.sparsity.ISTA(Op, Y, n, eps=alpha, # n est le nombre maximal d'itération, le vecteur beta contient la solution du problème d'optimisation, et finalement cost represente l'historique de la fonction de coût
                                                        tol=0, returninfo=True) 
        return(beta,niter,cost)

    
    # Hard thresholding function
def SoftThreshold(X, lamda):
    return np.sign(x) * np.maximum(np.abs(x) - lamda, 0)

def SoftThreshol(X, lamda):
    return np.sign(x) * np.maximum(np.abs(x) - lamda, 0)

def IHT(X, Y, lamda, max_iterations=100, tol=1e-6):
    m, n = Y.shape
    x = np.zeros(n)
    v = X.copy()
    history = []
    for i in range(max_iterations):
        x_new = SoftThreshold(D.T @ v + x_new, lamda)
        v = X - Y @ x_new
        if np.linalg.norm(x_new - x) < tol:
            history.append(0.5 * np.linalg.norm(X - Y @ x_new)**2 + lamda * np.linalg.norm(x_new, ord=1))
            break
        z = x_new.copy()
    return x, history