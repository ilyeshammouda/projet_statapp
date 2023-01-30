from classes import random
from classes import algo
from math import log
import matplotlib.pyplot as plt

X=random.matrix_normal(1000,2000)
beta=random.beta(1,0.1*2000,2000)
epsilon=random.vect_normal(1000)
Y=X@beta+epsilon

betahat_1,nbr_it,cost_1=algo.ista(X,Y,1000,4*((log(2000)**(1/2))))
betahat_2,nbr_it_2,cost_2=algo.ista(X,Y,1000,0.01)
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.plot(cost_1, color="red")
plt.plot(cost_2,color="blue")
plt.show()
