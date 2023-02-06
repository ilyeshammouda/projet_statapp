from classes import random
from classes import algo
from math import log
import matplotlib.pyplot as plt
Y,X,beta,epsilon=random.outcome(1000,2000,1,0.1*2000)
n=Y.shape

#betahat_1,nbr_it,cost_1=algo.ista(X,Y,1000,4*((log(2000)**(1/2))))
betahat_1,nbr_it_1,cost_1=algo.ista(X,Y,1000,0.01)
betahat_2,nbr_it_2,cost_2=algo.fista(X,Y,1000,0.01)
betahat_IHT,cost=algo.IHT(Y,X,1000,0.01)

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.plot(cost_1,label=" IST " ,color="red")
plt.plot(cost_2,label="FIST a=0.01",color="blue")
plt.plot(cost,label="IHT a=0.01",color="green")
plt.show()
