from classes import random
from classes import algo
from math import log
import matplotlib.pyplot as plt
import numpy




Y,X,beta,epsilon=random.outcome(1000,2000,1,0.1*2000)
p=Y.shape

n=100
alpha= numpy.linspace(0,0.1,100)
betahat_1,nbr_it,cost_1=algo.ista(X,Y,1000,4*((log(2000)**(1/2))))
for k in alpha: 
  betahat_1,nbr_it_1,cost_1=algo.ista(X,Y,n,k)
  nbr_it = list(range(1,n+1))
plt.title("valeur de la fonction de coût en fonction du nombre d'itérations avec l'algorithme ISTA")  # Titre du graphique
plt.ylabel('nombre itérations')  # Titre de l'axe y
plt.xlabel('valeur de la fonction de coût')
plt.plot(nbr_it,cost_1)
#ce graphique représente la vitesse de convergence en fonction du nombre d'itération de la fonction de coût de l'algorithme ISTA



#voici le même graphique appliqué à l'algorithme FISTA
for k in alpha: 
  betahat_2,nbr_it_2,cost_2=algo.fista(X,Y,n,k)
  nbr_it2 = list(range(1,n+1))

plt.title("valeur de la fonction de coût en fonction du nombre d'itérations avec l'algorithme FISTA")
plt.ylabel('nombre itérations')  # Titre de l'axe y
plt.xlabel('valeur de la fonction de coût')
plt.plot(nbr_it2,cost_1,color='red')


#tentons le calcul du coefficient de Pearson
x=cost_1
y=cost_2
from scipy.stats import pearsonr
coeff_pearson,_ = pearsonr(x,y)
print("le pourcentage de corrélation entre .... : {}".format(coeff_pearson))

#97% de pourcentage de corrélation c'est énorme haha, cela signifie que les deux fonctions de coût de l'algorithme ISTA et FISTA sont quasiment équivalentes
#le calcul de la pente nous permet de montrer que la valeur ajoutée de FISTA est sur la vitesse de convergence

#on peut faire un t-test aussi pour vérifier si ces 2 séries temporelles ont de fortes différences
#à valider avec Ilyes et Aziz

#calculons le slope des séries temporelles

slopeISTA = np.polyfit(nbr_it,cost_1,1)[0]
print('la pente de la fonction de coût générée par ISTA vaut', slopeISTA)


slopeFISTA = np.polyfit(nbr_it,cost_2,1)[0]
print( 'la pente de la fonction de coût générée par ISTA vaut', slopeFISTA)

#donc l'algorithme FISTA converge bien plus rapidement que l'algorithme FISTA puisque sa pente est plus grande 