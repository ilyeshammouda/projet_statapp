from classes import random
from classes import algo
from math import log
import matplotlib.pyplot as plt
import numpy
from scipy.stats import pearsonr
import pandas as pd
'''
 Dans cette partie on peut changer les valeurs pour lesquelles on souhaite exécuter le code. Les paramètres: n pour
le nombres d'itération qu'on souhaite fixer pour l'execution des différents algorithmes les paramètres pour définir 
la taille et les caractéristiques des vecteurs qui représentent les données simulées p et s représentent la taille 
de la matrice X qui simule un vecteur normal de paramètres E[X]=1 et V[X]=0.Quant à s et a sont les paramètres 
qu'il faut préciser pour la simulation de la variable $\beta$ qui simule un vecteur aléatoire de lois binomiale. 
s et a sont à préciser tel que s= 0,1*p et p> 2*s*log(n/2). $\alpha$ est un vecteur qui contient les différentes 
valeurs de alpha qu'on va utiliser pour exécuter les différents algorithmes.  
'''

n=100
alpha= numpy.linspace(0,0.1,100)
k=1000
p=2000
a=1
s=0.1*2000
alpha= numpy.linspace(0,0.1,100)

#X Y beta et epsilon sont les données simulées tel que Y=X.beta+epsilon
Y,X,beta,epsilon=random.outcome(1000,2000,1,0.1*2000)
p=Y.shape

# on crée une data frame pour la remplire par la suite par les différentes valeurs de la distance eucliedienne entre 
#l'estimateur calculer par les deux algorithmes et le vrai beta 

columns = ['alpha', 'distance euclidienne', 'algorithme']
df_ista = pd.DataFrame(columns=columns)
df_fista = pd.DataFrame(columns=columns)




'''
Dans cette partie on essaye de représenter la fonction coût pour 100 valeurs différentes d'alpha. 
Dans un premier temps on fait la représentation pour l'algorithme ISTA et FISTA d'un coté.
Dans un second temps on fait la repésentation des 2 courbes superposées.
Cette visualisation permet de suggérer que l'algorithme FISTA converge plus rapidement. 
La section suivante vise à mener une étude statistique sur ces deux fonction de coût

'''
#ces graphique représentent la vitesse de convergence en fonction du nombre d'itération de la fonction de coût de l'algorithme ISTA pour différentes valeurs d'alpha

for k in alpha: 
  betahat_ista,nbr_it_ista,cost_ista=algo.ista(X,Y,n,k)
  nbr_it = list(range(1,n+1))
  distance_eucliedienne=numpy.linalg.norm(beta -betahat_ista)
  values = [k,distance_eucliedienne,'ista']
  df_ista.loc[k] = values
  plt.figure(figsize=(10, 5), dpi=100)
  plt.plot(nbr_it,cost_ista)
  plt.title("valeur de la fonction de coût en fonction du nombre d'itérations avec l'algorithme ISTA pour alpha égale à  {}".format(k))  # Titre du graphique
  plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
  plt.xlabel('nombre itérations')
  plt.show()





#ces graphiques représentent la vitesse de convergence en fonction du nombre d'itération de la fonction de coût de l'algorithme IHT pour différentes valeurs d'alpha

for k in alpha: 
  beta_IHT,cost_IHT=algo.IHT(Y,X,n,k)
  plt.figure(figsize=(10, 5), dpi=100)
  nbr_it2 = list(range(1,n+1))
  plt.figure(figsize=(10, 5), dpi=100)
  plt.title("valeur de la fonction de coût en fonction du nombre d'itérations avec l'algorithme IHT pour alpha égale à  {}".format(k))
  plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
  plt.xlabel('nombre itérations')
  plt.plot(nbr_it2,cost_IHT,color='red')
  plt.legend(loc='lower left')
  plt.show()


#ces graphiques représentent la vitesse de convergence en fonction du nombre d'itération de la fonction de coût de l'algorithme FISTA pour différentes valeurs d'alpha
for k in alpha: 
  betahat_fista,nbr_it_fista,cost_fista=algo.fista(X,Y,n,k)
  distance_eucliedienne=numpy.linalg.norm(beta -betahat_fista)
  values = [k,distance_eucliedienne,'fista']
  df_fista.loc[k] = values
  plt.figure(figsize=(10, 5), dpi=100)
  nbr_it2 = list(range(1,n+1))
  plt.figure(figsize=(10, 5), dpi=100)
  plt.title("valeur de la fonction de coût en fonction du nombre d'itérations avec l'algorithme FISTA pour alpha égale à  {}".format(k))
  plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
  plt.xlabel('nombre itérations')
  plt.plot(nbr_it2,cost_fista,color='red')
  plt.show()

#graphiques superposés des courbes de coût de l'algorithme ISTA et FISTA 
#ici nous remarquons bien que l'algorithme FISTA est plus performant que l'algorithme ISTA 
#la fonction de coût décroît plus rapidement pour l'algorithme FISTA que pour l'algorithme ISTA
for k in alpha: 
  betahat_ista,nbr_it_ista,cost_ista=algo.ista(X,Y,n,k)
  betahat_fista,nbr_it_fista,cost_fista=algo.fista(X,Y,n,k)
  nbr_it = list(range(1,n+1))
  plt.figure(figsize=(10, 5), dpi=100)
  plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
  plt.xlabel('nombre itérations')
  plt.title("valeur de la fonction de coût en fonction du nombre d'itérations pour l'algorithme ISTA et FISTA pour alpha égale à  {}".format(k))
  plt.plot(nbr_it,cost_ista, color='blue',label="fonction coût de l'lagorithme ISTA")
  plt.plot(nbr_it,cost_fista, color='red',label="fonction coût de l'lagorithme FISTA")
  plt.legend(loc='lower left')
  plt.show()


#maintenant on crée un tableau qui résume la distance euclidienne entre le vrai beta et ceux estimer par l'algorithme ISTA et FISTA 

distance=df_ista
distance=distance.drop('algorithme', axis=1)
distance = distance.rename(columns={'distance euclidienne': 'distance euclidienne ista'})
distance['distance euclidienne fista']=df_fista['distance euclidienne']
print(distance)

# Étude statistique des résultats
'''
dans cette partie on essaye de calculer le coeficient de correlation entre la fonction de coût de l'algorithme ISTA et FISTA,
ainsi que la pente des ces deux fonctions afin de montrer que la convergence de FISTA est plus rapide que celle 
de l'algorithme ISTA
'''
#tentons le calcul du coefficient de Pearson
x=cost_ista
y=cost_fista
coeff_pearson,_ = pearsonr(x,y)
print("le pourcentage de corrélation entre .... : {}".format(coeff_pearson))

#97% de pourcentage de corrélation , cela signifie que les deux fonctions de coût de l'algorithme ISTA et FISTA sont quasiment équivalentes
#le calcul de la pente nous permet de montrer que la valeur ajoutée de FISTA est sur la vitesse de convergence
#je ne suis pas bien sûr de ce qu'on peut en faire mais je vais continuer à creuser

#on peut faire un t-test aussi pour vérifier si ces 2 séries temporelles ont de fortes différences
#à valider avec Ilyes et Aziz

#calculons le slope des séries temporelles

slopeISTA = numpy.polyfit(nbr_it,cost_ista,1)[0]
print('la pente de la fonction de coût générée par ISTA vaut', slopeISTA)


slopeFISTA = numpy.polyfit(nbr_it,cost_fista,1)[0]
print( 'la pente de la fonction de coût générée par FISTA vaut', slopeFISTA)

#donc l'algorithme FISTA converge bien plus rapidement que l'algorithme IISTA puisque sa pente est plus grande