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

#X Y beta et epsilon sont les données simulées tel que Y=X.beta+epsilon
Y,X,beta,epsilon=random.outcome(3000,4000,1,0.1*4000)
n=100

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
#ce graphique représente la vitesse de convergence en fonction du nombre d'itération de la fonction de coût de l'algorithme ISTA modifié

betahat_ista_modified,cost_ista_modified,loss_ista_modified=algo.ISTA(X,Y,beta,sparse='True')
nbr_it = list(range(1,len(cost_ista_modified)+1))
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(nbr_it,cost_ista_modified)
plt.title("valeur de la fonction de coût en fonction du nombre d'itérations avec l'algorithme ISTA modifié ")  # Titre du graphique
plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
plt.xlabel('nombre itérations')
plt.show()

#ce graphique représente la vitesse de convergence en fonction du nombre d'itération de la fonction de coût de l'algorithme ISTA 

betahat_ista,cost_ista,loss_ista=algo.ISTA(X,Y,beta)
nbr_it = list(range(1,len(cost_ista_modified)+1))
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(nbr_it,cost_ista)
plt.title("valeur de la fonction de coût en fonction du nombre d'itérations avec l'algorithme ISTA ")  # Titre du graphique
plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
plt.xlabel('nombre itérations')
plt.show()
#ce graphique représente la fonction loss  en fonction du nombre d'itération de la fonction de coût de l'algorithme ISTA 

plt.figure(figsize=(10, 5), dpi=100)
plt.plot(nbr_it,loss_ista_modified)
plt.title("valeur de la fonction de Loss en fonction du nombre d'itérations avec l'algorithme ISTA ")  # Titre du graphique
plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
plt.xlabel('nombre itérations')
plt.show()



# initialisation des paramètres pour l'algorithme IHT 
C=0.88
step=0.0001
max_iterations=3000
lamda=1.3
tol=1
# simulation de l'algorithme IHT modifié
betahat_iht_modified,cost_iht_modified,loss_iht_modified=algo.IHT(X, Y,beta ,C,step,max_iterations,lamda,sparse='True')
nbr_it_h = list(range(1,len(cost_iht_modified)+1))






#ce graphique représente la vitesse de convergence en fonction du nombre d'itération de la fonction de coût de l'algorithme IHT modifié
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(nbr_it_h,cost_iht_modified)
plt.title("valeur de la fonction de coût en fonction du nombre d'itérations avec l'algorithme IHT modifié")  # Titre du graphique)
plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
plt.xlabel('nombre itérations')
plt.show()

# représentation graphique de la fonction Loss de l'algorithme IHT modifié
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(nbr_it_h,loss_iht_modified)
plt.title("valeur de la fonction du loss en fonction du nombre d'itérations avec l'algorithme IHT modifié")  # Titre du graphique)
plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
plt.xlabel('nombre itérations')
plt.show()

# simulation de l'algorithme IHT 

betahat_iht,cost_iht,loss_iht=algo.IHT(X, Y,beta ,C,step,max_iterations,lamda)
nbr_it_h = list(range(1,len(cost_iht)+1))

#représentation de la fonction cost de l'algorithme IHT 
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(nbr_it_h,cost_iht)
plt.title("valeur de la fonction du cost en fonction du nombre d'itérations avec l'algorithme IHT" )  # Titre du graphique
plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
plt.xlabel('nombre itérations')
plt.show()

# représentation de la fonction Loss de l'algorithme IHT
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(nbr_it_h,loss_iht)
plt.title("valeur de la fonction du loss en fonction du nombre d'itérations avec l'algorithme IHT" )  # Titre du graphique
plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
plt.xlabel('nombre itérations')
plt.show()



#graphiques superposés des courbes de coût de l'algorithme IHT et ISTA 
#ici nous remarquons bien que l'algorithme IHT est plus performant que l'algorithme ISTA 

plt.figure(figsize=(10, 5), dpi=100)
plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
plt.xlabel('nombre itérations')
plt.title("valeur de la fonction de coût en fonction du nombre d'itérations pour l'algorithme ISTA et IHT")
plt.plot(nbr_it,cost_ista, color='blue', label='cost function ISTA')
plt.plot(nbr_it_h,cost_iht, color='red',label='cost fuction IHT')
plt.legend(loc='upper right')
plt.show()


#graphiques superposés des courbes de coût de l'algorithme IHT modifié  et ISTA modifié
#ici nous remarquons bien que dans ce cadre le IHT est plus précis 
plt.figure(figsize=(10, 5), dpi=100)
plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
plt.xlabel('nombre itérations')
plt.title("valeur de la fonction de coût en fonction du nombre d'itérations pour l'algorithme ISTA et IHT modifiés")
plt.plot(nbr_it,cost_ista_modified, color='blue', label='cost function ISTA')
plt.plot(nbr_it_h,cost_iht_modified, color='red',label='cost fuction IHT')
plt.legend(loc='upper right')
plt.show()

#graphiques superposés des courbes de coût de l'algorithme IHT modifié et normal,  ISTA modifié et normal 
#ici nous remarquons bien que dans ce cadre le IHT est plus précis et IHT modifié se rapproche de L'IHT normal


plt.figure(figsize=(10, 5), dpi=100)
plt.ylabel('valeur de la fonction de coût')  # Titre de l'axe y
plt.xlabel('nombre itérations')
plt.title("comparaison  de la fonction de coût en fonction du nombre d'itérations pour l'algorithme ISTA et IHT modifiés et normal")
plt.plot(nbr_it,cost_ista_modified, color='blue', label='cost function ISTA modifié')
plt.plot(nbr_it_h,cost_iht_modified, color='red',label='cost fuction IHT modifié')
plt.plot(nbr_it,cost_ista, color='g', label='cost function ISTA')
plt.plot(nbr_it_h,cost_iht, color='k',label='cost fuction IHT')

plt.legend(loc='upper right')
plt.show()



#maintenant on crée un tableau qui résume la distance euclidienne entre le vrai beta et ceux estimer par l'algorithme ISTA et FISTA 

distance=df_ista
distance=distance.drop('algorithme', axis=1)
distance = distance.rename(columns={'distance euclidienne': 'distance euclidienne ista'})
distance['distance euclidienne fista']=df_fista['distance euclidienne']
print(distance)




