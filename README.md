# projet_statapp
## Le fichier classe
Ce fichier contient une redéfinition des fonctions classiques pour simuler des matrices et des vecteurs aléatoires. Jusqu'à présent le but de ce fichier est de donner une forme plus simple et plus intuitive aux variables qui vont être utilisées par la suite. On y trouve aussi une méthode pour simuler l'algorithme ISTA en utilisant la bibliothèque pylops. Pour utiliser la méthode proposée il faut bien installer la version 1.5 de pylops comme précisé dans le code. Finalement, la fonction ISTA simule l'algorithme ISTA et retourne beta: le vecteur solution du problème d'optimisation, niter: le nombre d'itérations et cost: l'historique de la fonction de coût.
À présent ce fichier contient une version qu'on a codée nous même de l'algorithme ISTA et IHT. On a ajouté à ces algorithmes l'option "sparse" qui permet de retourner des estimateurs spareses 
## Le fichier simulated data
Il contient tout le travail sur les données simulées ainsi que les performances des deux algorithmes.
## Notebook
Le notebook propose une visualisation plus simple des courbes de performance, il est le plus à jour . 

