import numpy as np
from functions import randomWalk

# charge les fichiers csv
A = np.loadtxt('MatriceAdjacence.csv', delimiter=',')
#skiprows ignore la première ligbe du vecteur de personnalisation
v = np.loadtxt('VecteurPersonnalisation_Groupe7.csv', delimiter=',', skiprows=1) 

scores = randomWalk(A,0.9,v)

print("Scores PageRank (Random Walk):")
print(scores)