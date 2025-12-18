import numpy as np
from functions import randomWalk, pageRankPower, pageRankLinear

# charge les fichiers csv
A = np.loadtxt('MatriceAdjacence.csv', delimiter=',')
#skiprows ignore la première ligbe du vecteur de personnalisation
v = np.loadtxt('VecteurPersonnalisation_Groupe7.csv', delimiter=',', skiprows=1) 

scores1 = pageRankLinear(A, 0.9, v)
scores2 = pageRankPower(A, 0.9, v)
scores3 = randomWalk(A,0.9,v)

print("Scores PageRank (Power Method):")
print(scores2)

print("Scores PageRank (Linear Method):")
print(scores1)

print("Scores PageRank (Random Walk):")
print(scores3)
