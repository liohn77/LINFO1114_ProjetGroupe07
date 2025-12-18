import numpy as np
import matplotlib.pyplot as plt
from functions import randomWalk, pageRankPower, pageRankLinear

# charge les fichiers csv
A = np.loadtxt('MatriceAdjacence.csv', delimiter=',')
#skiprows ignore la première ligbe du vecteur de personnalisation
v = np.loadtxt('VecteurPersonnalisation_Groupe7.csv', delimiter=',', skiprows=1) 

score_exact = np.array([0.06613036, 0.09504818, 0.11723597, 0.13065298, 0.14513379, 0.10801892,
 0.07340477, 0.02547551, 0.11234878, 0.12655074])

scores1 = pageRankLinear(A, 0.9, v)
scores2 = pageRankPower(A, 0.9, v)
scores3 = randomWalk(A,0.9,v,score_exact)

print("Scores PageRank (Linear Method):")
print(scores1)

print("Scores PageRank (Power Method):")
print(scores2)

print("Scores PageRank (Random Walk):")
print(scores3)
