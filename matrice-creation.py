# Ce code DOIT être supprimé avant la soumission, il sert juste à créer ou re-créer la matrice d'adjacence

import numpy as np

# Définir l'ordre des noeuds
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
n = len(nodes)

# Créer une matrice 10x10 remplie de zéros
A = np.zeros((n, n))

A[0][1] = 5    # A → B : 5
A[0][7] = 3    # A → H : 3

A[1][0] = 5    # B → A : 5
A[1][2] = 1    # B → C : 1
A[1][8] = 2    # B → I : 2

A[2][3] = 2    # C → D : 2
A[2][8] = 5    # C → I : 5
A[2][9] = 3    # C → J : 3

A[3][2] = 3    # D → E : 2
A[3][9] = 3    # D → J : 3

A[4][3] = 5    # E → D : 5
A[4][5] = 4    # E → F : 4

A[5][4] = 2    # F → E : 2
A[5][6] = 5    # F → G : 5

A[6][5] = 2    # G → F : 2
A[6][8] = 3    # G → I : 3

A[7][6] = 2    # H → G : 2

A[8][0] = 1    # I → A : 1
A[8][1] = 4    # I → B : 4
A[8][7] = 4    # I → H : 4
A[8][9] = 4    # I → J : 4

A[9][4] = 4    # J → E : 4
A[9][5] = 1    # J → F : 1
A[9][8] = 2    # J → I : 2


print(A)

# Sauvegarder dans un fichier CSV
np.savetxt('MatriceAdjacence.csv', A, delimiter=',', fmt='%g')