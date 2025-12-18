import numpy as np

def pageRankLinear(A: np.matrix, alpha: 0.9, v: np.array) -> np.array:
    """
    Calcule le PageRank d'un graphe à partir de sa matrice d'adjacence A,
    avec facteur de téléportation alpha et vecteur de personnalisation v.
    
    Parameters:
        A (np.matrix): matrice d'adjacence (n x n)
        alpha (float): facteur de téléportation (0 < alpha < 1)
        v (np.array): vecteur de personnalisation (somme = 1)
    
    Returns:
        np.array: vecteur des scores PageRank
    """

    
    n = A.shape[0]
    
    # Step 1: build transition probability matrix P
    P = np.array(A, dtype=float)
    col_sums = P.sum(axis=0)
    
    # Step 2: normalize columns (stochastic matrix)
    for j in range(n):
        if col_sums[j] != 0:
            P[:, j] /= col_sums[j]
        else:
            P[:, j] = 1.0 / n  # handle dangling nodes
    
    # Step 3: solve linear system (I - alpha * P)x = (1 - alpha)v
    I = np.eye(n)
    b = (1 - alpha) * v
    x = np.linalg.solve(I - alpha * P, b)
    
    return x


def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
    """
    Cette fonction permet de calculer le pagerank par la fonction method (fonction n°2)
    """

    n = A.shape[0] # n est egal à la taille de la matrice carrée A


    P = np.zeros((n, n), dtype=float) # P est un matrice nxn avec uniquement des 0 elle servira de matrice de transition

    for j in range(n):
        colonnessommes = np.sum(A[:, j]) #colonnessommes sera la somme de chaque valeur de la colonne d'indice j de la matrice
        P[:, j] = A[:, j] / colonnessommes #ici on divise chaque valeur de colonnes par leur addition



    e = np.ones((n, 1)) #e = nx1 avec que des 1
    v = v.reshape((n, 1)) #v = nx1 --> des multiplications  plus tard
    G = alpha * P + (1 - alpha) * (e @ v.T) #Martice Google

    # Ici on veut créer le vecteur de départ x pour la power methofd
    x = np.sum(A, axis=0).astype(float)
    x = x / np.sum(x)
    x = x.reshape((n, 1))


    itérations_maximales= 1000 #j'ai mis cette limite d'itération pour  pas avoir de boucles infinies


    mini = 1e-15 #j'ai mis ça pour pouvoir arreter si lla  différence devient trop petite

    #on affiche les différentes matrices (demandé)
    print("Matrice d'adjacence A:\n", A)
    print("Matrice de transition P:\n", P)
    print("Matrice Google G:\n", G)

    for k in range(itérations_maximales):
        nouv_x = G.T @ x # nouv_x[k+1] = G^T · nouv_x[k]
        nouv_x = nouv_x / np.sum(nouv_x) #on noramlise

        if k < 3: #affiche les trois premières itérations de la power method (consignes)
            print(f"Iteration {k + 1}:\n", nouv_x.flatten())

        #arrete les itérations quand les valeurs du vecteur  page rank sont  stables 
        if np.linalg.norm(nouv_x - x, 1) < mini:
            break

        x = nouv_x #x devient le nouveau vecteur


    print("pagerank final:\n", nouv_x.flatten()) #affiche le dernier pagerank

    return nouv_x.flatten() #retourne le vecteur PageRank en un tableau en une dimention(1D)


def randomWalk(A, alpha, v) :
    """
    en simulant une marche aléatoire de longueur fixée à 1 000 000 pas sur le graphe dirigé 
    et pondéré, chaque étape on choisit le noeud suivant selon les probabilités de transition 
    définies par la matrice Google. Le score PageRank approximatif de chaque noeud est alors 
    obtenu en comptant la fréquence relative de visite de chaque page au cours de la marche
    aléatoire.
    """
    n = A.shape[0]
    
    # Matrice de probabilités de transition P
    col_sums=np.sum(A,axis=0) # somme des colonnes de A
    P=np.zeros((n,n)) # crée une matrice n*n remplies de 0

    for i in range(n) :
        P[:,i] = A[:,i]/col_sums[i] # je ne mets pas de condition comme A n'a pas de colonne nulle
    
    # Matrice Google
    G = alpha * P+(1-alpha) * np.outer(v,np.ones(n)) # pas de condition pour une colonne nulle car A n'est pas nulle
    
    # Compteur de visites
    count=np.zeros(n) # crée un vecteur de longueur n remplies de 0
    
    steps=1000000 # plus le nombre est grand plus le résultat est précis
    
    node=0
    count[node]+=1     
    
    # Marche aléatoire
    for j in range(steps) :
        transition_probs = G[:,node]
        next_node = np.random.choice(n,p=transition_probs) # choisit un index (0 à n-1) avec probabilités de transition_probs
        node=next_node
        count[node]+=1
    
    scores = count/steps
    
    return scores #retourne le vecteur des scores d'importance