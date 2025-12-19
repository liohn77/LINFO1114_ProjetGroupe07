import numpy as np
import matplotlib.pyplot as plt

def pageRankLinear(A: np.matrix, alpha: float, v: np.array) -> np.array:
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

    
   # Taille de la matrice carrée A
    n = A.shape[0]


    # Etape 1 : Construction de la matrice de transition P

    P = np.array(A, dtype=float)   # Copie de la matrice d'adjacence A
    col_sums = P.sum(axis=0)       # Somme de chaque colonne de P


    # etape 2 : Normalisation des colonnes de P (matrice scohastique)

    for j in range(n):
        if col_sums[j] != 0:
            # Si le noeud a des liens sortants on divise chaque valeur par la somme des liens
            P[:, j] /= col_sums[j]
        else:
            # si le noeud n'a pas de liens sortants on fait 1/n

            P[:, j] = 1.0 / n


    # etape 3 : Résolution du système linéaire pour obtenir le vecteur PageRank

    if alpha < 1:
        # cas général avec téléportation (0 < alpha < 1)
        # Résolution de (I - alpha * P) x = (1 - alpha) * v

        I = np.eye(n)                # matrice identité de taille n
        b = (1 - alpha) * v          # Côté droit du système linéaire
        x = np.linalg.solve(I - alpha * P, b)  # résolution du système linéaire
        # retourne le vecteur PageRank

    else:
        # cas particulier sans téléportation (alpha = 1) ((I - P)Tx = 0 avec eTx = 1)
        # résolution de (I - P)x = 0 avec la contrainte sum(x) = 1

        M = np.eye(n) - P            # matrice (I - P)
        b = np.zeros(n)              # côté droit initialisé à zéro

        # remplacer la dernière ligne pour imposer la contrainte sum(x) = 1
        # correspond à eTx = 1
        M[-1, :] = 1.0
        b[-1] = 1.0

        # résolution du système linéaire modifié
        # On utilise np.linalg.solve pour résoudre le système linéaire
        x = np.linalg.solve(M, b)

    # retourne le vecteur PageRank
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
    G = alpha * P + (1 - alpha) * (v @ e.T) #Martice Google

    # Ici on veut créer le vecteur de départ x pour la power methofd
    x = np.sum(A, axis=0).astype(float)
    x = x / np.sum(x)
    x = x.reshape((n, 1))


    iterations_maximales= 500000000 #j'ai mis cette limite d'itération pour  pas avoir de boucles infinies


    mini = 1e-12 #j'ai mis ça pour pouvoir arreter si lla  différence devient trop petite

    #on affiche les différentes matrices (demandé)
    print("Matrice d'adjacence A:\n", A)
    print("Matrice de transition P:\n", P)
    print("Matrice Google G:\n", G)

    for k in range(iterations_maximales):
        nouv_x = G @ x # nouv_x[k+1] = G · nouv_x[k]
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
    en simulant une marche aléatoire de longueur fixée à 5 000 000 pas sur le graphe dirigé 
    et pondéré, chaque étape on choisit le noeud suivant selon les probabilités de transition 
    définies par la matrice Google. Le score PageRank approximatif de chaque noeud est alors 
    obtenu en comptant la fréquence relative de visite de chaque page au cours de la marche
    aléatoire.
    """
    n = A.shape[0]

    # Vecteur des scores exacts (pour alpha=0.9) pour le calcul de l'erreur moyenne
    score_exact = np.array([0.06613036,0.09504818,0.11723597,0.13065298,0.14513379,0.10801892,0.07340477,0.02547551,0.11234878,0.12655074])

    
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

    # Checkpoints pour calculer l'erreur
    checkpoints = []
    for k in range(20) :
        checkpoints.append(k*steps//20)
    checkpoints.append(steps)
    steps_list = []
    errors_list = []
        
    node=0
    count[node]+=1     
    
    # Marche aléatoire
    for j in range(steps) :
        transition_probs = G[:,node]
        next_node = np.random.choice(n,p=transition_probs) # choisit un index (0 à n-1) avec probabilités de transition_probs
        node=next_node
        count[node]+=1
        
        # Calculer erreur moyenne à chaque checkpoint
        if (j+1) in checkpoints:
            score_approx = count / (j+1)
            erreur_moyenne = (1/n) * np.sum(np.abs(score_approx-score_exact))
            steps_list.append(j+1)
            errors_list.append(erreur_moyenne)
    
    scores = count/steps
    
    # Trace le graphique pour l'évolution de l'erreur moyenne
    plt.figure(figsize=(12, 6))
    plt.loglog(steps_list, errors_list, 'o-', linewidth=2.5, markersize=8, color='#e74c3c', label='ε(k)')
    plt.xlabel('Nombre de pas (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Erreur moyenne ε(k)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('pagerank_error_convergence.png', dpi=300, bbox_inches='tight') #sauvegarde le graphique
    
    return scores #retourne le vecteur des scores d'importance