import numpy as np

def randomWalk(A, alpha, v) :
    n = A.shape[0]
    
    # Matrice de probabilités de transition P
    col_sums=np.sum(A,axis=0) # somme des colonnes de A
    P=np.zeros((n,n)) # crée une matrice n*n remplies de 0

    for i in range(n) :
        P[:,i] = A[:,i]/col_sums[i] # je ne mets pas de condition comme A n'a pas de colonne nulle
    
    # Matrice Google
    G = alpha * P+(1-alpha) * np.outer(v,np.ones(n)) # produit extérieur de 2 vecteurs
    
    # Compteur de visites
    count=np.zeros(n) # crée un vecteur de longueur n remplies de 0
    
    steps=10000 # plus le nombre est grand plus le résultat est précis
    
    node=0
    count[node]+=1     
    
    # Marche aléatoire
    for j in range(steps) :
        transition_probs = G[:,node]
        next_node = np.random.choice(n,p=transition_probs) # choisit un index (0 à n-1) avec probabilités de transition_probs
        node=next_node
        count[node]+=1
    
    scores = count/steps
    
    return scores