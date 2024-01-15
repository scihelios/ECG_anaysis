import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from tqdm import tqdm
import time
import parametres as par
import linsub as ls

#Fonction F
def F(x):
    """
    x  =(X,param)
    X = (z, theta, omega)
    param = (alpha, b, m)
    """
    x = x.reshape((x.shape[0], 1))
    X, param = partitionnement(x)
    z = X[0]
    theta = X[1]
    omega = X[2]
    temp = param.fonction_gaussiennes()(0)
     
    n = len(x)
    val_F = np.zeros((n, 1))
    val_F[0, 0] = temp
    val_F[1, 0] = omega
    
    return val_F

def jacobien_diff_finies(x, h=1e-5):

    n = len(x)  # Nombre de variables
    J = np.zeros((n, n))  # Initialisation du jacobien

    for i in range(n):
        X_plus_h = x.copy()
        X_minus_h = x.copy()
        X_plus_h[i] += h
        X_minus_h[i] -= h

        # Calcul de la dérivée partielle avec différences finies centrées
        diff = (F(X_plus_h) - F(X_minus_h)) / (2 * h)
        J[:, i] = diff.reshape((n,))
        
    return J

def partitionnement(x):
    X = x[0:3,0]
    param = par.parametres()
    param.amplitudes = x[3:8,0]
    param.centres = x[8:13,0]
    param.ecarts_types = x[13:,0]
    param.nombre_gaussiennes = 5
    return X, param

def formation(X, param):
    x = np.zeros((X.shape[0] + param.nombre_gaussiennes * 3, 1))
    x[0:3,0] = X
    x[3:8,0] = np.array(param.amplitudes)
    x[8:13,0] = np.array(param.centres)
    x[13:,0] = np.array(param.ecarts_types)
    return x


def kalman(x0, y, P, Q, R, C, dt):
    """
    Paramètre:
    x0 : etat initial
    y : observation bruité
    P : matrice de covariance de l'état initial
    Q : matrice de covariance du bruit du processus
    R : matrice de covariance du bruit des mesures
    A : matrice de transition d'état (f)
    C : matrice de mesure (g)
    
    Retourne:
    Le signal filtré
    """
    x = np.zeros((x0.shape[0], y.shape[1]))
    x[:, 0] = x0.reshape((x.shape[0],))
    
    for k in range(y.shape[1]-1):
        #correction
        r = y[:,k] - (C.dot(x[:, k])) #ecart entre la mesure et le modèle théorique (xk moins)
        K = np.dot(P.dot(C.T), np.linalg.inv(np.dot(C.dot(P), C.T) + R)) #gain # +R?

        #etat du système
        x[:, k] += K.dot(r) #xk moins devient xk plus
        x[:, k+1] = (dt * F(x[:, k]) + x[:, k].reshape((x[:, k].shape[0], 1))).reshape((x[:, k].shape[0],))
        
        #mise à jour de P
        P = P - np.dot(K, C.dot(P)) #calcul de Pk_plus
        Ak = dt * jacobien_diff_finies(x[:, k]) + np.eye(x[:, k].shape[0])
        P = np.dot(Ak.dot(P), Ak.T) + Q #calcul de Pk_plus_un_moins
    
    return x

