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
    temp = param.fonction_gaussiennes()(theta)
     
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

def filtre_kalman(beat, P, Q, R, C, nombre_periode):
    nombre_points = len(beat)
    beat = ls.substract_linear(beat, 10)
    x_value = np.linspace(-np.pi, np.pi, nombre_points)
    dt = 2 * np.pi / nombre_points

    # Définition des paramètres initiaux
    param = par.parametres()
    param.nombre_gaussiennes = 5
    param.amplitudes = [8/120, -13/120, 105/120, -21/120, 69/120]
    param.centres = [-1.68, -1.37, -1.02, -0.73, 1.28]
    param.ecarts_types = [0.02, 0.04, 0.1, 0.05, 0.25]

    y = np.array([beat, x_value])

    # Mise en place de l'état initial
    x0 = np.zeros((18, 1))
    x0[0, 0] = beat[0]
    x0[1, 0] = x_value[0]
    x0[2, 0] = 1
    x0[3:8, 0] = np.array(param.amplitudes)
    x0[8:13, 0] = np.array(param.centres)
    x0[13:, 0] = np.array(param.ecarts_types)

    # Application du filtre de Kalman
    iteration_kalman = np.zeros((18,1))

    for i in range(nombre_periode):
        x_filtre = kalman(x0, y, P, Q, R, C, dt)
        X, param = partitionnement(x_filtre[:, -1].reshape((18, 1)))

        for j in range(5):
            param.centres[j] = (param.centres[j]) % (2 * np.pi) - np.pi

        X[0] = 0
        X[1] = -np.pi
        X[2] = 1
        x0 = formation(X, param)
        iteration_kalman = np.hstack((iteration_kalman, x_filtre))
    
    x = iteration_kalman[:, -1].reshape((18, 1))
    X, param = partitionnement(x)
    return param


