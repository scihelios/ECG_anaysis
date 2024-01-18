import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import parametres as par
import linsub as ls
import kalman as kalman
import pandas as pd


"""
Code permettant de tester le filtre de Kalman sur un signal gaussien bruité
"""
#Simulation d'une gaussiennes bruitées

input_folder = f'data/1/beats/24/776.npy'
bit_resolution = 12
max_range = 10 # mV
yscale_factor = max_range / (2**(bit_resolution+1))
pas = 10

#Paramètres des gaussiennes
param = par.parametres()

param.nombre_gaussiennes
param.amplitudes = [8/120, -13/120, 105/120, -21/120, 69/120]
param.centres = [-1.68, -1.37, -1.02, -0.73, 1.28]
param.ecarts_types = [0.02, 0.04, 0.1, 0.05, 0.25]


#nombre de fois que l'on va appliquer le filtre de Kalman sur un même signal
nombre_periode = 20
omega = 1
nombre_points = 1000
#Abscisse du tracé
val_gaussienne = param.signal_gaussiennes(nombre_points)

val_gaussienne = ls.substract_linear(val_gaussienne, pas)
n_point_theta = len(val_gaussienne)
val_theta = np.linspace(-np.pi, np.pi, n_point_theta)
dt = 2 * np.pi / omega / (n_point_theta + 1)



#bruitage des données, creation de y

y = np.array([val_gaussienne, val_theta])

#Création des matrices de covariance

#création de la matrice de mesure
C = np.zeros((2, 18))
C[0, 0] = 1
C[1, 1] = 1

#matrice de covariance du bruit

niveau_bruit = np.sqrt(np.var(val_gaussienne[0:50]))
P = niveau_bruit * np.eye(18) 
Q = np.zeros((18, 18))

# Paramètres de la matrice de covariance du bruit du processus
# x
Q[0, 0] = niveau_bruit/50
# theta
Q[1, 1] = 0
# omega

tab_cov = np.array(pd.read_csv('data/1/cov.csv'))[:,1:]
tab_corr = np.array(pd.read_csv('data/1/corr.csv'))[:,1:]


sigma = np.zeros((18,1))
sigma[0, 0] = niveau_bruit
sigma[1, 0] = 0
sigma[2, 0] = 0
# Amplitudes
sigma[3, 0] = 0.1
sigma[4, 0] = 0.1
sigma[5, 0] = 0.1
sigma[6, 0] = 0.1
sigma[7, 0] = 0.1
# Centres
sigma[8, 0] = np.pi/20
sigma[9, 0] = np.pi/20
sigma[10, 0] = np.pi/20
sigma[11, 0] = np.pi/20
sigma[12, 0] = np.pi/20
# Ecart-types
sigma[13, 0] = np.pi/20
sigma[14, 0] = np.pi/20
sigma[15, 0] = np.pi/20
sigma[16, 0] = np.pi/20
sigma[17, 0] = np.pi/20


rho = np.zeros((18, 18))
rho[0, 0] = 1
rho[1, 1] = 1
rho[2, 2] = 1

for i in range(10):
    rho[0, i+3] = 0.5
    rho[i+3, 0] = 0.5

for i in range(10):
    for j in range(15):
        rho[i+3, j+3] = tab_corr[i, j]



Q = sigma.T @ rho
Q = Q @ sigma


print(Q)

R = np.array([[niveau_bruit*10, 0], [0, niveau_bruit]])


#application du filtre de kalman

#création de la donné initiale x0
X = np.array([0, -np.pi, omega])

x0 = kalman.formation(X, param)


#on conserve toutes les iterations du filtre de kalman
iteration_kalman = np.zeros((18, 1))

for i in tqdm(range(nombre_periode)):
        
    #application du filtre

    x_filtre = kalman.kalman(x0, y, P, Q, R, C, dt)

    #on met à jour les paramètres initiaux
    X, param = kalman.partitionnement(x_filtre[:,-1].reshape((18, 1)))

    for j in range(5):
        param.centres[j] = param.centres[j] % (2*np.pi) - np.pi

    X[0] = 0
    X[1] = -np.pi
    X[2] = omega
    x0 = kalman.formation(X, param)
    
    #on converse l'iteration de kalman dans le tableau iteration_kalman
    iteration_kalman = np.hstack((iteration_kalman, x_filtre))
    

    
iteration_kalman = iteration_kalman[:, 1:]

X, param = kalman.partitionnement(iteration_kalman[:, -1].reshape((18, 1)))

print(param)
yscale_factor = 1
param.plot_pics(yscale_factor,xscale_factor=1)
plt.plot(val_theta, yscale_factor *  y[0,:], label="signal bruité", alpha = 0.5)
plt.plot(val_theta, yscale_factor * param.signal_gaussiennes(n_point_theta), label="signal filtré")
plt.legend()
plt.grid()
plt.show()
