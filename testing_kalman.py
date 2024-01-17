import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import parametres as par
import linsub as ls
import kalman as kalman


"""
Code permettant de tester le filtre de Kalman sur un signal gaussien bruité
"""
#Simulation d'une gaussiennes bruitées

input_folder = f'data/1/beats/1/1.npy'
bit_resolution = 12
max_range = 10 # mV
yscale_factor = max_range / (2**(bit_resolution+1))
pas = 10

#Paramètres des gaussiennes
param = par.parametres()

param.nombre_gaussiennes
param.amplitudes = [50, -10, 120, -20, 50]
param.centres = [-1.3, -0.4, 0, 0.2, 0.8]
param.ecarts_types = [0.2, 0.1, 0.1, 0.1, 0.2]

omega = 1
#Abscisse du tracé
val_gaussienne = np.load(input_folder)
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

niveau_bruit = np.var(val_gaussienne[0:50])
niveau_bruit = 5
P = niveau_bruit * np.eye(18) 
Q = niveau_bruit/10000 * np.ones((18, 18))
Q[0, 0] = niveau_bruit
Q[1, 1] = niveau_bruit/10
R = np.array([[niveau_bruit*50, 0], [0, 0]])


#application du filtre de kalman

#création de la donné initiale x0
X = np.array([0, -np.pi, omega])

x0 = kalman.formation(X, param)
#nombre de fois que l'on va appliquer le filtre de Kalman sur un même signal
nombre_periode = 40

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
#param.plot_pics(yscale_factor)
plt.plot(val_theta, yscale_factor *  y[0,:], label="signal bruité", alpha = 0.5)
plt.plot(val_theta, yscale_factor * param.signal_gaussiennes(n_point_theta), label="signal filtré")
plt.legend()
plt.grid()
plt.show()
