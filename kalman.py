# FICHIER AVEC TOUTES LES FONCTIONS RELATIVES AUX FILTRES DE KALMAN

import numpy as np
import matplotlib.pyplot as plt
import os
import parametres as param

#Donnée globale
omega = 1
n_point_theta = 800
dt = 2 * np.pi / omega * 1 / (n_point_theta - 1)

# Fonction Kalman appliquée à un signal simulé
def kalman_signal(signal, N_ite=1, dt=dt):
    """
    fonction qui à un signal applique le filtre de Kalman sur N itérations
    Retourne:
    param_kalman, iteration_Kalman
    """


    #Création des matrices de covariance

    #matrice de mesure
    C = creat_C()

    #matrice de covariance initiale
    P = creat_P(signal)

    #matrice de bruit de mesure
    R = creat_R()

    #matrice de covariance du model
    Q = creat_Q()

    #création de la donnée initiale x0 et de y
    x0 = creat_x0()
    y = np.array([signal])

    #application du filtre de Kalman
    param_kalman, iteration_Kalman, P_ite = kalman_N_periode(x0, y, P, Q, R, C, dt, conserve_ite=True, nombre_periode=N_ite)

    return param_kalman, iteration_Kalman, P_ite

def kalman_interface(signal, N_ite=1):
    liste_param, _, _ = kalman_signal(signal, N_ite)

    parametres = param.parametres()
    parametres.amplitudes = liste_param[0:5]
    parametres.ecarts_types = np.exp(liste_param[5:10])
    parametres.centres = 2*np.arctan(liste_param[10:15])
    return parametres


####################### Fonction de base ################################

#Application d'une itération du filtre de Kalman sur 1 signal
def kalman(x0, y, P, Q, R, C, dt):
    """
    Paramètre:
    x0 : etat initial
    y : observation bruité
    P : matrice de covariance de l'état initial
    Q : matrice de covariance du bruit du modèle
    R : matrice de covariance du bruit des mesures
    A : matrice de transition d'état 
    C : matrice de mesure 
    
    Retourne:
    Le signal filtré
    """
    x = np.zeros((x0.shape[0], y.shape[1]))
    x[:, 0] = x0.reshape((x.shape[0],))

    P_ite = np.zeros((x0.shape[0], y.shape[1]))
    P_ite[:, 0] = np.diag(P).reshape((x.shape[0],))

    for k in range(y.shape[1]-1):
        theta = k * dt - np.pi

        #correction
        r = y[:,k] - (C.dot(x[:, k])) #ecart entre la mesure et le modèle théorique (xk moins)
        K = np.dot(P.dot(C.T), np.linalg.inv(np.dot(C.dot(P), C.T) + R)) #gain 

        #etat du système
        x[:, k] += K.dot(r) #xk moins devient xk plus
        x[:, k+1] = (dt * F(x[:, k], theta) + x[:, k].reshape((x[:, k].shape[0], 1))).reshape((x[:, k].shape[0],))
        
        #mise à jour de P
        P = P - np.dot(K, C.dot(P)) #calcul de Pk_plus
        Ak = dt * jacobien_diff_finies(x[:, k], theta) + np.eye(x[:, k].shape[0])
        P = np.dot(Ak.dot(P), Ak.T) + Q #calcul de Pk_plus_un_moins
        P_ite[:, k+1] = np.diag(P).reshape((x.shape[0],))

    return x, P_ite

#applique N fois le filtre de kalman sur un signal, renvoit les paramètres déterminés
def kalman_N_periode(x0, y, P, Q, R, C, dt, nombre_periode = 20, conserve_ite = False):

    if conserve_ite == True:
        #on converse toute les iterations du filtre de kalman
        iteration_kalman = np.zeros((16, 1))
        P_ite_concat = np.zeros((16, 1))

    for i in range(nombre_periode):
            
        #application du filtre
        X_filtre, P_ite = kalman(x0, y, P, Q, R, C, dt)
        
        #on met à jour les paramètres initiaux
        new_para_init = recup_para(X_filtre)
        x0 = update_x0(new_para_init, y)

        if conserve_ite == True:
            #on converse l'iteration de kalman dans le tableau iteration_kalman
            iteration_kalman = np.hstack((iteration_kalman, X_filtre))
            P_ite_concat = np.hstack((P_ite_concat, P_ite))

    param_kalman = recup_para(X_filtre)

    if conserve_ite == True:
        iteration_kalman = iteration_kalman[:, 1:]
        P_ite_concat = P_ite_concat[:, 1:]
    
        return param_kalman, iteration_kalman, P_ite_concat
    
    return param_kalman

#Permet de tracer la gaussienne une fois les paramètres déterminés
def gaussienne(val_theta, param):
    """
    Paramètre:
    val_theta : liste des points où l'on doit trouver la valeur
    param = [tous les alphas, tous les b, tous les m]
    tab_alpha     : tableau des amplitudes
    tab_b         : tableau des ecarts type
    tab_m_tilde   : tableau des moyenne
    Retourne:
    val : La valeur de la gaussienne aux points val_theta 
    """
    val = []
    tab_alpha = param[:5]
    tab_sigma = param[5:10]
    tab_m = param[10: 15]
    
    #Boucle sur les points d'abscisse
    for theta in val_theta:
        
        temp = 0
        for i in range(5):
            
            delta_theta_i = (theta - tab_m[i])        
            temp += tab_alpha[i] * np.exp(-delta_theta_i ** 2 / (2 * tab_sigma[i] ** 2))
        
        #On ajoute la valeur de temp à val
        val.append(temp)
    
    return val


#Fonction F
def F(X, theta):
    """
    X = (z, tous les alphas, tous les b, tous les m_tilde)
    """
    tab_alpha = X[1:6]
    tab_b = X[6:11]
    tab_m_tilde = X[11:16]
    
    temp = 0
    for i in range(5):
        
        #on ramene l'angle entre -pi et pi
        delta_theta_i = (theta - 2*np.arctan(tab_m_tilde[i])) 
        
        temp += -tab_alpha[i] * omega * delta_theta_i / (np.exp(tab_b[i]) ** 2) * np.exp(-delta_theta_i ** 2 / (2 * np.exp(tab_b[i]) ** 2))
        
    val_F = np.zeros((16, 1))
    val_F[0, 0] = temp
    
    return val_F

def jacobien_diff_finies(X, theta, h=1e-5):

    n = len(X)  # Nombre de variables
    J = np.zeros((n, n))  # Initialisation du jacobien

    for i in range(n):
        X_plus_h = X.copy()
        X_minus_h = X.copy()
        X_plus_h[i] += h
        X_minus_h[i] -= h

        # Calcul de la dérivée partielle avec différences finies centrées
        diff = (F(X_plus_h, theta) - F(X_minus_h, theta)) / (2 * h)
        J[:, i] = diff.reshape((n,))
        
    return J

#On récupère les valeurs déterminées des coefficients
def recup_para(X_filtre, nb_point=5):
    """
    X_filtre : application du filtre de kalman 
    nb_point : nombre de point sur lequel on fait la moyenne des paramètres
    Retourne : une moyenne des alpha, b, m_tilde déterminés
    """
    return np.mean(X_filtre[1:, -nb_point:], axis=1)

def update_x0(new_para_init, y):
    
    x0 = np.zeros((16, 1))
    
    x0[0, 0] = y[0, 0]
    
    for i in range(15):
        x0[1+i, 0] = new_para_init[i]
        
    return x0


########################## Création des matrices de covariance #########################

#Matrice de covariance initiale
def creat_P(signal):

    # Determination du bruit du signal
    max_amplitude = max(abs(val) for val in signal)
    niveau_bruit = np.sqrt(np.var(signal[:5]))/max_amplitude
    niveau_bruit = 1e-6

    tab_cov_np = np.loadtxt('kalman-filter/tab_cov_P.txt')
    
    P = np.zeros((16, 16))

    P[0, 0] = niveau_bruit
    P[1:16, 1:16] = tab_cov_np * 0.001

    return P


#Si les valeurs dans Q sont petites, alors on croit le modèle, on suit donc moins les mesures.
def creat_Q():

    Q = np.zeros((16, 16))

    #z
    Q[0, 0] = 10**-10

    #alpha
    Q[1, 1] = 10**-6
    Q[2, 2] = 10**-6
    Q[3, 3] = 10**-7
    Q[4, 4] = 10**-6
    Q[5, 5] = 10**-6

    #b
    Q[6, 6] = 10**-6
    Q[7, 7] = 10**-6
    Q[8, 8] = 10**-8
    Q[9, 9] = 10**-4
    Q[10, 10] = 10**-8

    #m_tilde
    Q[11, 11] = 10**-6
    Q[12, 12] = 10**-5
    Q[13, 13] = 10**-8
    Q[14, 14] = 10**-8
    Q[15, 15] = 10**-5

    return Q
    


#Matrice de covariance des mesures
def creat_R():

    #10**-6 sans bruit 10**-3 avec 0.02
    R = np.array([[10**-3]]) 

    return R


#création de la matrice de mesure
def creat_C():
    
    C = np.zeros((1, 16))
    C[0, 0] = 1

    return C


#création vecteur initial
def creat_x0():

    x0 = np.zeros((16, 1))

    np_mean_std = np.loadtxt('mean_std_data.txt')

    for i in range(15):
        x0[i+1, 0] = np_mean_std[0, i]

    return x0

######################## Tracé des résultats ########################

def trace_resultat_Kalman(signal, param_kalman):
    """
    param_kalman : resultat de l'application de la fonction application_Kalman_signal_simu_i
    Retourne: 
    Trace le signal i et les gaussiennes de Kalman 
    """

    val_theta = np.linspace(-np.pi, np.pi, n_point_theta)

    for i in range(5):
        param_kalman[5+i] = np.exp(param_kalman[5+i])
        param_kalman[10+i] = 2 * np.arctan(param_kalman[10+i])

    fig = plt.figure(figsize=(5,5))
    plt.plot(val_theta, signal, label='Signal mesuré', color='grey')
    plt.plot(val_theta, gaussienne(val_theta, param_kalman), linestyle='-', label='Gaussienne obtenu par le fitre de Kalman', color='blue')
    plt.legend()
    # plt.show()
    # plt.close()
    
    # return fig

def trace_evol_param_intervalle_conf(param, iteration_Kalman, P_ite):

    """
    param : les vrais paramètres des gaussiennes
    iteration kalman : evolution des paramètres
    P_ite : matrice 19*800 avec la variance de chaque paramètre
    """

    ab = [i for i in range(iteration_Kalman.shape[1])]
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))

    for i, ax in enumerate(axes.flat):
        if i<5:
            ax.plot(ab, [param[i] for _ in range(len(ab))], color='black')
            ax.plot(ab, [iteration_Kalman[1+i, j] + 1.96 * np.sqrt(P_ite[1+i, j]) for j in range(iteration_Kalman.shape[1])], linestyle='--', color='orange')  
            ax.plot(ab, [iteration_Kalman[1+i, j] - 1.96 * np.sqrt(P_ite[1+i, j]) for j in range(iteration_Kalman.shape[1])], linestyle='--', color='orange')  
            ax.plot(ab, [iteration_Kalman[1+i, j] for j in range(iteration_Kalman.shape[1])], color='navy') 
            ax.set_title(f'Alpha{i+1}')  # Titre du sous-graphique
        if i>=5 and i<10:
            ax.plot(ab, [param[i] for _ in range(len(ab))], color='black')
            ax.plot(ab, [np.exp(iteration_Kalman[1+i, j] + 1.96 * np.sqrt(P_ite[1+i, j])) for j in range(iteration_Kalman.shape[1])], linestyle='--', color='orange')  
            ax.plot(ab, [np.exp(iteration_Kalman[1+i, j] - 1.96 * np.sqrt(P_ite[1+i, j])) for j in range(iteration_Kalman.shape[1])], linestyle='--', color='orange')  
            ax.plot(ab, [np.exp(iteration_Kalman[1+i, j]) for j in range(iteration_Kalman.shape[1])], color='navy') 
            ax.set_title(f'sigma{i+1-5}')  # Titre du sous-graphique
        if i>=10 and i<15:
            ax.plot(ab, [param[i] for _ in range(len(ab))], color='black')
            ax.plot(ab, [2 * np.arctan(iteration_Kalman[1+i, j] + 1.96 * np.sqrt(P_ite[1+i, j])) for j in range(iteration_Kalman.shape[1])], linestyle='--', color='orange')  
            ax.plot(ab, [2 * np.arctan(iteration_Kalman[1+i, j] - 1.96 * np.sqrt(P_ite[1+i, j])) for j in range(iteration_Kalman.shape[1])], linestyle='--', color='orange')  
            ax.plot(ab, [2 * np.arctan(iteration_Kalman[1+i, j]) for j in range(iteration_Kalman.shape[1])], color='navy') 
            ax.set_title(f'm{i+1-10}')  # Titre du sous-graphique
        
    # Ajustez l'espacement entre les sous-graphiques
    plt.tight_layout()

    # Affichez la figure
    # plt.show()
    # plt.close()


# Signaux synthétiques

def param_signal_simulé(i):
    """
    renvoie les paramètres du ieme signal simulé
    param = [les alphas, les b, les m, 0]
    """

    # Définir le chemin du fichier
    chemin_du_fichier = os.path.join("Signaux simulés", "param"+str(i))

    # Initialiser une liste pour stocker le contenu du fichier
    param = []

    # Lire le contenu du fichier
    with open(chemin_du_fichier, "r") as fichier:
        for ligne in fichier:
            param.append(float(ligne.strip()))  # strip() pour enlever les sauts de ligne et espaces en trop

    return param

def signal_simulé(i):

    #Abscisse du tracé
    val_theta = np.linspace(-np.pi, np.pi, n_point_theta)

    #Valeur de la somme des gaussiennes
    param = param_signal_simulé(i)
    val_gaussienne = gaussienne(val_theta, param) 

    #Paramètre du bruit
    max_amplitude = max(abs(val) for val in val_gaussienne)
    niveau_bruit = 0.02
    mu = 0
    sigma_bruit = max_amplitude * niveau_bruit

    #bruitage des données
    val_gaussienne = [val + np.random.normal(mu, sigma_bruit) for val in val_gaussienne]

    return val_gaussienne

def generation_parametres():
    np_mean_std = np.loadtxt('kalman-filter/mean_std_data.txt')
    #Paramètres des gaussiennes
    alpha1, alpha2, alpha3, alpha4, alpha5 = np.random.normal(np_mean_std[0, 0], np_mean_std[1, 0]), np.random.normal(np_mean_std[0, 1], np_mean_std[1, 1]), np.random.normal(np_mean_std[0, 2], np_mean_std[1, 2]), np.random.normal(np_mean_std[0, 3], np_mean_std[1, 3]), np.random.normal(np_mean_std[0, 4], np_mean_std[1, 4]) #amplitude
    b1, b2, b3, b4, b5 = np.random.normal(np_mean_std[0, 5], np_mean_std[1, 5]), np.random.normal(np_mean_std[0, 6], np_mean_std[1, 6]), np.random.normal(np_mean_std[0, 7], np_mean_std[1, 7]), np.random.normal(np_mean_std[0, 8], np_mean_std[1, 8]), np.random.normal(np_mean_std[0, 9], np_mean_std[1, 9])              #écart type
    m_tilde_1, m_tilde_2, m_tilde_3, m_tilde_4, m_tilde_5 = np.random.normal(np_mean_std[0, 10], np_mean_std[1, 10]), np.random.normal(np_mean_std[0, 11], np_mean_std[1, 11]), np.random.normal(np_mean_std[0, 12], np_mean_std[1, 12]), np.random.normal(np_mean_std[0, 13], np_mean_std[1, 13]), np.random.normal(np_mean_std[0, 14], np_mean_std[1, 14])          #moyenne, theta_i
    
    parametres = param.parametres()
    #param correspond aux vrais paramètres des gaussiennes
    parametres.amplitudes = [alpha1, alpha2, alpha3, alpha4, alpha5]
    parametres.ecarts_types = [np.exp(b1), np.exp(b2), np.exp(b3), np.exp(b4), np.exp(b5)]
    parametres.centres = [2 * np.arctan(m_tilde_1), 2 * np.arctan(m_tilde_2), 2 * np.arctan(m_tilde_3), 2 * np.arctan(m_tilde_4), 2 * np.arctan(m_tilde_5)]
    return parametres