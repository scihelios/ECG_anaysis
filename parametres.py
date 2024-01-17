import numpy as np
import matplotlib.pyplot as plt

correspondance_pics = ['Pic P', 'Pic Q', 'Pic R', 'Pic S', 'Pic T']

class parametres:
    def __init__(self):
        self.nombre_gaussiennes = 5
        self.amplitudes = []
        self.centres = []
        self.ecarts_types = []
        
    def fonction_gaussiennes(self):
        """
        Fonction permettant de créer une gaussienne à partir d'un dictionnaire de paramètres
        """
        if self.nombre_gaussiennes == 0:
            return lambda x : np.zeros(len(x))
        #return lambda x : sum([self.amplitudes[i] * (np.abs(x - self.centres[i]) < abs(self.ecarts_types[i])) for i in range(len(self.amplitudes))])
        return lambda x : sum([self.amplitudes[i] * np.exp(-1/2*((x - self.centres[i]) / self.ecarts_types[i])**2) for i in range(len(self.amplitudes))])
    
    def signal_gaussiennes(self, nombre_points):
        """
        Fonction permettant de créer un signal à partir d'un dictionnaire de paramètres
        """

        return self.fonction_gaussiennes()(np.linspace(-np.pi, np.pi, nombre_points))
    
    def init_parameters(self, signal):
        """
        Fonction permettant d'initialiser les paramètres des gaussiennes proches de ceux d'un ECG
        
        Args:
            nombre_gaussiennes (int): Le nombre de gaussiennes à initialiser
            signal (array): Le signal ECG sur lequel les gaussiennes seront ajustées
        
        Returns:
            dict: Un dictionnaire contenant les paramètres initialisés des gaussiennes
                - 'Amplitude' : Liste des amplitudes initialisées proches du maximum du signal ECG
                - 'Centre' : Liste des centres initialisés proches de points caractéristiques du signal ECG
                - 'Ecart-type' : Liste des écarts-types initialisés proches des largeurs caractéristiques des pics ECG
        """
        
        max_signal = np.max(signal)
        min_signal = np.min(signal)
        
        self.amplitudes = [max_signal for _ in range(self.nombre_gaussiennes-1)]+[max_signal]
        self.centres = [np.random.uniform(-np.pi, np.pi) for _ in range(self.nombre_gaussiennes-1)]+[np.argmax(signal)/len(signal)]
        self.ecarts_types = [np.random.uniform(0.1,0.3) for _ in range(self.nombre_gaussiennes)]

    def __repr__(self) -> str:
        s = f"Amplitude : \n"
        for value in self.amplitudes:
            s += f"{value:.2f} "
        s += f"\nCentre : \n"
        for value in self.centres:
            s += f"{180/np.pi*value:.2f} "
        s += f"\nEcart-type : \n"
        for value in self.ecarts_types:
            s += f"{180/np.pi*value:.2f} "
        return s
    
    def __str__(self) -> str:
        s = f"{'Pic':<15}{'Amplitude':^15}{'Centre (°)':^15}{'Ecart-type (°)':^15} \n"
        for i in range(5):
            s += f"{correspondance_pics[i]:<15}{self.amplitudes[i]:^15.2f}{180/np.pi*self.centres[i]:^15.2f}{180/np.pi*self.ecarts_types[i]:^15.2f} \n"
        return s

    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    
    def __dict__(self):
        return {'Amplitude' : self.amplitudes, 'Centre' : self.centres, 'Ecart-type' : self.ecarts_types}

    def get_params(self):
        if self.nombre_gaussiennes == 0:
            return None
        return self.amplitudes + self.centres + self.ecarts_types
    
    def set_params(self, params):
        self.amplitudes = params[:self.nombre_gaussiennes]
        self.centres = params[self.nombre_gaussiennes:2*self.nombre_gaussiennes]
        self.ecarts_types = params[2*self.nombre_gaussiennes:]

    def plot_pics(self, yscale_factor, xscale_factor=180/np.pi):
        for amplitude, centre in zip(self.amplitudes, self.centres):
            plt.plot([xscale_factor*centre, xscale_factor*centre], [0, yscale_factor * amplitude], color='black', linestyle='--', alpha=1)
        for i,centre in enumerate(np.sort(self.centres)):
            plt.text(xscale_factor * centre, yscale_factor * 1.2 * self.amplitudes[i],f"{correspondance_pics[i]}", rotation=90, fontsize=8)
