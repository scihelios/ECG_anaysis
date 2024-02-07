import numpy as np

class Vecteur:
    """
    Classe pour représenter un vecteur en 3D.
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return f'Point({self.x}, {self.y}, {self.z})'
    
    def polar(self):
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = np.arccos(self.z/r)
        phi = np.arctan2(self.y, self.x)
        return r, theta, phi

class Dipole:
    """
    Classe pour représenter un dipôle électrique en 3D.
    """
    def __init__(self, position, moment):
        self.position = position
        self.moment = moment
    
    def __repr__(self):
        return f'Dipole({self.position}, {self.moment})'
    
    def potentiel(self, point):
        """
        Retourne le potentiel électrique en un point donné.
        """
        r = np.sqrt((point.x - self.position.x)**2 + (point.y - self.position.y)**2 + (point.z - self.position.z)**2)
        return self.moment.z / r**2

    def add(self, point):
        """
        Ajoute un point à la position du dipôle.
        """
        self.position.x += point.x
        self.position.y += point.y
        self.position.z += point.z



