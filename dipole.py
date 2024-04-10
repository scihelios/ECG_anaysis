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
    
    def potentiel(self, x, y, z):
        """
        Retourne le potentiel électrique en un point donné.
        """
        r = np.sqrt((x - self.position[0])**2 + (y - self.position[1])**2 + (z - self.position[2])**2)
        return (self.moment[0] * x + self.moment[1] * y + self.moment[2] * z) / r**2







class Derivation:
    """
    Classe pour représenter une dérivation.
    """
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def __repr__(self):
        return f'Derivation({self.point1}, {self.point2})'
    
    def derive(self, dipole):
        return dipole.potentiel(self.point2) - dipole.potentiel(self.point1)


