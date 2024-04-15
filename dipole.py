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
        return f'Vecteur ({self.x}, {self.y}, {self.z})'
    
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
        return f'Dipole ({self.position}, {self.moment})'
    
    def potentiel(self, point):
        """
        Retourne le potentiel électrique en un point donné.
        """
        r = np.sqrt((point.x - self.position[0])**2 + (point.y - self.position[1])**2 + (point.z - self.position[2])**2)
        return 1/(4*np.pi) *(point.x * self.moment[0] +  point.y * self.moment[1] + point.z * self.moment[2])/r**3

    def difference_potentiel(self, derivation):
        return self.potentiel(derivation.point2) - self.potentiel(derivation.point1)






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


