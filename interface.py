import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.signal as sig
import extremum as ext
import parametres as par

learning_rate1 = {
    'Amplitude' : [1],
    'Centre' : [0.00001],
    'Ecart-type' : [0.0001]
}

# Fonction pour charger un signal depuis un fichier
def load_signal(file_path):
    # Code pour charger le signal depuis le fichier
    # Remplacez ceci par votre propre code de chargement de signal
    signal = np.load(file_path)
    return signal

# Fonction pour effectuer la descente de gradient avec les paramètres donnés
def gradient_descent(signal, learning_rate = 1, max_iterations = 1):
    # Code pour effectuer la descente de gradient
    # Remplacez ceci par votre propre code de descente de gradient
    # Vous pouvez utiliser la fonction minimize de scipy pour cela
    # Par exemple : result = minimize(your_cost_function, initial_parameters, method='gradient', options={'maxiter': max_iterations})
    param, filt_beat = ext.gradient_descent_calibre(signal, learning_rate=learning_rate1) # Remplacez cela par votre propre résultat de descente de gradient
    result = param.signal_gaussiennes(len(signal))
    return result

# Fonction appelée lorsque le bouton "Choisir un fichier" est cliqué
def choose_file():
    file_path = filedialog.askopenfilename(title="Choisir un fichier")
    if file_path:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, file_path)



# Fonction pour visualiser la courbe
def plot_curve(signal, result):
    plt.figure(figsize=(6, 4))
    plt.plot(signal, label="Signal original")
    

    
    plt.legend()
    plt.xlabel("Temps")
    plt.ylabel("Amplitude")
    plt.title("Analyse d'ECG")
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=4, column=0, columnspan=3, pady=10)

# Fonction appelée lorsque le bouton "Analyser" est cliqué
def analyze_signal():
    file_path = entry_file_path.get()
    signal = load_signal(file_path)

    learning_rate = float(entry_learning_rate.get())
    max_iterations = int(entry_max_iterations.get())

    result = gradient_descent(signal, learning_rate, max_iterations)

    plot_curve(signal, result)

# Création de la fenêtre principale
window = tk.Tk()
window.title("Analyse d'ECG")

# Création des widgets
label_file_path = tk.Label(window, text="Chemin du fichier:")
entry_file_path = tk.Entry(window, width=50)
button_choose_file = tk.Button(window, text="Choisir un fichier", command=choose_file)

label_learning_rate = tk.Label(window, text="Taux d'apprentissage:")
entry_learning_rate = tk.Entry(window, width=10)

label_max_iterations = tk.Label(window, text="Nombre maximal d'itérations:")
entry_max_iterations = tk.Entry(window, width=10)

button_analyze = tk.Button(window, text="Analyser", command=analyze_signal)

# Placement des widgets dans la fenêtre
label_file_path.grid(row=0, column=0, padx=10, pady=5)
entry_file_path.grid(row=0, column=1, padx=10, pady=5)
button_choose_file.grid(row=0, column=2, padx=10, pady=5)

label_learning_rate.grid(row=1, column=0, padx=10, pady=5)
entry_learning_rate.grid(row=1, column=1, padx=10, pady=5)

label_max_iterations.grid(row=2, column=0, padx=10, pady=5)
entry_max_iterations.grid(row=2, column=1, padx=10, pady=5)

button_analyze.grid(row=3, column=0, columnspan=3, pady=10)

# Lancement de la boucle principale
window.mainloop()
