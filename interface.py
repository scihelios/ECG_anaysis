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
import linsub as ls


class Interface:
    def __init__(self) -> None:
        # Création de la fenêtre principale
        window = tk.Tk()
        window.title("Analyse d'ECG")
        self.window = window


        '''
        Frame pour les boutons
        '''

        self.frame_buttons = tk.Frame(window, borderwidth=2, relief=tk.GROOVE)
        self.frame_buttons.grid(row=1, column=0, columnspan=3, padx=10, pady=5)

        # Entrée pour choisir un fichier dans le dossier "/data/1/beats/{numero_patient}"
        self.label_file_path = tk.Label(self.frame_buttons, text="Chemin du fichier")
        self.label_file_path.grid(row=1, column=0, padx=10, pady=5)
        self.entry_file_path = tk.Entry(self.frame_buttons)
        self.entry_file_path.insert(0, "data/1/beats/1/1.npy")
        self.entry_file_path.grid(row=1, column=1, padx=10, pady=5)
        self.button_choose_file = tk.Button(self.frame_buttons, text="Choisir un fichier", command= lambda : choose_file(self))
        self.button_choose_file.grid(row=1, column=2, padx=10, pady=5)
        self.beat = np.load(self.entry_file_path.get())
        self.beat = ls.substract_linear(self.beat, 10)

        
        '''
        Fenêtre pour les paramètres de la descente de gradient
        '''

        self.frame_parameters_gradient = tk.Frame(window, borderwidth=2, relief=tk.GROOVE)
        self.frame_parameters_gradient.grid(row=2, column=0, padx=10, pady=5)
        self.label_parameters = tk.Label(self.frame_parameters_gradient, text="Paramètres de la descente de gradient")
        self.label_parameters.grid(row=0, column=0, padx=10, pady=5)

        # Entrée pour le nombre d'itérations
        self.label_max_iterations = tk.Label(self.frame_parameters_gradient, text="Nombre d'itérations")
        self.label_max_iterations.grid(row=1, column=0, padx=10, pady=5)
        self.entry_max_iterations = tk.Entry(self.frame_parameters_gradient)
        self.entry_max_iterations.insert(0, "100")
        self.entry_max_iterations.grid(row=1, column=1, padx=10, pady=5)


        # Entrée pour choisir les 3 taux d'apprentissage
        self.label_learning_rate = tk.Label(self.frame_parameters_gradient, text="Taux d'apprentissage")
        self.label_learning_rate.grid(row=2, column=0, padx=10, pady=5)
        self.entry_learning_rate1 = tk.Entry(self.frame_parameters_gradient)
        self.entry_learning_rate1.insert(0, "1")
        self.entry_learning_rate1.grid(row=3, column=1, padx=10, pady=5)
        self.entry_learning_rate2 = tk.Entry(self.frame_parameters_gradient)
        self.entry_learning_rate2.insert(0, "0.00001")
        self.entry_learning_rate2.grid(row=3, column=2, padx=10, pady=5)
        self.entry_learning_rate3 = tk.Entry(self.frame_parameters_gradient)
        self.entry_learning_rate3.insert(0, "0.0001")
        self.entry_learning_rate3.grid(row=3, column=3, padx=10, pady=5)

        # Bouton pour lancer la descente de gradient
        self.button_gradient = tk.Button(self.frame_parameters_gradient, text="Descente de gradient", command= lambda : self.gradient_descent())
        self.button_gradient.grid(row=4, column=0, padx=10, pady=5)


        '''
        Fenêtre pour les paramètres du filtre de Kalman
        '''

        self.frame_parameters_kalman = tk.Frame(window, borderwidth=2, relief=tk.GROOVE)
        self.frame_parameters_kalman.grid(row=2, column=1, padx=10, pady=5)
        self.label_parameters = tk.Label(self.frame_parameters_kalman, text="Paramètres du filtre de Kalman")
        self.label_parameters.grid(row=0, column=0, padx=10, pady=5)

        # Entrée pour le nombre d'itérations
        self.label_max_iterations = tk.Label(self.frame_parameters_kalman, text="Nombre d'itérations")
        self.label_max_iterations.grid(row=1, column=0, padx=10, pady=5)
        self.entry_max_iterations = tk.Entry(self.frame_parameters_kalman)
        self.entry_max_iterations.insert(0, "10")
        self.entry_max_iterations.grid(row=1, column=1, padx=10, pady=5)

        # Entrée pour définir les 3 covariances pour les amplitudes, les centres et les écarts-types
        self.label_covariance = tk.Label(self.frame_parameters_kalman, text="Covariance")
        self.label_covariance.grid(row=2, column=0, padx=10, pady=5)
        self.entry_covariance1 = tk.Entry(self.frame_parameters_kalman)
        self.entry_covariance1.insert(0, "0.01")
        self.entry_covariance1.grid(row=3, column=1, padx=10, pady=5)
        self.entry_covariance2 = tk.Entry(self.frame_parameters_kalman)
        self.entry_covariance2.insert(0, "0.01")
        self.entry_covariance2.grid(row=3, column=2, padx=10, pady=5)
        self.entry_covariance3 = tk.Entry(self.frame_parameters_kalman)
        self.entry_covariance3.insert(0, "0.01")
        self.entry_covariance3.grid(row=3, column=3, padx=10, pady=5)

        """
        Définition des plots
        """

        self.fig_gradient, self.ax_gradient = plt.subplots(figsize=(6, 4))
        self.fig_kalman, self.ax_kalman = plt.subplots(figsize=(6, 4))

        self.canvas_gradient = FigureCanvasTkAgg(self.fig_gradient, master=window)
        self.canvas_gradient_widget = self.canvas_gradient.get_tk_widget()
        self.canvas_gradient_widget.grid(row=4, column=0, padx=10, pady=5)
        self.ax_gradient.set_xlabel("Phase (°)")
        self.ax_gradient.set_ylabel("Amplitude")
        self.ax_gradient.grid()


        self.canvas_kalman = FigureCanvasTkAgg(self.fig_kalman, master=window)
        self.canvas_kalman_widget = self.canvas_kalman.get_tk_widget()
        self.canvas_kalman_widget.grid(row=4, column=1, padx=10, pady=5)
        self.ax_kalman.set_xlabel("Phase (°)")
        self.ax_kalman.set_ylabel("Amplitude")
        self.ax_kalman.grid()

        self.plot_gradient(self.beat, "Signal")
        self.plot_kalman(self.beat, "Signal")


        self.window.mainloop()

    def plot_gradient(self, signal, label):
        """
        Fonction permettant de tracer le signal et le résultat de la descente de gradient
        """
        x_values = np.linspace(-np.pi, np.pi, len(signal))
        self.ax_gradient.plot(x_values, signal, label=label)
        self.ax_gradient.legend()
        self.canvas_gradient.draw()

    def plot_kalman(self, signal, label):
        x_values = np.linspace(-np.pi, np.pi, len(signal))
        self.ax_kalman.plot(x_values, signal, label=label)
        self.ax_kalman.legend()
        self.canvas_kalman.draw()
    
    def gradient_descent(self):
        learning_rate1 = float(self.entry_learning_rate1.get())
        learning_rate2 = float(self.entry_learning_rate2.get())
        learning_rate3 = float(self.entry_learning_rate3.get())
        max_iterations = int(self.entry_max_iterations.get())
        learning_rate = {"Amplitude" : learning_rate1, "Centre" : learning_rate2, "Ecart-type" : learning_rate3}
        self.param_gradient = par.parametres()
        self.param_gradient, self.filt_beat = ext.gradient_descent_calibre(self.beat, learning_rate, max_iterations)
        self.plot_gradient(self.param_gradient.signal_gaussiennes(len(self.beat)), "Signal filtré")




def choose_file(interface):
    """
    Fonction permettant de choisir un fichier dans le dossier "/data/1/beats/{numero_patient}"
    """
    filename = filedialog.askopenfilename(initialdir="data/1/beats/1", title="Select a File", filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
    interface.entry_file_path.delete(0, tk.END)
    interface.entry_file_path.insert(0, filename)
    beat = np.load(filename)
    interface.plot_gradient(beat, "Signal") 
    interface.plot_kalman(beat, "Signal")


Interface = Interface()

#