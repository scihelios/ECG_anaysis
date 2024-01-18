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
import tkinter.ttk as ttk


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
        self.label_file_path.grid(row=1, column=0, padx=20, pady=15)
        self.entry_file_path = tk.Entry(self.frame_buttons)
        self.entry_file_path.insert(0, "data/1/beats/24/776.npy")
        self.entry_file_path.grid(row=1, column=1, padx=10, pady=5)
        self.button_choose_file = tk.Button(self.frame_buttons, text="Choisir un fichier", command= self.choose_file)
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
        self.entry_max_iterations.insert(0, "10")
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

        # Bouton pour lancer le filtre de Kalman
        self.button_kalman = tk.Button(self.frame_parameters_kalman, text="Filtre de Kalman", command= self.choose_file)
        self.button_kalman.grid(row=4, column=0, padx=10, pady=5)


        """
        Définition des plots
        """

        self.fig_gradient, self.ax_gradient = plt.subplots(figsize=(6, 4))
        self.fig_kalman, self.ax_kalman = plt.subplots(figsize=(6, 4))

        self.canvas_gradient = FigureCanvasTkAgg(self.fig_gradient, master=window)
        self.canvas_gradient_widget = self.canvas_gradient.get_tk_widget()
        self.canvas_gradient_widget.grid(row=4, column=0, padx=10, pady=5)

        self.canvas_kalman = FigureCanvasTkAgg(self.fig_kalman, master=window)
        self.canvas_kalman_widget = self.canvas_kalman.get_tk_widget()
        self.canvas_kalman_widget.grid(row=4, column=1, padx=10, pady=5)

        self.update_plots()


        """
        Affichage des paramètres des gaussiennes
        """

        self.frame_table_gradient = tk.Frame(window, borderwidth=2, relief=tk.GROOVE)
        self.frame_table_gradient.grid(row=5, column=0, padx=10, pady=5)
        self.label_table_gradient = tk.Label(self.frame_table_gradient, text="Paramètres des gaussiennes")
        self.label_table_gradient.grid(row=0, column=0, padx=10, pady=5)
        self.tree_gradient = ttk.Treeview(self.frame_table_gradient, columns=(1,2,3,4,5,6), show="headings", height="3")
        self.tree_gradient.grid(row=1, column=0, padx=10, pady=5)
        self.tree_gradient.heading(1, text="Caractéristique")
        self.tree_gradient.heading(2, text="Pic P")
        self.tree_gradient.heading(3, text="Pic Q")
        self.tree_gradient.heading(4, text="Pic R")
        self.tree_gradient.heading(5, text="Pic S")
        self.tree_gradient.heading(6, text="Pic T")
        self.tree_gradient.column(1, width=100)
        self.tree_gradient.column(2, width=100, anchor="center")
        self.tree_gradient.column(3, width=100, anchor="center")
        self.tree_gradient.column(4, width=100, anchor="center")
        self.tree_gradient.column(5, width=100, anchor="center")
        self.tree_gradient.column(6, width=100, anchor="center")


        self.frame_table_kalman = tk.Frame(window, borderwidth=2, relief=tk.GROOVE)
        self.frame_table_kalman.grid(row=5, column=1, padx=10, pady=5)
        self.label_table_kalman = tk.Label(self.frame_table_kalman, text="Paramètres des gaussiennes")
        self.label_table_kalman.grid(row=0, column=0, padx=10, pady=5)
        self.tree_kalman = ttk.Treeview(self.frame_table_kalman, columns=(1,2,3,4,5,6), show="headings", height="3")
        self.tree_kalman.grid(row=1, column=0, padx=10, pady=5)
        self.tree_kalman.heading(1, text="Caractéristique")
        self.tree_kalman.heading(2, text="Pic P")
        self.tree_kalman.heading(3, text="Pic Q")
        self.tree_kalman.heading(4, text="Pic R")
        self.tree_kalman.heading(5, text="Pic S")
        self.tree_kalman.heading(6, text="Pic T")
        self.tree_kalman.column(1, width=100)
        self.tree_kalman.column(2, width=100, anchor="center")
        self.tree_kalman.column(3, width=100, anchor="center")
        self.tree_kalman.column(4, width=100, anchor="center")
        self.tree_kalman.column(5, width=100, anchor="center")
        self.tree_kalman.column(6, width=100, anchor="center")


        self.param_gradient = par.parametres()
        self.param_gradient.amplitudes = [0, 0, 0, 0, 0]
        self.param_gradient.centres = [0, 0, 0, 0, 0]
        self.param_gradient.ecarts_types = [0, 0, 0, 0, 0]

        self.param_kalman = par.parametres()
        self.param_kalman.amplitudes = [0, 0, 0, 0, 0]
        self.param_kalman.centres = [0, 0, 0, 0, 0]
        self.param_kalman.ecarts_types = [0, 0, 0, 0, 0]

        self.update_parameters_gaussiennes()

        self.window.mainloop()

    def plot_gradient(self, signal, label, color = 'b'):
        """
        Fonction permettant de tracer le signal et le résultat de la descente de gradient
        """
        x_values = np.linspace(-np.pi, np.pi, len(signal))
        self.ax_gradient.plot(x_values, signal, label=label, color = color)
        self.ax_gradient.legend()
        self.canvas_gradient.draw()

    def plot_kalman(self, signal, label, color = 'b'):
        x_values = np.linspace(-np.pi, np.pi, len(signal))
        self.ax_kalman.plot(x_values, signal, label=label, color = color)
        self.ax_kalman.legend()
        self.canvas_kalman.draw()
    
    def gradient_descent(self):
        learning_rate1 = float(self.entry_learning_rate1.get())
        learning_rate2 = float(self.entry_learning_rate2.get())
        learning_rate3 = float(self.entry_learning_rate3.get())
        max_iterations = int(self.entry_max_iterations.get())
        learning_rate = {"Amplitude" : learning_rate1, "Centre" : learning_rate2, "Ecart-type" : learning_rate3}
        self.param_gradient, self.filt_beat = ext.gradient_descent_calibre(self.beat, learning_rate, max_iterations)
        self.plot_gradient(self.param_gradient.signal_gaussiennes(len(self.beat)), "Signal filtré", color = 'r')
        self.update_parameters_gaussiennes()

    def update_plots(self):
        self.ax_gradient.clear()
        self.ax_kalman.clear()

        self.ax_gradient.set_xlabel("Phase (°)")
        self.ax_gradient.set_ylabel("Amplitude")
        self.ax_gradient.grid()

        self.ax_kalman.set_xlabel("Phase (°)")
        self.ax_kalman.set_ylabel("Amplitude")
        self.ax_kalman.grid()

        self.plot_gradient(self.beat, "Signal")
        self.plot_kalman(self.beat, "Signal")

    def update_parameters_gaussiennes(self):
        self.tree_gradient.delete(*self.tree_gradient.get_children())
        self.tree_kalman.delete(*self.tree_kalman.get_children())
        self.tree_gradient.insert("", "end", values=("Amplitude", round(self.param_gradient.amplitudes[0],0), round(self.param_gradient.amplitudes[1],0), round(self.param_gradient.amplitudes[2],0), round(self.param_gradient.amplitudes[3],0), round(self.param_gradient.amplitudes[4],0)))
        self.tree_gradient.insert("", "end", values=("Centre (°)", round(180/np.pi * self.param_gradient.centres[0],2), round(180/np.pi * self.param_gradient.centres[1],2), round(180/np.pi * self.param_gradient.centres[2],2), round(180/np.pi * self.param_gradient.centres[3],2), round(180/np.pi * self.param_gradient.centres[4],2)))
        self.tree_gradient.insert("", "end", values=("Ecart-type (°)", round(180/np.pi * self.param_gradient.ecarts_types[0],2), round(180/np.pi * self.param_gradient.ecarts_types[1],2), round(180/np.pi * self.param_gradient.ecarts_types[2],2), round(180/np.pi * self.param_gradient.ecarts_types[3],2), round(180/np.pi * self.param_gradient.ecarts_types[4],2)))
        self.tree_kalman.insert("", "end", values=("Amplitude", round(self.param_kalman.amplitudes[0],0), round(self.param_kalman.amplitudes[1],0), round(self.param_kalman.amplitudes[2],0), round(self.param_kalman.amplitudes[3],0), round(self.param_kalman.amplitudes[4],0)))
        self.tree_kalman.insert("", "end", values=("Centre (°)", round(180/np.pi * self.param_kalman.centres[0],2), round(180/np.pi * self.param_kalman.centres[1],2), round(180/np.pi * self.param_kalman.centres[2],2), round(180/np.pi * self.param_kalman.centres[3],2), round(180/np.pi * self.param_kalman.centres[4],2)))
        self.tree_kalman.insert("", "end", values=("Ecart-type (°)", round(180/np.pi * self.param_kalman.ecarts_types[0],2), round(180/np.pi * self.param_kalman.ecarts_types[1],2), round(180/np.pi * self.param_kalman.ecarts_types[2],2), round(180/np.pi * self.param_kalman.ecarts_types[3],2), round(180/np.pi * self.param_kalman.ecarts_types[4],2)))

    def choose_file(self):
        """
        Fonction permettant de choisir un fichier dans le dossier "/data/1/beats/{numero_patient}"
        """
        filename = filedialog.askopenfilename(initialdir="data/1/beats/1", title="Select a File", filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
        self.entry_file_path.delete(0, tk.END)
        self.entry_file_path.insert(0, filename)
        beat = np.load(filename)
        self.beat = ls.substract_linear(beat, 10)
        self.update_plots()




Interface = Interface()

#