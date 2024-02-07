"""
Script permettant de traiter un signal (découpàage, analyse battement par battement, extraction des caractéristiques et prédiction)
"""

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
import kalman as kalman
import customtkinter as ctk
import filter as fil
import time

ctk.set_appearance_mode("light")
#ctk.set_default_color_theme("r")

class Interface_signal:
    def __init__(self) -> None:
        # Création de la fenêtre principale
        window = ctk.CTk()
        window.title("Analyse d'ECG")
        self.window = window

        """
        Configuration de la grille
        """

        window.columnconfigure(0, weight=1)
        window.columnconfigure(1, weight=1)
        window.rowconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        window.rowconfigure(2, weight=1)
        window.rowconfigure(3, weight=1)




        '''
        Frame pour les boutons
        '''

        self.frame_buttons = ctk.CTkFrame(window)
        self.frame_buttons.grid(row=1, column=0, columnspan = 2, sticky="nsew", padx=5, pady=5)

        # Entrée pour choisir un fichier dans le dossier "/data/1/beats/{numero_patient}"
        self.label_file_path = ctk.CTkLabel(self.frame_buttons, text="Chemin du fichier")
        self.label_file_path.grid(row=1, column=0, padx=20, pady=15)
        self.entry_file_path = ctk.CTkEntry(self.frame_buttons)
        self.entry_file_path.insert(0, "data/1/full_signal/3.npy")
        self.entry_file_path.grid(row=1, column=1, sticky = "nsew", columnspan = 3, padx=10, pady=5)
        self.button_choose_file = ctk.CTkButton(self.frame_buttons, text="Choisir un fichier", command= self.choose_file)
        self.button_choose_file.grid(row=1, column=4, padx=10, pady=5)
        self.signal = np.load(self.entry_file_path.get())
        self.signal = ls.substract_linear(self.signal, 10)
        self.signal = np.array(self.signal)/np.max(self.signal)


        
        '''
        Fenêtre pour les paramètres du découpage
        '''

        self.frame_parameters_decoupage = ctk.CTkFrame(window)
        self.frame_parameters_decoupage.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.label_parameters = ctk.CTkLabel(self.frame_parameters_decoupage, text="Paramètres du découpage")
        self.label_parameters.grid(row=0, column=0, padx=10, pady=5)

        # Entrée pour la hauteur minimale de détection des pics
        self.label_hauteur_decoupage = ctk.CTkLabel(self.frame_parameters_decoupage, text="Hauteur du découpage")
        self.label_hauteur_decoupage.grid(row=1, column=0, padx=10, pady=5)
        self.entry_hauteur_decoupage = ctk.CTkEntry(self.frame_parameters_decoupage)
        self.entry_hauteur_decoupage.insert(0, "0.50")
        self.entry_hauteur_decoupage.grid(row=1, column=1, padx=10, pady=5)

        # Entrée pour la distance minimale entre deux pics

        self.label_distance_decoupage = ctk.CTkLabel(self.frame_parameters_decoupage, text="Distance du découpage")
        self.label_distance_decoupage.grid(row=2, column=0, padx=10, pady=5)
        self.entry_distance_decoupage = ctk.CTkEntry(self.frame_parameters_decoupage)
        self.entry_distance_decoupage.insert(0, "100")
        self.entry_distance_decoupage.grid(row=2, column=1, padx=10, pady=5)
    

        # Bouton pour lancer la descente de gradient
        self.button_decoupage = ctk.CTkButton(self.frame_parameters_decoupage, text="Détection des pics", command= lambda : self.detection_pics())
        self.button_decoupage.grid(row=4, column=0, padx=10, pady=5)

        # Bouton pour effectuer le découpage
        self.button_decoupage = ctk.CTkButton(self.frame_parameters_decoupage, text="Découpage des pics", command= lambda : self.decoupage())
        self.button_decoupage.grid(row=4, column=1, padx=10, pady=5)


        """
        Définition des plots
        """

        self.fig_signal, self.ax_signal = plt.subplots(figsize=(6, 3))
        self.canvas_signal = FigureCanvasTkAgg(self.fig_signal, master=window) 
        self.canvas_signal_widget = self.canvas_signal.get_tk_widget()
        self.canvas_signal_widget.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)



        self.update_plot_signal()


        """
        Frame pour l'analyse battement par battement
        """

        self.frame_beat_by_beat = ctk.CTkFrame(window)
        self.frame_beat_by_beat.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.label_beat_by_beat = ctk.CTkLabel(self.frame_beat_by_beat, text="Analyse battement par battement")
        self.label_beat_by_beat.grid(row=0, column=0, padx=10, pady=5)

        self.fig_beat, self.ax_beat = plt.subplots(figsize=(6, 3))
        self.canvas_beat = FigureCanvasTkAgg(self.fig_beat, master=window)
        self.canvas_beat_widget = self.canvas_beat.get_tk_widget()
        self.canvas_beat_widget.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)

        # Affichage du nombre de battements détectés
        self.label_beat_number = ctk.CTkLabel(self.frame_beat_by_beat, text="Nombre de battements détectés")
        self.label_beat_number.grid(row=1, column=0, padx=10, pady=5)
        self.label_beat_number_value = ctk.CTkLabel(self.frame_beat_by_beat, text="0")
        self.label_beat_number_value.grid(row=1, column=1, padx=10, pady=5)

        # Affichae superposé des battements
        self.button_plot_beats = ctk.CTkButton(self.frame_beat_by_beat, text="Afficher les battements", command= lambda : self.plot_beats())
        self.button_plot_beats.grid(row=2, column=0, padx=10, pady=5)



        """
        Frame pour l'analyse des battements
        """

        self.frame_beat_analysis = ctk.CTkFrame(window)
        self.frame_beat_analysis.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        self.label_beat_analysis = ctk.CTkLabel(self.frame_beat_analysis, text="Analyse des battements")
        self.label_beat_analysis.grid(row=0, column=0, padx=10, pady=5)

        self.label_learning_rate = ctk.CTkLabel(self.frame_beat_analysis, text="Taux d'apprentissage")
        self.label_learning_rate.grid(row=2, column=0, padx=10, pady=5)
        self.entry_learning_rate1 = ctk.CTkEntry(self.frame_beat_analysis)
        self.entry_learning_rate1.insert(0, "1")
        self.entry_learning_rate1.grid(row=3, column=1, padx=10, pady=5)
        self.entry_learning_rate2 = ctk.CTkEntry(self.frame_beat_analysis)
        self.entry_learning_rate2.insert(0, "0.00001")
        self.entry_learning_rate2.grid(row=3, column=2, padx=10, pady=5)
        self.entry_learning_rate3 = ctk.CTkEntry(self.frame_beat_analysis)
        self.entry_learning_rate3.insert(0, "0.01")
        self.entry_learning_rate3.grid(row=3, column=3, padx=10, pady=5)
        

        # Bouton pour lancer l'analyse des battements

        self.button_analyse_beat = ctk.CTkButton(self.frame_beat_analysis, text="Analyser les battements", command= lambda : self.analyse_beats())
        self.button_analyse_beat.grid(row=1, column=0, padx=10, pady=5)

        """
        Frame pour l'affichage des résultats
        """

        self.fig_params, self.ax_params = plt.subplots(figsize=(6, 3))
        self.canvas_params = FigureCanvasTkAgg(self.fig_params, master=window)
        self.canvas_params_widget = self.canvas_params.get_tk_widget()
        self.canvas_params_widget.grid(row=4, column=1, sticky="nsew", padx=5, pady=5)


        self.beats = []
        # self.decoupage()
        # self.plot_beats()
        # self.analyse_beats()
        # self.plot_params()








        self.window.mainloop()

    def choose_file(self):
        """
        Fonction permettant de choisir un fichier dans le dossier "/data/1/full_signal"
        """
        filename = filedialog.askopenfilename(initialdir="data/1/full_signal", title="Select a File", filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
        self.entry_file_path.delete(0, tk.END)
        self.entry_file_path.insert(0, filename)
        self.signal = np.load(self.entry_file_path.get())
        self.signal = ls.substract_linear(self.signal, 10)
        self.signal = np.array(self.signal)/np.max(self.signal)
        self.update_plot_signal()

    def update_plot_signal(self):
        """
        Fonction permettant de mettre à jour le plot du signal
        """
        self.ax_signal.clear()
        self.ax_signal.plot(self.signal, color = 'red')
        self.ax_signal.grid()
        self.canvas_signal.draw()
    
    def detection_pics(self):
        self.update_plot_signal()
        self.hauteur = float(self.entry_hauteur_decoupage.get())
        self.distance = float(self.entry_distance_decoupage.get())
        self.pics, _ = sig.find_peaks(self.signal, height=self.hauteur, distance=self.distance)
        self.ax_signal.plot(self.pics, self.signal[self.pics], "o", color = 'green')
        self.canvas_signal.draw()
        self.beats = []
        peaks, _ = sig.find_peaks(self.signal, distance=self.distance,height=self.hauteur)

        # On découpe le signal en battements cardiaques entre chaque pic
        for i in range(len(peaks)):
            # Indice du pic précédent
            if i == 0:
                peak_a = 0
            else:
                peak_a = peaks[i-1]
            # Indice du pic central
            peak_b = peaks[i]
            # Indice du pic suivant
            if i == len(peaks)-1:
                peak_c = len(self.signal)
            else:
                peak_c = peaks[i+1]
            
            indice_start = (peak_a + peak_b) // 2
            indice_end = (peak_b + peak_c) // 2

            beat = self.signal[indice_start:indice_end]
            self.beats.append(beat)
        
        self.beats.pop(0)
        self.beats.pop(-1)

    def decoupage(self):
        self.detection_pics()
        self.label_beat_number_value.configure(text=str(len(self.beats)))

    def plot_beats(self):
        self.ax_beat.clear()
        for beat in self.beats:
            self.ax_beat.plot(np.linspace(-180,180,len(beat)), beat, color = 'red', alpha = 0.5)
        self.ax_beat.grid()
        self.canvas_beat.draw()
    
    def analyse_beats(self):
        self.list_params = []
        learning_rate1 = float(self.entry_learning_rate1.get())
        learning_rate2 = float(self.entry_learning_rate2.get())
        learning_rate3 = float(self.entry_learning_rate3.get())
        max_iterations = 20
        pas = 10
        learning_rate = {"Amplitude" : learning_rate1, "Centre" : learning_rate2, "Ecart-type" : learning_rate3}
        for beat in self.beats:
            self.list_params.append(ext.gradient_descent_calibre(np.array(beat), learning_rate, pas, max_iterations)[0])
        self.plot_params()

    def plot_params(self):
        self.ax_params.clear()
        self.ax_params.title.set_text("Evolution des paramètres des battements")
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        labels = ['Pic P', 'Pic Q', 'Pic R', 'Pic S', 'Pic T']
        for j in range(5):
            y_param = []
            for param in self.list_params:
                y_param.append(param.centres[j])
            self.ax_params.plot(np.linspace(0,len(y_param),len(y_param)), y_param, color = colors[j], label = labels[j])
        self.ax_params.grid()
        self.ax_params.legend()
        self.canvas_params.draw()


interface = Interface_signal()
