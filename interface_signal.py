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


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

class Interface_signal:
    def __init__(self) -> None:
        # Création de la fenêtre principale
        window = ctk.CTk()
        window.title("Analyse d'ECG")
        self.window = window

        '''
        Frame pour les boutons
        '''

        self.frame_buttons = ctk.CTkFrame(window)
        self.frame_buttons.grid(row=1, column=0, columnspan=3, padx=10, pady=5)

        # Entrée pour choisir un fichier dans le dossier "/data/1/beats/{numero_patient}"
        self.label_file_path = ctk.CTkLabel(self.frame_buttons, text="Chemin du fichier")
        self.label_file_path.grid(row=1, column=0, padx=20, pady=15)
        self.entry_file_path = ctk.CTkEntry(self.frame_buttons)
        self.entry_file_path.insert(0, "data/1/full_signal/1.npy")
        self.entry_file_path.grid(row=1, column=1, padx=10, pady=5)
        self.button_choose_file = ctk.CTkButton(self.frame_buttons, text="Choisir un fichier", command= self.choose_file)
        self.button_choose_file.grid(row=1, column=2, padx=10, pady=5)
        self.signal = np.load(self.entry_file_path.get())
        self.signal = ls.substract_linear(self.signal, 10)
        self.signal = np.array(self.signal)/np.max(self.signal)
        
        '''
        Fenêtre pour les paramètres du découpage
        '''

        self.frame_parameters_decoupage = ctk.CTkFrame(window)
        self.frame_parameters_decoupage.grid(row=2, column=0, padx=10, pady=5)
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




        """
        Définition des plots
        """

        self.fig_signal, self.ax_signal = plt.subplots(figsize=(6, 4))
        self.canvas_signal = FigureCanvasTkAgg(self.fig_signal, master=window)
        self.canvas_signal_widget = self.canvas_signal.get_tk_widget()
        self.canvas_signal_widget.grid(row=4, column=0, padx=10, pady=5)

        self.update_plot_signal()



        self.window.mainloop()

    def choose_file(self):
        """
        Fonction permettant de choisir un fichier dans le dossier "/data/1/full_signal"
        """
        filename = filedialog.askopenfilename(initialdir="data/1/full_signal", title="Select a File", filetypes=(("npy files", "*.npy"), ("all files", "*.*")))
        self.entry_file_path.delete(0, tk.END)
        self.entry_file_path.insert(0, filename)
        self.signal = np.load(filename)
        self.signal = ls.substract_linear(self.signal, 10)
        self.signal = np.array(self.signal)/np.max(self.signal)
        self.update_plot_signal()

    def update_plot_signal(self):
        """
        Fonction permettant de mettre à jour le plot du signal
        """
        self.ax_signal.clear()
        self.ax_signal.plot(self.signal)
        self.canvas_signal.draw()
    
    def detection_pics(self):
        self.update_plot_signal()
        self.hauteur = float(self.entry_hauteur_decoupage.get())
        self.distance = float(self.entry_distance_decoupage.get())
        self.pics, _ = sig.find_peaks(self.signal, height=self.hauteur, distance=self.distance)
        self.ax_signal.plot(self.pics, self.signal[self.pics], "x", color = 'red')
        self.canvas_signal.draw()
        self.beats = []
        for i in range(len(self.pics)-1):
            self.beats.append(self.signal[self.pics[i]:self.pics[i+1]])
        self.beats.append(self.signal[self.pics[-1]:])
    


interface = Interface_signal()
