
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json
error_progression = []
# Define the Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Define the combined model with 5 Gaussian functions
def combined_gaussian(x, *params):
    return sum([gaussian(x, *params[i:i+3]) for i in range(0, len(params), 3)])

with open("signal10001.json", "r") as json_file:
    data = json.load(json_file)
    x_data = np.array(data["signal"])
    param = np.array(data["gaussienne"])
    print(param)
    print(x_data)
    plt.plot([i for i in range(len(x_data))],combined_gaussian([i for i in range(len(x_data))],*param))
    plt.show()