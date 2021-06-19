from dla_rand import DLA_init, plot_grid
import numpy as np


simulations = 30

stick_list = [1, 0.9, 0.7, 0.5, 0.2, 0.1, 0.05, 0.01]
for stick in stick_list:   
    for sim in range(simulations):
        grid = DLA_init(100, 250, stick)
        np.save("results/changing_stickness/"+str(stick)+","+str(sim), grid)