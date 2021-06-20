from dla_rand_new import DLA_init
import numpy as np


simulations = 50

stick_list = [1, 0.9, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.01, 0.005]
for stick in stick_list:   
    for sim in range(simulations):
        grid = DLA_init(100, 400, stick)
        np.save("results/changing_stickness_2/"+str(stick)+","+str(sim), grid)
        
        
# change walker number

simulations = 50

walker_list = [50, 100, 250, 500, 1000, 3000, 4500, 6000, 8500, 10000]
for walk in walker_list:   
    for sim in range(simulations):
        grid = DLA_init(500, walk, 1)
        np.save("results/changing_walkers/"+str(walk)+","+str(sim), grid) 
        
        