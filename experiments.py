from dla_rand_new import DLA_init
import numpy as np

# change stickiness

# =============================================================================
# simulations = 50
# 
# stick_list = [1, 0.9, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.01, 0.005]
# for stick in stick_list:   
#     for sim in range(simulations):
#         sim = sim + 50
#         grid = DLA_init(100, 400, stick)
#         np.save("results/changing_stickness_400_walkers/"+str(stick)+","+str(sim), grid)
# =============================================================================
        
        
# change walker number

#simulations = 50

#walker_list = [50, 100, 250, 500, 1000, 3000, 4500, 6000, 8500, 10000]
#for walk in walker_list:   
#    for sim in range(simulations):
#        grid = DLA_init(500, walk, 1)
#        np.save("results/changing_walkers/"+str(walk)+","+str(sim), grid) 
        


### for comparison with other cities

simulations = 10

stick_list = [1, 0.9, 0.7, 0.5, 0.3, 0.15, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]
for stick in stick_list:   
    for sim in range(simulations):
        grid = DLA_init(100, 500, stick)
        np.save("results/for_city_comprison/"+str(stick)+","+str(sim), grid)
        