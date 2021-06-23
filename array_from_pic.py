import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import pylab as pl


def pic2array(image_grid):
    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
 
    image=rgb2gray(pl.imread(f"{image_grid}.png"))
    plt.imshow(image, cmap='binary')
    plt.show()

    grid = np.zeros((image.shape[0], image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > 0 and (image[i,j] < 1):
            # if image[i,j] > 0:
                grid[i,j] = 1
                
    print(grid.shape[0], grid.shape[1])
    plt.imshow(grid, cmap='binary')
    plt.show()

pic2array('test_0.1')