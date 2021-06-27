import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def course_grain_convolution(grid, threshold=0):
    blured = gaussian_filter(grid, sigma=1, order=2)

    tmp = blured > threshold

    return tmp.astype(int)



if __name__ == "__main__":
    grid = np.load("results/changing_walkers/250,48.npy")
    plt.matshow(grid)
    plt.show()
    plt.matshow(course_grain_convolution(grid))
    plt.show()
