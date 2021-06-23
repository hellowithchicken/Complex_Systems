import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from copy import deepcopy
from array2gif import write_gif
from celluloid import Camera
import sys


grid = np.load('10k_3periods.npy')
print(grid.shape[0])

pixels = []
c_values = []

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if grid[i,j] > 0:
            pixels.append((i, j))
            c_values.append(grid[i,j])

sorted_values = np.argsort(c_values)

pixels_x = [x[0] for x in pixels]
pixels_y = [y[1] for y in pixels]

min_x, max_x = min(pixels_x), max(pixels_x)
min_y, max_y = min(pixels_y), max(pixels_y)

x_len = max_x - min_x
y_len = max_y - min_y
# get the pixel coordinates/colors in order
ordered_pixels = []
ordered_colors = []
for i in range(len(sorted_values)):
    pixel = pixels[sorted_values[i]]
    ordered_pixels.append(pixel)
    color = c_values[sorted_values[i]]
    ordered_colors.append(color)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticks([])
ax.set_yticks([])
camera = Camera(fig)
data_set = []
images = []
array = np.zeros((x_len+1, y_len+1))
for i in range(len(ordered_pixels)):
    c_value = c_values[i]
    if i%100 == 0:
        for pixel, color in zip(ordered_pixels[:i], ordered_colors[:i]):
            array[pixel[0]-min_x, pixel[1]-min_y] = color
        data_set.append(deepcopy(array))
        image = ax.imshow(array, cmap='afmhot', animated=True)
        images.append([image])
        plt.pause(0.01)
        camera.snap()

animation = camera.animate()
animation.save('animation10k3.gif', writer='PillowWriter', fps=10)    
