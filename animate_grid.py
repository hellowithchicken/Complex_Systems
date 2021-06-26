import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from celluloid import Camera

def create_gif(gif_array, walkers_per_frame=10, fps=10, colored=False):
    """
    Loads a saved numpy array and returns a GIF, also shows the animation during creation, but this can be closed
    gif_array: .npy array to be loaded in, leave out the .npy extension, this cannot be a binary array
    walkers_per_frame: amount of walkers in the grid to be attached per frame
    fps: frames per second for GIF
    colored: True for color, False for black and white
    """

    print(f'Creating GIF of {gif_array}. You may cancel the pop-up animation.')
    grid = np.load(f'{gif_array}.npy')

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

    # plots the figure and snapshots every frame
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    camera = Camera(fig)
    data_set = []
    images = []
    array = np.zeros((x_len+1, y_len+1))
    for i in range(len(ordered_pixels)):
        if i%walkers_per_frame == 0:
            for pixel, color in zip(ordered_pixels[:i], ordered_colors[:i]):
                if colored == False:
                    array[pixel[0]-min_x, pixel[1]-min_y] = 1#color
                else:
                    array[pixel[0]-min_x, pixel[1]-min_y] = color
            data_set.append(deepcopy(array))
            image = ax.imshow(array, cmap='afmhot', animated=True)
            images.append([image])
            plt.pause(0.01)
            camera.snap()

    animation = camera.animate()
    animation.save(f'{gif_array}.gif', writer='PillowWriter', fps=fps)   
    print('Done!') 

if __name__ == '__main__':

    # give the string of the array you want to make a gif out of and feed it to create_gif
    gif_array = '400_1'
    create_gif(gif_array)
    