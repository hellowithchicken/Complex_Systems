import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import pylab as pl

def initialize_grid(size):
    """makes grid of chosen size"""

    # make the grid
    grid = np.zeros((size,size))

    # fill in the 'anchor point' for the structure
    grid[int(size/2)][int(size/2)] = 1

    # for _ in range(5):
    #     x, y = np.random.uniform(0, len(grid)-1 , 2)
    #     grid[int(x)][int(y)] = 1

    return grid

def check_adhere(x_pos, y_pos, grid):
    """checks if the walker can adhere and does so, if not returns False"""

    # not allowing for adhering ON the structure itself
    if grid[x_pos][y_pos] > 0:
        return False

    if y_pos + 1 <= len(grid) - 1:
        if grid[x_pos][y_pos + 1] > 0:
            return True
    if y_pos - 1 >= 0:
        if grid[x_pos][y_pos - 1] > 0:
            return True
    if x_pos + 1 <= len(grid) - 1:
        if grid[x_pos+1][y_pos] > 0:
            return True
    if x_pos - 1 >= 0:
        if grid[x_pos-1][y_pos] > 0:
            return True
    
    return False

def move(x_pos, y_pos, grid):
    """moves the random walker, avoiding edges and the structure"""
    
    while True:
        x_new, y_new = x_pos, y_pos
        movement = np.random.choice([0,1,2,3])

        if movement == 0:
            x_new = x_pos + 1

        elif movement == 1:
            x_new = x_pos - 1
        
        elif movement == 2:
            y_new = y_pos + 1
        
        else:
            y_new = y_pos - 1
        
        # check if the new move doesn't hit a boundary or structure
        if (x_new < 0) or (x_new > len(grid) - 1):
            continue
        if (y_new < 0) or (y_new > len(grid) - 1):
            continue
        if grid[x_new][y_new] > 0:
            continue

        return x_new, y_new


def walker(grid, stick_prob, time, n_walkers):
    """spawns a new random walker and walks it to the structure"""

    # initiate position, randomly in a circle around the structure
    while True:
        polar_coord = np.random.uniform(0, 2*np.pi)
        x_pos, y_pos = int(len(grid)/2+(len(grid)/2)*np.cos(polar_coord)), int(len(grid)/2+(len(grid)/2)*np.sin(polar_coord))

        if grid[x_pos][y_pos] == 0:
            break

    while True:

        # check if the walker can adhere
        if (check_adhere(x_pos, y_pos, grid) == True) and (np.random.uniform() <= stick_prob):
            grid[x_pos][y_pos] = time/n_walkers*0.8 + 0.2
            break

        # if didn't adhere, move random walker
        else:
            x_pos, y_pos = move(x_pos, y_pos, grid)


def DLA_init(gridsize, n_walkers, stick_prob):
    """runs the DLA sim"""

    grid = initialize_grid(gridsize)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     result = [executor.submit(walker, grid, stick_prob) for _ in range(n_walkers)]
    #     for i in concurrent.futures.as_completed(result):
    #         grid = i.result()

    for i in tqdm(range(n_walkers)):
        walker(grid, stick_prob, i, n_walkers)

    return grid

def fractal_dim(grid):

    Lx = grid.shape[1]
    Ly = grid.shape[0]

    lengths = np.linspace(0.1, 1, 20)
    square_counts = []
    for r in lengths:
        point_counter = 0

        # select the corners of the box to count our grid points in
        x_start, x_end = int(Lx/2 - r*Lx/2), int(Lx/2 + r*Lx/2)
        y_start, y_end = int(Ly/2 - r*Ly/2), int(Ly/2 + r*Ly/2)
        for gridpoint in np.hstack(grid[x_start:x_end, y_start:y_end]):
            if gridpoint > 0:
                point_counter += 1
        
        square_counts.append(point_counter)

    log_x = [np.log(x*Lx) for x in lengths]
    log_y = [np.log(y) for y in square_counts]
    coeffs=np.polyfit(log_x, log_y, 1)
    print("Fractal dimension is roughly ", coeffs[0])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(log_x, log_y)
    # ax.set_xlabel('lengths')
    # plt.show()


def plot_grid(grid):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    cax = ax.matshow(grid, cmap='afmhot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.axis('off')
    #plt.colorbar()
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)







if __name__ == '__main__':

    grid = DLA_init(100, 250, 0.7)

    grid = grid[~np.all(grid == 0, axis=1)]
    grid = grid[:, ~np.all(grid == 0, axis=0)]

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    cax = ax.matshow(grid, cmap='afmhot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.axis('off')
    #plt.colorbar()
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    #plt.imshow(grid)
    plt.savefig('test.png')

    fractal_dim(grid)