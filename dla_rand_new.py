import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import pylab as pl
from scipy import signal

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

def move(x_pos, y_pos, grid, time, n_walkers, radius):
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

        # make sure the particle doesn't move away too far from the structure, so limit it to a circle
        if np.sqrt((x_new-len(grid)*0.5)**2+(y_new-len(grid)*0.5)**2) > radius + 5:
            continue

        return x_new, y_new


def walker(grid, stick_prob, time, n_walkers, curr_farthest_dist):
    """spawns a new random walker and walks it to the structure"""

    # initiate position, randomly in a circle around the structure, radius scaling with time to reduce comp time
    while True:
        polar_coord = np.random.uniform(0, 2*np.pi)
        #radius = 10*(1-time/n_walkers) + 0.5*len(grid)*time/n_walkers - 1
        radius = curr_farthest_dist + 10
        x_pos, y_pos = int(len(grid)/2 + radius*np.cos(polar_coord)), int(len(grid)/2 + radius*np.sin(polar_coord))

        if grid[x_pos][y_pos] == 0:
            break

    while True:

        # check if the walker can adhere
        if (check_adhere(x_pos, y_pos, grid) == True) and (np.random.uniform() <= stick_prob):
            #grid[x_pos][y_pos] = time/n_walkers*0.8 + 0.2
            grid[x_pos][y_pos] = 1
            break

        # if didn't adhere, move random walker
        else:
            x_pos, y_pos = move(x_pos, y_pos, grid, time, n_walkers, radius)

    # update the farthest distance from center
    dist_from_center = np.sqrt((x_pos-0.5*len(grid))**2 + (y_pos-0.5*len(grid))**2)
    if dist_from_center > curr_farthest_dist:
        farthest_distance = dist_from_center

    else:
        farthest_distance = curr_farthest_dist

    return farthest_distance


def DLA_init(gridsize, n_walkers, stick_prob):
    """runs the DLA sim"""

    grid = initialize_grid(gridsize)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     result = [executor.submit(walker, grid, stick_prob) for _ in range(n_walkers)]
    #     for i in concurrent.futures.as_completed(result):
    #         grid = i.result()

    # set a maximum distance for spawning which gets updates with every walker
    farthest_distance = 5
    for i in tqdm(range(n_walkers)):
        farthest_distance = walker(grid, stick_prob, i, n_walkers, farthest_distance)

    return grid

def fractal_dim(grid, Lx, Ly):

    Lx_orig = grid.shape[1]
    Ly_orig = grid.shape[0]

    # to keep the box square, else the box counting method is a little iffy
    if Lx > Ly:
        Lx = Ly
    else:
        Ly = Lx

    lengths = np.linspace(0.2, 1, 10)
    square_counts = []
    for r in lengths:
        point_counter = 0

        # select the corners of the box to count our grid points in
        x_start, x_end = int(Lx_orig/2 - r*Lx/2), int(Lx_orig/2 + r*Lx/2)
        y_start, y_end = int(Ly_orig/2 - r*Ly/2), int(Ly_orig/2 + r*Ly/2)
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
    return coeffs[0]

def get_fractal_dim(image_grid):
    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    image=rgb2gray(pl.imread(f"{image_grid}.png"))
    # plt.imshow(image)
    # plt.show()

    # finding all the non-zero pixels
    pixels=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 0:
                pixels.append((i,j))

    Lx=image.shape[1]
    Ly=image.shape[0]
    #print (Lx, Ly)
    pixels=pl.array(pixels)
    #print (pixels.shape)

    # computing the fractal dimension
    #considering only scales in a logarithmic list
    scales=np.logspace(0.01, 1, num=10, endpoint=False, base=2)
    Ns=[]
    # looping over several scales
    for scale in scales:
        #print ("======= Scale :",scale)
        # computing the histogram
        H, edges=np.histogramdd(pixels, bins=(np.arange(0,Lx,scale),np.arange(0,Ly,scale)))
        Ns.append(np.sum(H>0))

    # linear fit, polynomial of degree 1
    coeffs, cov =np.polyfit(np.log(scales), np.log(Ns), 1, cov=True)
    print ("The fractal dimension is", -coeffs[0]) #the fractal dimension is the OPPOSITE of the fitting coefficient
    #np.savetxt("scaling.txt", list(zip(scales,Ns)))
    return -coeffs[0], np.sqrt(cov[0,0])

def plot_grid(grid):
    grid_reduced = grid[~np.all(grid == 0, axis=1)]
    grid_reduced = grid_reduced[:, ~np.all(grid == 0, axis=0)]
    Lx, Ly = grid_reduced.shape[1], grid_reduced.shape[0]
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    cax = ax.matshow(grid_reduced, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.axis('off')
    #plt.colorbar()
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    #plt.imshow(grid)
    #plt.savefig(f'test_{stickiness}.png')
    plt.show()


if __name__ == '__main__':

    fractal_dimension_list = []
    fractal_dimension_std_list = []
    fractal_dimension_list_big = []
    fractal_dimension_std_list_big = []
    stickiness_list = np.linspace(0.1, 1, 10)
    stickiness_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1]
    for stickiness in stickiness_list:
        frac_list = []
        frac_list_big = []

        # for i in range(5):
        grid = DLA_init(300, 500, stickiness)

        grid_reduced = grid[~np.all(grid == 0, axis=1)]
        grid_reduced = grid_reduced[:, ~np.all(grid == 0, axis=0)]
        #grid_reduced = grid
        Lx, Ly = grid_reduced.shape[1], grid_reduced.shape[0]
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot()
        #cax = ax.matshow(grid_reduced, cmap='afmhot')
        cax = ax.matshow(grid_reduced, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.axis('off')
        #plt.colorbar()
        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)
        #plt.imshow(grid)
        plt.savefig(f'test_{stickiness}.png')
        plt.close()

        frac_dim, std = get_fractal_dim(f'test_{stickiness}')
        fractal_dimension_list.append(frac_dim)
        fractal_dimension_std_list.append(std)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(stickiness_list, fractal_dimension_list, label='N=400')
    ax.fill_between(stickiness_list, np.array(fractal_dimension_list) - np.array(fractal_dimension_std_list), np.array(fractal_dimension_list) + np.array(fractal_dimension_std_list), alpha=0.5)
    #ax.plot(stickiness_list, fractal_dimension_list_big, label='N=1000')
    #ax.fill_between(stickiness_list, np.array(fractal_dimension_list_big) - np.array(fractal_dimension_std_list_big), np.array(fractal_dimension_list_big) + np.array(fractal_dimension_std_list_big), alpha=0.5)
    ax.set_xlabel('Sticking probability')
    ax.set_ylabel('Fractal dimension')
    ax.set_title('Fractal dimension for different sticking probabilities and system sizes')
    #plt.legend()
    plt.show()
