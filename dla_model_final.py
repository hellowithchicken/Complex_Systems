import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pylab as pl
import os

def initialize_grid(size):
    """makes grid of chosen size"""

    # make the grid
    grid = np.zeros((size,size))

    # fill in the 'anchor point' for the structure
    grid[int(size/2)][int(size/2)] = 1

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


def walker(grid, stick_prob, time, n_walkers, curr_farthest_dist, binary=False):
    """spawns a new random walker and walks it to the structure"""

    # initiate position, randomly in a circle around the structure, radius scaling with structure size to reduce comp time
    while True:
        polar_coord = np.random.uniform(0, 2*np.pi)
        radius = curr_farthest_dist + 10
        x_pos, y_pos = int(len(grid)/2 + radius*np.cos(polar_coord)), int(len(grid)/2 + radius*np.sin(polar_coord))

        if grid[x_pos][y_pos] == 0:
            break

    while True:

        # check if the walker can adhere
        if (check_adhere(x_pos, y_pos, grid) == True) and (np.random.uniform() <= stick_prob):

            # either time-scaled value in the cell or simpy a 1 depending on whether we want the picture colored
            if binary == False:
                grid[x_pos][y_pos] = time/n_walkers*0.8 + 0.2
            else:
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


def DLA_init(gridsize, n_walkers, stick_prob = 1, binary=False, dynamic_scaler=False):
    """
    runs the DLA sim with chosen gridsize, amount of walkers and sticking probability
    if dynamic stickiness is prefered, set dynamic_scaler to 'sin', 
    'exponential', 'linear', 'dampened_sin', 'waves', or 'step_function'
    if not, set it to False. If dynamic_scaler is used, stick_prob is neglected.


    Caution: the dynamic stickiness doesn't automatically adjust to different grid sizes and walkers,
    so the stickiness_scaler function must be adapted manually.

    Set binary to True if a black/white grid is prefered, else False for colored
    colored grids are necessary for grid GIF animation.

    Returns numpy array of the created structure.
    """

    grid = initialize_grid(gridsize)
    
    # set a maximum distance for spawning which gets updated with every walker
    farthest_distance = 5
    for i in tqdm(range(n_walkers)):

        # change the sticking probability dynamically
        if dynamic_scaler is not False:
            stick_prob = stickiness_scaler(i, dynamic_scaler, n_walkers)

        farthest_distance = walker(grid, stick_prob, i, n_walkers, farthest_distance, binary=binary)

    return grid

def stickiness_scaler(time, scaling, n_walkers):
    """
    this function contains different scalings for the stickiness as a function of
    the amount of walkers and current number or walker
    it is advised to make a scaling manually, as they need to be adapted to the amount of walkers.
    """
    if scaling == 'sin':
        return 0.5*np.sin(0.0012*(time-1250))+0.501
    if scaling == 'exponential':
        return np.exp(-0.002*time)+0.003
    if scaling == 'linear':
        return 1 - time/n_walkers + 0.001
    if scaling == 'dampened_sin':
        if time > 0.5*n_walkers:
            return (0.5*np.sin(0.0012*(time-1250))+0.501)*(np.exp(-0.002*(time-2500))+0.003)
        else:
            return 0.5*np.sin(0.0012*(time-1250))+0.501

    if scaling == 'waves':
        cycles = 2
        return 0.5*np.sin(cycles*((time+1875)*2*np.pi)/n_walkers)+0.501

    if scaling == 'step_function':
        # amount of periods, 1 period meaning a stickiness of high or low setting
        periods = 3
        
        # it is advised to play around with the timings themselves here
        periods_time = [0] + list(np.logspace(0.85, 1, num=periods, base=n_walkers))
        #periods_time = [0, 1500, 2500, 5500, 6500, 10000]

        # stickiness switch
        stickiness_switch=1
        for t in range(len(periods_time)-1):
            if (time>=periods_time[t]) and (time<periods_time[t+1]):
                if stickiness_switch == 1:
                    return 0.01
                else:
                    return 1
            stickiness_switch *= -1

def get_fractal_dim(image_grid):
    """takes in a saved png of a structure and returns its fractal dimension"""

    def rgb2gray(rgb):
        """converts colored picture to gray-scale"""
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
 
    image=rgb2gray(pl.imread(f"{image_grid}.png"))

    # finding all colored pixels
    pixels=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 0:
                pixels.append((i,j))
    
    Lx=image.shape[1]
    Ly=image.shape[0]
    pixels=pl.array(pixels)
  
    
    # computing the fractal dimension
    # considering only scales in a logarithmic list
    scales=np.logspace(0.01, 9, num=10, endpoint=False, base=2)
    Ns=[]

    # looping over several scales
    for scale in scales:
       
        # computing the histogram
        H, edges=np.histogramdd(pixels, bins=(np.arange(0,Lx,scale),np.arange(0,Ly,scale)))
        Ns.append(np.sum(H>0))
    
    # linear fit, polynomial of degree 1
    coeffs, cov =np.polyfit(np.log(scales), np.log(Ns), 1, cov=True)
    
    return -coeffs[0], np.sqrt(cov[0,0])

def plot_grid(grid):
    """
    takes DLA numpy grid and visualises it
    """
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

def get_fractal_dla(grid):
    """
    takes a DLA grid and applies box counting to calculate the fractal dimension
    and standard deviation
    """
    # cut off the edges of the grid
    grid_reduced = grid[~np.all(grid == 0, axis=1)]
    grid_reduced = grid_reduced[:, ~np.all(grid_reduced == 0, axis=0)]

    # make a figure to save
    fig = plt.figure()
    ax = fig.add_subplot()

    # choose the colormap wanted, suggestions are 'binary' and 'afmhot', for fractal analysis, should be 'binary'
    cax = ax.matshow(grid_reduced, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.axis('off')
   
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    
    # save the figure for fractal analysis, high dpi is advised, more accurate but time-consuming!
    plt.savefig('interim_fractal.png', dpi=1000)
    plt.close()

    frac_dim, std = get_fractal_dim('interim_fractal')
    print(f'Found fractal dimension of {frac_dim} with standard deviation {std}.')
    os.remove('interim_fractal.png') 
    return frac_dim, std


if __name__ == '__main__':

    # choose stickiness setting, if dynamic scaling is used, this value can be anything
    stickiness = 0.3
    
    # choose grid size and amount of walkers
    size = 500
    n_walkers = 420

    # for accurate fractal analysis, binary should be True
    grid = DLA_init(size, n_walkers, stickiness, binary=True, dynamic_scaler=False)

    # save the numpy array as .npy for GIF, if wanted
    np.save(f'{n_walkers}_{stickiness}', grid)

    # cut off the edges of the grid
    grid_reduced = grid[~np.all(grid == 0, axis=1)]
    grid_reduced = grid_reduced[:, ~np.all(grid_reduced == 0, axis=0)]

    # make a figure to save
    fig = plt.figure()
    ax = fig.add_subplot()

    # choose the colormap wanted, suggestions are 'binary' and 'afmhot', for fractal analysis, should be 'binary'
    cax = ax.matshow(grid_reduced, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.axis('off')
   
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    
    # save the figure for fractal analysis, high dpi is advised, more accurate but time-consuming!
    plt.savefig(f'{n_walkers}_{stickiness}.png', dpi=1000)
    plt.close()
    
    # use the saved picture to get the fractal dimension and its std, if high dpi is used
    # for the saved figure this may take a minute
    frac_dim, std = get_fractal_dim(f'{n_walkers}_{stickiness}')
    print(f'Found fractal dimension of {frac_dim} with standard deviation {std}.')