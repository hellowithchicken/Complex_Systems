import numpy as np
import numba
import matplotlib.pyplot as plt
import pylab as pl


def get_new_walker(grid):

    # there's no difference in sticking probability because we have fully random walkers that are also spawned randomly, 
    # should spawn them from the outside at least to see a diff I guess
    while True:
        #walker_x, walker_y = np.random.randint(0, len(grid)-1, 2)
        #walker_x, walker_y = np.random.randint(0, len(grid)-1, 1), len(grid)-1
        polar_coord = np.random.uniform(0, 2*np.pi)
        walker_x, walker_y = int(len(grid)/2+(len(grid)/2)*np.cos(polar_coord)), int(len(grid)/2+(len(grid)/2)*np.sin(polar_coord))
        if grid[walker_x, walker_y] == 1:
            continue
        else:
            return walker_x, walker_y

def walk(walker_x, walker_y, N, grid):
    direction = np.random.randint(0, 4)
    if direction == 0:
        walker_y+=1
    if direction == 1:
        walker_y-=1
    if direction == 2:
        walker_x+=1
    if direction == 3:
        walker_x-=1

    if walker_y >= N or walker_y < 0:
        walker_x, walker_y = get_new_walker(grid)
        return walk(walker_x, walker_y, N, grid)
    if walker_x >= N:
        walker_x = 0
    if walker_x < 0:
        walker_x = N-1

    return walker_x, walker_y

def next_to_structure(walker_x, walker_y, grid):

    if grid[walker_x, walker_y] == 1:
        return False
    if walker_y < len(grid) - 1:
        if grid[walker_x, walker_y + 1] == 1:
            return True
    if walker_y > 0:
        if grid[walker_x, walker_y - 1] == 1:
            return True
    if walker_x < len(grid) - 1:
        if grid[walker_x + 1, walker_y] == 1:
            return True
    if walker_x > 0:
        if grid[walker_x - 1, walker_y] == 1:
            return True
    return False

def dla_monte_carlo(N, num_walkers = 5, sticking_prob = 1):
    """DLA model using Monte Carlo simulation
    Args:
        N (int): lattice size
        num_walkers(int): number of walkers that will be used
    """
    print("DLA Monte Carlo model with N = {}.".format(N))
    print("")

    grid = np.zeros((N, N))
    struct_x = int(N/2)
    struct_y = int(N/2)
    grid[struct_x, struct_y] = 1

    for _ in range(num_walkers):
        walker_x, walker_y = get_new_walker(grid)
        stuck = False

        while not stuck:
            if next_to_structure(walker_x, walker_y, grid):
                if np.random.uniform(0, 1) < sticking_prob:
                    stuck = True
                    grid[walker_x, walker_y] = 1
            else:
              walker_x, walker_y = walk(walker_x, walker_y, N, grid)
    return grid

def get_frac_dim(image_file):

    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
 
    # image=rgb2gray(pl.imread(f"{image_file}"))
    # plt.plot(image)
    # plt.show()
    #image=pl.imread(f"{image_file}")
    image = image_file
    # finding all the non-zero pixels
    pixels=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j]>0:
                pixels.append((i,j))
    
    Lx=image.shape[1]
    Ly=image.shape[0]
    print (Lx, Ly)
    pixels=pl.array(pixels)
    print (pixels.shape)
    
    # computing the fractal dimension
    #considering only scales in a logarithmic list
    scales=np.logspace(0.01, 1, num=10, endpoint=False, base=2)
    Ns=[]
    # looping over several scales
    for scale in scales:
        print ("======= Scale :",scale)
        # computing the histogram
        H, edges=np.histogramdd(pixels, bins=(np.arange(0,Lx,scale),np.arange(0,Ly,scale)))
        H, edges=np.histogramdd(pixels, bins=(np.arange(0,Lx,scale),np.arange(0,Ly,scale)))
        Ns.append(np.sum(H>0))
    
    # linear fit, polynomial of degree 1
    coeffs=np.polyfit(np.log(scales), np.log(Ns), 1)
    
    # pl.plot(np.log(scales),np.log(Ns), 'o', mfc='none')
    # pl.plot(np.log(scales), np.polyval(coeffs,np.log(scales)))
    # pl.xlabel('log $\epsilon$')
    # pl.ylabel('log N')
    # pl.savefig('sierpinski_dimension.pdf')
    
    print ("The fractal dimension is", -coeffs[0]) #the fractal dimension is the OPPOSITE of the fitting coefficient
    #np.savetxt("scaling.txt", list(zip(scales,Ns)))

if __name__ == "__main__":

    print("in main")
    grid = dla_monte_carlo(100, 1450, 0.01)

    # get rid of all zero rows and columns for the fractal dimension box count

    grid = grid[~np.all(grid == 0, axis=1)]
    grid = grid[:, ~np.all(grid == 0, axis=0)]

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    cax = ax.matshow(grid, cmap='binary')
    ax.set_facecolor('red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.axis('off')
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    #plt.imshow(grid)
    plt.savefig('test.png')


    get_frac_dim(grid)

    
