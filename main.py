import numpy as np
import numba
import matplotlib.pyplot as plt

from dla_monte_carlo import dla_monte_carlo
from dla_prob_model import dla_prob_model

def main():

    dla_prob_model()

    # TODO: implement monte carlo model
    N = 100      # lattice size
    dla_monte_carlo(N)

    return


if __name__ == "__main__":
    main()
