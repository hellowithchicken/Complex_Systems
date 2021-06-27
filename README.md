# Generating urban-like networks with DLA
### Ignas Krikštaponis, Xiang Li, Sebastiaan Kruize, Youri Moll


![](figures/process.png)

## Getting started

To install all necessary packages, run:
```
  pip3 install requirements.txt
```

## Modules
- `dla_model_final.py` - functions for generating DLA arrays (including dynamic stickiness) and calculating fractal dimension
- `networks.py` - functions for generating networks from DLA, retrieving real-life networks, converting city networks into fractals and performing analysis
- `animate_grid.py` - function for generating animated gifs from DLA numpy arrays
- `fractal_city.py` - code for retrieving fractals from real-life city networks.

### Other

- `experiments.py` - code used for running experiments with changing stickiness and number of walkers; and saving the results into .npy files
- `result_analysis.py` - code used for reading .npy DLA arrays in  `results` folder and extracting network statistics from them
- `analyse_cities.py` - code used for gathring network statistics for selected real-world cities.

### Results

Saved numpy arrays generated with experiments can be found in the `results` folder:

- `changing_stickness_250_walkers` DLA numpy arrays generated with `experiments.py` changing stickiness with 250 walkers. file format `x,y.npy`: x - stickiness, y - simulation number
- `changing_stickness_400_walkers` DLA numpy arrays generated with `experiments.py` changing stickiness with 400 walkers. file format `x,y.npy`: x - stickiness, y - simulation number
- `changing_walkers` DLA numpy arrays generated with `experiments.py` changing number of walkers with stickiness of 1. file format `x,y.npy`: x - number of walkers, y - simulation number
- `real_cities` - csv files with network statistics for selected real-world cities generated with `analyse_cities.py`
- `periods` - a number of DLA .npy arrays generated with periodic stickiness.

## Usage examples

Notebook with usage examples of the code can be found in `examples.ipynb` 
