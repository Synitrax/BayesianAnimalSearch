import numpy as np
import scipy.ndimage as ndimage

# 1. Setup the Environment
grid_size = 10
# Initialize uniform prior: Every cell has an equal 1% chance
prior = np.full((grid_size, grid_size), 1.0 / (grid_size**2))

# The "True" location (Unknown to the model)
animal_loc = (3, 7) 

# Detection probability (Sensor effectiveness in dense forest)
# Only 30% chance to see it even if we are in the right square
q = 0.3 

def search_cell(coord, true_loc, detection_prob):
    """Simulates a physical search of a cell."""
    if coord == true_loc:
        return np.random.random() < detection_prob
    return False

def kinetic_update(grid):
    """Simulates animal movement by 'blurring' the probability."""
    # The animal moves slightly, so probability 'leaks' to neighbors
    return ndimage.gaussian_filter(grid, sigma=0.5)

# 2. Run a Search Step
target_cell = (3, 7) # We decide to search here
found = search_cell(target_cell, animal_loc, q)

if not found:
    # 3. Apply Bayesian Update
    # Probability of not finding it in the target cell
    p_not_found_given_here = 1 - q
    
    # Update the searched cell
    prior[target_cell] *= p_not_found_given_here
    
    # Renormalize: All probabilities must sum to 1
    prior /= prior.sum()
    
    # 4. Animal moves (Kinetic Update)
    prior = kinetic_update(prior)

print(f"New Max Probability Cell: {np.unravel_index(prior.argmax(), prior.shape)}")
