import numpy as np
import matplotlib.pyplot as plt

# Adjacency matrix for TSP problem
INF = float('inf')
graph = [
    [0, 12, 10, INF, INF, INF, 12],
    [12, 0, 8, 12, INF, INF, INF],
    [10, 8, 0, 11, 3, INF, 9],
    [INF, 12, 11, 0, 11, 10, INF],
    [INF, INF, 3, 11, 0, 6, 7],
    [INF, INF, INF, 10, 6, 0, 9],
    [12, INF, 9, INF, 7, 9, 0]
]

NUM_CITIES = len(graph)
NUM_NODES = NUM_CITIES  # Match number of neurons to cities
learning_rate = 0.1  # Initial learning rate
radius = 3  # Initial neighborhood radius
iterations = 50  # Number of training iterations

# STEP 1: Represent cities as random 2D coordinates for visualization
city_coords = np.random.rand(NUM_CITIES, 2) * 100  # Randomly generate city coordinates (scaled to 100x100)

# STEP 2: Initialize neurons (SOM nodes) as random positions in 2D space
weights = np.random.rand(NUM_NODES, 2)  # Each neuron is a point in 2D space

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b)**2))

# STEP 3: SOM training loop
for t in range(iterations):
    # Decay learning rate and neighborhood radius
    alpha = learning_rate * np.exp(-t / iterations)
    sigma = radius * np.exp(-t / iterations)
    
    # Select a random city as input
    city_idx = np.random.choice(NUM_CITIES)
    city_pos = city_coords[city_idx]  # Use city coordinates as the input to the SOM
    
    # Find the winning neuron (neuron closest to the input city)
    distances = [euclidean_distance(city_pos, weights[i]) for i in range(NUM_NODES)]
    winner_idx = np.argmin(distances)  # Index of the winning neuron
    
    # Update the weights for the winner and its neighbors
    for i in range(NUM_NODES):
        dist_to_winner = abs(i - winner_idx)
        if dist_to_winner < sigma:  # Neighborhood function
            # Influence of the update decreases with distance from the winner
            influence = np.exp(-dist_to_winner**2 / (2 * sigma**2))
            weights[i] += alpha * influence * (city_pos - weights[i])  # Update weights

# STEP 4: Form the TSP route by mapping neurons to cities
route = sorted(range(NUM_NODES), key=lambda x: weights[x, 0])
route = [x + 1 for x in route]  # Convert to 1-based city indices

# Ensure starting city is City 1
route = [1] + [city for city in route if city != 1] + [1]  # Start and end at City 1
