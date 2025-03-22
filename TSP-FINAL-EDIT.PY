import sys

NUM_CITIES = 7
INF = float('inf')

# Adjacency matrix representing the distances between cities
# INF is used to indicate no direct connection between two cities
graph = [
    [0, 12, 10, INF, INF, INF, 12],  
    [12, 0, 8, 12, INF, INF, INF],   
    [10, 8, 0, 11, 3, INF, 9],      
    [INF, 12, 11, 0, 11, 10, INF],    
    [INF, INF, 3, 11, 0, 6, 7],      
    [INF, INF, INF, 10, 6, 0, 9],    
    [12, INF, 9, INF, 7, 9, 0]    
]

# Memoization tables to store intermediate results
# dp stores the minimum cost for a given state (visited cities mask, current city)
dp = [[-1] * NUM_CITIES for _ in range(1 << NUM_CITIES)]
# parent stores the previous city to help reconstruct the path
parent = [[-1] * NUM_CITIES for _ in range(1 << NUM_CITIES)]

def tsp(mask, pos):
    """
    Recursive function to solve the Traveling Salesperson Problem (TSP) using dynamic programming
    with bit masking.

    Args:
    mask: A bitmask representing the set of visited cities.
    pos: The current city.

    Returns:
    The minimum cost to visit all cities and return to the start.
    """
    # Base case: If all cities have been visited, return the cost to go back to the starting city
    if mask == (1 << NUM_CITIES) - 1:
        return graph[pos][0] if graph[pos][0] > 0 else INF

    # If the result for the current state is already computed, return it
    if dp[mask][pos] != -1:
        return dp[mask][pos]

    # Initialize minimum cost and the next city
    min_cost = INF
    next_city = -1
    
    # Try visiting all unvisited cities
    for city in range(NUM_CITIES):
        # If the city is not visited (bit is not set in the mask) and there's a valid edge
        if not (mask & (1 << city)) and graph[pos][city] > 0:
            # Calculate the cost of visiting this city and the remaining cities
            new_cost = graph[pos][city] + tsp(mask | (1 << city), city)
            if new_cost < min_cost:
                min_cost = new_cost
                next_city = city

    # Store the result in the dp table for future reference
    dp[mask][pos] = min_cost
    # Store the path in the parent table to reconstruct the tour later
    parent[mask][pos] = next_city
    return min_cost

def print_tour():
    """
    Function to reconstruct and print the optimal tour from the parent table.
    """
    mask, pos = 1, 0  # Start at city 0 (represented as bit 0 set in the mask)
    tour = [1]  # Start city is 1 (0-indexed in code, 1-based in output)
    
    # Traverse the parent table to reconstruct the optimal tour
    while True:
        next_city = parent[mask][pos]  # Find the next city in the path
        if next_city == -1:  # If no more cities to visit, break the loop
            break
        tour.append(next_city + 1)  # Convert to 1-based index for output
        mask |= (1 << next_city)  # Mark the next city as visited in the mask
        pos = next_city  # Move to the next city
    
    tour.append(1)  # Return to the starting city
    print("Optimal Tour:", " -> ".join(map(str, tour)))

def main():
    """
    Driver function to solve the TSP and print the results.
    """
    # Start the TSP from city 0 with only the starting city visited
    min_cost = tsp(1, 0)
    print(f"Minimum Travel Cost: {min_cost}")
    print_tour()

if __name__ == "__main__":
    main()

