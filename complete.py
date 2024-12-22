import random
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import heapq

# Constants for grid dimensions
GRID_ROWS = 9
GRID_COLS = 10

# Pathfinding visualization variables
grid = []
open_set = []
closed_set = set()
path = []
walls = []

class Node:
    def __init__(self, row, col, g, h, f, parent=None):
        self.row = row
        self.col = col
        self.g = g  # Cost to reach this node
        self.h = h  # Estimated cost to reach the goal
        self.f = f  # Total cost (g + h)
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f  # This is required to use heapq to get the node with the smallest f

def create_grid():
    return np.zeros((GRID_ROWS, GRID_COLS), dtype=int)

def generate_random_walls(num_walls):
    available_positions = [(i, j) for i in range(GRID_ROWS) for j in range(GRID_COLS)]
    wall_positions = random.sample(available_positions, num_walls)
    walls = set()
    for row, col in wall_positions:
        walls.add((row, col))
    return walls

def get_neighbors(node):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for dr, dc in directions:
        r, c = node.row + dr, node.col + dc
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            neighbors.append((r, c))
    return neighbors

def calculate_heuristic(node, goal_node):
    # Manhattan distance
    return abs(node.row - goal_node.row) + abs(node.col - goal_node.col)

def a_star(start, goal):
    global open_set, closed_set, path
    path.clear()  # Clear previous path

    # Initialize the open set and closed set
    open_set = []
    heapq.heappush(open_set, (start.f, start))  # Use a heap to efficiently get the node with the lowest f
    closed_set = set()

    while open_set:
        _, current_node = heapq.heappop(open_set)  # Pop the node with the lowest f value
        if (current_node.row, current_node.col) in closed_set:
            continue

        closed_set.add((current_node.row, current_node.col))

        # If we reached the goal
        if current_node.row == goal.row and current_node.col == goal.col:
            # Reconstruct the path from goal to start
            while current_node:
                path.append((current_node.row, current_node.col))
                current_node = current_node.parent
            path.reverse()
            return True

        # Check neighbors
        for neighbor_row, neighbor_col in get_neighbors(current_node):
            if (neighbor_row, neighbor_col) in walls or (neighbor_row, neighbor_col) in closed_set:
                continue

            g = current_node.g + 1  # Cost from start to the neighbor
            h = calculate_heuristic(Node(neighbor_row, neighbor_col, 0, 0, 0), goal)
            f = g + h

            neighbor_node = Node(neighbor_row, neighbor_col, g, h, f, current_node)

            # Add the neighbor to the open set
            heapq.heappush(open_set, (f, neighbor_node))

    return False  # No path found

def visualize_grid(start, goal):
    fig, ax = plt.subplots(figsize=(10, 9))

    # Draw grid
    ax.set_xticks(np.arange(0, GRID_COLS, 1))
    ax.set_yticks(np.arange(0, GRID_ROWS, 1))
    ax.grid(True)

    # Mark walls as black
    for row, col in walls:
        ax.add_patch(plt.Rectangle((col, row), 1, 1, color="black"))

    # Mark start node
    ax.add_patch(plt.Rectangle((start.col, start.row), 1, 1, color="green"))

    # Mark end node
    ax.add_patch(plt.Rectangle((goal.col, goal.row), 1, 1, color="red"))

    # Mark path in blue
    for (row, col) in path:
        ax.add_patch(plt.Rectangle((col, row), 1, 1, color="blue"))

    plt.gca().invert_yaxis()  # To match the grid's row and column system
    plt.show()

def main():
    st.set_page_config(page_title="Pathfinding Visualization", page_icon=":package:")
    st.title("Pathfinding Visualization")

    # Sidebar inputs
    start_node_input = st.sidebar.number_input("Enter the start node number", min_value=11, max_value=99)
    end_node_input = st.sidebar.number_input("Enter the end node number", min_value=11, max_value=99)
    num_walls = st.sidebar.number_input("Enter the number of walls", min_value=0, max_value=50)

    # Generate random walls
    global walls
    walls = generate_random_walls(num_walls)
    st.sidebar.write(f"Walls placed at: {sorted(list(walls))}")

    # Get the row and column of the start and end nodes
    start_row = (start_node_input // 10) - 1
    start_col = start_node_input - (10 + start_row * 10)
    end_row = (end_node_input // 10) - 1
    end_col = end_node_input - (10 + end_row * 10)

    # Create start and end nodes
    start = Node(start_row, start_col, 0, 0, 0)
    goal = Node(end_row, end_col, 0, 0, 0)

    # Run A* pathfinding
    if a_star(start, goal):
        st.write(f"Path found: {path}")
    else:
        st.write("No path found.")

    visualize_grid(start, goal)

if __name__ == "__main__":
    main()
