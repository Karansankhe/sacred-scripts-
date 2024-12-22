import os
from groqflow import ChatGroq

# Load Groq API Key
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the A* Pathfinding Algorithm
def astar_with_llm(grid, start, goal, dynamic_obstacles):
    """
    A* pathfinding enhanced with LLM for heuristic optimization and obstacle handling.
    :param grid: 2D list representing the environment (0 for free, 1 for obstacles).
    :param start: Tuple (x, y) for the start position.
    :param goal: Tuple (x, y) for the target position.
    :param dynamic_obstacles: List of dynamic obstacle positions.
    :return: List of path coordinates.
    """
    import heapq
    import math

    # Priority queue for A*
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    def heuristic(a, b):
        """LLM-enhanced heuristic."""
        base_distance = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        response = llm.ask(f"Optimize the heuristic value for a path from {a} to {b} in a grid with dynamic obstacles: {dynamic_obstacles}.")
        llm_adjustment = float(response)  # Parse LLM response for adjustment
        return base_distance + llm_adjustment

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        neighbors = get_neighbors(current, grid)
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + 1  # Assuming uniform cost

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found

def get_neighbors(node, grid):
    """Get valid neighbors for a given node."""
    x, y = node
    neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
    return [(nx, ny) for nx, ny in neighbors if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0]
