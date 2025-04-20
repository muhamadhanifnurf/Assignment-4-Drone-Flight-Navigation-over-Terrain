import time
from heapq import heappush, heappop
import matplotlib.pyplot as plt
import numpy as np

# Grid dengan elevasi dan zona larangan (#)
terrain = [
    ["S", "1", "2", "#", "2"],
    ["2", "3", "2", "2", "2"],
    ["#", "3", "#", "2", "#"],
    ["1", "1", "2", "3", "2"],
    ["1", "2", "3", "4", "G"]
]

# Konversi nilai string ke int untuk biaya kecuali 'S', 'G', dan '#'
def elevation_cost(cell):
    if cell == "S" or cell == "G":
        return 1
    elif cell == "#":
        return float('inf')
    else:
        return int(cell)

# Temukan posisi start dan goal
start4, goal4 = None, None
for i in range(len(terrain)):
    for j in range(len(terrain[0])):
        if terrain[i][j] == "S":
            start4 = (i, j)
        elif terrain[i][j] == "G":
            goal4 = (i, j)

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def a_star_terrain(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heappush(open_set, (manhattan(start, goal), 0, start, [start]))
    visited = set()
    node_count = 0

    start_time = time.time()

    while open_set:
        f, g, current, path = heappop(open_set)
        node_count += 1
        if current == goal:
            elapsed_time = (time.time() - start_time) * 1000
            return path, g, node_count, elapsed_time
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] != "#" and (nx, ny) not in visited:
                    step_cost = elevation_cost(grid[nx][ny])
                    new_g = g + step_cost
                    new_f = new_g + manhattan((nx, ny), goal)
                    heappush(open_set, (new_f, new_g, (nx, ny), path + [(nx, ny)]))
    return None, float('inf'), node_count, 0

def gbfs_terrain(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heappush(open_set, (manhattan(start, goal), start, [start]))
    visited = set()
    node_count = 0

    start_time = time.time()

    while open_set:
        h, current, path = heappop(open_set)
        node_count += 1
        if current == goal:
            elapsed_time = (time.time() - start_time) * 1000
            return path, node_count, elapsed_time
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] != "#" and (nx, ny) not in visited:
                    heappush(open_set, (manhattan((nx, ny), goal), (nx, ny), path + [(nx, ny)]))
    return None, node_count, 0

def visualize_grid(grid, path):
    """
    Visualisasi grid dengan jalur yang ditemukan.
    """
    visual_grid = [row[:] for row in grid]  # Salin grid
    for x, y in path:
        if visual_grid[x][y] not in ("S", "G"):
            visual_grid[x][y] = "*"
    print("\nGrid Visualization:")
    for row in visual_grid:
        print(" ".join(row))

def plot_heatmap(grid, path):
    """
    Visualisasi heatmap untuk grid dengan jalur.
    """
    heatmap = np.zeros((len(grid), len(grid[0])))
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "#":
                heatmap[i][j] = -1  # Zona larangan
            elif grid[i][j] in ("S", "G"):
                heatmap[i][j] = 0
            else:
                heatmap[i][j] = int(grid[i][j])
    for x, y in path:
        heatmap[x][y] = 0.5  # Tandai jalur
    plt.imshow(heatmap, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Elevation")
    plt.title("Terrain Heatmap with Path")
    plt.show()

# Menjalankan algoritma A* untuk mencari jalur dengan mempertimbangkan biaya elevasi
start_time = time.perf_counter()
path4_a_star, cost4_a_star, nodes4_a_star, elapsed4_a_star = a_star_terrain(terrain, start4, goal4)
end_time = time.perf_counter()
elapsed4_a_star = (end_time - start_time) * 1000  # Hitung waktu eksekusi dalam milidetik

print("\n=== A* Search ===")
if path4_a_star:
    print("Path:", path4_a_star)
    print("Cost:", cost4_a_star)
    print("Nodes visited:", nodes4_a_star)
    print("Time (ms):", round(elapsed4_a_star, 4))  # Waktu dengan presisi tinggi
    visualize_grid(terrain, path4_a_star)
    plot_heatmap(terrain, path4_a_star)
else:
    print("No path found using A*.")

# Menjalankan algoritma GBFS untuk mencari jalur berdasarkan heuristik tanpa mempertimbangkan biaya elevasi
start_time = time.perf_counter()
path4_gbfs, nodes4_gbfs, elapsed4_gbfs = gbfs_terrain(terrain, start4, goal4)
end_time = time.perf_counter()
elapsed4_gbfs = (end_time - start_time) * 1000  # Hitung waktu eksekusi dalam milidetik

print("\n=== Greedy Best-First Search (GBFS) ===")
if path4_gbfs:
    print("Path:", path4_gbfs)
    print("Nodes visited:", nodes4_gbfs)
    print("Time (ms):", round(elapsed4_gbfs, 4))  # Waktu dengan presisi tinggi
    visualize_grid(terrain, path4_gbfs)
    plot_heatmap(terrain, path4_gbfs)
else:
    print("No path found using GBFS.")