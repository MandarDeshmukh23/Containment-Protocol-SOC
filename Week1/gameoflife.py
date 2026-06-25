import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 50

grid = np.random.choice([0, 1], size=(N, N), p=[0.8, 0.2])

# grid = np.zeros((50, 50), dtype=int)

# glider = np.array([
#     [0,1,0],
#     [0,0,1],
#     [1,1,1]
# ])

# grid[10:13,10:13] = glider


def count_neighbors(grid):
    return (
        np.roll(grid, 1, 0) +
        np.roll(grid, -1, 0) +
        np.roll(grid, 1, 1) +
        np.roll(grid, -1, 1) +
        np.roll(np.roll(grid, 1, 0), 1, 1) +
        np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) +
        np.roll(np.roll(grid, -1, 0), -1, 1)
    )


def update(frame):
    global grid

    neighbors = count_neighbors(grid)

    new_grid = (
        ((grid == 1) & ((neighbors == 2) | (neighbors == 3)))
        |
        ((grid == 0) & (neighbors == 3))
    )

    grid = new_grid.astype(int)

    img.set_data(grid)

    return [img]


fig, ax = plt.subplots()

img = ax.imshow(grid, cmap='binary')

ani = FuncAnimation(
    fig,
    update,
    interval=100,
    blit=True
)

plt.show()


