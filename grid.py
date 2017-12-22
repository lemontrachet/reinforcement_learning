import numpy as np
import pygame
from time import sleep



def neighbours(x, y):
    return [(a, b) for (a, b) in
            [(x - 1, y),
             (x - 2, y),
             (x - 3, y),
             (x + 1, y),
             (x + 2, y),
             (x + 3, y),
             (x, y - 1),
             (x, y - 2),
             (x, y - 3),
             (x, y + 1),
             (x, y + 2),
             (x, y + 3)]
            if 0 <= a < DIM
            and 0 <= b < DIM]

def calc_value(i, grid, threshold):
    current = grid[i[0]]
    neighbours = sum([grid[n] for n in i[1]])
    if neighbours > threshold and np.random.random() > 0.3:
        return current * 1.5 + np.random.random()
    elif np.random.random() > 0.99:
        return current + np.random.random()
    return current * 0.7


def grow(grid, threshold):
    indices = [((x, y), neighbours(x, y)) for x in range(DIM) for y in range(DIM)]
    return np.array([calc_value(i, grid, threshold) for i in indices]).reshape(DIM, DIM)


if __name__ == "__main__":
    DIM = 600
    grid = np.random.randn(DIM, DIM) - 2.5
    grid[grid < 0.01] = 0

    pygame.init()
    display = pygame.display.set_mode((600, 600))
    grid_d = 255 * grid / grid.max()
    surf = pygame.surfarray.make_surface(grid_d)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        threshold = np.mean(grid) * 50
        print(threshold)
        grid1 = grow(grid, threshold) / 1000
        grid_d = 255 * grid1 / grid1.max()
        surf = pygame.surfarray.make_surface(grid_d)
        display.blit(surf, (0, 0))
        pygame.display.update()
        grid = grid1
        sleep(0.1)
    pygame.quit()
