import time
import pygame
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use('TkAgg')

WINDOW_SIZE = [800, 800]
num_particles = 150
steps = 1000000
G = 6.67e-11
# G = 6.67e-8


class Particle:
    def __init__(self, x, y, mass):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.mass = mass

    def acceleration(self, other_p):
        dx = other_p.x - self.x
        dy = other_p.y - self.y
        dsq = dx * dx + dy * dy  # distance squared
        if dsq < (3 * AU) ** 2:
            dsq = (3 * AU) ** 2
        dr = np.sqrt(dsq)
        f = G * self.mass * other_p.mass / dsq if dsq > 1e-10 else 0
        # f = ma, so... a = f / m
        self.ax = f * dx / dr
        self.ay = f * dy / dr

    def update(self, other_p):
        # calc acc
        self.acceleration(other_p)
        self.vx += self.ax
        self.vy += self.ay
        self.x += self.vx
        self.y += self.vy


pygame.init()
window = pygame.display.set_mode((800, 800))
planets = []
AU = 1.5e11
X_LIM = 180 * AU
Y_LIM = 180 * AU
xs = np.random.uniform(low=0, high=X_LIM, size=(num_particles,))
ys = np.random.uniform(low=0, high=Y_LIM, size=(num_particles,))
# masses = np.random.normal(low=1e20, high=1e24, size=(num_particles,))
# masses = np.abs(np.random.normal(loc=1e15, scale=1e8, size=num_particles))
masses = np.ones(num_particles) * 1e20
for i in range(num_particles):
    planets.append(Particle(xs[i], ys[i], masses[i]))

FPS_AIM = 60
run = True
while run:
    start_time = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    window.fill(0)

    rect = pygame.Rect(window.get_rect().center, (0, 0)).inflate(*([min(window.get_size()) // 2] * 2))
    for p in planets:
        for p_ in planets:
            if p is p_:
                continue
            else:
                p.update(p_)
        # map x, y between 0 and windowsize
        # screenx = p.x /

        winx = int((((int(p.x) - 0) * float(800)) / X_LIM) + 0)
        winy = int((((int(p.y) - 0) * float(800)) / Y_LIM) + 0)
        window.set_at((winx, winy), (255, 0, 0))
        window.set_at((winx+1, winy), (255, 0, 0))
        window.set_at((winx - 1, winy), (255, 0, 0))
        window.set_at((winx, winy + 1), (255, 0, 0))
        window.set_at((winx, winy - 1), (255, 0, 0))
        # window.set_at((int(p.x), int(p.y)), (255, 0, 0))

    pygame.display.flip()

    end_time = time.time()
    frame_duration = end_time - start_time
    if frame_duration < 1 / 60:
        time.sleep(1 / 60 - frame_duration)

pygame.quit()
exit()
