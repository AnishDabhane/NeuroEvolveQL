from neat.genome import Genome
from neat.gene_history import GeneHistory
import pygame
import random

gh = GeneHistory(5, 4)

# inputs = [[0.1, 0.2, 0.3, 0.4, 0.5],[0.3, 0.12, 0.39, 0.44, 0.85]]
# target=[[0.153,0.76,0.124,0.28],[0.13,0.63,0.224,0.281]]

inputs = [[round(random.uniform(0,1),2) for _ in range(5)] for _ in range(32)]
target = [[0.2 if random.uniform(0,1)<0.5 else 0.7 for _ in range(4)] for _ in range(32)]

g = Genome(gh)
for i in range(50):
    g.mutate()

print(f"INPUTS:",inputs)
pred=g.feed_forward(inputs)
print(f"OUTPUTS:",pred)
print(f"TARGET:",target)
mse=g.mean_squared_error(target,pred)
print("Mean squared error:", mse)
print(g)

screen = pygame.display.set_mode((600, 600))
running = True
while running:
    for event in pygame.event.get():
        if event == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_b:
                g.backpropogate(inputs, target)
            if event.key == pygame.K_a:
                g.add_node()
            if event.key == pygame.K_p:
                print(g)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:
        running = False

    # W Show
    screen.fill((255, 255, 255))
    g.show(screen)
    pygame.display.update()

pygame.quit()