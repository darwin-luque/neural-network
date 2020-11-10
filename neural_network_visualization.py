import pygame

screen = {'height': 600, 'width': 600}
win = pygame.display.set_mode((screen['width'], screen['height']))


run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break
    win.fill((255, 255, 255))
    pygame.display.update()

pygame.quit()