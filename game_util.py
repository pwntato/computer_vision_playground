import pygame
import numpy as np
from datetime import datetime

def get_human_action(keys):
    if keys[pygame.K_SPACE] and keys[pygame.K_LEFT]:
        action = 5
    elif keys[pygame.K_SPACE] and keys[pygame.K_RIGHT]:
        action = 4
    elif keys[pygame.K_LEFT]:
        action = 3
    elif keys[pygame.K_RIGHT]:
        action = 2
    elif keys[pygame.K_SPACE]:
        action = 1
    else:
        action = 0
    return action

def render_frame(view_scale, choose_random, width, screen, font, observation, score, high_score, recent_scores, tries, start_time):
    screen.fill((0,0,0))

    observation = observation.swapaxes(0, 1)
    observation = np.repeat(observation, view_scale, axis=0)
    observation = np.repeat(observation, view_scale, axis=1)
    surface = pygame.surfarray.make_surface(observation)

    screen.blit(surface, (0, 0))

    text_offset = width * view_scale

    text_surface = font.render(f"Score: {int(score)}", True, (255, 255, 255))
    screen.blit(text_surface, dest=(text_offset, 50))

    text_surface = font.render(f"Time: {datetime.now() - start_time}", True, (255, 255, 255))
    screen.blit(text_surface, dest=(text_offset, 100))

    if score > high_score:
      high_score = score
    text_surface = font.render(f"High score: {int(high_score)}", True, (255, 255, 255))
    screen.blit(text_surface, dest=(text_offset, 150))

    text_surface = font.render(f"Tries: {tries}", True, (255, 255, 255))
    screen.blit(text_surface, dest=(text_offset, 200))

    text_surface = font.render(f"Choose random: {int(choose_random * 100)}%", True, (255, 255, 255))
    screen.blit(text_surface, dest=(text_offset, 250))

    recent_scores = recent_scores[-100:]
    if len(recent_scores) > 0:
      text_surface = font.render(
          f"Rolling average: {int(sum(recent_scores) / len(recent_scores))}", 
          True, 
          (255, 255, 255)
        )
      screen.blit(text_surface, dest=(text_offset, 300))

    pygame.display.flip()

    # handle closing the window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
        
    return True
