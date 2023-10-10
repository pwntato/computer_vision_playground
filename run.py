#! /usr/bin/env python3

import gymnasium as gym
import pygame
import numpy as np

human = True

height, width = 210, 160

pygame.init()
screen = pygame.display.set_mode((width, height))

env = gym.make("SpaceInvaders-v4", render_mode="rgb_array")
observation, info = env.reset()

running = True
while running:
    action = 0 # no action

    if human:
        # slow down the game
        pygame.time.wait(10)

        # handle keyboard input
        keys = pygame.key.get_pressed()
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
    else:
        action = env.action_space.sample() # random action, replace with ML agent

    # take action in environment
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    # show frame
    surface = pygame.surfarray.make_surface(observation.swapaxes(0, 1))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # handle closing the window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            running = False

env.close()
