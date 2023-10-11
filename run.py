#! /usr/bin/env python3

import gymnasium as gym
import pygame
import torch
import numpy as np

human = True
view_scale = 4

height, width = 210, 160

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pygame.init()
screen = pygame.display.set_mode((width * view_scale, height * view_scale))

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

    # covert observation to grey scale tensor
    greyscale = torch.tensor(observation.mean(axis=-1), device=device, dtype=torch.float32)

    # show frame
    observation = observation.swapaxes(0, 1)
    observation = np.repeat(observation, view_scale, axis=0)
    observation = np.repeat(observation, view_scale, axis=1)
    surface = pygame.surfarray.make_surface(observation)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # handle closing the window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            running = False

env.close()
