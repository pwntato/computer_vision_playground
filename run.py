#! /usr/bin/env python3

import gymnasium as gym
import pygame
import torch
import torch.nn.functional as F
import numpy as np

from model import SpaceInvadersModel

human = False
view_scale = 4

height, width = 210, 160

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SpaceInvadersModel().to(device)

pygame.init()
screen = pygame.display.set_mode((width * view_scale, height * view_scale))

env = gym.make("SpaceInvaders-v4", render_mode="rgb_array")
observation, info = env.reset()

action = 0 # no action
running = True
while running:
    # take action in environment
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

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
        # covert observation to grey scale tensor
        greyscale = torch.tensor(observation.mean(axis=-1), device=device, dtype=torch.float32)
        greyscale = greyscale / 255.0 # normalize to [0, 1]
        # make sure the tensor is square
        #greyscale = F.pad(greyscale, (0, max(0, greyscale.shape[0] - greyscale.shape[1]), 0, max(0, greyscale.shape[1] - greyscale.shape[0])))

        # don't update gradients until after the game is over
        with torch.no_grad():
            # add batch dimension (need to stack a few frames together)
            greyscale = greyscale.unsqueeze(0)
            # add channel dimension
            greyscale = greyscale.unsqueeze(0)

            # run through model (either argmax or sample from distribution)
            result = model(greyscale)
            #action = result.argmax().item()
            action = torch.distributions.Categorical(result).sample().item()

        action = env.action_space.sample() # random action, replace with ML agent

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
