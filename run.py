#! /usr/bin/env python3

import gymnasium as gym
import pygame
import torch
import torch.nn.functional as F
import numpy as np

from model import SpaceInvadersModel
from util import prep_observation_for_model

human = False
view_scale = 4

learning_rate = 1e-4

height, width = 210, 160

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SpaceInvadersModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

pygame.init()
screen = pygame.display.set_mode((width * view_scale, height * view_scale))

env = gym.make("SpaceInvaders-v4", render_mode="rgb_array")
observation, info = env.reset()

action = 0 # no action
state = prep_observation_for_model(observation, device)
score = 0
running = True
while running:
    # take action in environment
    observation, reward, terminated, truncated, info = env.step(action)
    score += reward

    if terminated or truncated:
        action = 0
        observation, info = env.reset()
        state = prep_observation_for_model(observation, device)
        score = 0

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
        next_state = prep_observation_for_model(observation, device)

        # don't update gradients until after the game is over
        with torch.no_grad():
            # add batch dimension (need to stack a few frames together)
            next_state = next_state.unsqueeze(0)
            # add channel dimension
            next_state = next_state.unsqueeze(0)

            # run through model (either argmax or sample from distribution)
            result = model(next_state)
            #action = result.argmax().item()
            action = torch.distributions.Categorical(result).sample().item()
            #print(f"result: {result} action: {action} argmax: {result.argmax().item()}")

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
