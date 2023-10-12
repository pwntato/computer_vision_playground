#! /usr/bin/env python3

import gymnasium as gym
import pygame
import torch
import torch.nn.functional as F
import random
import numpy as np

from model import SpaceInvadersModel
from util import prep_observation_for_model, q_values_to_action

human = False
view_scale = 4

learning_rate = 1e-4
discount = 0.99
choose_random = 1.0
choose_random_min = 0.01
choose_random_decay = 0.995

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
        if random.random() < choose_random:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = model(state)
                action = q_values_to_action(q_values)

    # take action in environment
    observation, reward, terminated, truncated, info = env.step(action)
    # need to stack a few frames together...
    next_state = prep_observation_for_model(observation, device)
    score += reward

    # update model
    if not human:
        # run through model
        next_q_values = model(next_state)
        target_q_value = reward + discount * next_q_values.max().item() * (not terminated)
        target_q_value = torch.tensor(target_q_value, device=device, dtype=torch.float32, requires_grad=True)

        q_values = model(state)
        q_value = q_values[0, action]
        q_value = torch.tensor(q_value, device=device, dtype=torch.float32, requires_grad=True)

        loss = F.smooth_l1_loss(q_value, target_q_value)

        #print(f"loss: {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    state = next_state

    if terminated or truncated:
        action = 0
        observation, info = env.reset()
        state = prep_observation_for_model(observation, device)
        score = 0
        choose_random = max(choose_random_min, choose_random * choose_random_decay)

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
