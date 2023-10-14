#! /usr/bin/env python3

import gymnasium as gym
import pygame
import torch
import torch.nn.functional as F
import random
import numpy as np
from datetime import datetime
from collections import deque

from model import SpaceInvadersModel
from util import prep_observation_for_model, q_values_to_action, frames_to_tensor

human = False
view_scale = 4

learning_rate = 1e-4
frame_count = 4
discount = 0.99
choose_random = 0#1.0
choose_random_min = 0.01
choose_random_decay = 0.999
skip_frames = 4

height, width = 210, 160

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SpaceInvadersModel(frames=frame_count).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

pygame.init()
screen = pygame.display.set_mode(((width * view_scale) + 400, height * view_scale))
font = pygame.font.Font(pygame.font.get_default_font(), 36)

frames = deque(maxlen=frame_count)
next_frames = deque(maxlen=frame_count)

env = gym.make("SpaceInvaders-v4", render_mode="rgb_array")
observation, info = env.reset()

action = 0 # no action
state = prep_observation_for_model(observation, device)
for _ in range(frame_count):
    frames.append(state)
    next_frames.append(state)
score = 0
high_score = 0
recent_scores = []
tries = 0
frame_number = 0
start_time = datetime.now()
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
        if frame_number % skip_frames == 0:
            if random.random() < choose_random:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(frames_to_tensor(frames))
                    action = q_values_to_action(q_values)

    # take action in environment
    observation, reward, terminated, truncated, info = env.step(action)
    # need to stack a few frames together...
    next_state = prep_observation_for_model(observation, device)
    next_frames.append(next_state)
    score += reward

    # update model
    if not human:
        # run through model
        next_q_values = model(frames_to_tensor(next_frames))
        target_q_value = reward + discount * next_q_values.max().item() * (not terminated)
        target_q_value = torch.tensor(target_q_value, device=device, dtype=torch.float32, requires_grad=True)

        q_values = model(frames_to_tensor(frames))
        #print(f"q_values: {q_values}")
        q_value = q_values[0, action]
        q_value.requires_grad_(True)

        loss = F.smooth_l1_loss(q_value, target_q_value)

        #print(f"loss: {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    state = next_state
    frames = next_frames.copy()

    frame_number += 1

    if terminated or truncated:
        recent_scores.append(score)
        recent_scores = recent_scores[-100:] # keep last 100 scores
        print(f"Try {tries}: score {score} high score {high_score} rolling average {int(sum(recent_scores) / len(recent_scores))}")

        tries += 1
        frame_number = 0
        action = 0
        start_time = datetime.now()
        observation, info = env.reset()
        state = prep_observation_for_model(observation, device)
        for _ in range(frame_count):
            frames.append(state)
            next_frames.append(state)
        score = 0
        choose_random = max(choose_random_min, choose_random * choose_random_decay)

    # render frame to screen
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
            running = False

env.close()
