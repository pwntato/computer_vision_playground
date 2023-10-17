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
from util import prep_observation_for_model, q_values_to_action, frames_to_tensor, random_stack_sample

# Pass action history to model

human = False
view_scale = 4

learning_rate = 1e-4
frame_count = 4
discount = 0.99
choose_random = 1.0
choose_random_min = 0.01
choose_random_decay = 0.999
skip_frames = 4
batch_size = 32
keep_frame_stack_history = 1000

height, width = 210, 160

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SpaceInvadersModel(frames=frame_count).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

pygame.init()
screen = pygame.display.set_mode(((width * view_scale) + 400, height * view_scale))
font = pygame.font.Font(pygame.font.get_default_font(), 36)

frames = deque(maxlen=frame_count)
frame_stack_history = {
    "state": deque(maxlen=keep_frame_stack_history),
    "next_state": deque(maxlen=keep_frame_stack_history),
    "reward": deque(maxlen=keep_frame_stack_history),
}

env = gym.make("SpaceInvaders-v4", render_mode="rgb_array")
observation, info = env.reset()

action = 0 # no action
state = prep_observation_for_model(observation, device)
for _ in range(frame_count):
    frames.append(state)
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
                    q_values = model(frames_to_tensor(frames).unsqueeze(0).to(device))
                    action = q_values_to_action(q_values)

    # take action in environment
    observation, reward, terminated, truncated, info = env.step(action)
    # need to stack a few frames together...
    next_state = prep_observation_for_model(observation, device)
    score += reward

    # update model
    if not human:
        # need to have random sample and compare to random sample next
        random_state_sample, random_next_state_sample, random_reward_sample = random_stack_sample(frame_stack_history, batch_size-1, device)

        state_stack = frames_to_tensor(frames).clone().unsqueeze(0).to(device)
        frame_stack_history["state"].append(state_stack)
        frames.append(next_state)
        next_state_stack = frames_to_tensor(frames).clone().unsqueeze(0).to(device)
        frame_stack_history["next_state"].append(next_state_stack)
        reward = torch.tensor(reward, device=device, dtype=torch.float32, requires_grad=True).unsqueeze(0).to(device)
        frame_stack_history["reward"].append(reward)

        state_stack_sample = None
        if len(random_state_sample) == 0:
            state_stack_sample = state_stack
        else:
            state_stack_sample = random_state_sample
            state_stack_sample = torch.cat((state_stack_sample, state_stack), dim=0)
        q_values = model(state_stack_sample)
        q_value = q_values[:, action]
        q_value.requires_grad_(True)
        #print(f"q_value: {q_value.shape}")

        # run through model
        next_state_stack_sample = None
        if len(random_next_state_sample) == 0:
            next_state_stack_sample = next_state_stack
        else:
            next_state_stack_sample = random_next_state_sample
            next_state_stack_sample = torch.cat((next_state_stack_sample, next_state_stack), dim=0)
        next_q_values = model(next_state_stack_sample)

        reward_stack_sample = None
        if len(random_reward_sample) == 0:
            reward_stack_sample = reward
        else:
            reward_stack_sample = random_reward_sample
            reward_stack_sample = torch.cat((reward_stack_sample, reward.unsqueeze(0)), dim=0)

        target_q_value = reward_stack_sample + discount * next_q_values.max().item() * (not terminated)
        if len(target_q_value.shape) > 1:
            target_q_value = target_q_value.squeeze(-1)
        target_q_value.requires_grad_(True)

        #print(f"q_value: {q_value.shape} target_q_value: {target_q_value.shape}")

        loss = F.smooth_l1_loss(q_value, target_q_value)

        #print(f"loss: {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    state = next_state

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
