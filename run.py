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
from util import prep_observation_for_model, q_values_to_action, frames_to_tensor, random_stack_sample, get_sample_stack
from game_util import get_human_action, render_frame

# Pass action history to model

human = False
view_scale = 4

learning_rate = 1e-4
frame_count = 4
discount = 0.99
choose_random = 1.0
choose_random_min = 0.0
choose_random_decay = 0.995#0.999
skip_frames = 4
batch_size = 64
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
frame_skip_reward = 0
running = True
while running:
    if human:
        # slow down the game
        pygame.time.wait(10)
        # handle keyboard input
        action = get_human_action(pygame.key.get_pressed())
    else:
        if frame_number % skip_frames == 0:
            frame_skip_reward = 0

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
    frame_skip_reward += reward

    # update model
    if not human and (frame_number % skip_frames == 0 or terminated or truncated):
        # need to have random sample and compare to random sample next
        random_state_sample, random_next_state_sample, random_reward_sample = random_stack_sample(frame_stack_history, batch_size-1, device)

        state_stack = frames_to_tensor(frames).clone().unsqueeze(0).to(device)
        frame_stack_history["state"].append(state_stack)
        frames.append(next_state)
        next_state_stack = frames_to_tensor(frames).clone().unsqueeze(0).to(device)
        frame_stack_history["next_state"].append(next_state_stack)
        reward = torch.tensor(reward, device=device, dtype=torch.float32, requires_grad=True).unsqueeze(0).to(device)
        frame_stack_history["reward"].append(reward)

        state_stack_sample = get_sample_stack(random_state_sample, state_stack)
        q_values = model(state_stack_sample)
        q_value = q_values[:, action]
        q_value.requires_grad_(True)
        #print(f"q_value: {q_value.shape}")

        # run through model
        next_state_stack_sample = get_sample_stack(random_next_state_sample, next_state_stack)
        next_q_values = model(next_state_stack_sample)

        reward_stack_sample = get_sample_stack(random_reward_sample, reward.unsqueeze(0))
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
    running = render_frame(view_scale, choose_random, width, screen, font, observation, score, high_score, recent_scores, tries, start_time)

env.close()
