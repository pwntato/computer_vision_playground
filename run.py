#! /usr/bin/env python3

import gymnasium as gym
import pygame
import torch
import torch.nn.functional as F
import random
from datetime import datetime
from collections import deque

from model import AtariModel
from util import prep_observation_for_model, frames_to_tensor, get_sample_stack
from game_util import render_frame

game = "SpaceInvaders-v4" # pick from https://gymnasium.farama.org/environments/atari/complete_list/

learning_rate = 1e-4
frame_count = 4 # number of frames to stack so the model can perceive movement
discount = 0.99
choose_random = 1.0 # epsilon
choose_random_min = 0.0
choose_random_decay = 0.995
skip_frames = 1 # number of frames to skip between actions, 1 means every frame
batch_size = 50 # number of samples to take from history for training
randomize_episode_batches = True # whether to randomize the order of samples in each episode
loss_function = F.smooth_l1_loss

height, width = 210, 160
view_scale = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make(game, render_mode="rgb_array")

model = AtariModel(n_actions=env.action_space.n, frames=frame_count, hidden_layers=0).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

pygame.init()
screen = pygame.display.set_mode(((width * view_scale) + 400, height * view_scale))
font = pygame.font.Font(pygame.font.get_default_font(), 36)

frames = deque(maxlen=frame_count)

observation, info = env.reset()

episode = []
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
    # only take action every N frames
    if frame_number % skip_frames == 0:
        frame_skip_reward = 0

        if random.random() < choose_random:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = model(frames_to_tensor(frames).unsqueeze(0).to(device))
                action = q_values.argmax().item()
                #action = torch.distributions.Categorical(q_values).sample().item()
                #print(f"q_values: {q_values} action: {action}")

    # take action in environment
    observation, reward, terminated, truncated, _ = env.step(action)
    next_state = prep_observation_for_model(observation, device)
    score += reward
    frame_skip_reward += reward
    reward = frame_skip_reward
    episode.append((frames_to_tensor(frames), action, reward))

    state = next_state
    frame_number += 1

    if terminated or truncated:
        # Adjust model weights, monte carlo style
        returns = []
        g_return = 0
        for i in range(len(episode) - 1, -1, -1):
            stack, action, reward = episode[i]
            g_return = reward + discount * g_return
            returns.append((stack, action, g_return))
        if randomize_episode_batches:
            random.shuffle(returns)
        else:
            returns.reverse()

        for i in range(0, len(returns), batch_size):
            batch = returns[i:i+batch_size]
            stack_batch = torch.stack([x[0] for x in batch]).to(device)
            action_batch = torch.tensor([x[1] for x in batch], device=device)
            g_return_batch = torch.tensor([x[2] for x in batch], device=device)

            q_values = model(stack_batch)
            q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            loss = loss_function(q_values, g_return_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        episode = []

        # Game over, reset tracking variables
        if score > high_score:
            high_score = score
    
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
