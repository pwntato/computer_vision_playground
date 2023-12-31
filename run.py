#! /usr/bin/env python3

import gymnasium as gym
import pygame
import torch
import torch.nn.functional as F
import random
from datetime import datetime
from collections import deque
import cv2

from model import AtariModel
from util import prep_observation_for_model, frames_to_tensor
from game_util import render_frame

game = "ALE/MsPacman-v5" # pick from https://gymnasium.farama.org/environments/atari/complete_list/

render = True                       # whether to render the game to the screen, False is faster, but won't create video
record_tries = 0                    # how many tries to record, 0 for no video which is faster
save_best_as = "best.pt"            # where to save the best model, None to not save
load_model = None                   # where to load a model from, None to not load, loading disables training

learning_rate = 1e-4                # how fast to learn
frame_count = 4                     # number of frames to stack so the model can perceive movement
discount = 0.95                     # gamma, how much to discount future rewards from current actions
choose_random = 1.0                 # epsilon, how often to choose a random action
choose_random_min = 0.01            # minimum epsilon, after this it will always choose the best action
choose_random_decay = 0.995         # how much to decay epsilon after each episode (multiplied, not subtracted)
skip_frames = 1                     # number of frames to skip between actions, 1 means every frame
batch_size = 100                    # number of samples to process at once
randomize_episode_batches = True    # whether to randomize the order of samples in each episode
loss_function = F.mse_loss          # loss function to use
optimizer = torch.optim.SGD         # optimizer to use
hidden_layers = 1                   # number of hidden linear layers in the model
no_score_penalty = 1                # how much to penalize for not scoring

height, width = 210, 160
view_scale = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make(game, render_mode="rgb_array")

model = AtariModel(n_actions=env.action_space.n, frames=frame_count, hidden_layers=hidden_layers).to(device)
optimizer = optimizer(model.parameters(), lr=learning_rate)

if load_model is not None:
    save_best_as = None
    choose_random = 0
    model.load_state_dict(torch.load(load_model))

pygame.init()
screen_width = (width * view_scale) + 400
screen_height = height * view_scale
screen = pygame.display.set_mode((screen_width, screen_height)) if render else None
video_writer = cv2.VideoWriter("atari.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 75, (screen_width, screen_height)) if record_tries > 0 and render else None
font = pygame.font.Font(pygame.font.get_default_font(), 36)

frames = deque(maxlen=frame_count)

observation, info = env.reset()

episode = [] # list of (state, action, reward) tuples for monte carlo learning
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
    # only take action every skip_frames frames
    if frame_number % skip_frames == 0:
        frame_skip_reward = 0

        if random.random() < choose_random:
            # choose random action
            action = env.action_space.sample()
        else:
            # choose best action
            with torch.no_grad():
                q_values = model(frames_to_tensor(frames).unsqueeze(0).to(device))
                action = q_values.argmax().item()

    # take action in environment
    observation, reward, terminated, truncated, _ = env.step(action)
    next_state = prep_observation_for_model(observation, device)
    score += reward
    if reward == 0:
        reward = -no_score_penalty     # punish for not scoring
    frame_skip_reward += reward     # accumulate reward over skipped frames
    reward = frame_skip_reward
    episode.append((frames_to_tensor(frames), action, reward))

    frames.append(state)
    state = next_state
    frame_number += 1

    if terminated or truncated:
        if score > high_score:
            high_score = score

            # save best model after done choosing random, BEFORE updating weights
            if save_best_as is not None and choose_random <= choose_random_min:
                torch.save(model.state_dict(), save_best_as)

        if load_model is None:
            # adjust model weights, monte carlo style
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

        # print stats to console
        recent_scores.append(score)
        recent_scores = recent_scores[-100:] # keep last 100 scores
        print(f"Try {tries}: score {score} high score {high_score} rolling average {int(sum(recent_scores) / len(recent_scores))}")

        # game over, reset tracking variables
        episode = []
        tries += 1
        frame_number = 0
        action = 0
        start_time = datetime.now()
        observation, info = env.reset()
        state = prep_observation_for_model(observation, device)
        for _ in range(frame_count):
            frames.append(state)
        score = 0

        # decay choose_random
        choose_random = choose_random * choose_random_decay
        if choose_random > 0 and choose_random <= choose_random_min:
            choose_random = 0
            high_score = 0  # reset high score when done choosing random

    if render:
        # render frame to screen
        running = render_frame(view_scale, choose_random, width, screen, font, observation, score, high_score, recent_scores, tries, start_time)

        # record video
        if video_writer is not None and tries <= record_tries:
            frame = pygame.display.get_surface()
            view = pygame.surfarray.array3d(frame)
            view = view.transpose([1, 0, 2])
            image = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
            video_writer.write(image)

            if tries >= record_tries:
                video_writer.release()
                video_writer = None

env.close()
