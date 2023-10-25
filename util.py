import torch
import random

def prep_observation_for_model(observation, device):
    result = observation.mean(axis=-1)
    result = torch.tensor(result, device=device, dtype=torch.float32) / 255.0
    return result

def frames_to_tensor(frames):
    result = torch.stack(tuple(frames))
    return result
