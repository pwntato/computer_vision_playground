import torch
import random

def prep_observation_for_model(observation, device):
    result = observation.mean(axis=-1)
    result = torch.tensor(result, device=device, dtype=torch.float32) / 255.0
    #result = result - result.mean()
    return result

def frames_to_tensor(frames):
    result = torch.stack(tuple(frames))
    return result

def get_sample_stack(random_sample, current_value):
    stack_sample = None
    if len(random_sample) == 0:
        stack_sample = current_value
    else:
        stack_sample = random_sample
        stack_sample = torch.cat((stack_sample, current_value), dim=0)
    return stack_sample
    