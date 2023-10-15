import torch
import random

def prep_observation_for_model(observation, device):
    result = observation.mean(axis=-1)
    result = torch.tensor(result, device=device, dtype=torch.float32) / 255.0
    #result = result - result.mean()
    return result

def q_values_to_action(q_values):
    #return q_values.argmax().item()
    return torch.distributions.Categorical(q_values).sample().item()

def frames_to_tensor(frames):
    result = torch.stack(tuple(frames))
    return result

def random_stack_sample(frame_stack_history, batch_size, device):
    result = []
    for _ in range(min(batch_size, len(frame_stack_history))):
        result.append(random.choice(frame_stack_history[:-1]))
    if len(result) > 0:
        result = torch.stack(result).to(device)
    return result
