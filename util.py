import torch
import random

def prep_observation_for_model(observation, device):
    result = observation.mean(axis=-1)
    result = torch.tensor(result, device=device, dtype=torch.float32) / 255.0
    #result = result - result.mean()
    return result

def q_values_to_action(q_values):
    return q_values.argmax().item()
    #return torch.distributions.Categorical(q_values).sample().item()

def frames_to_tensor(frames):
    result = torch.stack(tuple(frames))
    return result

def random_stack_sample(frame_stack_history, batch_size, device):
    state = []
    next_state = []
    reward = []
    for _ in range(min(batch_size, len(frame_stack_history["state"]))):
        index = random.randint(0, len(frame_stack_history["state"])-1)
        state.append(frame_stack_history["state"][index])
        next_state.append(frame_stack_history["next_state"][index])
        reward.append(frame_stack_history["reward"][index].unsqueeze(0))
    if len(state) > 0:
        state = torch.cat(state, dim=0).to(device)
        next_state = torch.cat(next_state, dim=0).to(device)
        reward = torch.cat(reward, dim=0).to(device)
    #print(f"state: {state} next_state: {next_state} reward: {reward}")
    return (state, next_state, reward)

def get_sample_stack(random_sample, current_value):
    stack_sample = None
    if len(random_sample) == 0:
        stack_sample = current_value
    else:
        stack_sample = random_sample
        stack_sample = torch.cat((stack_sample, current_value), dim=0)
    return stack_sample
    