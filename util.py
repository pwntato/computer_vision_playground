import torch

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
    # add batch dimension
    result = result.unsqueeze(0)
    return result
