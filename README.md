# Atari Computer Vison Model Playground
[Ms Pacman demo video](https://youtu.be/2jsLlFzWb2s) shows a trained model playing Ms Pacman.

### Summary

This project uses a neural network made up of convolutional and linear layers that take pixel input and use computer vison to learn how to play Atari games provided by Gymnasium. 

### How it works

Gymnasium provides an interface to Atari games that accepts a range of whole numbers that represent actions and provides a 2D array of RGB pixels as well as the score change per frame. The playground converts the RGB array into a grayscale array because the color in Atari games tends not to be important to game play. The model is given a configurable number of frames at a time so the model has the opportunity to sense movement. 

A stack of grayscale pixel arrays is passed to the model. Those arrays pass through a series of convolutional layers. These layers use a square kernel of values to determine each pixel value for the next layer based on the pixels near it, the ones overlapped by the kernel. The kernel is then moved by the stride to calculate the next pixel value. A stride of 1 would keep the pixel arrays the same size, but larger strides result in fewer pixels passed to the next layer. As the number of pixels is reduced, the number of channels is increased. The arrays start at 210x160 pixels with a channel for each grayscale frame passed in. 

For example, if the model takes 4 frames of 210x160 grayscale pixels and uses a stride 4 convolutional layer that also quadruples the number of channels:

| Layer | Input Shape | Output Shape |
| --- | --- | --- |
| Conv2d | (batch_size, 4, 210, 160) | (batch_size, 16, 52, 40) |
| Conv2d | (batch_size, 16, 52, 40) | (batch_size, 64, 13, 10) |
| Conv2d | (batch_size, 64, 13, 10) | (batch_size, 256, 3, 2) |
| Conv2d | (batch_size, 256, 3, 2) | (batch_size, 1024, 1, 1) |

These convolutional layers keep shrinking the pixel array until it is only 1x1, but with hundreds of channels. This flat array is then fed to a configurable number of linear layers, with a final linear layer that reduces the array to the number of actions supported by the Atari game. Finally, that array is run through softmax layer that takes all of the values and keeps them proportional, but makes it so they all add up to 1. 

The index of the highest value in that array is the action chosen by the model.

This action is given to the Atari game through Gymnasium, which returns the RGB pixel array of the next frame as well as how the score was changed by the action. 

This uses the Monte Carlo system where model weights are updated after the game is finished, but before the next game. The Q value, or expected score, is then calculated by adding the current frame score plus the score of the next frame multiplied by the discount. This helps the model understand how past actions impact future scores. The Q value is compared to the output of the model for the same frame using a loss function. The result of the loss function is used to update the weights of the model before the next game is started.

### Epsilon greedy policy
This implementation supports an epsilon greedy policy. That means that random actions will be taken a certain percentage of the time early in training. These actions and their results are still fed to the model to train from, even though they weren't selected by the model itself. This causes the model to see more different situations before selecting actions entirely on its own. This rate get smaller after every game until it reaches 0 and the model is making all of the action decisions.

### Note about convolutional layers used
Technically the convolutional layers used in this implementation are residual networks, or resnets. Instead of tracking and updating the overall value, it focuses on the difference between the input and the output. This can help make training more stable.

## How to use the playground yourself
First, pick a game you would like to play with from [Gymnasium](https://gymnasium.farama.org/environments/atari/complete_list/) and assign the string from the import to the `game` variable. 

This playground supports a number of options in the form of configuration variables. 

| Parameter | Description |
| --- | --- |
| `render` | Whether to render the game to the screen, False is faster, but won't create video |
| `record_tries` | How many tries to record, 0 for no video which is faster |
| `save_best_as` | Where to save the best model, None to not save |
| `load_model` | Where to load a model from, None to not load, loading disables training |
| `learning_rate` | How fast to learn |
| `frame_count` | Number of frames to stack so the model can perceive movement |
| `discount` | Gamma, how much to discount future rewards from current actions |
| `choose_random` | Epsilon, how often to choose a random action |
| `choose_random_min` | Minimum epsilon, after this it will always choose the best action |
| `choose_random_decay` | How much to decay epsilon after each episode (multiplied, not subtracted) |
| `skip_frames` | Number of frames to skip between actions, 1 means every frame |
| `batch_size` | Number of samples to process at once |
| `randomize_episode_batches` | Whether to randomize the order of samples in each episode |
| `loss_function` | Loss function to use |
| `optimizer` | Optimizer to use |
| `hidden_layers` | Number of hidden linear layers in the model |
| `no_score_penalty` | How much to penalize for not scoring |

From there, feel free to reimplement `AtariModel` with your own model. There are already some commented out configurations that support different numbers of convolutional layers to try.

