import gym
import airgym

import math
import numpy as np

import matplotlib.pyplot as plt

import DQN

# set epsilon decay
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
'''
plt.figure()
plt.plot([epsilon_by_frame(i) for i in range(10000)])
plt.show()
'''
def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

# set environment
print("set environment...")
env = gym.make(
        "airgym:airsim-car-sample-v0",
        ip_address="127.0.0.1",
        image_shape=(1, 84, 84),
    )
print("done!")

#define model
gamma = 0.99
batch_size = 64
model = DQN.Net(env.image_shape, env.action_space.n, gamma)
#use cuda
if DQN.USE_CUDA:
    model = model.cuda()
# set optimizer
optimizer = DQN.optim.Adam(model.parameters())
# set replay buffer
replay_buffer = DQN.ReplayBuffer(10000)

#train
num_frames = 1000000
losses = []
all_rewards = []
episode_reward = 0
success_streak = 0

obs = env.reset()
print("start training")
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(obs, epsilon)
    next_obs, reward, done, state = env.step(action)
    '''
    print("GPS: {}".format(state["position"]))
    print("XYZ: {}".format(state["pose"].position.to_numpy_array()))
    print("------------------------------------------------")
    '''
    replay_buffer.push(obs, action, reward, next_obs, done)
    
    obs = next_obs
    episode_reward += reward
    
    if done:
        obs = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        if state["info"] == "success":
            success_streak += 1
        else:
            success_streak = 0
        if success_streak == 10:
            break
        
    if len(replay_buffer) > batch_size:
        loss = model.compute_td_loss(batch_size, replay_buffer)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if frame_idx % 200 == 0:
        print("frame index: {}".format(frame_idx))
        print("reward: {}, loss: {}".format(all_rewards[-1], loss))

# Save policy weights
plot(frame_idx, all_rewards, losses)
model.save("dqn_airsim_car_policy")