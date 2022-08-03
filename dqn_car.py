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
        target=np.array([-82.4, -102.7, 0])
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
num_frames = 500000
losses = []
all_rewards = []
episode_reward = 0
success_streak = 0

obs = env.reset()
state = env.state
print("start training")
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(obs, state["target_diff"], epsilon)
    next_obs, reward, done, next_state = env.step(action)
    replay_buffer.push(obs, state["target_diff"], action, reward, next_obs, next_state["target_diff"], done)
    
    obs = next_obs
    state = next_state
    episode_reward += reward
    
    if len(replay_buffer) > batch_size:
        loss = model.compute_td_loss(batch_size, replay_buffer)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if done:
        all_rewards.append(episode_reward)
        episode_reward = 0
        print("episode {}: {}".format(len(all_rewards), state["result"]))
        print("reward: {}".format(all_rewards[-1]))
        if state["result"] == "success":
            success_streak += 1
        else:
            success_streak = 0
        if success_streak == 10:
            break
        obs = env.reset()
        state = env.state

        model.save("model.pt", frame_idx, optimizer.state_dict(), loss)        
        
    if frame_idx % 200 == 0:
        print("frame index: {}, loss: {}".format(frame_idx, loss))

# Save policy weights
plot(frame_idx, all_rewards, losses)
model.save("dqn_airsim_car_policy.pt")