import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import numpy as np
import random

from collections import deque

USE_CUDA = torch.cuda.is_available()
CUDA = lambda args: args.cuda() if USE_CUDA else args

class Net(nn.Module):
    def __init__(self, input_shape, num_actions, gamma):
        super(Net, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size()+2, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, img, state):
        x = self.features(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, state), dim=1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, obs, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                obs     = CUDA(torch.FloatTensor(np.float32(obs)).unsqueeze(0))
                state   = CUDA(torch.FloatTensor(np.float32(state)).unsqueeze(0))
                q_value = self.forward(obs, state)
                action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action

    def compute_td_loss(self, batch_size, replay_buffer):
        obs, state, action, reward, next_obs, next_state, done = replay_buffer.sample(batch_size)
        obs        = CUDA(torch.FloatTensor(obs))
        state      = CUDA(torch.FloatTensor(state))
        next_obs   = CUDA(torch.FloatTensor(next_obs))
        next_state = CUDA(torch.FloatTensor(next_state))
        action     = CUDA(torch.LongTensor(action))
        reward     = CUDA(torch.FloatTensor(reward))
        done       = CUDA(torch.FloatTensor(done))

        q_values      = self.forward(obs, state)
        next_q_values = self.forward(next_obs, next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = (q_value - CUDA(expected_q_value.data)).pow(2).mean()
        
        return loss

    def save(self, path, epoch=None, optim_state=None, loss=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optim_state,
            'loss': loss,
        }, path)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, state, action, reward, next_obs, next_state, done):
        obs        = np.expand_dims(obs, 0)
        state      = np.expand_dims(state, 0)
        next_obs   = np.expand_dims(next_obs, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((obs, state, action, reward, next_obs, next_state, done))
    
    def sample(self, batch_size):
        obs, state, action, reward, next_obs, next_state, done = zip(*random.sample(self.buffer, batch_size))
        obs        = np.concatenate(obs, dtype=np.float32) / 255
        state      = np.concatenate(state, dtype=np.float32)
        next_obs   = np.concatenate(next_obs, dtype=np.float32) / 255
        next_state = np.concatenate(next_state, dtype=np.float32)

        return obs, state, action, reward, next_obs, next_state, done
    
    def __len__(self):
        return len(self.buffer)