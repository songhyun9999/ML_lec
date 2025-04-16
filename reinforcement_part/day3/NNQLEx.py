import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np


env = gym.make('FrozenLake-v1')

def onehot2Tensor(state):
    tmp = np.zeros(16)
    tmp[state] = 1
    vector = np.array(tmp,dtype=np.float32)
    tensor = torch.from_numpy(vector)

    return tensor

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16,64)
        self.fc2 = nn.Linear(64,96)
        self.fc3 = nn.Linear(96,96)
        self.fc4 = nn.Linear(96,64)
        self.fc5 = nn.Linear(64,4)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        y = self.fc5(out)

        return y


model = QNet()

def applyModel(input_tensor):
    output_tensor = model(input_tensor)
    output_array = output_tensor.data.numpy()

    return output_tensor, output_array


rewardAll = 0
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)
rlist = []

for episode in range(3000):
    current_state = env.reset()
    episode_reward = 0
    total_loss = 0

    for t in range(100):
        current_tensor = onehot2Tensor(current_state[0])
        current_output_tensor, current_output_array = applyModel(current_tensor)

        if np.random.rand() < 0.1:
            action = env.action_space.sample()
        else:
            action = np.argmax(current_output_array)

        next_state,reward,terminated,truncated,_ = env.step(action)
        next_state_tensor = onehot2Tensor(next_state)

        next_output_tensor, next_output_array = applyModel(next_state_tensor)
        target = reward + 0.99*np.max(next_output_array)

        q_array = np.copy(current_output_array)
        q_array[action] = target
        target_tensor = torch.tensor(q_array)

        optimizer.zero_grad()
        loss = loss_func(current_output_tensor,target_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        current_state = [next_state]
        done = terminated | truncated
        if done:
            episode_reward += reward
            break
    rlist.append(episode_reward)
    print(f'episode:{episode+1}\ttotal_loss:{total_loss:.5f}')


import matplotlib.pyplot as plt

plt.plot(rlist)
plt.show()





