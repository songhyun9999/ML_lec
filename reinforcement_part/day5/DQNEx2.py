import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import os

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        #########################################################################################
        self.fc1 = nn.Linear(state_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)
        #########################################################################################

    def forward(self, x):
    #########################################################################################
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
    #########################################################################################

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.memory.append(experience)

    def sample(self):
    #########################################################################################
        mini_batch = random.sample(self.memory,k=self.batch_size)
        state_lst, action_lst, reward_lst, next_state_lst, done_lst = zip(*mini_batch)

        state_lst = torch.tensor(np.array(state_lst),dtype=torch.float32).to(device)
        action_lst = torch.tensor(action_lst,dtype=torch.int64).unsqueeze(-1).to(device)
        reward_lst = torch.tensor(reward_lst,dtype=torch.float32).unsqueeze(-1).to(device)
        next_state_lst = torch.tensor(np.array(next_state_lst),dtype=torch.float32).to(device)
        done_lst = torch.tensor(done_lst,dtype=torch.uint8).unsqueeze(-1).to(device)

        return state_lst, action_lst, reward_lst, next_state_lst, done_lst
    #########################################################################################

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size, seed, buffer_size=100000, batch_size=64, gamma=0.99, lr=0.001, tau=0.01, update_every=4):
    #########################################################################################
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size

        self.qnetwork_local = QNetwork(state_size,action_size).to(device)
        self.qnetwork_target = QNetwork(state_size,action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=lr)

        self.memory = ReplayBuffer(buffer_size,batch_size)
        self.t_step = 0
    #########################################################################################

    def step(self, state, action, reward, next_state, done):
    #########################################################################################
        self.memory.add((state,action,reward,next_state,done))
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    #########################################################################################

    def action(self, state, eps=0.):
    #########################################################################################
        state = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    #########################################################################################

    def learn(self, experiences):
    #########################################################################################
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1-dones))
        Q_expected = self.qnetwork_local(states).gather(1,actions)

        loss = F.mse_loss(Q_expected,Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 초창기 학습 안정화
        self.soft_update(self.qnetwork_local,self.qnetwork_target,self.tau)
    #########################################################################################


    def soft_update(self, local_model, target_model, tau):
    #########################################################################################
        # 학습 안정화를 위해 일정 비율만큼만 parameter 갱신
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param + (1.0-tau) * target_param.data)
    #########################################################################################


# Check if the model file exists
load_model = os.path.isfile('dqn_mountaincar.pth')

# Create and train the DQN Agent
env = gym.make('MountainCar-v0', render_mode="rgb_array")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
seed = 0
agent = DQNAgent(state_size, action_size, seed)

n_episodes = 10000
max_t = 200
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

scores = []

if load_model:
    #########################################################################################
    env = gym.make('MountainCar-v0',render_mode='human')
    agent.qnetwork_local.load_state_dict(torch.load('dqn_mountaincar.pth',map_location=device))
    print('Model loaded')
    #########################################################################################
else:
    for i_episode in range(1, n_episodes + 1):
        #########################################################################################
        state = env.reset()[0]
        eps = max(eps_end, eps_start * (eps_decay**i_episode))
        score = 0
        for t in range(max_t):
            action = agent.action(state,eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score += reward
            if done:
                break
        #########################################################################################

        scores.append(score)
        print(f"Episode {i_episode}/{n_episodes} - Score: {score:.2f}")

    # Save the trained model
    torch.save(agent.qnetwork_local.state_dict(), 'dqn_mountaincar.pth')
    print("Model saved.")



# Render the trained agent
state = env.reset()[0]
env.render()
done = False
while not done:
    action = agent.action(state, eps=0.0)  # Use epsilon=0 for deterministic actions
    state, _, terminated, truncated, _ = env.step(action)
    done = terminated | truncated
    env.render()
env.close()

# Plotting the rewards if training was done
if not load_model:
    plt.plot(np.arange(1, n_episodes + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title('Score per Episode')
    plt.show()