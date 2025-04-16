import gymnasium as gym
import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

learning_rate = 0.0005
discount_factor = 0.98
buffer_limit = 50000
batch_size = 32

class ReplayMemory():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, action_lst, reward_lst, next_state_lst, done_lst = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done = transition
            state_lst.append(state)
            action_lst.append([action])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_lst.append([done])

        return torch.tensor(state_lst, dtype=torch.float), torch.tensor(action_lst), \
               torch.tensor(reward_lst), torch.tensor(next_state_lst, dtype=torch.float), \
               torch.tensor(done_lst)

    def size(self):
        return len(self.buffer)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        y = self.fc3(out)
        return y

    def sample_action(self, state, epsilon):
        out = self.forward(state)
        rvalue = random.random()
        if rvalue < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    for i in range(10):
        state, action, reward, next_state, done = memory.sample(batch_size)

        q_out = q(state)
        q_a = q_out.gather(1, action)
        max_q_next_value = q_target(next_state).max(1)[0].unsqueeze(1)
        target = reward + discount_factor * max_q_next_value * done
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


q = QNet()
q_target = QNet()
q_target.load_state_dict(q.state_dict()) #동기화
memory = ReplayMemory()
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

print_interval = 20
score = 0.0
env = gym.make('CartPole-v1')

for episode in range(10000):
    epsilon = max(0.01, 0.1 - 0.01*(episode/200)) #10% => 1%
    state = env.reset()[0]
    done = False

    while not done:
        action = q.sample_action(torch.from_numpy(state).float(), epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        done_mask = 0.0 if done else 1.0
        memory.put((state, action, reward/100.0, next_state, done_mask))
        state = next_state

        score += reward
        if done:
            break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

    if episode % print_interval == 0 and episode!=0:
        q_target.load_state_dict(q.state_dict())
        print('n_episode:{}, score:{:.1f}, n_buffer:{}, epsilon:{:.1f}%'.format(
            episode, score/print_interval, memory.size(), epsilon*100
        ))
        score = 0.0
env.close()









