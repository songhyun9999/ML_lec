import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1')
print(env)
print('state_size:', env.observation_space.n)
print('action_size:', env.action_space.n)
print('start point:',env.reset()[0])

Q_table = np.zeros([env.observation_space.n, env.action_space.n]) #16 * 4
print(Q_table)
print(Q_table.shape)
print()

lr = 0.8
df = 0.95
episodes = 2000
rlist = []

for i in range(episodes):
    current_state = env.reset()
    rewardAll = 0
    j = 0
    while j < 99:
        j+=1

        action = np.argmax(Q_table[current_state[0], :] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        next_state, reward, terminated, truncated, info =  env.step(action)

        done = terminated | truncated

        Q_table[current_state[0], action] = Q_table[current_state[0], action] + \
            lr * (reward + df * np.max(Q_table[next_state, :]) - Q_table[current_state[0], action])

        rewardAll += reward
        current_state = [next_state]
        if done:
            break

    rlist.append(rewardAll)
print(rlist)

import matplotlib.pyplot as plt
plt.bar(x=range(len(rlist)), height=rlist)
plt.show()
print()
print(Q_table)



