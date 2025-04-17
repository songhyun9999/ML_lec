import numpy as np
import random
import gymnasium as gym
from rich.markup import render


def Q_learning_train(env,alpha,gamma,epsilon,episodes):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    for i in range(episodes):
        state = env.reset()[0]
        epochs,reward = 0,0
        done = False

        while not done:
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state,reward,terminated,truncated,info = env.step(action)
            done = terminated | truncated

            old_value = q_table[state,action]
            next_s_max = np.max(q_table[next_state])

            new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_s_max) # 이전 값을 어느정도 참조할지 결정
            q_table[state,action] = new_value

            state = next_state

        if i % 100 == 0:
            print('episodes: ',i)

    policy = np.ones([env.observation_space.n, env.action_space.n])/env.action_space.n

    for state in range(env.observation_space.n):
        best_act = np.argmax(q_table[state])
        policy[state] = np.eye(env.action_space.n)[best_act]

    print('training end!!!!!!\n\n\n')
    return policy, q_table


env = gym.make('Taxi-v3', render_mode='rgb_array')
env.reset()
Q_learning_policy = Q_learning_train(env,0.2,0.95,0.1,100000)

import matplotlib.pyplot as plt

def view_policy(policy):
    current_state = env.reset()[0]
    frames = []
    frames.append(env.render())
    reward = None

    while reward != 20:
        next_state, reward, terminated, truncated, info = env.step(np.argmax(policy[0][current_state]))
        current_state = next_state
        frames.append(env.render())

    for i, frame in enumerate(frames):
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f'frame {i}')
        plt.pause(0.3)
    plt.show()

view_policy(Q_learning_policy)