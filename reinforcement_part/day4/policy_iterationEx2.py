import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("Taxi-v3")
n_states = env.observation_space.n
n_actions = env.action_space.n

# 초기 정책: 모든 상태에서 모든 행동을 균등하게 선택
policy = np.ones([n_states, n_actions]) / n_actions

def policy_evaluation(policy, env, gamma=0.99, theta=1e-5):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, _ in env.P[s][a]: #상태, 행동 → env.P[state][action] = [(probability, next_state, reward, done)]
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(V, env, gamma=0.99):
    policy_stable = True
    new_policy = np.zeros([n_states, n_actions])
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, _ in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        best_action = np.argmax(action_values)
        new_policy[s] = np.eye(n_actions)[best_action]
        if not np.array_equal(new_policy[s], policy[s]):
            policy_stable = False
    return new_policy, policy_stable

i = 0
# 정책 반복
while True:
    i+=1
    print('학습 진행:', i)
    V = policy_evaluation(policy, env)
    policy, stable = policy_improvement(V, env)
    if stable:
        break

import time
import matplotlib.pyplot as plt
# 학습된 정책 테스트
# env = gym.make("Taxi-v3" ,render_mode="human")
env = gym.make("Taxi-v3", render_mode='rgb_array')
total_rewards = 0
for episode in range(100):
    state, _ = env.reset()
    frames = []
    frames.append(env.render())
    # env.render()
    done = False
    while not done:
        action = np.argmax(policy[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        frames.append(env.render())
        # env.render()
        total_rewards += reward
    print(f'{episode+1}')

    for i,frame in enumerate(frames):
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f'frame {i}')
        plt.pause(0.3)
    plt.show()

print(f"100 에피소드 동안 평균 보상: {total_rewards / 100}")
