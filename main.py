import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

import manager
from manager import batch_size
from DQN import Transition

from collections import namedtuple, deque

env = gym.make("ALE/MsPacman-v5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HyperParams
gamma = .99
learning_rate = 1e-4
tau = 0.005

eps_start = 1.0
eps_end = 0.1
eps_decay = 0.999995

num_episodes = 5000
max_step = 10000

n_observations = env.observation_space.shape[2]
n_actions = env.action_space.n

agent_net, target_net, i_episode, \
    steps_done, episode_rewards, eps_thresholds, memory = manager.load_model(n_observations, n_actions, device=device)

optimizer = optim.Adam(agent_net.parameters(), learning_rate)
criterion = nn.SmoothL1Loss()

def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = agent_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = reward_batch + gamma * next_state_values

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(agent_net.parameters(), 100)
    optimizer.step()


def select_action(state):
    global steps_done

    sample = random.random()
    eps_threshold = max(eps_end, eps_start * (eps_decay ** steps_done))
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return torch.argmax(agent_net(state)).view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=device)

while i_episode < num_episodes:
    manager.plot_durations(episode_rewards, eps_thresholds)

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, 3, 210, 160)

    for i_step in range(max_step):
        action = select_action(state)

        observation, reward, terminated, truncated, _ = env.step(action)
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).view(1, 3, 210, 160)

        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        agent_net_state_dict = agent_net.state_dict()
        for key in agent_net_state_dict:
            target_net_state_dict[key] = agent_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break
    
    episode_rewards.append(i_step)
    eps_thresholds.append(max(eps_end, eps_start * (eps_decay ** steps_done)))

    print("Episode : {}, Reward : {}, eps_threshold : {:.5}, memory_length : {:,}, cuda_memory : {:.4}".format(i_episode, episode_rewards[i_episode], eps_thresholds[i_episode], len(memory), torch.cuda.memory_allocated() / (1024 ** 3)))
    i_episode += 1

    manager.save_model(i_episode, episode_rewards, eps_thresholds, steps_done, 
                       memory, agent_net, target_net)