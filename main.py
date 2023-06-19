import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import random
import math
import os

from datetime import datetime
from collections import namedtuple, deque

env = gym.make("ALE/MsPacman-v5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# HyperParams
batch_size = 128
memory_size = 15000
gamma = .99
learning_rate = 1e-4
tau = 0.005

save_rate = 5
save_path = ".pacman_model"
save_point = False

eps_start = 1.0
eps_end = 0.1
eps_decay = 0.999995

num_episodes = 5000
max_step = 10000

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def setSize(self, count):
        self.memory = deque(self.memory, maxlen=count)

class Agent(nn.Module):

    def __init__(self, input, output):
        super(Agent, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22 * 16 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, output)
        )
    def forward(self, x):
        return self.layer1(x)

n_observations = env.observation_space.shape[2]
n_actions = env.action_space.n


agent_net = Agent(n_observations, n_actions).to(device)
target_net = Agent(n_observations, n_actions).to(device)

if os.path.exists(f"{save_path}/model-latest.pt"):
    checkpoint = torch.load(f"{save_path}/model-latest.pt")

    agent_net.load_state_dict(checkpoint['agent_state_dict'])
    target_net.load_state_dict(checkpoint['target_state_dict'])

    i_episode = checkpoint['epoch']
    steps_done = checkpoint['steps_done']

    episode_rewards = checkpoint['episode_rewards']
    memory = checkpoint['replaybuffer']
    memory.setSize(memory_size)
    
    print("Model Loaded")
else:
    target_net.load_state_dict(agent_net.state_dict())
    
    i_episode = 0
    steps_done = 0

    episode_rewards = []
    memory = ReplayMemory(memory_size)

optimizer = optim.Adam(agent_net.parameters(), learning_rate)
criterion = nn.SmoothL1Loss()


print(steps_done)
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

def plot_durations(show_result=False):
    fig = plt.figure(1)
    durations_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    fig.canvas.flush_events()
    plt.pause(0.01)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


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
    plot_durations()

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
    

    eps_threshold = max(eps_end, eps_start * (eps_decay ** steps_done))
    print("Episode : {}, Reward : {}, eps_threshold : {:.5}, memory_length : {:,}, cuda_memory : {:.4}".format(i_episode, episode_rewards[i_episode], eps_threshold, len(memory), torch.cuda.memory_allocated() / (1024 ** 3)))
    i_episode += 1

    if i_episode % save_rate == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if save_point:
            date = datetime.today().strftime("%Y%m%d%H%M%S")
            torch.save({
                    'epoch': i_episode,
                    'episode_rewards': episode_rewards,
                    'steps_done': steps_done,
                    'replaybuffer': memory,
                    'agent_state_dict': agent_net.state_dict(),
                    'target_state_dict': target_net.state_dict()
                   }, 
                   f'{save_path}/model-{date}.pt')

        torch.save({
                    'epoch' : i_episode,
                    'episode_rewards': episode_rewards,
                    'steps_done': steps_done,
                    'replaybuffer': memory,
                    'agent_state_dict': agent_net.state_dict(),
                    'target_state_dict': target_net.state_dict()
                   },
                   f'{save_path}/model-latest.pt')