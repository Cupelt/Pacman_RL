import os
import torch
import matplotlib
import matplotlib.pyplot as plt

from DQN import ReplayMemory, Agent

from datetime import datetime

save_rate = 5
save_path = "pacman_model"
save_point = False

batch_size = 128
memory_size = 15000

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(episode_rewards, eps_thresholds, show_result=False):
    fig = plt.figure(1, figsize=(6.4, 9.6))
    durations_t = torch.tensor(episode_rewards, dtype=torch.float)
    eps_thresholds_t = torch.tensor(eps_thresholds, dtype=torch.float)

    # eps_threshold 그리기
    plt.subplot(2, 1, 1)
    if show_result:
        plt.title('Result')
    else:
        plt.title('Training...')
    plt.plot(eps_thresholds_t.numpy(), color='#e35f62')
    plt.xlabel('Episode')
    plt.ylabel('eps_threshold')

    # 생존 시간 그리기
    plt.subplot(2, 1, 2)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), 'b')
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), 'r')

    fig.canvas.flush_events()
    plt.pause(0.01)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def load_model(n_observations, n_actions, device=None):
    agent_net = Agent(n_observations, n_actions).to(device)
    target_net = Agent(n_observations, n_actions).to(device)

    if os.path.exists(f"{save_path}/model-latest.pt"):
        checkpoint = torch.load(f"{save_path}/model-latest.pt")

        agent_net.load_state_dict(checkpoint['agent_state_dict'])
        target_net.load_state_dict(checkpoint['target_state_dict'])

        i_episode = checkpoint['epoch']
        steps_done = checkpoint['steps_done']

        episode_rewards = checkpoint['episode_rewards']
        eps_thresholds = checkpoint['eps_thresholds']
        memory = checkpoint['replaybuffer']
        memory.setSize(memory_size)
        
        print("Model Loaded")
    else:
        target_net.load_state_dict(agent_net.state_dict())
        
        i_episode = 0
        steps_done = 0

        eps_thresholds = []
        episode_rewards = []
        memory = ReplayMemory(memory_size)
    
    return agent_net, target_net, i_episode, steps_done, episode_rewards, eps_thresholds, memory

def save_model(i_episode, episode_rewards, eps_thresholds, steps_done, memory, agent_net, target_net):
    if i_episode % save_rate != 0:
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if save_point:
        date = datetime.today().strftime("%Y%m%d%H%M%S")
        torch.save({
                'epoch': i_episode,
                'episode_rewards': episode_rewards,
                'eps_thresholds': eps_thresholds,
                'steps_done': steps_done,
                'replaybuffer': memory,
                'agent_state_dict': agent_net.state_dict(),
                'target_state_dict': target_net.state_dict()
                }, 
                f'{save_path}/model-{date}.pt')

    torch.save({
                'epoch' : i_episode,
                'episode_rewards': episode_rewards,
                'eps_thresholds': eps_thresholds,
                'steps_done': steps_done,
                'replaybuffer': memory,
                'agent_state_dict': agent_net.state_dict(),
                'target_state_dict': target_net.state_dict()
                },
                f'{save_path}/model-latest.pt')