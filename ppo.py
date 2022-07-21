import torch
import torch.optim as optim
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
from torch.distributions import MultivariateNormal
from argparse import Namespace
import env
from model_utils import PPO


def ppo(environment, hp):
    import numpy as np
    np.random.seed(123)
    torch.manual_seed(211)

    ppo_net = PPO(hp.num_state_features)
    ppo_optimizer = optim.Adam(ppo_net.parameters(), lr=hp.lr)

    ppo_net = ppo_net.to(hp.device)
    ppo_net.train()

    moving_avg_reward = 0

    # plot settings:
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("moving average reward with ppo")
    ax.set_xlabel('episode')
    ax.set_ylabel('moving average reward')
    (line,) = ax.plot([], [])
    moving_avg_reward_pool = []
    episode_pool = []
    moving_avg_reward_pool_lim = None
    ax.set(xlim=(0, 1))

    for i in range(int(hp.num_episode/hp.batch_num)):
        # generate hp.batch_num trajectories, with each trajectory of the length hp.episode_size
        batch_states, batch_log_probs, batch_returns, mean_reward = rollout(environment, ppo_net, hp)
        moving_avg_reward += (mean_reward.item() - moving_avg_reward) / (i + 1)
        values, _ = ppo_net(batch_states)
        advantages = batch_returns - values.squeeze().detach()

        for _ in range(hp.num_update_per_itr):
            values, policy_means = ppo_net.forward(batch_states)
            distros = MultivariateNormal(policy_means, hp.cov_mat.to(hp.device))
            actions = distros.sample()
            curr_log_probs = distros.log_prob(actions)

            ratios = torch.exp(curr_log_probs - batch_log_probs)
            actor_loss = (
                    - torch.min(ratios * advantages, torch.clamp(ratios, 1 - hp.clip, 1 + hp.clip) * advantages)
                          ).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
            loss = actor_loss + critic_loss

            ppo_optimizer.zero_grad()
            loss.backward()
            ppo_optimizer.step()

        episode = (i + 1) * hp.batch_num
        sys.stdout.write("Episode: {}, moving average reward: {}\n".format(episode, moving_avg_reward))
        sys.stdout.write("mean: {}, action: {}\n".format(policy_means.mean().item(), actions[0].mean().item()))
        # update plot settings
        if moving_avg_reward_pool_lim is None:
            moving_avg_reward_pool_lim = [moving_avg_reward, moving_avg_reward]
        elif moving_avg_reward > moving_avg_reward_pool_lim[1]:
            moving_avg_reward_pool_lim[1] = moving_avg_reward
        elif moving_avg_reward < moving_avg_reward_pool_lim[0]:
            moving_avg_reward_pool_lim[0] = moving_avg_reward
        ax.set(xlim=(-5, episode + 5),
               ylim=(moving_avg_reward_pool_lim[0] - 10, moving_avg_reward_pool_lim[1] + 10))
        # add data, then plot
        episode_pool.append(episode)
        moving_avg_reward_pool.append(moving_avg_reward)
        line.set_data(episode_pool, moving_avg_reward_pool)
        # reserve time to plot the data
        plt.pause(0.2)

    plt.show(block=True)
    torch.save(ppo_net.state_dict(), hp.model_save_path)


def rollout(environment, net, hp):
    batch_states = []
    batch_log_probs = []
    batch_rewards = []

    for _ in range(hp.batch_num):
        # rewards per episode
        ep_rewards = []
        state = torch.from_numpy(environment.reset())
        state = torch.cat(
            (state[:-1], torch.tensor([state[-1] == 0, state[-1] == 1, state[-1] == 2], dtype=torch.float))
        ).to(hp.device)

        # run an episode
        for _ in range(hp.episode_size):
            batch_states.append(state)

            # compute action and log_prob
            _, policy_mean = net.forward(state)
            distro = MultivariateNormal(policy_mean, hp.cov_mat.to(hp.device))
            action = distro.sample().detach()
            log_prob = distro.log_prob(action).detach()

            # compute reward and go to next state
            r, state = environment.step(action.item())
            state = torch.from_numpy(state)
            state = torch.cat(
                (state[:-1], torch.tensor([state[-1] == 0, state[-1] == 1, state[-1] == 2], dtype=torch.float))
            ).to(hp.device)

            ep_rewards.append(r)
            batch_log_probs.append(log_prob)

        batch_rewards.append(ep_rewards)

    batch_states = torch.stack(batch_states)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(hp.device)
    batch_returns = compute_returns(batch_rewards, hp.gamma).to(hp.device)

    return batch_states, batch_log_probs, batch_returns, torch.tensor(batch_rewards).mean()


def compute_returns(batch_rewards, gamma):
    batch_returns = []
    # iterate through each episode
    for ep_rewards in reversed(batch_rewards):
        discounted_reward = 0
        for r in reversed(ep_rewards):
            discounted_reward = r + discounted_reward * gamma
            batch_returns.insert(0, discounted_reward)

    return torch.tensor(batch_returns, dtype=torch.float)


hyperparameter = Namespace(
    lr=3e-4,
    gamma=0.99,
    num_episode=3000,
    batch_num=5,
    episode_size=300,
    num_update_per_itr=5,
    num_state_features=21,
    price_min=400,
    price_max=2700,
    clip=0.2,
    cov_mat=torch.diag(torch.full(size=(1,), fill_value=50.)),
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    model_save_path='./data/ppo_model.pth'
)
glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
ppo(env_train, hyperparameter)
