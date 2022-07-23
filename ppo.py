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
from data_utils import rollout_r_p_a, rollout


def ppo(environment, hp):
    # import numpy as np
    # np.random.seed(123)
    # torch.manual_seed(211)

    cov_mat = hp.cov_mat.to(hp.device)

    ppo_net = PPO(hp.num_state_features, hp.price_min, hp.price_max)
    ppo_optimizer = optim.Adam(ppo_net.parameters(), lr=hp.lr)

    ppo_net = ppo_net.to(hp.device)

    # record model status before training
    mean_reward, p_mean, a_mean = rollout_r_p_a(environment, ppo_net, hp)
    moving_avg_reward = mean_reward
    sys.stdout.write("Iteration: {}, moving average reward: {}\n".format(0, moving_avg_reward))
    sys.stdout.write("policy_mean: {}, action: {}\n".format(p_mean, a_mean))
    p_mean, a_mean = 0, 0

    # plot settings:
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("moving average reward with ppo")
    ax.set_xlabel('iteration')
    ax.set_ylabel('moving average reward')
    (line,) = ax.plot([], [])
    moving_avg_reward_pool = [moving_avg_reward]
    episode_pool = [0]
    moving_avg_reward_pool_lim = [moving_avg_reward, moving_avg_reward]
    ax.set(xlim=(-10, 10), ylim=(moving_avg_reward - 10, moving_avg_reward + 10))
    line.set_data(episode_pool, moving_avg_reward_pool)
    # reserve time to plot the data
    plt.pause(0.2)

    ppo_net.train()
    for i in range(hp.num_itr):
        # generate hp.batch_num trajectories, with each trajectory of the length hp.episode_size
        batch_states, batch_log_probs, batch_returns, mean_reward = rollout(environment, ppo_net, hp)
        moving_avg_reward += (mean_reward.item() - moving_avg_reward) / (i + 1)
        values, _ = ppo_net(batch_states)
        advantages = batch_returns - values.squeeze().detach()

        for _ in range(hp.num_update_per_itr):
            values, policy_means = ppo_net.forward(batch_states)
            distros = MultivariateNormal(policy_means, cov_mat)
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

            p_mean = policy_means.mean().item()
            a_mean = actions.mean().item()

        itr = i + 1
        sys.stdout.write("Iteration: {}, moving average reward: {}\n".format(itr, moving_avg_reward))
        sys.stdout.write("policy_mean: {}, action: {}\n".format(p_mean, a_mean))
        p_mean, a_mean = 0, 0

        # update plot settings
        if moving_avg_reward > moving_avg_reward_pool_lim[1]:
            moving_avg_reward_pool_lim[1] = moving_avg_reward
        elif moving_avg_reward < moving_avg_reward_pool_lim[0]:
            moving_avg_reward_pool_lim[0] = moving_avg_reward
        ax.set(xlim=(-10, itr + 10),
               ylim=(moving_avg_reward_pool_lim[0] - 10, moving_avg_reward_pool_lim[1] + 10))
        # add data, then plot
        episode_pool.append(itr)
        moving_avg_reward_pool.append(moving_avg_reward)
        line.set_data(episode_pool, moving_avg_reward_pool)
        # reserve time to plot the data
        plt.pause(0.2)

    plt.show(block=True)
    torch.save(ppo_net.state_dict(), hp.model_save_path)


hyperparameter = Namespace(
    lr=3e-4,
    gamma=0.99,
    num_itr=600,
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
