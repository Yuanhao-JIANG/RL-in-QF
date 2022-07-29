import torch
import torch.optim as optim
import torch.nn as nn
import sys
import statsmodels.api as sm
from torch.distributions import MultivariateNormal
from argparse import Namespace
import pandas as pd
import env
from model_utils import PPO
from data_utils import rollout_ppo


def ppo(environment, hp):
    cov_mat = hp.cov_mat.to(hp.device)

    ppo_net = PPO(hp.num_state_features, hp.price_min, hp.price_max)
    ppo_optimizer = optim.Adam(ppo_net.parameters(), lr=hp.lr)

    ppo_net = ppo_net.to(hp.device)

    moving_avg_reward = 0
    moving_avg_reward_pool = []

    ppo_net.train()
    for i in range(hp.num_itr):
        # generate hp.batch_num trajectories, with each trajectory of the length hp.episode_size
        batch_states, batch_log_probs, batch_returns, p_mean, a_mean, mean_reward = \
            rollout_ppo(environment, ppo_net, hp)
        moving_avg_reward += (mean_reward.item() - moving_avg_reward) / (i + 1)
        values, _ = ppo_net(batch_states)
        advantages = batch_returns - values.squeeze().detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

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

        sys.stdout.write("Iteration: {}, moving average reward: {}\n".format(i, moving_avg_reward))
        sys.stdout.write("policy_mean: {}, action: {}\n".format(p_mean.item(), a_mean.item()))
        moving_avg_reward_pool.append(moving_avg_reward)

    # save training result to csv file, and save the model
    df = pd.DataFrame([moving_avg_reward_pool])
    df.to_csv(hp.csv_out_path, mode='a', header=False)
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
    cov_mat=torch.diag(torch.full(size=(1,), fill_value=100.)),
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    model_save_path='./data/ppo_model.pth',
    csv_out_path='./data/ppo_out.csv'
)
glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
ppo(env_train, hyperparameter)
