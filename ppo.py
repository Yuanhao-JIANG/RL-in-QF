import torch
import torch.optim as optim
import torch.nn as nn
import sys
import statsmodels.api as sm
from torch.distributions import MultivariateNormal
from argparse import Namespace
import pandas as pd
import env
from model_utils import Actor, Critic
from data_utils import rollout_ppo


def ppo(environment, hp):
    cov_mat = hp.cov_mat.to(hp.device)

    actor, critic = Actor(hp.num_state_features, hp.price_min, hp.price_max), Critic(hp.num_state_features)
    actor_optimizer, critic_optimizer = \
        optim.Adam(actor.parameters(), lr=hp.actor_lr), optim.Adam(critic.parameters(), lr=hp.critic_lr)

    actor, critic = actor.to(hp.device), critic.to(hp.device)

    moving_avg_reward_pool = []

    actor.train()
    critic.train()
    for i in range(hp.num_itr):
        batch_states, batch_log_probs, batch_returns, p_mean, a_mean, moving_avg_reward = \
            rollout_ppo(environment, actor, hp)
        values = critic(batch_states)
        advantages = batch_returns - values.squeeze().detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(hp.num_update_per_itr):
            policy_means = actor.forward(batch_states)
            values = critic.forward(batch_states)
            distros = MultivariateNormal(policy_means, cov_mat)
            actions = distros.sample()
            curr_log_probs = distros.log_prob(actions)

            ratios = torch.exp(curr_log_probs - batch_log_probs)
            actor_loss = (
                - torch.min(ratios * advantages, torch.clamp(ratios, 1 - hp.clip, 1 + hp.clip) * advantages)
            ).mean()
            policy_edge_loss = nn.MSELoss()(
                policy_means.squeeze(),
                torch.tensor([(hp.price_min + hp.price_max) / 2] * len(policy_means), dtype=torch.float).to(hp.device)
            )
            actor_loss += policy_edge_loss * 1e-3
            critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        sys.stdout.write("Iteration: {}, moving average reward: {}\n".format(i, moving_avg_reward.item()))
        sys.stdout.write("policy_mean: {}, action: {}\n".format(p_mean.item(), a_mean.item()))
        print(f'actor_loss: {actor_loss}, critic_loss: {critic_loss}, policy_edge_loss: {policy_edge_loss}')
        moving_avg_reward_pool.append(moving_avg_reward.item())

    # save training result to csv file, and save the model
    df = pd.DataFrame([moving_avg_reward_pool])
    df.to_csv(hp.csv_out_path, mode='a', header=False)
    torch.save(actor.state_dict(), hp.actor_save_path)
    torch.save(critic.state_dict(), hp.critic_save_path)


hyperparameter = Namespace(
    actor_lr=3e-4,
    critic_lr=3e-4,
    gamma=0.99,
    num_itr=600,
    batch_num=5,
    episode_size=300,
    moving_avg_num=300,
    num_update_per_itr=5,
    num_state_features=21,
    price_min=200,
    price_max=2000,
    clip=0.2,
    cov_mat=torch.diag(torch.full(size=(1,), fill_value=100.)),
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    actor_save_path='./data/ppo_actor.pth',
    critic_save_path='./data/ppo_critic.pth',
    csv_out_path='./data/ppo_out.csv'
)
glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
ppo(env_train, hyperparameter)
# while True:
#     ppo(env_train, hyperparameter)
