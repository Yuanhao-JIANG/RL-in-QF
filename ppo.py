import torch
import torch.optim as optim
import torch.nn as nn
import sys
import statsmodels.api as sm
from argparse import Namespace
import pandas as pd
import env
from model_utils import Actor, Critic


def ppo(environment, hp):
    actor, critic = \
        Actor(hp.num_state_features, (hp.price_max - hp.price_min) / hp.price_binwidth), Critic(hp.num_state_features)
    actor_optimizer, critic_optimizer = \
        optim.Adam(actor.parameters(), lr=hp.actor_lr), optim.Adam(critic.parameters(), lr=hp.critic_lr)

    actor, critic = actor.to(hp.device), critic.to(hp.device)

    moving_avg_reward_pool = []

    actor.train()
    critic.train()
    for i in range(hp.num_itr):
        # collect set of trajectories
        price_mean = 0
        batch_states = []
        batch_log_probs = []
        batch_returns = []
        for _ in range(hp.batch_size):
            # run a trajectory
            ep_returns = []
            state = torch.from_numpy(environment.reset()).to(hp.device)
            for _ in range(hp.episode_size):
                batch_states.append(state)

                # compute action by current policy
                policy_distro = actor.forward(state)
                action = policy_distro.sample()
                log_prob = policy_distro.log_prob(action)
                price = hp.price_min + action * hp.price_binwidth

                # compute reward and go to next state
                r, state = environment.step(price.item())
                state = torch.from_numpy(state).to(hp.device)

                ep_returns.append(r)
                price_mean += price
                batch_log_probs.append(log_prob)
            batch_returns.append(ep_returns)
            moving_avg_reward = sum(ep_returns) / len(ep_returns)
            price_mean = price_mean / hp.episode_size
        batch_states = torch.stack(batch_states).to(hp.device)
        batch_log_probs = torch.stack(batch_log_probs).to(hp.device).detach()

        # compute returns and advantages
        for e in reversed(range(len(batch_returns))):
            for t in reversed(range(len(batch_returns[e]) - 1)):
                batch_returns[e][t] += hp.gamma * batch_returns[e][t + 1]
        batch_returns = torch.flatten(torch.tensor(batch_returns, dtype=torch.float)).to(hp.device)
        values = critic(batch_states)
        advantages = (batch_returns - values.squeeze()).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # update network
        for _ in range(hp.num_updates_per_itr):
            policy_distros = actor.forward(batch_states)
            actions = policy_distros.sample()
            curr_log_probs = policy_distros.log_prob(actions)
            values = critic.forward(batch_states)

            ratios = torch.exp(curr_log_probs - batch_log_probs)
            actor_loss = (
                - torch.min(ratios * advantages, torch.clamp(ratios, 1 - hp.clip, 1 + hp.clip) * advantages)
            ).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        sys.stdout.write("Iteration: {}, price_mean: {}, moving average reward: {}\n"
                         .format(i, price_mean, moving_avg_reward))
        print(f'actor_loss: {actor_loss}, critic_loss: {critic_loss}')
        moving_avg_reward_pool.append(moving_avg_reward)

    # save training result to csv file, and save the model
    df = pd.DataFrame([moving_avg_reward_pool])
    df.to_csv(hp.csv_out_path, mode='a', header=False, index=False)
    torch.save(actor.state_dict(), hp.actor_save_path)
    torch.save(critic.state_dict(), hp.critic_save_path)


hyperparameter = Namespace(
    actor_lr=3e-4,
    critic_lr=1e-3,
    gamma=0.99,
    num_itr=600,
    batch_size=5,
    episode_size=300,
    num_updates_per_itr=5,
    num_state_features=21,
    price_min=200,
    price_max=2000,
    price_binwidth=15,
    clip=0.2,
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
