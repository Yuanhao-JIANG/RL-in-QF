import torch
import sys
import torch.optim as optim
import statsmodels.api as sm
import env
from argparse import Namespace
import pandas as pd
from model_utils import Actor


def reinforce(environment, hp):
    actor = Actor(hp.num_state_features, (hp.price_max - hp.price_min) / hp.price_binwidth)
    actor_optimizer = optim.Adam(actor.parameters(), lr=hp.actor_lr)

    actor = actor.to(hp.device)

    moving_avg_reward_pool = []

    actor.train()
    for i in range(hp.num_itr):
        moving_avg_reward = 0
        ep_states = []
        ep_actions = []
        ep_returns = []
        actor_loss_mean = 0
        # run a trajectory
        state = torch.from_numpy(environment.reset()).to(hp.device)
        for step in range(hp.episode_size):
            ep_states.append(state)
            policy_distro = actor.forward(state)
            action = policy_distro.sample()
            price = hp.price_min + action * hp.price_binwidth

            r, state = environment.step(price.item())
            state = torch.from_numpy(state).to(hp.device)

            ep_actions.append(action)
            ep_returns.append(r)
            moving_avg_reward += r

        # compute and normalize returns
        for t in reversed(range(len(ep_returns) - 1)):
            ep_returns[t] += hp.gamma * ep_returns[t + 1]
        ep_returns = torch.tensor(ep_returns, dtype=torch.float)
        ep_returns = (ep_returns - ep_returns.mean()) / (ep_returns.std(dim=0) + 1e-10)

        # update policy network
        for step in range(hp.episode_size):
            policy_distro = actor.forward(ep_states[step])
            log_prob = policy_distro.log_prob(ep_actions[step])
            actor_loss = - (hp.gamma ** step) * ep_returns[step] * log_prob
            actor_loss_mean += actor_loss

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

        actor_loss_mean = actor_loss_mean / hp.episode_size
        price_mean = hp.price_min + (sum(ep_actions) / len(ep_actions)) * hp.price_binwidth
        moving_avg_reward = moving_avg_reward / hp.episode_size

        if i % 5 == 0:
            sys.stdout.write("Iteration: {}, price_mean: {}, moving average reward: {}\n"
                             .format(i, price_mean, moving_avg_reward))
            print(f'actor_loss: {actor_loss_mean}')
            moving_avg_reward_pool.append(moving_avg_reward)

    # save training result to csv file, and save the model
    df = pd.DataFrame([moving_avg_reward_pool])
    df.to_csv(hp.csv_out_path, mode='a', header=False, index=False)
    torch.save(actor.state_dict(), hp.actor_save_path)


hyperparameter = Namespace(
    actor_lr=3e-4,
    gamma=0.99,
    num_itr=3000,
    episode_size=300,
    num_state_features=21,
    price_min=200,
    price_max=2000,
    price_binwidth=15,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    actor_save_path='./data/reinforce_actor.pth',
    csv_out_path='./data/reinforce_out.csv'
)
glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
reinforce(env_train, hyperparameter)
# while True:
#     reinforce(env_train, hyperparameter)
