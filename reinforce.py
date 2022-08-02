import torch
import sys
import torch.optim as optim
import statsmodels.api as sm
import env
from argparse import Namespace
import pandas as pd
from model_utils import Actor
from data_utils import rollout_reinforce


def reinforce(environment, hp):
    actor = Actor(hp.num_state_features, (hp.price_max - hp.price_min) / hp.price_binwidth)
    actor_optimizer = optim.Adam(actor.parameters(), lr=hp.actor_lr)

    actor = actor.to(hp.device)

    moving_avg_reward_pool = []

    actor.train()
    for i in range(hp.num_itr):
        # generate trajectory
        batch_log_probs, discounted_batch_returns, price_mean, moving_avg_reward = \
            rollout_reinforce(environment, actor, hp)
        discounted_batch_returns = (discounted_batch_returns - discounted_batch_returns.mean()) / \
                                   (discounted_batch_returns.std() + 1e-10)

        # compute the return and loss
        actor_loss = (- discounted_batch_returns * batch_log_probs).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if i % 5 == 0:
            sys.stdout.write("Iteration: {}, price_mean: {}, moving average reward: {}\n"
                             .format(i, price_mean.item(), moving_avg_reward.item()))
            print(f'actor_loss: {actor_loss}')
            moving_avg_reward_pool.append(moving_avg_reward.item())

    # save training result to csv file, and save the model
    df = pd.DataFrame([moving_avg_reward_pool])
    df.to_csv(hp.csv_out_path, mode='a', header=False)
    torch.save(actor.state_dict(), hp.actor_save_path)


hyperparameter = Namespace(
    actor_lr=3e-4,
    gamma=0.99,
    num_itr=3000,
    batch_num=1,
    episode_size=300,
    moving_avg_num=300,
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
