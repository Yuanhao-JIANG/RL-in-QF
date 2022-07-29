import torch
import sys
import torch.optim as optim
import statsmodels.api as sm
import env
from argparse import Namespace
import pandas as pd
from model_utils import Reinforce
from data_utils import rollout_reinforce


def reinforce(environment, hp):
    net = Reinforce(hp.num_state_features, hp.price_min, hp.price_max)
    optimizer = optim.Adam(net.parameters(), lr=hp.lr)

    net = net.to(hp.device)

    moving_avg_reward_pool = []

    net.train()
    for i in range(hp.num_itr):
        # generate trajectory
        batch_log_probs, discounted_batch_returns, p_mean, a_mean, moving_avg_reward = \
            rollout_reinforce(environment, net, hp)
        discounted_batch_returns = (discounted_batch_returns - discounted_batch_returns.mean()) / \
                                   (discounted_batch_returns.std() + 1e-10)

        # compute the return and loss
        losses = discounted_batch_returns * batch_log_probs
        for j in range(len(losses)):
            losses[j] *= - (hp.gamma ** j)

        loss = losses.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            sys.stdout.write("Iteration: {}, moving average reward: {}\n".format(i, moving_avg_reward))
            sys.stdout.write("policy_mean: {}, action: {}\n".format(p_mean.item(), a_mean.item()))
            moving_avg_reward_pool.append(moving_avg_reward)

    # save training result to csv file, and save the model
    df = pd.DataFrame([moving_avg_reward_pool])
    df.to_csv(hp.csv_out_path, mode='a', header=False)
    torch.save(net.state_dict(), hp.model_save_path)


hyperparameter = Namespace(
    lr=3e-4,
    gamma=0.99,
    num_itr=3000,
    batch_num=1,
    episode_size=300,
    moving_avg_num=300,
    num_state_features=21,
    price_min=200,
    price_max=2000,
    cov_mat=torch.diag(torch.full(size=(1,), fill_value=100.)),
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    model_save_path='./data/reinforce_model.pth',
    csv_out_path='./data/reinforce_out.csv'
)
glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
reinforce(env_train, hyperparameter)
