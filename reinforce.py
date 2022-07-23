import torch
import sys
import torch.optim as optim
import statsmodels.api as sm
import env
import matplotlib.pyplot as plt
from argparse import Namespace
from model_utils import Reinforce
from data_utils import rollout_r_p_a, rollout_with_gradient


# train method
def reinforce(environment, hp):
    # import numpy as np
    # np.random.seed(123)
    # torch.manual_seed(211)

    net = Reinforce(hp.num_state_features, hp.price_min, hp.price_max)
    optimizer = optim.Adam(net.parameters(), lr=hp.lr)

    net = net.to(hp.device)

    moving_avg_reward = 0

    # plot settings:
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("moving average reward with reinforce")
    ax.set_xlabel('iteration')
    ax.set_ylabel('moving average reward')
    (line,) = ax.plot([], [])
    moving_avg_reward_pool = []
    episode_pool = []
    moving_avg_reward_min = float('inf')
    moving_avg_reward_max = float('-inf')

    net.train()
    for i in range(hp.num_itr):
        # generate trajectory
        batch_states, batch_log_probs, batch_returns, p_mean, a_mean, mean_reward = \
            rollout_with_gradient(environment, net, hp, policy_only=True)
        moving_avg_reward += (mean_reward.item() - moving_avg_reward) / (i + 1)

        # compute the return and loss
        losses = batch_returns * batch_log_probs
        for j in range(len(losses)):
            losses[j] *= - (hp.gamma ** j)

        loss = losses.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            sys.stdout.write("Iteration: {}, moving average reward: {}\n".format(i, moving_avg_reward))
            sys.stdout.write("policy_mean: {}, action: {}\n".format(p_mean.item(), a_mean.item()))

            # update plot settings
            moving_avg_reward_min, moving_avg_reward_max = \
                min(moving_avg_reward, moving_avg_reward_min), max(moving_avg_reward, moving_avg_reward_max)
            ax.set(xlim=(-10, i + 10), ylim=(moving_avg_reward_min - 10, moving_avg_reward_max + 10))
            # add data, then plot
            episode_pool.append(i)
            moving_avg_reward_pool.append(moving_avg_reward)
            line.set_data(episode_pool, moving_avg_reward_pool)
            # reserve time to plot the data
            plt.pause(0.2)

    # this make sure the plot won't quit automatically after finish plotting
    plt.show(block=True)
    torch.save(net.state_dict(), hp.model_save_path)


hyperparameter = Namespace(
    lr=3e-4,
    gamma=0.99,
    num_itr=3000,
    batch_num=1,
    episode_size=300,
    num_state_features=21,
    price_min=400,
    price_max=2700,
    cov_mat=torch.diag(torch.full(size=(1,), fill_value=50.)),
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    model_save_path='./data/reinforce_model.pth'
)
glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
reinforce(env_train, hyperparameter)
