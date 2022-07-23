import torch
import sys
import torch.optim as optim
import statsmodels.api as sm
import env
import matplotlib.pyplot as plt
from argparse import Namespace
from model_utils import Reinforce
from torch.distributions import MultivariateNormal
from data_utils import rollout_r_p_a


# train method
def reinforce(environment, hp):
    # import numpy as np
    # np.random.seed(123)
    # torch.manual_seed(211)

    cov_mat = hp.cov_mat.to(hp.device)

    net = Reinforce(hp.num_state_features, hp.price_min, hp.price_max)
    optimizer = optim.Adam(net.parameters(), lr=hp.lr)

    net = net.to(hp.device)

    mean_reward, p_mean, a_mean = rollout_r_p_a(environment, net, hp, policy_only=True)
    moving_avg_reward = mean_reward
    sys.stdout.write("Iteration: {}, moving average reward: {}\n".format(0, moving_avg_reward))
    sys.stdout.write("policy_mean: {}, action: {}\n".format(p_mean, a_mean))
    p_mean, a_mean = 0, 0

    # plot settings:
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("moving average reward with reinforce")
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

    net.train()
    for i in range(hp.num_itr):
        state = torch.from_numpy(environment.reset())
        state = torch.cat(
            (state[:-1], torch.tensor([state[-1] == 0, state[-1] == 1, state[-1] == 2], dtype=torch.float))
        ).to(hp.device)
        log_prob_seq = []
        reward_seq = []

        # generate trajectory
        for step in range(hp.episode_size):
            policy_mean = net.forward(state)

            distro = MultivariateNormal(policy_mean, cov_mat.to(hp.device))
            action = distro.sample()
            log_prob_seq.append(torch.stack([distro.log_prob(action)]))

            # compute reward, go to next state
            reward, new_state = environment.step(action.item())
            reward_seq.append(reward)
            new_state = torch.from_numpy(new_state)
            new_state = torch.cat(
                (new_state[:-1],
                 torch.tensor([new_state[-1] == 0, new_state[-1] == 1, new_state[-1] == 2], dtype=torch.float))
            ).to(hp.device)

            state = new_state
            moving_avg_reward += (reward - moving_avg_reward) / \
                                 (hp.batch_num * hp.episode_size + i * hp.episode_size + step + 1)
            p_mean += policy_mean.item()
            a_mean += action.item()

        # compute the return and loss
        losses = []
        returns = reward_seq.copy()
        for step in reversed(range(hp.episode_size)):
            if step != hp.episode_size - 1:
                returns[step] += hp.gamma * returns[step + 1]
            losses.append(-(hp.gamma ** step) * returns[step] * log_prob_seq[step])

        loss = torch.cat(losses).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        itr = i + 1
        if itr % hp.batch_num == 0:
            sys.stdout.write("Iteration: {}, moving average reward: {}\n".format(itr, moving_avg_reward))
            sys.stdout.write("policy_mean: {}, action: {}\n".format(
                p_mean / (hp.batch_num * hp.episode_size), a_mean / (hp.batch_num * hp.episode_size))
            )
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

    # this make sure the plot won't quit automatically after finish plotting
    plt.show(block=True)
    torch.save(net.state_dict(), hp.model_save_path)


hyperparameter = Namespace(
    lr=3e-4,
    gamma=0.99,
    num_itr=3000,
    batch_num=5,
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
