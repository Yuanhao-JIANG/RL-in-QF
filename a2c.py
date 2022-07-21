import torch
from torch.distributions import MultivariateNormal
import sys
import torch.optim as optim
import statsmodels.api as sm
import env
import matplotlib.pyplot as plt
from model_utils import ActorCritic


# train method
def a2c(environment):
    # import numpy as np
    # np.random.seed(123)
    # torch.manual_seed(211)

    learning_rate = 3e-5
    gamma = 0.99
    num_steps = 300
    max_episodes = 3000
    num_state_features = 21
    price_min = 400
    price_max = 2700
    cov_mat = torch.diag(torch.full(size=(1,), fill_value=50.))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    actor_critic = ActorCritic(num_state_features)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    actor_critic = actor_critic.to(device)
    actor_critic.train()

    moving_avg_reward = 0

    # plot settings:
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("moving average reward with a2c")
    ax.set_xlabel('episode')
    ax.set_ylabel('moving average reward')
    (line,) = ax.plot([], [])
    moving_avg_reward_pool = []
    episode_pool = []
    moving_avg_reward_pool_lim = None
    ax.set(xlim=(0, 1))

    for episode in range(max_episodes):
        state = torch.from_numpy(environment.reset())
        state = torch.cat(
            (state[:-1], torch.tensor([state[-1] == 0, state[-1] == 1, state[-1] == 2], dtype=torch.float))
        ).to(device)

        p_mean, a_mean = 0, 0
        for step in range(num_steps):
            value, policy_mean = actor_critic.forward(state)

            distro = MultivariateNormal(policy_mean, cov_mat.to(device))
            action = distro.sample()
            log_prob = distro.log_prob(action)

            # compute reward, go to next state to compute v'
            reward, new_state = environment.step(action.item())
            new_state = torch.from_numpy(new_state)
            new_state = torch.cat(
                (new_state[:-1],
                 torch.tensor([new_state[-1] == 0, new_state[-1] == 1, new_state[-1] == 2], dtype=torch.float))
            ).to(device)
            value_next, _ = actor_critic.forward(new_state)

            advantage = (reward + gamma * value_next[0] - value[0]).detach()
            critic_loss = - advantage * value[0]
            actor_loss = - (gamma ** step) * advantage * log_prob
            ac_loss = actor_loss + critic_loss

            ac_optimizer.zero_grad()
            ac_loss.backward()
            ac_optimizer.step()

            state = new_state
            moving_avg_reward += (reward - moving_avg_reward) / (episode * num_steps + step + 1)
            p_mean += policy_mean.item()
            a_mean += action.item()

        if episode % 5 == 0:
            sys.stdout.write("Episode: {}, moving average reward: {}\n".format(episode, moving_avg_reward))
            sys.stdout.write("mean: {}, action: {}\n".format(p_mean/num_steps, a_mean/num_steps))

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

    # this make sure the plot won't quit automatically after finish plotting
    plt.show(block=True)

    torch.save(actor_critic.state_dict(), './data/a2c_model.pth')


glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
a2c(env_train)
