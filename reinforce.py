import torch
import sys
import torch.optim as optim
import statsmodels.api as sm
import env
import matplotlib.pyplot as plt
from model_utils import Reinforce
from torch.distributions import Categorical


# train method
def reinforce(environment):
    # np.random.seed(123)
    # torch.manual_seed(211)

    learning_rate = 1e-3
    gamma = 0.99
    num_steps = 300
    max_episodes = 3000
    num_state_features = 21
    price_low = 400
    price_high = 2700
    price_step = 20
    num_actions = int((price_high - price_low) / price_step)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = Reinforce(num_state_features, num_actions)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    net = net.to(device)
    net.train()

    moving_avg_reward = 0

    # plot settings:
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("moving average reward with reinforce")
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
        log_prob_seq = []
        reward_seq = []

        # generate trajectory
        for step in range(num_steps):
            policy_distro = net.forward(state)

            distro = Categorical(policy_distro)
            action = distro.sample()
            c = action.item() * price_step + price_low
            log_prob_seq.append(torch.stack([distro.log_prob(action)]))

            # compute reward, go to next state
            reward, new_state = environment.step(c)
            reward_seq.append(reward)
            new_state = torch.from_numpy(new_state)
            new_state = torch.cat(
                (new_state[:-1],
                 torch.tensor([new_state[-1] == 0, new_state[-1] == 1, new_state[-1] == 2], dtype=torch.float))
            ).to(device)

            state = new_state

            moving_avg_reward += (reward - moving_avg_reward) / (episode * num_steps + step + 1)

        # compute the return and loss
        loss = []
        returns = reward_seq.copy()
        for step in reversed(range(num_steps)):
            if step != num_steps - 1:
                returns[step] += gamma * returns[step + 1]
            loss.append(- (gamma ** step) * returns[step] * log_prob_seq[step])

        # update policy model parameters
        for step in range(num_steps):
            optimizer.zero_grad()
            loss[step].backward()
            optimizer.step()

        # loss = torch.cat(loss).sum()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        if episode % 5 == 0:
            sys.stdout.write("Episode: {}, moving average reward: {}\n".format(episode, moving_avg_reward))

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

    torch.save(net.state_dict(), './data/reinforce_model.pth')


glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
reinforce(env_train)
