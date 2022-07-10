import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import statsmodels.api as sm
import env
import matplotlib.pyplot as plt


class LogLayer(nn.Module):
    def __init__(self):
        super(LogLayer, self).__init__()

    def forward(self, t):
        return torch.log(t+torch.abs(torch.min(t)).detach()+1)


class Net(nn.Module):
    def __init__(self, num_state_features, num_actions):
        super(Net, self).__init__()
        self.num_actions = num_actions

        # value
        self.value_net = nn.Sequential(
            nn.Linear(num_state_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # policy
        self.policy_net = nn.Sequential(
            nn.Linear(num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            LogLayer(),     # to make numbers not differ too much
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        state = Variable(state.float())

        value = self.value_net(state)
        policy_distro = self.policy_net(state)

        with torch.no_grad():
            v = self.value_net(state)

        return value, policy_distro, v


# train method
def policy_itr(environment):
    # np.random.seed(123)
    # torch.manual_seed(211)

    learning_rate = 3e-2
    gamma = 0.99
    num_steps = 100
    max_episodes = 2000
    num_state_features = 18
    price_low = 400
    price_high = 2700
    price_step = 20
    num_actions = int((price_high - price_low)/price_step)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = Net(num_state_features, num_actions)
    net_optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    net = net.to(device)
    net.train()

    plt.ion()

    for episode in range(max_episodes):
        state = torch.from_numpy(environment.reset()).to(device)
        total_reward = 0

        for steps in range(num_steps):
            # policy evaluation
            value, policy_distro, _ = net.forward(state)
            value, policy_distro = value.to(device)[0], policy_distro.to(device)
            c = torch.multinomial(policy_distro.cpu(), 1, replacement=True)[0] * price_step + price_low
            # print(policy_distro)
            # exit()

            target = torch.tensor(0.).to(device)
            q = []
            for i in range(len(policy_distro)):
                if policy_distro[i] > 1e-3:
                    r, s_expect = environment.peek_expected(i * price_step + price_low)
                    _, _, v = net.forward(torch.from_numpy(s_expect).to(device))
                    target += policy_distro[i].detach() * (r + gamma * v[0])
                    if steps % 20 == 0:
                        q.append(r + gamma * v[0].cpu())

            loss = (target - value).pow(2)
            if q:
                print(len(q), ', ')
                idx_max = np.argmax(np.array(q))
                loss -= policy_distro[idx_max]

            net_optimizer.zero_grad()
            loss.backward()
            net_optimizer.step()

            # compute reward, and go to next state
            reward, state = environment.step(c.numpy())
            state = torch.from_numpy(state).to(device)
            total_reward += reward

        if episode % 5 == 0:
            sys.stdout.write("Episode: {}, avg_reward: {}\n".format(episode, total_reward/num_steps))
            plt.plot(episode, total_reward / num_steps, 'o')
            plt.pause(0.25)

    torch.save(net.state_dict(), './data/policy_itr__model.pth')


glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
policy_itr(env_train)
